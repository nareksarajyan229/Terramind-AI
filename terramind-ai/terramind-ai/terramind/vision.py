"""Computer-vision pipeline: land-cover classification, issue detection, overlays.

Engines (chosen automatically, best available first):
  1. ``segmentation`` — PyTorch U-Net (terramind.segmentation) when trained weights exist.
  2. ``yolo``         — Ultralytics YOLO object detector for discrete features
                        (clearings, tailings ponds, dumps, gullies, blooms, burn scars).
  3. ``spectral``     — explainable OpenCV/NumPy rules on spectral indices. Always available,
                        used on its own or as a fallback/complement to the models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import indices as ix

# Order defines the integer label in class maps (shared with the U-Net).
CLASSES: list[tuple[str, str, tuple[int, int, int]]] = [
    ("healthy_vegetation", "Healthy vegetation", (46, 125, 50)),
    ("stressed_vegetation", "Stressed / dry vegetation", (192, 202, 51)),
    ("bare_soil", "Bare / degraded soil", (161, 120, 90)),
    ("clean_water", "Clear water", (30, 136, 229)),
    ("turbid_water", "Turbid / polluted water", (121, 85, 72)),
    ("algal_bloom", "Algal bloom", (0, 230, 160)),
    ("burned_dark", "Burned / dark surface", (38, 50, 56)),
    ("built_mining", "Built-up / mining / rock", (189, 189, 189)),
]
CLASS_KEYS = [c[0] for c in CLASSES]
CLASS_LABELS = {c[0]: c[1] for c in CLASSES}
CLASS_COLORS = {c[0]: c[2] for c in CLASSES}
C = {k: i for i, k in enumerate(CLASS_KEYS)}

ISSUES = {
    "deforestation": {"label": "Deforestation / forest clearing", "color": (255, 82, 82)},
    "water_pollution": {"label": "Water pollution / eutrophication", "color": (0, 176, 255)},
    "drought": {"label": "Drought stress", "color": (255, 193, 7)},
    "soil_degradation": {"label": "Soil degradation / erosion", "color": (255, 112, 67)},
    "mining": {"label": "Mining / tailings / exposed rock", "color": (224, 64, 251)},
    "wildfire": {"label": "Burn scar / fire damage", "color": (255, 61, 0)},
}

MAX_SIDE = 1024


@dataclass
class Region:
    issue: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    area_ha: float | None
    confidence: float
    source: str = "spectral"

    def to_dict(self) -> dict:
        return {"issue": self.issue, "bbox": list(self.bbox), "area_px": self.area_px,
                "area_ha": None if self.area_ha is None else round(self.area_ha, 2),
                "confidence": round(self.confidence, 3), "source": self.source}


@dataclass
class AnalysisResult:
    image: np.ndarray                 # RGB uint8 (resized working copy)
    class_map: np.ndarray             # HxW int labels
    fractions: dict[str, float]
    metrics: dict[str, float]
    issues: dict[str, dict[str, Any]]
    regions: list[Region]
    engine: str
    gsd_m: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def total_area_ha(self) -> float | None:
        if not self.gsd_m:
            return None
        h, w = self.class_map.shape
        return h * w * self.gsd_m ** 2 / 10_000

    def summary(self) -> dict:
        return {
            "engine": self.engine,
            "image_size": list(self.class_map.shape[::-1]),
            "gsd_m": self.gsd_m,
            "total_area_ha": None if self.total_area_ha is None else round(self.total_area_ha, 2),
            "fractions": {k: round(v, 4) for k, v in self.fractions.items()},
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "issues": {k: {kk: vv for kk, vv in v.items() if kk != "mask"} for k, v in self.issues.items()},
            "regions": [r.to_dict() for r in self.regions],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------

def load_image(data: bytes | np.ndarray, filename: str = "") -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """Return (rgb_uint8, multispectral_bands_or_None). Supports JPG/PNG and GeoTIFFs.

    For GeoTIFFs with >=4 bands, Sentinel-2 order is assumed when 10+ bands
    (B1..B12) and B,G,R,NIR order when 4 bands. Requires rasterio for GeoTIFF.
    """
    if isinstance(data, np.ndarray):
        arr = data
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return _to_uint8(arr[..., :3]), None

    name = filename.lower()
    if name.endswith((".tif", ".tiff")):
        try:
            import rasterio  # optional
            from rasterio.io import MemoryFile

            with MemoryFile(data) as mem, mem.open() as ds:
                stack = ds.read().astype(np.float32)  # (bands, H, W)
            bands = _bands_from_stack(stack)
            if bands is not None:
                rgb = np.dstack([bands["red"], bands["green"], bands["blue"]])
                return _to_uint8(ix.to_float01(rgb)), bands
            arr = np.moveaxis(stack[:3], 0, -1)
            return _to_uint8(ix.to_float01(arr)), None
        except ImportError:
            pass  # fall through to OpenCV, which reads 8/16-bit RGB TIFFs

    buf = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Could not decode image. Use JPG, PNG or GeoTIFF.")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.shape[2] == 4:
        bgr = bgr[..., :3]
    rgb = cv2.cvtColor(_to_uint8(bgr), cv2.COLOR_BGR2RGB)
    return rgb, None


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return (ix.to_float01(arr) * 255).astype(np.uint8)


def _bands_from_stack(stack: np.ndarray) -> dict[str, np.ndarray] | None:
    n = stack.shape[0]
    scale = 10000.0 if stack.max() > 2.0 else 1.0
    s = stack / scale
    if n >= 12:  # Sentinel-2 L2A full stack B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12 (12) or with B10 (13)
        off = 1 if n == 13 else 0
        return {"blue": s[1], "green": s[2], "red": s[3], "red_edge": s[4], "nir": s[7],
                "swir1": s[10 + off], "swir2": s[11 + off]}
    if n == 4:
        return {"blue": s[0], "green": s[1], "red": s[2], "nir": s[3]}
    return None


def _resize(rgb: np.ndarray, bands: dict | None, max_side: int = MAX_SIDE):
    h, w = rgb.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f >= 1.0:
        return rgb, bands, 1.0
    size = (int(w * f), int(h * f))
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    if bands:
        bands = {k: cv2.resize(v, size, interpolation=cv2.INTER_AREA) for k, v in bands.items()}
    return rgb, bands, f


# --------------------------------------------------------------------------------
# Spectral classifier
# --------------------------------------------------------------------------------

def classify_rgb(rgb01: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """Rule-based land-cover map from visible bands. sensitivity>1 flags more problems."""
    f = ix.compute_rgb_indices(rgb01)
    h, s, v, tex, exg, vr = f["hue"], f["sat"], f["val"], f["texture"], f["exg"], f["vari"]
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]

    smooth = tex < 0.03 * sensitivity
    very_smooth = tex < 0.018 * sensitivity

    blue_water = smooth & (h >= 85) & (h <= 135) & (s > 0.12) & (b >= r)
    dark_water = very_smooth & (v < 0.28) & (b >= r * 0.95)
    water = blue_water | dark_water
    near_water = cv2.dilate(water.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)

    green_like = (h >= 30) & (h <= 95) & (exg > 0.03)
    green_smooth = green_like & very_smooth & (s > 0.18)

    # Algal bloom: smooth green surface either touching water or forming a large uniform sheet.
    algal = np.zeros_like(water)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(green_smooth.astype(np.uint8), 8)
    total = rgb01.shape[0] * rgb01.shape[1]
    for i in range(1, n):
        comp = lbl == i
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.0005 * total:
            continue
        if (comp & near_water).sum() > 0.2 * area or area > 0.03 * total:
            algal |= comp

    built = (s < 0.14) & (v > 0.5) & ~water
    near_built = cv2.dilate(built.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
    turbid = smooth & (h >= 5) & (h < 35) & (s > 0.2) & near_water & ~algal
    turbid |= very_smooth & (h >= 3) & (h < 22) & (s > 0.5) & near_built  # acid/iron-stained drainage

    veg = green_like & ~algal
    healthy = veg & ((exg > 0.10 / sensitivity) | (vr > 0.18 / sensitivity))
    stressed = (veg & ~healthy) | ((h >= 18) & (h < 32) & (s > 0.25) & (exg > -0.02) & ~near_water & ~smooth)

    burned = (v < 0.22) & ~water & (s < 0.4)

    cmap = np.full(rgb01.shape[:2], C["bare_soil"], np.uint8)  # default: bare / degraded
    cmap[built] = C["built_mining"]
    cmap[burned] = C["burned_dark"]
    cmap[stressed] = C["stressed_vegetation"]
    cmap[healthy] = C["healthy_vegetation"]
    cmap[water] = C["clean_water"]
    cmap[turbid] = C["turbid_water"]
    cmap[algal] = C["algal_bloom"]
    return _smooth_labels(cmap)


def classify_multispectral(bands: dict[str, np.ndarray], sensitivity: float = 1.0) -> np.ndarray:
    """Land-cover map from true spectral indices (Sentinel-2/Landsat reflectance)."""
    ind = ix.compute_ms_indices(bands)
    shape = next(iter(bands.values())).shape
    nd = ind.get("ndvi", np.zeros(shape, np.float32))
    water_idx = ind.get("mndwi", ind.get("ndwi", np.full(shape, -1, np.float32)))
    bs = ind.get("bsi", -nd)
    chl = ind.get("ndci", np.zeros(shape, np.float32))
    burn = ind.get("nbr", np.ones(shape, np.float32))
    nir = bands.get("nir", np.zeros(shape, np.float32))

    water = water_idx > 0.05
    algal = water & (chl > 0.10 / sensitivity)
    turbid = water & ~algal & (bands["red"] > bands["green"] * 0.95) & (bands["red"] > 0.06)
    healthy = ~water & (nd > 0.45 / sensitivity)
    stressed = ~water & ~healthy & (nd > 0.2)
    burned = ~water & (burn < -0.05 * sensitivity) & (nd < 0.2)
    built = ~water & (nd < 0.15) & (bs < 0.0) & (nir > 0.2)

    cmap = np.full(shape, C["bare_soil"], np.uint8)
    cmap[built] = C["built_mining"]
    cmap[stressed] = C["stressed_vegetation"]
    cmap[healthy] = C["healthy_vegetation"]
    cmap[burned] = C["burned_dark"]
    cmap[water] = C["clean_water"]
    cmap[turbid] = C["turbid_water"]
    cmap[algal] = C["algal_bloom"]
    return _smooth_labels(cmap)


def _smooth_labels(cmap: np.ndarray, k: int = 5) -> np.ndarray:
    """Majority-like smoothing via median filter to remove salt-and-pepper noise."""
    return cv2.medianBlur(cmap, k)


# --------------------------------------------------------------------------------
# Issue extraction
# --------------------------------------------------------------------------------

def _regions_from_mask(mask: np.ndarray, issue: str, min_area: int, gsd: float | None,
                       conf_map: np.ndarray | None = None) -> list[Region]:
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    out: list[Region] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        conf = float(np.clip(conf_map[lbl == i].mean(), 0, 1)) if conf_map is not None else 0.6
        out.append(Region(issue, (int(x), int(y), int(w), int(h)), int(area),
                          None if gsd is None else area * gsd ** 2 / 10_000, conf))
    out.sort(key=lambda r: r.area_px, reverse=True)
    return out


def extract_issues(cmap: np.ndarray, rgb01: np.ndarray, gsd: float | None,
                   min_region_frac: float = 0.002) -> tuple[dict, list[Region], dict[str, float], dict[str, float]]:
    total = cmap.size
    frac = {k: float((cmap == C[k]).sum() / total) for k in CLASS_KEYS}
    m = {k: cmap == C[k] for k in CLASS_KEYS}
    min_area = max(30, int(min_region_frac * total))

    veg = m["healthy_vegetation"] | m["stressed_vegetation"]
    nonveg_land = m["bare_soil"] | m["burned_dark"] | m["built_mining"]
    veg_density = cv2.blur(veg.astype(np.float32), (61, 61))
    water_all = m["clean_water"] | m["turbid_water"] | m["algal_bloom"]

    gray = (rgb01.mean(axis=2) * 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 150).astype(np.float32) / 255
    edge_density = cv2.blur(edges, (21, 21))

    clearing = _clearings(nonveg_land, m["healthy_vegetation"], min_area)
    masks = {
        "deforestation": clearing,
        "water_pollution": m["turbid_water"] | m["algal_bloom"],
        "drought": (m["stressed_vegetation"] | (m["bare_soil"] & (veg_density < 0.35))) & ~clearing,
        "soil_degradation": m["bare_soil"] & ~clearing & ((edge_density > 0.08) | (veg_density < 0.15)),
        "mining": m["built_mining"],
        "wildfire": m["burned_dark"],
    }
    conf_maps = {
        "deforestation": np.clip(0.4 + veg_density * 0.6, 0, 1),
        "water_pollution": np.full(cmap.shape, 0.75, np.float32),
        "drought": 1 - veg_density * 0.5,
        "soil_degradation": np.clip(0.45 + edge_density * 3, 0, 1),
        "mining": np.full(cmap.shape, 0.55, np.float32),
        "wildfire": np.full(cmap.shape, 0.6, np.float32),
    }

    veg_total = frac["healthy_vegetation"] + frac["stressed_vegetation"]
    water_total = frac["clean_water"] + frac["turbid_water"] + frac["algal_bloom"]
    land_total = max(1e-6, 1 - water_total)
    metrics = {
        "vegetation_cover": veg_total,
        "vegetation_health": frac["healthy_vegetation"] / veg_total if veg_total > 0 else 0.0,
        "water_cover": water_total,
        "water_quality": frac["clean_water"] / water_total if water_total > 0 else 1.0,
        "bare_soil_share": frac["bare_soil"] / land_total,
        "edge_density_bare": float(edge_density[m["bare_soil"]].mean()) if m["bare_soil"].any() else 0.0,
        "mean_exg": float(ix.excess_green(rgb01)[veg].mean()) if veg.any() else 0.0,
    }

    # Severity 0–100 per issue, from area share and context.
    def sev(x: float) -> float:
        return float(np.clip(x, 0, 1) * 100)

    land_mask_frac = {k: float(v.sum() / total) for k, v in masks.items()}
    severity = {
        "deforestation": sev(land_mask_frac["deforestation"] / max(frac["healthy_vegetation"] + land_mask_frac["deforestation"], 1e-6) * 3.0),
        "water_pollution": max(
            sev(np.sqrt(1 - metrics["water_quality"]) * 1.2) if water_total > 0.01 else 0.0,
            sev(frac["turbid_water"] / land_total * 25) if water_total <= 0.02 else 0.0,  # seeps / small plumes
        ),
        "drought": sev(land_mask_frac["drought"] / land_total * 0.9),
        "soil_degradation": sev(land_mask_frac["soil_degradation"] / land_total * 1.6 + metrics["edge_density_bare"]),
        "mining": sev(land_mask_frac["mining"] / land_total * 3.0),
        "wildfire": sev(land_mask_frac["wildfire"] / land_total * 3.0),
    }

    regions: list[Region] = []
    issues: dict[str, dict] = {}
    for key, mask in masks.items():
        regs = _regions_from_mask(mask, key, min_area, gsd, conf_maps[key])
        regions.extend(regs[:25])
        area_frac = land_mask_frac[key]
        detected = severity[key] >= 15 and len(regs) > 0
        issues[key] = {
            "label": ISSUES[key]["label"],
            "detected": bool(detected),
            "severity": round(severity[key], 1),
            "area_fraction": round(area_frac, 4),
            "area_ha": None if gsd is None else round(area_frac * total * gsd ** 2 / 10_000, 2),
            "n_regions": len(regs),
            "confidence": round(float(np.mean([r.confidence for r in regs])) if regs else 0.0, 3),
            "mask": mask,
        }
    return issues, regions, frac, metrics


def _clearings(nonveg: np.ndarray, healthy: np.ndarray, min_area: int, ring_px: int = 21,
               min_ring_veg: float = 0.55) -> np.ndarray:
    """Non-vegetated patches surrounded mostly by healthy canopy = probable forest clearings."""
    out = np.zeros(nonveg.shape, bool)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(nonveg.astype(np.uint8), 8)
    kernel = np.ones((ring_px, ring_px), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        pad = ring_px
        y0, y1, x0, x1 = max(0, y - pad), y + h + pad, max(0, x - pad), x + w + pad
        comp = (lbl[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(comp, kernel).astype(bool) & ~comp.astype(bool)
        if ring.any() and healthy[y0:y1, x0:x1][ring].mean() >= min_ring_veg:
            out[y0:y1, x0:x1] |= comp.astype(bool)
    return out


# --------------------------------------------------------------------------------
# Overlay rendering
# --------------------------------------------------------------------------------

def colorize(cmap: np.ndarray) -> np.ndarray:
    lut = np.array([c[2] for c in CLASSES], np.uint8)
    return lut[cmap]


def render_overlay(rgb: np.ndarray, cmap: np.ndarray, regions: list[Region],
                   alpha: float = 0.45, show_boxes: bool = True, max_boxes: int = 15,
                   issues: dict | None = None) -> np.ndarray:
    """Blend the class map over the image and box the largest regions of detected issues."""
    out = cv2.addWeighted(rgb, 1 - alpha, colorize(cmap), alpha, 0)
    if not show_boxes:
        return out
    if issues is not None:
        regions = [r for r in regions if issues.get(r.issue, {}).get("detected") or r.source != "spectral"]
    scale = max(rgb.shape[:2]) / 900
    for reg in sorted(regions, key=lambda r: r.area_px, reverse=True)[:max_boxes]:
        x, y, w, h = reg.bbox
        col = ISSUES[reg.issue]["color"]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, max(1, int(2 * scale)))
        label = f"{reg.issue.replace('_', ' ')} {reg.confidence:.0%}"
        fs = 0.45 * max(scale, 0.7)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(y, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty), col, -1)
        cv2.putText(out, label, (x + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (15, 15, 15), 1, cv2.LINE_AA)
    return out


def issue_heatmap(rgb: np.ndarray, issues: dict) -> np.ndarray:
    """Combined severity heat map of all detected issues."""
    acc = np.zeros(rgb.shape[:2], np.float32)
    for v in issues.values():
        if v.get("detected"):
            acc += v["mask"].astype(np.float32) * (v["severity"] / 100)
    acc = cv2.GaussianBlur(acc, (0, 0), 6)
    acc = (np.clip(acc / max(acc.max(), 1e-6), 0, 1) * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(acc, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.5, heat, 0.5, 0)


# --------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------

def analyze(rgb: np.ndarray, bands: dict | None = None, *, gsd_m: float | None = 10.0,
            sensitivity: float = 1.0, engine: str = "auto",
            segmenter=None, detector=None) -> AnalysisResult:
    """Run the full pipeline.

    engine: "auto" | "spectral" | "segmentation" | "yolo"
    segmenter: object with .predict(rgb_uint8) -> class_map (see segmentation.py)
    detector:  object with .detect(rgb_uint8) -> list[Region]  (see detector.py)
    """
    notes: list[str] = []
    rgb, bands, f = _resize(rgb, bands)
    if f < 1.0 and gsd_m:
        gsd_m = gsd_m / f
        notes.append(f"Image downscaled ×{f:.2f} for processing; effective resolution {gsd_m:.1f} m/px.")
    rgb01 = rgb.astype(np.float32) / 255.0

    used = "spectral"
    cmap = None
    if engine in ("auto", "segmentation") and segmenter is not None and segmenter.ready:
        try:
            cmap = segmenter.predict(rgb)
            used = "segmentation (U-Net)"
        except Exception as exc:  # pragma: no cover
            notes.append(f"Segmentation model failed ({exc}); used spectral rules.")
    elif engine == "segmentation":
        notes.append("No trained segmentation weights found; used spectral rules.")

    if cmap is None:
        if bands and {"nir", "red"} <= bands.keys():
            cmap = classify_multispectral(bands, sensitivity)
            used = "spectral (multispectral indices)"
        else:
            cmap = classify_rgb(rgb01, sensitivity)
            used = "spectral (visible-band indices)"
            notes.append("RGB-only input: water/vegetation are estimated from visible-band proxies. "
                         "Upload a Sentinel-2 GeoTIFF for NDVI/NDWI/NDCI-based results.")

    issues, regions, frac, metrics = extract_issues(cmap, rgb01, gsd_m)

    if engine in ("auto", "yolo") and detector is not None and detector.ready:
        try:
            dets = detector.detect(rgb, gsd_m)
            regions.extend(dets)
            for d in dets:
                it = issues.get(d.issue)
                if it is not None:
                    it["detected"] = True
                    it["n_regions"] += 1
                    it["severity"] = round(max(it["severity"], 40 + 40 * d.confidence), 1)
            used += " + YOLO"
        except Exception as exc:  # pragma: no cover
            notes.append(f"YOLO detector failed ({exc}).")
    elif engine == "yolo":
        notes.append("No trained YOLO weights found; showing spectral regions only.")

    return AnalysisResult(rgb, cmap, frac, metrics, issues, regions, used, gsd_m, notes)
