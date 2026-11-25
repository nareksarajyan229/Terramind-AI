"""Google Earth Engine service: Sentinel-2 indices, forest loss, rainfall anomaly,
surface water and map tiles for any location in Armenia.

Setup (free for non-commercial / research use):
  1. Register a Cloud project for Earth Engine: https://earthengine.google.com/
  2. ``earthengine authenticate``  (or set GEE_SERVICE_ACCOUNT + GEE_KEY_FILE)
  3. Set GEE_PROJECT in .env
Without this the app runs fully on the uploaded-image engine and reference data.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

from .config import settings

try:
    import ee

    EE_IMPORTED = True
except ImportError:
    EE_IMPORTED = False

_STATE = {"ready": False, "error": ""}

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CHIRPS = "UCSB-CHG/CHIRPS/DAILY"
JRC_WATER = "JRC/GSW1_4/GlobalSurfaceWater"
MODIS_NDVI = "MODIS/061/MOD13Q1"
FIRMS = "FIRMS"

VIS = {
    "true_color": {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.2},
    "ndvi": {"min": -0.2, "max": 0.8, "palette": ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"]},
    "ndwi": {"min": -0.5, "max": 0.5, "palette": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]},
    "ndci": {"min": -0.1, "max": 0.3, "palette": ["#08306b", "#2171b5", "#41ab5d", "#d9f0a3", "#ffff00"]},
    "bsi": {"min": -0.3, "max": 0.3, "palette": ["#1a9850", "#fee08b", "#d73027"]},
    "forest_loss": {"min": 0, "max": 1, "palette": ["#ff0000"]},
}


def init() -> bool:
    """Initialise Earth Engine once. Returns True when usable."""
    if _STATE["ready"]:
        return True
    if not EE_IMPORTED:
        _STATE["error"] = "earthengine-api-ն տեղադրված չէ"
        return False
    if not settings.gee_project:
        _STATE["error"] = "GEE_PROJECT-ը նշված չէ"
        return False
    try:
        if settings.gee_service_account and settings.gee_key_file:
            creds = ee.ServiceAccountCredentials(settings.gee_service_account, settings.gee_key_file)
            ee.Initialize(creds, project=settings.gee_project)
        else:
            ee.Initialize(project=settings.gee_project)
        _STATE.update(ready=True, error="")
    except Exception as exc:  # auth missing, project not registered, no network…
        _STATE["error"] = str(exc).splitlines()[0][:200]
    return _STATE["ready"]


def status() -> tuple[bool, str]:
    return _STATE["ready"], _STATE["error"]


# --------------------------------------------------------------------------------
# Image builders
# --------------------------------------------------------------------------------

def _mask_s2(img):
    scl = img.select("SCL")
    good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(good)


def s2_composite(aoi, start: str, end: str, max_cloud: int = 30):
    col = (ee.ImageCollection(S2).filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)).map(_mask_s2))
    return col.median().clip(aoi)


def add_indices(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndci = img.normalizedDifference(["B5", "B4"]).rename("NDCI")
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    bsi = img.expression("((S+R)-(N+B))/((S+R)+(N+B))",
                         {"S": img.select("B11"), "R": img.select("B4"),
                          "N": img.select("B8"), "B": img.select("B2")}).rename("BSI")
    return img.addBands([ndvi, ndwi, mndwi, ndci, nbr, bsi])


def _aoi(lat: float, lon: float, radius_m: float):
    return ee.Geometry.Point([lon, lat]).buffer(radius_m)


def _season(year: int | None = None) -> tuple[str, str]:
    """Peak growing season (June–September) of the latest complete year."""
    today = dt.date.today()
    y = year or (today.year if today.month >= 10 else today.year - 1)
    return f"{y}-06-01", f"{y}-09-30"


# --------------------------------------------------------------------------------
# Location analysis
# --------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def location_stats(lat: float, lon: float, radius_m: float = 3000, year: int | None = None) -> dict:
    """Return mean indices and change metrics for a circle around (lat, lon)."""
    aoi = _aoi(lat, lon, radius_m)
    start, end = _season(year)
    y = int(start[:4])
    cur = add_indices(s2_composite(aoi, start, end))
    base = add_indices(s2_composite(aoi, f"{y - 4}-06-01", f"{y - 4}-09-30"))

    water = cur.select("MNDWI").gt(0.05)
    land = water.Not()
    stats = {}
    red = ee.Reducer.mean()
    kw = dict(geometry=aoi, scale=20, maxPixels=1e9, bestEffort=True)
    stats.update(cur.select(["NDVI", "BSI", "NBR"]).updateMask(land).reduceRegion(red, **kw).getInfo())
    w = cur.select(["NDCI", "MNDWI"]).updateMask(water).reduceRegion(red, **kw).getInfo()
    stats["NDCI_water"] = w.get("NDCI")
    stats["water_fraction"] = water.reduceRegion(red, **kw).getInfo().get("MNDWI")
    stats["NDVI_baseline"] = base.select("NDVI").updateMask(land).reduceRegion(red, **kw).getInfo().get("NDVI")
    stats["water_fraction_baseline"] = base.select("MNDWI").gt(0.05).reduceRegion(red, **kw).getInfo().get("MNDWI")

    # Forest loss (Hansen GFC): share of 2000 forest lost
    gfc = ee.Image(settings.hansen_asset).clip(aoi)
    forest2000 = gfc.select("treecover2000").gt(30)
    loss = gfc.select("loss").And(forest2000)
    area = ee.Image.pixelArea()
    fa = forest2000.multiply(area).reduceRegion(ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9).getInfo()
    la = loss.multiply(area).reduceRegion(ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9).getInfo()
    f_ha = (list(fa.values())[0] or 0) / 1e4
    l_ha = (list(la.values())[0] or 0) / 1e4
    stats["forest_2000_ha"] = round(f_ha, 1)
    stats["forest_loss_ha"] = round(l_ha, 1)

    # Rainfall anomaly for the 12 months before the season end vs. 2001–2020 mean
    endd = ee.Date(end)
    chirps = ee.ImageCollection(CHIRPS)
    recent = chirps.filterDate(endd.advance(-1, "year"), endd).sum()
    clim = ee.ImageCollection([chirps.filterDate(f"{yy}-10-01", f"{yy + 1}-09-30").sum()
                               for yy in range(2001, 2020)]).mean()
    pr = recent.reduceRegion(red, geometry=aoi, scale=5000, bestEffort=True).getInfo().get("precipitation")
    pc = clim.reduceRegion(red, geometry=aoi, scale=5000, bestEffort=True).getInfo().get("precipitation")
    stats["rain_12m_mm"] = pr
    stats["rain_clim_mm"] = pc
    stats["rain_anomaly_pct"] = None if not (pr and pc) else round((pr / pc - 1) * 100, 1)
    stats["period"] = f"{start} → {end}"
    stats["radius_m"] = radius_m
    return stats


def issues_from_stats(s: dict) -> dict:
    """Convert GEE statistics into the same issue structure produced by the vision engine."""
    def clamp(x):
        return max(0.0, min(100.0, x))

    ndvi = s.get("NDVI") or 0
    ndvi0 = s.get("NDVI_baseline") or ndvi
    bsi = s.get("BSI") or 0
    ndci = s.get("NDCI_water")
    wf = s.get("water_fraction") or 0
    wf0 = s.get("water_fraction_baseline") or wf
    rain = s.get("rain_anomaly_pct")
    nbr = s.get("NBR") or 0.3
    f2000 = s.get("forest_2000_ha") or 0
    loss = s.get("forest_loss_ha") or 0

    sev = {
        "deforestation": clamp((loss / f2000 * 400) if f2000 > 5 else 0),
        "water_pollution": clamp(((ndci or 0) - 0.02) * 400) if wf > 0.01 and ndci is not None else 0.0,
        "drought": clamp(max(0, (ndvi0 - ndvi)) * 400 + (max(0, -(rain or 0)) * 1.5)),
        "soil_degradation": clamp((bsi + 0.05) * 250 + max(0, 0.3 - ndvi) * 120),
        "mining": 0.0,  # not separable from indices alone — use imagery / YOLO
        "wildfire": clamp((0.1 - nbr) * 300) if ndvi < 0.25 else 0.0,
    }
    if wf0 and wf0 > 0.02 and wf < wf0 * 0.8:
        sev["drought"] = clamp(sev["drought"] + (1 - wf / wf0) * 80)

    labels = {
        "deforestation": f"Անտառի կորուստ՝ {loss:.0f} հա {f2000:.0f} հա-ից (2000 թ.-ից)",
        "water_pollution": f"Քլորոֆիլի ինդեքսը (NDCI) ջրում՝ {ndci:.3f}" if ndci is not None else "Ջրավազան չկա",
        "drought": f"NDVI {ndvi:.2f}՝ չորս տարի առաջվա {ndvi0:.2f}-ի համեմատ, տեղումների շեղում՝ "
                   f"{'անհայտ' if rain is None else f'{rain:+.0f}%'}",
        "soil_degradation": f"Մերկ հողի ինդեքս՝ {bsi:.3f}",
        "mining": "Ինդեքսներով չի գնահատվում",
        "wildfire": f"Այրվածության ինդեքս (NBR)՝ {nbr:.2f}",
    }
    from .vision import ISSUES

    return {k: {"label": ISSUES[k]["label"], "detected": v >= 15, "severity": round(v, 1),
                "evidence": labels[k]} for k, v in sev.items()}


@lru_cache(maxsize=16)
def ndvi_timeseries(lat: float, lon: float, radius_m: float = 3000, years: int = 6) -> list[dict]:
    """Monthly MODIS NDVI (250 m) for the last N years."""
    aoi = _aoi(lat, lon, radius_m)
    end = dt.date.today().replace(day=1)
    start = end.replace(year=end.year - years)
    col = ee.ImageCollection(MODIS_NDVI).filterDate(str(start), str(end)).select("NDVI")

    def per_image(img):
        v = img.reduceRegion(ee.Reducer.mean(), aoi, 250).get("NDVI")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "ndvi": v})

    feats = col.map(per_image).filter(ee.Filter.notNull(["ndvi"])).getInfo()["features"]
    return [{"date": f["properties"]["date"], "ndvi": f["properties"]["ndvi"] / 10000} for f in feats]


def tile_layer(layer: str, lat: float, lon: float, radius_m: float = 15000, year: int | None = None) -> str:
    """Return an XYZ tile URL for folium."""
    aoi = _aoi(lat, lon, radius_m)
    start, end = _season(year)
    if layer == "forest_loss":
        img = ee.Image(settings.hansen_asset).select("loss").selfMask().clip(aoi)
    else:
        comp = add_indices(s2_composite(aoi, start, end))
        img = comp if layer == "true_color" else comp.select(layer.upper())
        if layer == "ndci":
            img = img.updateMask(comp.select("MNDWI").gt(0.05))
    return img.getMapId(VIS[layer])["tile_fetcher"].url_format


def thumbnail_png(lat: float, lon: float, radius_m: float = 3000, year: int | None = None,
                  dims: int = 768) -> bytes:
    """True-colour Sentinel-2 PNG of the area — fed into the vision engine."""
    import urllib.request

    aoi = _aoi(lat, lon, radius_m)
    start, end = _season(year)
    img = s2_composite(aoi, start, end).visualize(**VIS["true_color"])
    url = img.getThumbURL({"region": aoi.bounds(), "dimensions": dims, "format": "png"})
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 — Earth Engine URL
        return r.read()
