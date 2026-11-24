"""Spectral and colour indices computed with NumPy/OpenCV.

Works with plain RGB photos/screenshots (visible-band proxies) and with
multispectral rasters (Sentinel-2 / Landsat) when NIR and SWIR bands are present.
All functions return float32 arrays in roughly [-1, 1] unless noted.
"""
from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


def to_float01(img: np.ndarray) -> np.ndarray:
    """Scale uint8/uint16/float imagery to float32 in [0, 1] (per-image robust stretch for >8-bit)."""
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        return np.clip(img, 0, 1)
    if img.max() <= 255.0:
        return img / 255.0
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img - lo) / max(hi - lo, EPS), 0, 1)


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b) / (a + b + EPS)).astype(np.float32)


# --- visible-band (RGB) indices -------------------------------------------------

def excess_green(rgb: np.ndarray) -> np.ndarray:
    """ExG on chromatic coordinates: 2g - r - b. Vegetation > ~0.05."""
    s = rgb.sum(axis=2) + EPS
    r, g, b = rgb[..., 0] / s, rgb[..., 1] / s, rgb[..., 2] / s
    return (2 * g - r - b).astype(np.float32)


def vari(rgb: np.ndarray) -> np.ndarray:
    """Visible Atmospherically Resistant Index (G-R)/(G+R-B), clipped to [-1, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.clip((g - r) / (g + r - b + EPS), -1, 1).astype(np.float32)


def redness(rgb: np.ndarray) -> np.ndarray:
    """Iron-oxide / bare-soil redness proxy (R-B)/(R+B)."""
    return _nd(rgb[..., 0], rgb[..., 2])


def local_texture(gray: np.ndarray, k: int = 7) -> np.ndarray:
    """Local standard deviation — water and blooms are smooth, canopy is rough."""
    mu = cv2.blur(gray, (k, k))
    mu2 = cv2.blur(gray * gray, (k, k))
    return np.sqrt(np.clip(mu2 - mu * mu, 0, None)).astype(np.float32)


# --- multispectral indices -------------------------------------------------------

def ndvi(nir, red):
    return _nd(nir, red)


def ndwi(green, nir):
    return _nd(green, nir)


def mndwi(green, swir1):
    return _nd(green, swir1)


def bsi(blue, red, nir, swir1):
    return _nd(swir1 + red, nir + blue)


def ndci(red_edge, red):
    """Normalized Difference Chlorophyll Index — chlorophyll-a / algae in water."""
    return _nd(red_edge, red)


def nbr(nir, swir2):
    return _nd(nir, swir2)


def compute_rgb_indices(rgb01: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor((rgb01 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = rgb01.mean(axis=2)
    return {
        "exg": excess_green(rgb01),
        "vari": vari(rgb01),
        "redness": redness(rgb01),
        "brightness": gray,
        "texture": local_texture(gray),
        "hue": hsv[..., 0],          # 0–180 (OpenCV)
        "sat": hsv[..., 1] / 255.0,
        "val": hsv[..., 2] / 255.0,
    }


def compute_ms_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """bands keys (any subset): blue, green, red, red_edge, nir, swir1, swir2 — reflectance 0–1."""
    out: dict[str, np.ndarray] = {}
    b = {k: v.astype(np.float32) for k, v in bands.items()}
    if {"nir", "red"} <= b.keys():
        out["ndvi"] = ndvi(b["nir"], b["red"])
    if {"green", "nir"} <= b.keys():
        out["ndwi"] = ndwi(b["green"], b["nir"])
    if {"green", "swir1"} <= b.keys():
        out["mndwi"] = mndwi(b["green"], b["swir1"])
    if {"blue", "red", "nir", "swir1"} <= b.keys():
        out["bsi"] = bsi(b["blue"], b["red"], b["nir"], b["swir1"])
    if {"red_edge", "red"} <= b.keys():
        out["ndci"] = ndci(b["red_edge"], b["red"])
    if {"nir", "swir2"} <= b.keys():
        out["nbr"] = nbr(b["nir"], b["swir2"])
    return out
