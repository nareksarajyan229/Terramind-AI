"""Generate synthetic satellite-like demo tiles for trying TerraMind AI offline.

These are illustrations, not real imagery. For real data, export Sentinel-2 tiles from
the Interactive Map page (Google Earth Engine) or from Copernicus Browser.

    python scripts/generate_samples.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "assets" / "samples"
RNG = np.random.default_rng(42)
H = W = 768


def noise(scale: int, amp: float) -> np.ndarray:
    small = RNG.normal(0, 1, (H // scale + 2, W // scale + 2)).astype(np.float32)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC) * amp


def fill(color, tex_amp=12.0, grain=6.0) -> np.ndarray:
    base = np.ones((H, W, 3), np.float32) * np.array(color, np.float32)
    t = noise(8, tex_amp)[..., None] + noise(2, grain)[..., None]
    return base + t


def blob_mask(cx, cy, r, irregular=0.35, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rad = r * (1 + irregular * rng.uniform(-1, 1, ang.size))
    pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return cv2.GaussianBlur(m.astype(np.float32), (9, 9), 0)


def compose(base, layer, mask):
    return base * (1 - mask[..., None]) + layer * mask[..., None]


def save(name, img):
    OUT.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("wrote", OUT / name)


def forest_clearing():
    img = fill((34, 78, 36), tex_amp=18, grain=14)  # rough canopy
    for i, (cx, cy, r) in enumerate([(220, 260, 70), (520, 420, 95), (380, 620, 50), (610, 150, 40)]):
        img = compose(img, fill((150, 112, 78), 8, 5), blob_mask(cx, cy, r, 0.4, i))
    road = np.zeros((H, W), np.uint8)
    cv2.polylines(road, [np.array([[0, 300], [220, 260], [420, 380], [520, 420], [768, 520]])], False, 1, 9)
    img = compose(img, fill((170, 140, 105), 4, 3), cv2.GaussianBlur(road.astype(np.float32), (5, 5), 0))
    save("forest_clearing_tavush.png", img)


def lake_bloom():
    land = fill((176, 158, 104), 14, 8)  # dry grassland
    lake = blob_mask(400, 400, 290, 0.18, 7)
    water = np.ones((H, W, 3), np.float32) * np.array((28, 66, 120), np.float32) + noise(16, 2)[..., None]
    img = compose(land, water, lake)
    bloom = blob_mask(250, 300, 120, 0.5, 3) * lake
    bloom_col = np.ones((H, W, 3), np.float32) * np.array((52, 140, 92), np.float32) + noise(24, 3)[..., None]
    img = compose(img, bloom_col, bloom * 0.95)
    plume = blob_mask(620, 560, 70, 0.6, 5) * lake
    img = compose(img, np.ones((H, W, 3), np.float32) * np.array((150, 112, 66), np.float32), plume * 0.9)
    save("lake_sevan_bloom.png", img)


def drought_farmland():
    img = fill((182, 160, 112), 10, 6)
    for r in range(0, H, 96):
        for c in range(0, W, 128):
            k = RNG.random()
            col = (58, 120, 48) if k < 0.25 else (160, 150, 70) if k < 0.6 else (168, 128, 92)
            img[r + 3:r + 93, c + 3:c + 125] = fill(col, 6, 5)[r + 3:r + 93, c + 3:c + 125]
    gul = np.zeros((H, W), np.uint8)
    for _ in range(9):
        x, y = RNG.integers(0, W), RNG.integers(0, H)
        pts = [(x, y)]
        for _ in range(12):
            x += int(RNG.integers(-20, 20)); y += int(RNG.integers(10, 30)); pts.append((x, y))
        cv2.polylines(gul, [np.array(pts)], False, 1, 3)
    img = compose(img, fill((96, 74, 56), 3, 3), gul.astype(np.float32))
    save("ararat_drought_farmland.png", img)


def mining_tailings():
    img = fill((126, 128, 82), 14, 10)  # sparse vegetation
    pit = blob_mask(260, 300, 150, 0.25, 11)
    img = compose(img, fill((175, 170, 165), 10, 12), pit)
    for r, c in [(110, 200), (75, 160)]:
        ring = blob_mask(260, 300, r, 0.1, 2) - blob_mask(260, 300, c, 0.1, 2)
        img = compose(img, fill((140, 135, 130), 6, 6), np.clip(ring, 0, 1) * 0.6)
    pond = blob_mask(560, 520, 130, 0.2, 13)
    img = compose(img, np.ones((H, W, 3), np.float32) * np.array((214, 214, 206), np.float32) + noise(20, 2)[..., None], pond)
    seep = blob_mask(640, 660, 55, 0.5, 17)
    img = compose(img, np.ones((H, W, 3), np.float32) * np.array((168, 98, 44), np.float32), seep * 0.9)
    save("mining_tailings_lori.png", img)


if __name__ == "__main__":
    forest_clearing(); lake_bloom(); drought_farmland(); mining_tailings()
