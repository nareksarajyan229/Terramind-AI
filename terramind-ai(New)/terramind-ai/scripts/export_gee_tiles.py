"""Export Sentinel-2 training tiles for Armenian hotspots from Google Earth Engine to Google Drive.

    python scripts/export_gee_tiles.py --year 2024 --radius 5000

Each hotspot yields a 12-band GeoTIFF (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 + NDVI + MNDWI) that
you can label for YOLO / U-Net training or upload directly to the Image Analysis page.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terramind import armenia, gee  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--radius", type=float, default=5000)
    ap.add_argument("--folder", default="terramind_tiles")
    args = ap.parse_args()
    if not gee.init():
        sys.exit(f"Earth Engine not available: {gee.status()[1]}")
    import ee

    start, end = gee._season(args.year)
    for h in armenia.HOTSPOTS:
        aoi = ee.Geometry.Point([h.lon, h.lat]).buffer(args.radius).bounds()
        img = gee.add_indices(gee.s2_composite(aoi, start, end))
        bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI", "MNDWI"]
        task = ee.batch.Export.image.toDrive(image=img.select(bands).toFloat(), description=f"tm_{h.id}_{start[:4]}",
                                             folder=args.folder, region=aoi, scale=10, maxPixels=1e9)
        task.start()
        print("started", h.id)


if __name__ == "__main__":
    main()
