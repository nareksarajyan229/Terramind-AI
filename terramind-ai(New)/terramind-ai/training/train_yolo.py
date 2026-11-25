"""Train the TerraMind YOLO detector (Ultralytics) on labelled satellite / drone tiles.

    pip install ultralytics
    python training/train_yolo.py --data training/dataset.yaml --model yolov8s.pt --epochs 100

Labelling: annotate 640×640 tiles in YOLO format with the free, open-source tools
CVAT or Label Studio. Good starting imagery for Armenia: Sentinel-2 L2A (10 m) via
Earth Engine / Copernicus Browser for large features (tailings ponds, clearings,
blooms, burn scars), and drone or high-resolution aerial tiles for small ones (gullies,
waste dumps, logging roads). Aim for 200+ instances per class; include negative tiles.

Weights are written to runs/terramind/<name>/weights/best.pt — copy that file to
models/terramind_yolo.pt (or set YOLO_WEIGHTS) and the app will pick it up.
Ultralytics is AGPL-3.0; keep your deployment open-source or obtain a licence.
"""
import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/dataset.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="Pre-trained checkpoint to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="terramind")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project="runs/terramind", name=args.name, patience=25,
                # Satellite imagery has no canonical "up": rotate and flip freely.
                degrees=90, flipud=0.5, fliplr=0.5, mosaic=1.0, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3)
    metrics = model.val()
    print("mAP50-95:", metrics.box.map, "mAP50:", metrics.box.map50)

    best = Path(model.trainer.best)
    dest = Path("models/terramind_yolo.pt")
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    print(f"Copied {best} -> {dest}")


if __name__ == "__main__":
    main()
