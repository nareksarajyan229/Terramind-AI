"""YOLO object detector for discrete environmental features (Ultralytics, AGPL-3.0).

Expected class names in the trained model (see training/dataset.yaml):
  forest_clearing, logging_road, algal_bloom, turbid_plume, tailings_pond,
  open_pit, waste_dump, erosion_gully, burn_scar, dry_riverbed
Each is mapped to one of the platform's issue categories.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .vision import Region

YOLO_TO_ISSUE = {
    "forest_clearing": "deforestation",
    "logging_road": "deforestation",
    "algal_bloom": "water_pollution",
    "turbid_plume": "water_pollution",
    "tailings_pond": "mining",
    "open_pit": "mining",
    "waste_dump": "mining",
    "erosion_gully": "soil_degradation",
    "burn_scar": "wildfire",
    "dry_riverbed": "drought",
}


class Detector:
    def __init__(self, weights: str | None = None, conf: float = 0.3, imgsz: int = 640):
        self.ready = False
        self.reason = ""
        self.conf = conf
        self.imgsz = imgsz
        try:
            from ultralytics import YOLO
        except ImportError:
            self.reason = "ultralytics not installed"
            return
        if not weights or not Path(weights).exists():
            self.reason = "No trained YOLO weights found"
            return
        self.model = YOLO(weights)
        self.ready = True

    def detect(self, rgb: np.ndarray, gsd_m: float | None = None) -> list[Region]:
        res = self.model.predict(rgb[..., ::-1], conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        names = res.names
        out: list[Region] = []
        for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy()):
            issue = YOLO_TO_ISSUE.get(names[int(cls)])
            if issue is None:
                continue
            x1, y1, x2, y2 = box.astype(int)
            area = int((x2 - x1) * (y2 - y1))
            out.append(Region(issue, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), area,
                              None if gsd_m is None else area * gsd_m ** 2 / 10_000, float(conf),
                              source=f"yolo:{names[int(cls)]}"))
        return out
