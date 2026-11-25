"""PyTorch U-Net for 8-class environmental land-cover segmentation.

The class order matches ``terramind.vision.CLASSES``. Train it with
``training/train_segmentation.py`` (e.g. on LoveDA / DeepGlobe re-labelled to these
classes plus your own Armenian tiles), then place the weights at ``models/terramind_unet.pt``
or point SEG_WEIGHTS to them. Without weights the app uses the spectral engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:  # app still runs without torch
    TORCH_OK = False

N_CLASSES = 8
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


if TORCH_OK:

    class _DoubleConv(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        """Compact U-Net (~7.8M params at base=32) — trains on a single consumer GPU or Colab."""

        def __init__(self, in_ch: int = 3, n_classes: int = N_CLASSES, base: int = 32):
            super().__init__()
            ch = [base, base * 2, base * 4, base * 8, base * 16]
            self.downs = nn.ModuleList()
            prev = in_ch
            for c in ch:
                self.downs.append(_DoubleConv(prev, c))
                prev = c
            self.ups = nn.ModuleList()
            self.up_convs = nn.ModuleList()
            for c in reversed(ch[:-1]):
                self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
                self.up_convs.append(_DoubleConv(c * 2, c))
                prev = c
            self.head = nn.Conv2d(prev, n_classes, 1)

        def forward(self, x):
            skips = []
            for i, down in enumerate(self.downs):
                x = down(x)
                if i < len(self.downs) - 1:
                    skips.append(x)
                    x = F.max_pool2d(x, 2)
            for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
                x = up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = conv(torch.cat([skip, x], dim=1))
            return self.head(x)


class Segmenter:
    """Inference wrapper. ``ready`` is True only when torch and trained weights are present."""

    def __init__(self, weights: str | None = None, device: str | None = None, tile: int = 512):
        self.ready = False
        self.reason = ""
        self.tile = tile
        if not TORCH_OK:
            self.reason = "PyTorch-ը տեղադրված չէ"
            return
        if not weights or not Path(weights).exists():
            self.reason = "Մարզված մոդել չկա"
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state.get("model", state))
        self.model.eval()
        self.ready = True

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """rgb uint8 HxWx3 -> class map HxW (uint8). Tiled to bound memory."""
        h, w = rgb.shape[:2]
        x = ((rgb.astype(np.float32) / 255.0) - MEAN) / STD
        out = np.zeros((h, w), np.uint8)
        t = self.tile
        with torch.no_grad():
            for y0 in range(0, h, t):
                for x0 in range(0, w, t):
                    patch = x[y0:y0 + t, x0:x0 + t]
                    ph, pw = patch.shape[:2]
                    pad = np.zeros((t, t, 3), np.float32)
                    pad[:ph, :pw] = patch
                    tens = torch.from_numpy(pad).permute(2, 0, 1)[None].to(self.device)
                    pred = self.model(tens).argmax(1)[0].cpu().numpy().astype(np.uint8)
                    out[y0:y0 + ph, x0:x0 + pw] = pred[:ph, :pw]
        return out
