"""Train the TerraMind U-Net land-cover segmenter (PyTorch).

    pip install torch torchvision
    python training/train_segmentation.py --data datasets/seg --epochs 60

Dataset layout (any size images; random 512×512 crops are taken):
    datasets/seg/{train,val}/images/*.png     RGB tiles
    datasets/seg/{train,val}/masks/*.png      single-channel masks, values 0–7 in this order:
        0 healthy_vegetation  1 stressed_vegetation  2 bare_soil  3 clean_water
        4 turbid_water        5 algal_bloom          6 burned_dark 7 built_mining
        255 = ignore

Open datasets you can re-label to these classes: LoveDA, DeepGlobe Land Cover, and
Dynamic World / ESA WorldCover labels exported from Earth Engine over Armenia (use
NDCI > 0.1 inside water for algal_bloom, dNBR for burned). Tip: bootstrap masks with the
spectral engine (`terramind.vision.classify_multispectral` on Sentinel-2) and correct them
by hand in CVAT — far faster than labelling from scratch.

Weights are saved to models/terramind_unet.pt, which the app loads automatically.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from terramind.segmentation import MEAN, N_CLASSES, STD, UNet  # noqa: E402


class TileDataset(Dataset):
    def __init__(self, root: Path, crop: int = 512, train: bool = True):
        self.imgs = sorted((root / "images").glob("*.png"))
        self.root, self.crop, self.train = root, crop, train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.root / "masks" / p.name), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        c = self.crop
        if h < c or w < c:
            img = cv2.copyMakeBorder(img, 0, max(0, c - h), 0, max(0, c - w), cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, max(0, c - h), 0, max(0, c - w), cv2.BORDER_CONSTANT, value=255)
            h, w = mask.shape
        y = random.randint(0, h - c) if self.train else (h - c) // 2
        x = random.randint(0, w - c) if self.train else (w - c) // 2
        img, mask = img[y:y + c, x:x + c], mask[y:y + c, x:x + c]
        if self.train:
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = np.clip(img.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255)
        img = (np.ascontiguousarray(img).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(np.ascontiguousarray(mask)).long()


def miou(conf: np.ndarray) -> float:
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    valid = union > 0
    return float((inter[valid] / union[valid]).mean()) if valid.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/seg")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/terramind_unet.pt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data)
    tl = DataLoader(TileDataset(root / "train"), args.batch, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(TileDataset(root / "val", train=False), args.batch, num_workers=2)

    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tl))
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
        model.eval()
        conf = np.zeros((N_CLASSES, N_CLASSES), np.int64)
        with torch.no_grad():
            for x, y in vl:
                p = model(x.to(dev)).argmax(1).cpu().numpy().ravel()
                t = y.numpy().ravel()
                keep = t != 255
                conf += np.bincount(t[keep] * N_CLASSES + p[keep], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        score = miou(conf)
        print(f"epoch {ep + 1:3d}  loss {tot / len(tl):.4f}  val mIoU {score:.3f}")
        if score > best:
            best = score
            Path(args.out).parent.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "miou": score, "epoch": ep + 1}, args.out)
    print(f"best mIoU {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
