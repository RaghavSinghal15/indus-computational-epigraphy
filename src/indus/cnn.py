"""A5: a small convolutional network that identifies one cut-out sign.

Why a CNN: template matching (A2) compares a photo crop with ONE clean drawing
per sign and is at chance on real seals (reports/a4_real_seals.md). A CNN
learns what a sign looks like when carved, lit from the side and worn.

Training happens in two stages (scripts/train_sign_cnn.py):
  1. pretrain on carved renders of all 417 templates (render.py, 50 per sign),
     so every sign has been seen at least in synthetic form;
  2. fine-tune on real crops cut from the labelled CISI seals (real_crops.py),
     still mixing in renders so rare signs are not forgotten.

Inputs are 64 x 64 grayscale. Each image is standardised on its own (mean 0,
std 1) so the colour and brightness of the stone don't matter.
Class k of the network output is Mahadevan sign k + 1.

PyTorch basics used here:
  * nn.Conv2d(c_in, c_out, 3, padding=1) slides 3x3 filters over the image;
    nn.BatchNorm2d rescales each channel to keep training stable; ReLU zeroes
    negatives; MaxPool2d(2) halves height and width.
  * After four blocks a 64x64 image is a 4x4 grid of 256 features;
    AdaptiveAvgPool2d(1) averages the grid, and nn.Linear maps the 256
    numbers to 417 scores (one per sign). Softmax of the scores = probabilities.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 417
IMG = 64


class SignCNN(nn.Module):
    """64x64 grayscale crop -> 417 sign scores (logits)."""

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(256, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ----------------------------------------------------------------- images


def to_input(gray_uint8: np.ndarray) -> np.ndarray:
    """(N, 64, 64) uint8 -> (N, 1, 64, 64) float32, each image standardised."""
    x = gray_uint8.astype(np.float32)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True) + 1e-3
    return ((x - mean) / std)[:, None]


def load_carved_renders(folder) -> tuple[np.ndarray, np.ndarray]:
    """All carved renders under folder/<sign>/ as (images uint8 (N,64,64), labels = sign numbers).

    Reading 20,850 PNGs takes a minute, so the result is cached next to the folder.
    """
    cache = folder.parent / "synthetic_carved_64.npz"
    if cache.exists():
        d = np.load(cache)
        return d["images"], d["labels"]
    images, labels = [], []
    for sign in range(1, N_CLASSES + 1):
        for path in sorted((folder / str(sign)).glob("*.png")):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            images.append(cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA))
            labels.append(sign)
    if not images:
        raise FileNotFoundError(f"no renders in {folder}; run scripts/make_carved_synthetic.py first")
    images, labels = np.stack(images), np.array(labels)
    np.savez_compressed(cache, images=images, labels=labels)
    return images, labels


# ----------------------------------------------------------- augmentation


def augment(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Random small changes to a batch (N,1,64,64), different for every image, on any device.

    Geometry: rotation up to 10 degrees, scale 0.8-1.15, shift up to 12%,
    slight shear; done with affine_grid/grid_sample, which builds one 2x3
    transformation matrix per image and resamples the image through it.
    NO left-right flips: a mirrored sign can be a different sign.
    Photometric: contrast and brightness jitter, blur, noise.
    """
    n, dev = x.shape[0], x.device
    s = strength
    angle = (torch.rand(n, device=dev) * 2 - 1) * math.radians(10) * s
    scale = 1 + (torch.rand(n, device=dev) * 0.35 - 0.2) * s
    shear = (torch.rand(n, device=dev) * 2 - 1) * 0.1 * s
    tx, ty = [(torch.rand(n, device=dev) * 2 - 1) * 0.24 * s for _ in range(2)]   # grid units are -1..1
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin + shear, tx], 1),
                         torch.stack([sin, cos, ty], 1)], 1)                   # (n, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # padding_mode="border" repeats edge pixels, so no black corners appear
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    # blur half the images a little (3x3 average mixed with the original)
    blur = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
    mix = (torch.rand(n, 1, 1, 1, device=dev) < 0.5) * torch.rand(n, 1, 1, 1, device=dev)
    x = x * (1 - mix) + blur * mix

    contrast = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.3 * s
    bright = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.3 * s
    x = x * contrast + bright + torch.randn_like(x) * 0.1 * s
    # re-standardise so augmented images look like what the network sees at test time
    return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-3)


# ---------------------------------------------------------------- predict


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int = 512, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Top-k sign numbers and their probabilities for prepared inputs x (N,1,64,64)."""
    model.eval()
    dev = next(model.parameters()).device
    ids, probs = [], []
    for i in range(0, len(x), batch):
        p = F.softmax(model(torch.from_numpy(x[i:i + batch]).to(dev)), dim=1)
        top = p.topk(k, dim=1)
        ids.append(top.indices.cpu().numpy() + 1)       # class index -> sign number
        probs.append(top.values.cpu().numpy())
    return np.concatenate(ids), np.concatenate(probs)


def load_model(path, device: str = "cpu") -> SignCNN:
    """A trained SignCNN from a .pt file written by scripts/train_sign_cnn.py."""
    model = SignCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
