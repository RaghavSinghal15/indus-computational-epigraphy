"""Synthetic inscriptions built from real sign crops (issue #32, the professor's suggestion).

Idea: we know hundreds of real signs exactly (hand-drawn boxes) and thousands of
real sign sequences (the ICIT corpus). So cut the real signs out and paste them,
in the order of a real sequence, onto a real seal surface. Every composite comes
with its exact sequence and box positions, which is what a whole-band reader
(CRNN + CTC) or a detector needs to train on.

How one composite is made:
  1. Background: the text band of a training seal with its signs painted out
     (cv2.inpaint fills each sign box from the surrounding stone).
  2. Signs: for each sign of the sequence, a random real crop of that sign from a
     training seal, scaled by the same factor as its own seal's band, so small
     signs (strokes) stay small and big ones stay big.
  3. Layout: placed left to right in VISUAL order (the reverse of reading order,
     because seal impressions read right to left), with random gaps and jitter.
  4. Blend: each crop's brightness is matched to the stone under it and its
     edges are feathered, so no hard rectangles show.
Only training seals are ever used (crops and backgrounds), never val/test/Harappa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from indus.segment import load_image


@dataclass
class Crop:
    image: np.ndarray        # BGR pixels of one real sign, with a small margin
    band_height: int         # height of the text band on its source seal (for scale)
    seal: str


@dataclass
class Background:
    image: np.ndarray        # BGR text band with the signs painted out
    seal: str


@dataclass
class Bank:
    crops: dict[int, list[Crop]] = field(default_factory=dict)   # Wells code -> crops
    backgrounds: list[Background] = field(default_factory=list)


def build_bank(boxes: pd.DataFrame, margin: int = 3) -> Bank:
    """Sign crops and sign-free backgrounds from hand-boxed TRAINING seals (pass only those rows)."""
    bank = Bank()
    for (seal, image_path), rows in boxes.groupby(["cisi", "image"]):
        img = load_image(image_path)
        h, w = img.shape[:2]
        y0, y1 = max(0, rows.y0.min() - margin), min(h, rows.y1.max() + margin)
        x0, x1 = max(0, rows.x0.min() - margin), min(w, rows.x1.max() + margin)
        band_h = y1 - y0
        if band_h < 10 or x1 - x0 < 10:
            continue
        for r in rows.itertuples():
            c = img[max(0, r.y0 - margin):min(h, r.y1 + margin), max(0, r.x0 - margin):min(w, r.x1 + margin)]
            if c.shape[0] >= 6 and c.shape[1] >= 3:
                bank.crops.setdefault(int(r.wells), []).append(Crop(c.copy(), band_h, seal))
        # paint the signs out of the band to get bare stone
        band = img[y0:y1, x0:x1].copy()
        mask = np.zeros(band.shape[:2], np.uint8)
        for r in rows.itertuples():
            cv2.rectangle(mask, (r.x0 - x0 - 2, r.y0 - y0 - 2), (r.x1 - x0 + 2, r.y1 - y0 + 2), 255, -1)
        # inpaint: fill masked pixels from their surroundings (Telea's method)
        bank.backgrounds.append(Background(cv2.inpaint(band, mask, 7, cv2.INPAINT_TELEA), seal))
    return bank


def covered(sequence, bank: Bank) -> bool:
    """True if every sign of the sequence has at least one real crop in the bank."""
    return all(s in bank.crops for s in sequence)


def compose(signs_reading: list[int], bank: Bank, rng: np.random.Generator, height: int = 96):
    """One synthetic band. Returns (BGR image, boxes in visual order, source seals)."""
    bg = bank.backgrounds[rng.integers(len(bank.backgrounds))]
    visual = list(signs_reading)[::-1]                    # impressions read right to left
    picks = []
    for s in visual:
        options = bank.crops[s]
        picks.append(options[rng.integers(len(options))])

    # scale every crop as if its own band were `height` pixels tall (keeps relative sign sizes)
    jitter = rng.uniform(0.85, 1.0)
    pieces = []
    for p in picks:
        f = height * jitter / p.band_height * rng.uniform(0.92, 1.08)
        ph = max(6, min(height - 2, round(p.image.shape[0] * f)))
        pw = max(3, round(p.image.shape[1] * ph / p.image.shape[0]))
        pieces.append(cv2.resize(p.image, (pw, ph), interpolation=cv2.INTER_AREA))
    gaps = [round(height * rng.uniform(0.0, 0.08)) for _ in pieces]   # real signs sit close together
    pad = round(height * rng.uniform(0.03, 0.12))
    width = 2 * pad + sum(pc.shape[1] for pc in pieces) + sum(gaps[:-1])

    canvas = cv2.resize(bg.image, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    boxes, x = [], pad
    for pc, gap in zip(pieces, gaps):
        ph, pw = pc.shape[:2]
        y = int(rng.integers(0, max(1, height - ph + 1)))
        y = min(y, height - ph)
        region = canvas[y:y + ph, x:x + pw]
        piece = pc.astype(np.float32)
        piece += region.mean(axis=(0, 1)) - piece.mean(axis=(0, 1))   # match brightness to the stone
        alpha = np.zeros((ph, pw), np.float32)
        inset = max(1, min(ph, pw) // 6)
        alpha[inset:ph - inset, inset:pw - inset] = 1
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, inset))[..., None]   # feathered edges
        canvas[y:y + ph, x:x + pw] = alpha * piece + (1 - alpha) * region
        boxes.append((x, y, x + pw, y + ph))
        x += pw + gap

    # whole-image photo effects: lighting, blur, grain
    canvas = canvas * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    if rng.random() < 0.5:
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.3, 1.0))
    canvas += rng.normal(0, rng.uniform(0, 6), canvas.shape)
    out = np.clip(canvas, 0, 255).astype(np.uint8)
    sources = sorted({p.seal for p in picks} | {bg.seal})
    return out, boxes, sources
