"""Render whole carved ROWS of signs, then cut them exactly like real seals.

Why: single-sign renders (render.py) are centred, square and alone. Real
training and test crops are pieces of a text band cut by real_crops.band_crops:
they have grey padding bars, bits of the neighbouring signs, and sometimes the
wrong piece altogether (the cut slipped). A CNN trained only on clean single
renders gets 4.5% on real crops. If the synthetic crops go through the SAME
cutting code, they look like real crops, mistakes included.

One row:
  1. pick a sequence of signs (left to right on the impression);
  2. for each sign: jitter the drawing (render._jitter_shape), trim it to its
     strokes, scale it to the row height (wide signs capped at 1.8 x height),
     thicken the strokes;
  3. place them side by side with random gaps and small vertical offsets;
     sometimes add junk at an end (a curved stroke like the unicorn's horn,
     or a straight edge like the seal's frame);
  4. light the whole row as ONE stone surface (render.render_surface);
  5. cut into n crops: usually with the equal-slice cutter used on real seals
     (labels by position, as for real seals), sometimes near the true borders.
"""

from __future__ import annotations

import cv2
import numpy as np

from indus.data import template_path
from indus.real_crops import band_crops, square_crop
from indus.render import _jitter_shape, _thicken, render_surface

ROW_HEIGHT = 96            # sign height in pixels (before the margins)


def _template(sign: int, cache={}) -> np.ndarray:
    """Binary template drawing (0/255), read once per sign."""
    if sign not in cache:
        gray = cv2.imread(str(template_path(sign)), cv2.IMREAD_GRAYSCALE)
        cache[sign] = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    return cache[sign]


def sign_shape(sign: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Jittered, thickened strokes of one sign, trimmed to the strokes, about `height` tall (uint8 0/255)."""
    shape = _jitter_shape(_template(sign), rng)
    ys, xs = np.nonzero(shape > 127)
    if len(xs) == 0:
        return np.zeros((height, height // 4), np.uint8)
    shape = shape[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = shape.shape
    scale = height * rng.uniform(0.7, 1.0) / h
    scale = min(scale, 1.8 * height / w)                      # very wide signs: limit the width
    shape = cv2.resize(shape, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
    return _thicken(shape, rng)


def _junk(canvas: np.ndarray, side: str, rng: np.random.Generator) -> None:
    """Draw a horn-like curve or a frame edge into one end of the canvas (in place)."""
    h, w = canvas.shape
    x = w - 1 - int(rng.uniform(0.02, 0.12) * h) if side == "right" else int(rng.uniform(0.02, 0.12) * h)
    thick = max(2, int(rng.uniform(0.04, 0.1) * h))
    if rng.random() < 0.5:     # curved stroke rising from the bottom, like the unicorn's horn
        axes = (int(rng.uniform(0.2, 0.5) * h), int(rng.uniform(0.5, 0.9) * h))
        cv2.ellipse(canvas, (x, h), axes, 0, 180 + rng.uniform(0, 40), 270, 255, thick)
    else:                      # straight edge, like the frame of the seal
        cv2.line(canvas, (x, 0), (x, h - 1), 255, thick)


def render_row(signs: list[int], rng: np.random.Generator):
    """(grayscale band, true (x0, x1) of each sign) for signs given left to right."""
    H = ROW_HEIGHT
    shapes = [sign_shape(s, H, rng) for s in signs]
    top, bottom = int(rng.uniform(0.08, 0.25) * H), int(rng.uniform(0.08, 0.25) * H)
    gaps = [int(rng.uniform(0.05, 0.3) * H) for _ in shapes[1:]]
    left, right = int(rng.uniform(0.1, 0.5) * H), int(rng.uniform(0.1, 0.5) * H)
    width = left + sum(s.shape[1] for s in shapes) + sum(gaps) + right
    canvas = np.zeros((top + H + bottom, width), np.uint8)
    x, bounds = left, []
    for i, s in enumerate(shapes):
        h, w = s.shape
        y = top + int(rng.uniform(0, 1) * (H - h))             # anywhere in the line, mostly near full height
        canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], s)
        bounds.append((x, x + w))
        x += w + (gaps[i] if i < len(gaps) else 0)
    if rng.random() < 0.3:
        _junk(canvas, "right" if rng.random() < 0.6 else "left", rng)
    bgr = render_surface(canvas.astype(np.float32) / 255, rng, sunken=bool(rng.random() < 0.5))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bounds


def row_crops(signs: list[int], rng: np.random.Generator, equal_frac: float = 0.7):
    """Crops of one rendered row in READING order, with their labels (reading order too)."""
    gray, bounds = render_row(signs, rng)
    if rng.random() < equal_frac:
        crops, _ = band_crops(gray, len(signs))                 # the real-seal cutter, mistakes and all
    else:
        crops = []
        for x0, x1 in bounds:                                  # near the true borders
            w = x1 - x0
            crops.append(square_crop(gray, x0 + int(rng.uniform(-0.15, 0.15) * w),
                                     x1 + int(rng.uniform(-0.15, 0.15) * w)))
        crops = crops[::-1]
    return crops, list(signs)[::-1]
