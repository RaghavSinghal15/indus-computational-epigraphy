"""A2: identify a cut-out sign by comparing it with the 417 Mahadevan templates.

How it works (no training, one reference drawing per sign):

1. **Turn the image into strokes.** Templates are white strokes on black. A
   photo crop is carved stone, so we reuse A1's carving mask to get the same
   kind of picture: white where the stone is carved, black elsewhere.
   (`strokes_from_drawing` for clean drawings, `strokes_from_photo` for crops.)

2. **Normalize.** Trim to the strokes, pad to a square (keeping the sign's
   proportions), shrink to 48 x 48 and blur. The blur matters: two drawings of
   the same sign never line up pixel for pixel, and blurring lets "nearly in
   the same place" still count as a match.

3. **Compare with every template (normalized cross-correlation).** Each
   normalized image is turned into one long vector, the mean is subtracted and
   it is scaled to length 1. The dot product of two such vectors is then the
   correlation between the two images: 1 means identical, 0 unrelated,
   negative means one is bright where the other is dark. All 417 scores come
   from one matrix-vector product.

4. **Return the 5 best** templates with their scores (top-5).

Measured on P4's synthetic images (distorted copies of the templates, so this
is a best case): see `scripts/a2_evaluate.py`. On real photo crops it is much
weaker; the drawings and carved stone look too different. That gap is what
the CNN trained on synthetic data is meant to close.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from indus.data import template_path
from indus.segment import carving_mask

N_TEMPLATES = 417
SIZE = 48           # normalized images are SIZE x SIZE pixels
BLUR_SIGMA = 3      # Gaussian blur in pixels at SIZE; best of 0-6 on synthetic data
PHOTO_HEIGHT = 96   # photo crops are resized to this height before masking


@dataclass
class Match:
    """One candidate identification for a sign."""
    mahadevan_id: int
    score: float    # correlation with the template, from -1 to 1 (higher is better)


def strokes_from_drawing(gray: np.ndarray) -> np.ndarray:
    """Binary image (0 or 255) of a clean drawing: white strokes on black."""
    # Otsu picks the threshold automatically. Templates are already white on
    # black, so this only cleans up grey pixels left by blurring or resizing.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def strokes_from_photo(crop_bgr: np.ndarray) -> np.ndarray:
    """Binary image (0 or 255) of the carved strokes in a photo crop of one sign."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    scale = PHOTO_HEIGHT / gray.shape[0]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return carving_mask(gray)


def normalize(binary: np.ndarray) -> np.ndarray:
    """Flat vector of length SIZE*SIZE with mean 0 and length 1 (all zeros if empty)."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return np.zeros(SIZE * SIZE, np.float32)

    # trim to the strokes, then centre on a square canvas so the shape keeps
    # its proportions (a tall sign stays tall)
    trimmed = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = trimmed.shape
    side = max(h, w)
    square = np.zeros((side, side), np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = trimmed

    small = cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), BLUR_SIGMA)  # (0, 0): kernel size chosen from sigma

    v = small.ravel()
    v -= v.mean()
    length = np.linalg.norm(v)
    return v / length if length > 0 else v


@lru_cache(maxsize=1)
def template_matrix() -> np.ndarray:
    """Array of shape (417, SIZE*SIZE): row i-1 is the normalized template for sign i.

    Cached, so the 417 files are read only once per Python session.
    """
    rows = []
    for sign in range(1, N_TEMPLATES + 1):
        gray = cv2.imread(str(template_path(sign)), cv2.IMREAD_GRAYSCALE)
        rows.append(normalize(strokes_from_drawing(gray)))
    return np.stack(rows)


def rank_templates(vector: np.ndarray, k: int = 5) -> list[Match]:
    """The k best-matching templates for a normalized vector, best first."""
    scores = template_matrix() @ vector          # one correlation per template
    best = np.argsort(-scores)[:k]               # indices of the k highest scores
    return [Match(int(i) + 1, float(scores[i])) for i in best]


def classify_drawing(gray: np.ndarray, k: int = 5) -> list[Match]:
    """Top-k template matches for a clean drawing (white strokes on black)."""
    return rank_templates(normalize(strokes_from_drawing(gray)), k)


def classify_crop(crop_bgr: np.ndarray, mirrored: bool = False, k: int = 5) -> list[Match]:
    """Top-k template matches for a photo crop of one sign (BGR, as from A1).

    Templates show signs as they appear on an impression. A photo of the seal
    itself shows every sign mirrored, so set `mirrored=True` for those and the
    crop is flipped back before comparing.
    """
    if crop_bgr.size == 0:                       # empty crop: nothing to match
        return rank_templates(np.zeros(SIZE * SIZE, np.float32), k)
    strokes = strokes_from_photo(crop_bgr)
    if mirrored:
        strokes = strokes[:, ::-1]               # flip left-right
    return rank_templates(normalize(strokes), k)
