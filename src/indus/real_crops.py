"""Cut real sign crops out of labelled seal photos, using the known sign count.

A1 has to guess how many signs a seal has, and on real photos it guesses wrong
92% of the time. For TRAINING and TESTING the classifier we don't need to
guess: the labelled dataset tells us a seal has n signs, so we cut its text
band into exactly n pieces and give each piece its label.

How the n pieces are chosen ("forced segmentation"):
  1. find the text band (pipeline.find_text_row) and its carving mask,
  2. trim empty margins left and right,
  3. make the column profile (carved pixels per column, smoothed),
  4. start from n equal-width slices (signs on a seal are roughly evenly
     spaced), then move each cut to the lowest point of the profile within
     +-30% of one slice width, so it lands in the gap between two signs.

Pieces are returned in reading order (right to left on the impression).
"""

from __future__ import annotations

import cv2
import numpy as np

from indus.pipeline import find_text_row
from indus.segment import WORKING_HEIGHT, carving_mask, column_profile

CROP_SIZE = 64      # crops are returned as CROP_SIZE x CROP_SIZE grayscale


def _band_profile(band_gray: np.ndarray):
    """Carving mask and smoothed column profile of a text band, at WORKING_HEIGHT."""
    scale = WORKING_HEIGHT / band_gray.shape[0]
    small = cv2.resize(band_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = carving_mask(small)
    return scale, mask, column_profile(mask)


def forced_cuts(profile: np.ndarray, n: int, window: float = 0.3) -> list[int]:
    """n-1 cut positions: equal slices, each cut moved to the profile's lowest
    point within +-window of one slice width."""
    w = len(profile)
    if n <= 1:
        return []
    step = w / n
    cuts = []
    for i in range(1, n):
        ideal = i * step
        lo, hi = int(max(1, ideal - window * step)), int(min(w - 1, ideal + window * step))
        x = lo + int(np.argmin(profile[lo:hi + 1]))
        if cuts and x <= cuts[-1]:              # keep cuts in order
            x = cuts[-1] + 1
        cuts.append(x)
    return cuts


def square_crop(gray: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Columns x0:x1 of a band (full band height), padded to a square with the
    piece's median (stone) colour and resized to CROP_SIZE x CROP_SIZE."""
    x0 = min(max(x0, 0), gray.shape[1] - 1)             # keep at least 1 column inside the image
    x1 = min(max(x1, x0 + 1), gray.shape[1])
    piece = gray[:, x0:x1]
    h, w = piece.shape
    side = max(h, w)
    sq = np.full((side, side), int(np.median(piece)), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = piece
    return cv2.resize(sq, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def sign_crops(image_bgr: np.ndarray, n: int, mirrored: bool = False):
    """Exactly n crops (CROP_SIZE x CROP_SIZE gray) from a seal photo, in reading order,
    plus the text band box and the cut positions (for drawing)."""
    band = find_text_row(image_bgr)
    gray = cv2.cvtColor(band.crop(image_bgr), cv2.COLOR_BGR2GRAY)
    if mirrored:
        gray = gray[:, ::-1]
    scale, mask, profile = _band_profile(gray)

    # trim columns with (almost) no carving at either end
    ink = np.nonzero(profile > 0.05 * profile.max())[0] if profile.max() > 0 else np.arange(len(profile))
    left, right = int(ink[0]), int(ink[-1]) + 1
    sub = profile[left:right]
    cuts = [left + c for c in forced_cuts(sub, n)]
    edges = [left, *cuts, right]

    crops = [square_crop(gray, int(a / scale), int(b / scale)) for a, b in zip(edges[:-1], edges[1:])]
    crops = crops[::-1]                       # left-to-right pieces -> reading order (right to left)
    return crops, band, [int(e / scale) for e in edges]
