"""A1: split a photographed row of Indus signs into one box per sign.

How it works (classical image processing, no training data):

1. **Carving mask.** Seal photos are not dark ink on light paper. The signs
   are carved relief, lit from one side, so each stroke shows up as a bright
   highlight next to a dark shadow on a stone background whose colour varies
   from photo to photo. So instead of "find dark pixels" we ask "which pixels
   differ strongly from their surroundings?":
     - estimate the local background with a heavy blur,
     - take |image - background|  (catches highlights AND shadows),
     - threshold that automatically with Otsu's method,
     - drop tiny specks (dust, stone texture).

2. **Column profile.** Count carved pixels in each column of the image.
   Signs give high counts; gaps between signs give low counts. Background
   texture means gaps are rarely empty, so we look for VALLEYS in the
   (smoothed) profile rather than for columns with zero ink.

3. **Cut at valleys.** Each stretch between two valleys is one sign. Its box
   is tightened to the carved pixels inside it.

Boxes are returned right to left, i.e. in reading order for a photo of a seal
IMPRESSION. For a photo of the seal itself the text is mirrored: flip the
image horizontally first (`mirrored=True`).

Known weaknesses: signs that touch get merged into one box (the most common
error on real photos); hollow signs and signs made of separate strokes
(e.g. "||||") can be split; strong shadows or photo edges can produce a junk box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

# All size settings are fractions of the row height, so the same values work
# for a 100-pixel-tall photo and a 1000-pixel-tall one.
WORKING_HEIGHT = 120          # rows are resized to this height before processing
MIN_SPECK_FRAC = 0.06         # specks smaller than (0.06 * height)^2 pixels are noise
EDGE_BAND_FRAC = 0.12         # ignore top/bottom 12% when building the profile
SMOOTH_FRAC = 0.10            # profile smoothing window
MIN_SIGN_WIDTH_FRAC = 0.35    # two valleys can't be closer than this
VALLEY_PROMINENCE_FRAC = 0.18 # how deep a dip must be to count as a gap
MAX_SIGN_WIDTH_FRAC = 0.9     # a piece wider than this is probably two touching signs
SPLIT_DIP_FRAC = 0.7          # ...split it at its lowest column if that is below 70% of its peak
MIN_SIGN_INK_FRAC = 0.005     # a slice with fewer carved pixels than 0.005 * height^2 is empty
                              # (0.02 dropped small thin signs like the jar; no change on the 22 photos)


@dataclass
class Box:
    """One detected sign, in pixel coordinates of the ORIGINAL image."""
    x0: int
    y0: int
    x1: int
    y1: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0:self.y1, self.x0:self.x1]


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as a BGR array (OpenCV's channel order)."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def carving_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (0 or 255) of pixels that belong to carved strokes."""
    h = gray.shape[0]
    k = int(h * 0.6) | 1                      # blur kernel size must be odd
    background = cv2.GaussianBlur(gray, (k, k), 0)
    detail = cv2.absdiff(gray, background)
    detail = cv2.GaussianBlur(detail, (3, 3), 0)
    # Otsu picks the threshold that best separates two groups of pixel values
    _, mask = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # connectedComponentsWithStats labels every separate white blob and gives
    # its area, so we can delete the small ones
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = (h * MIN_SPECK_FRAC) ** 2
    keep = np.zeros_like(mask)
    for i in range(1, n):                     # label 0 is the background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def column_profile(mask: np.ndarray) -> np.ndarray:
    """Smoothed count of carved pixels per column (central band only)."""
    h = mask.shape[0]
    band = mask[int(EDGE_BAND_FRAC * h):int((1 - EDGE_BAND_FRAC) * h)] > 0
    profile = band.sum(axis=0).astype(float)
    k = max(3, int(SMOOTH_FRAC * h))
    return np.convolve(profile, np.ones(k) / k, mode="same")   # moving average


def _cuts_through_a_sign(mask: np.ndarray, x: int) -> bool:
    """True if column x passes through the middle of one connected, sign-sized shape.

    Hollow signs (squares, diamonds, the jar) have an empty interior that looks
    like a gap in the column profile. But a hollow sign is ONE connected shape
    that extends well past the cut on both sides, while a real gap between two
    signs is not crossed by any such shape.
    """
    h = mask.shape[0]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in set(labels[:, x].tolist()) - {0}:
        left = stats[label, cv2.CC_STAT_LEFT]
        right = left + stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        width = stats[label, cv2.CC_STAT_WIDTH]
        spans_both_sides = left <= x - 0.15 * h and right >= x + 0.15 * h
        # One sign is roughly as tall as the row and at most about as wide.
        # Thin shadow lines (too short) and texture-glued chains of several
        # signs (too wide) must not block a cut.
        sign_sized = height >= 0.3 * h and width <= 1.0 * h
        if spans_both_sides and sign_sized:
            return True
    return False


def _split_wide_pieces(cuts: list[int], profile: np.ndarray, h: int) -> list[int]:
    """Add cuts inside pieces too wide to be one sign.

    Touching signs leave only a shallow dip in the column profile, too shallow
    to pass the prominence test. But a piece much wider than one sign almost
    certainly holds two, so we cut it at its lowest point anyway, as long as
    that point is a real dip (well below the piece's peak). Repeats until no
    piece is too wide or no dip is left.
    """
    max_width = int(MAX_SIGN_WIDTH_FRAC * h)
    margin = int(MIN_SIGN_WIDTH_FRAC * h)          # don't cut off a sliver at either end
    cuts = list(cuts)
    changed = True
    while changed:
        changed = False
        for x0, x1 in zip(cuts[:-1], cuts[1:]):
            if x1 - x0 <= max_width or x1 - x0 <= 2 * margin:
                continue
            inner = profile[x0 + margin:x1 - margin]
            x = x0 + margin + int(np.argmin(inner))
            if profile[x] < SPLIT_DIP_FRAC * profile[x0:x1].max():
                cuts = sorted(cuts + [x])
                changed = True
                break                               # cuts changed; start the scan again
    return cuts


def find_sign_boxes(
    image: np.ndarray,
    mirrored: bool = False,
    protect_hollow_signs: bool = False,
    split_wide_boxes: bool = True,
) -> list[Box]:
    """Boxes around each sign in a text-row image, in reading order.

    `image` is a BGR array of a single row of signs. Boxes are always in the
    coordinates of the image you pass in.

    Reading order is right to left for a photo of a seal IMPRESSION (the
    default). For a photo of the seal itself the text is mirrored, so set
    `mirrored=True` and the boxes come back left to right instead.

    `protect_hollow_signs=True` refuses to cut through a single compact shape,
    which stops hollow signs (squares, diamonds) being split down the middle.
    It is OFF by default: on the 22 sample photos the main error is merging
    neighbouring signs, and this rule merges more (fewer boxes on 6 photos,
    clearly better on 2). Decide properly once D2 ground truth can measure it.

    `split_wide_boxes=True` (default) cuts any piece wider than 0.9 x the row
    height at its deepest dip, since it probably holds two touching signs. On
    the samples it fixed a merge on photo 96 and changed nothing else of note.
    A lower limit (0.6) also cut real multi-stroke signs like "||||" in half.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = WORKING_HEIGHT / gray.shape[0]
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = small.shape

    mask = carving_mask(small)
    profile = column_profile(mask)
    if profile.max() == 0:
        return []

    # find_peaks finds local maxima; on the negated profile those are valleys
    valleys, _ = find_peaks(
        -profile,
        distance=max(1, int(MIN_SIGN_WIDTH_FRAC * h)),
        prominence=VALLEY_PROMINENCE_FRAC * profile.max(),
    )
    cuts = [0, *valleys.tolist(), w]
    if split_wide_boxes:
        cuts = _split_wide_pieces(cuts, profile, h)
    if protect_hollow_signs:   # checked last, so it also vetoes cuts made by the split
        cuts = [0, *[x for x in cuts[1:-1] if not _cuts_through_a_sign(mask, x)], w]

    boxes = []
    for x0, x1 in zip(cuts[:-1], cuts[1:]):
        piece = mask[:, x0:x1] > 0
        if piece.sum() < MIN_SIGN_INK_FRAC * h * h:   # nearly empty slice: not a sign
            continue
        cols = np.where(piece.any(axis=0))[0]
        rows = np.where(piece.any(axis=1))[0]
        bx0, bx1 = x0 + cols[0], x0 + cols[-1] + 1
        by0, by1 = rows[0], rows[-1] + 1
        # scale back to the original image's pixel coordinates
        boxes.append(Box(*(int(round(v / scale)) for v in (bx0, by0, bx1, by1))))

    # Boxes were collected left to right. An impression reads right to left;
    # a mirrored photo of the seal itself reads left to right.
    if not mirrored:
        boxes.reverse()
    return boxes


def draw_boxes(image: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Copy of `image` with each box drawn and numbered in reading order."""
    out = image.copy()
    thickness = max(1, image.shape[0] // 60)
    for i, b in enumerate(boxes, start=1):
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (0, 0, 255), thickness)
        cv2.putText(out, str(i), (b.x0 + 3, b.y0 + 3 + 12 * thickness),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thickness, (0, 255, 0), thickness)
    return out
