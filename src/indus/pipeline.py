"""A3: photo in, ordered list of Mahadevan sign IDs out.

    photo ──► find the text band ──► A1: one box per sign ──► A2: top-5 per box
              (whole-seal photos)    (reading order)          (best guess first)

A whole seal photo shows an animal with the text along the top edge. A1 only
works on the text row, so for whole seals we first find that row: carved
pixels are dense in a band at the top (the signs), then thin out (the gap
above the animal). `find_text_row` cuts at that drop.

Use `read_seal(image)` for a whole-seal photo, `read_text_row(image)` when the
photo already shows just the row of signs. Set `mirrored=True` for a photo of
the seal itself (carved stone, mirrored) rather than an impression.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from indus.classify import Match, classify_crop
from indus.segment import Box, carving_mask, find_sign_boxes

BAND_SEARCH = (0.15, 0.45)   # the text band ends somewhere between 15% and 45% of the height
BAND_DROP_FRAC = 0.4         # ...where carving density first falls below 40% of its peak
FIND_ROW_HEIGHT = 300        # seal photos are resized to this height to find the band


@dataclass
class SignReading:
    """One sign: where it is (in the photo passed in) and its top-k guesses."""
    box: Box
    matches: list[Match]

    @property
    def best(self) -> int:
        """Mahadevan ID of the top guess."""
        return self.matches[0].mahadevan_id


def find_text_row(image: np.ndarray) -> Box:
    """Box around the row of signs at the top of a whole-seal photo."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = FIND_ROW_HEIGHT / gray.shape[0]
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = small.shape

    # fraction of carved pixels in each row, smoothed over 9 rows
    mask = carving_mask(small)
    density = np.convolve((mask > 0).mean(axis=1), np.ones(9) / 9, mode="same")

    lo, hi = int(BAND_SEARCH[0] * h), int(BAND_SEARCH[1] * h)
    peak_row = int(np.argmax(density[:hi]))
    peak = density[peak_row]
    start = max(lo, peak_row)
    drops = np.nonzero(density[start:hi] < BAND_DROP_FRAC * peak)[0]
    # bottom of the band: the first big drop, or failing that the lowest point
    bottom = start + int(drops[0]) if len(drops) else lo + int(np.argmin(density[lo:hi]))
    # the drop can start while still inside the bottoms of the signs, so keep
    # going down while density keeps falling, to the floor of the gap
    while bottom + 1 < hi and density[bottom + 1] < density[bottom]:
        bottom += 1

    # top of the band: first row that is reasonably dense, plus a small margin
    top = int(np.argmax(density[:bottom] > 0.3 * peak))
    top = max(0, top - int(0.02 * h))

    # left and right: columns with any real amount of carving inside the band
    cols = np.nonzero((mask[top:bottom] > 0).mean(axis=0) > 0.02)[0]
    x0, x1 = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (0, w)
    return Box(*(int(round(v / scale)) for v in (x0, top, x1, bottom)))


def read_text_row(image: np.ndarray, mirrored: bool = False, k: int = 5) -> list[SignReading]:
    """Signs in a photo of a text row, in reading order, each with its top-k guesses."""
    readings = []
    for box in find_sign_boxes(image, mirrored=mirrored):
        matches = classify_crop(box.crop(image), mirrored=mirrored, k=k)
        readings.append(SignReading(box, matches))
    return readings


def read_seal(image: np.ndarray, mirrored: bool = False, k: int = 5) -> tuple[Box, list[SignReading]]:
    """Text band and signs of a whole-seal photo. Boxes are in whole-photo coordinates."""
    band = find_text_row(image)
    readings = read_text_row(band.crop(image), mirrored=mirrored, k=k)
    for r in readings:   # shift boxes from band coordinates back to the whole photo
        b = r.box
        r.box = Box(b.x0 + band.x0, b.y0 + band.y0, b.x1 + band.x0, b.y1 + band.y0)
    return band, readings


def draw_readings(image: np.ndarray, readings: list[SignReading], band: Box | None = None) -> np.ndarray:
    """Copy of `image` with the text band (blue), each sign's box (red), its reading
    position (green, above the box) and its best-guess Mahadevan ID (red, below)."""
    if not readings:
        return image.copy()
    # size lines and text from the sign height, so labels stay small next to the signs
    sign_h = int(np.median([r.box.y1 - r.box.y0 for r in readings]))
    thick = max(1, sign_h // 50)
    font = max(0.3, sign_h / 150)
    # white margin above and below, so labels never fall off the edge
    pad = int(30 * font) + 4
    out = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if band is not None:
        cv2.rectangle(out, (band.x0, band.y0 + pad), (band.x1, band.y1 + pad), (255, 0, 0), thick)
    for i, r in enumerate(readings, start=1):
        x0, y0, x1, y1 = r.box.x0, r.box.y0 + pad, r.box.x1, r.box.y1 + pad
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), thick)
        cv2.putText(out, str(i), (x0 + 2, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 160, 0), thick)
        cv2.putText(out, str(r.best), (x0 + 2, y1 + int(22 * font)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 255), thick)
    return out
    # size lines and text from the sign height, so labels stay small next to the signs
    sign_h = int(np.median([r.box.y1 - r.box.y0 for r in readings]))
    thick = max(1, sign_h // 50)
    font = max(0.3, sign_h / 150)
    if band is not None:
        cv2.rectangle(out, (band.x0, band.y0), (band.x1, band.y1), (255, 0, 0), thick)
    for i, r in enumerate(readings, start=1):
        b = r.box
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (0, 0, 255), thick)
        cv2.putText(out, str(i), (b.x0 + 2, max(10, b.y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 160, 0), thick)
        cv2.putText(out, str(r.best), (b.x0 + 2, b.y1 + int(22 * font)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 255), thick)
    return out
