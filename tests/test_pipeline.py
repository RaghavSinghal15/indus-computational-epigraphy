"""Checks for A3, the full pipeline. Rows built from template drawings give a
case where the right answer is known."""

import cv2
import numpy as np

from indus.data import template_path
from indus.pipeline import find_text_row, read_seal, read_text_row


def drawn_row(ids: list[int]) -> np.ndarray:
    """BGR image of template drawings placed left to right, dark strokes on light."""
    tiles = [255 - cv2.imread(str(template_path(i)), cv2.IMREAD_GRAYSCALE) for i in ids]
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_GRAY2BGR)


def drawn_seal() -> np.ndarray:
    """A 300x300 'seal': a row of four signs along the top, a large 'animal' below."""
    seal = np.full((300, 300), 200, np.uint8)
    for n, sign in enumerate([267, 51, 380, 17]):
        strokes = cv2.imread(str(template_path(sign)), cv2.IMREAD_GRAYSCALE)
        strokes = cv2.dilate(strokes, np.ones((5, 5), np.uint8))       # thicker, like carving
        strokes = cv2.resize(strokes[25:125, 25:125], (70, 70))         # trim margins, 70 px tall
        x = 10 + 70 * n
        patch = seal[10:80, x:x + 70]
        patch[strokes > 127] = 60                                        # dark strokes on stone
    cv2.ellipse(seal, (150, 210), (110, 60), 0, 0, 360, 60, 8)      # body
    cv2.line(seal, (70, 240), (70, 295), 60, 8)                       # legs
    cv2.line(seal, (230, 240), (230, 295), 60, 8)
    return cv2.cvtColor(seal, cv2.COLOR_GRAY2BGR)


# Signs without hollow parts, so A1's known hollow-sign split doesn't interfere.
ROW = [267, 51, 380, 17]          # left to right as drawn


def test_reads_a_row_of_templates_in_reading_order():
    # an impression reads right to left, so the expected sequence is reversed
    assert [r.best for r in read_text_row(drawn_row(ROW))] == ROW[::-1]


def test_mirrored_photo_gives_the_same_sequence():
    image = drawn_row(ROW)
    normal = [r.best for r in read_text_row(image)]
    mirrored = [r.best for r in read_text_row(image[:, ::-1].copy(), mirrored=True)]
    assert normal == mirrored


def test_find_text_row_stops_above_the_animal():
    band = find_text_row(drawn_seal())
    assert band.y0 <= 20            # starts at the top of the signs (ink begins near row 16)
    assert 70 <= band.y1 <= 140     # ends below the signs, above the animal (body starts at 150)


def test_read_seal_boxes_lie_inside_the_text_band():
    band, readings = read_seal(drawn_seal())
    assert len(readings) >= 3
    for r in readings:
        b = r.box
        assert band.x0 <= b.x0 < b.x1 <= band.x1 and band.y0 <= b.y0 < b.y1 <= band.y1
