"""Compositor (#32): synthetic bands from real sign crops."""
import numpy as np

from indus.composite import Background, Bank, Crop, compose, covered


def tiny_bank():
    rng = np.random.default_rng(0)
    bank = Bank()
    for code in (740, 2, 817):
        img = rng.integers(0, 255, (40, 25, 3), dtype=np.uint8)
        bank.crops[code] = [Crop(img, band_height=50, seal="M-1")]
    bank.backgrounds.append(Background(np.full((50, 200, 3), 128, np.uint8), seal="M-2"))
    return bank


def test_one_box_per_sign_inside_the_image_in_visual_order():
    bank = tiny_bank()
    img, boxes, sources = compose([817, 2, 740], bank, np.random.default_rng(1))
    h, w = img.shape[:2]
    assert len(boxes) == 3 and h == 96
    assert all(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h for x0, y0, x1, y1 in boxes)
    assert [b[0] for b in boxes] == sorted(b[0] for b in boxes)     # left to right
    assert sources == ["M-1", "M-2"]


def test_covered_needs_every_sign():
    bank = tiny_bank()
    assert covered([740, 2], bank) and not covered([740, 999], bank)
