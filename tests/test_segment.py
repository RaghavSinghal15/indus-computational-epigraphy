"""Checks for A1 segmentation. These test that the code behaves sensibly,
not how accurate it is (that needs the D2 ground truth)."""

import cv2
import numpy as np

from indus.data import REPO_ROOT
from indus.segment import find_sign_boxes, load_image

SAMPLES = REPO_ROOT / "data" / "raw" / "seal_samples_ocr2017"


def synthetic_row(n_signs: int = 4) -> np.ndarray:
    """A light-grey strip with n hollow square 'signs', evenly spaced."""
    img = np.full((100, 100 * n_signs, 3), 200, np.uint8)
    for i in range(n_signs):
        x = 100 * i + 30
        cv2.rectangle(img, (x, 20), (x + 40, 80), (60, 60, 60), 6)
    return img


def test_hollow_signs_split_without_protection():
    # Documents the known weakness: each hollow square is cut in two.
    assert len(find_sign_boxes(synthetic_row(4))) == 8


def test_hollow_signs_kept_whole_with_protection():
    assert len(find_sign_boxes(synthetic_row(4), protect_hollow_signs=True)) == 4


def test_boxes_are_in_reading_order_right_to_left():
    boxes = find_sign_boxes(synthetic_row(4), protect_hollow_signs=True)
    xs = [b.x0 for b in boxes]
    assert xs == sorted(xs, reverse=True)


def test_mirrored_flag_reverses_reading_order_but_not_positions():
    img = synthetic_row(3)
    normal = find_sign_boxes(img, protect_hollow_signs=True)
    mirrored = find_sign_boxes(img, mirrored=True, protect_hollow_signs=True)
    assert len(normal) == len(mirrored) == 3
    # same boxes in the same places, read in the opposite order
    assert mirrored == normal[::-1]


def test_runs_on_every_sample_and_boxes_stay_inside_the_image():
    paths = sorted(SAMPLES.glob("text_row_*.png"))
    assert len(paths) == 22
    for p in paths:
        img = load_image(p)
        h, w = img.shape[:2]
        boxes = find_sign_boxes(img)
        assert len(boxes) >= 1, p.name
        for b in boxes:
            assert 0 <= b.x0 < b.x1 <= w and 0 <= b.y0 < b.y1 <= h, p.name
