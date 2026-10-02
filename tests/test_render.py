"""Checks for the carved-sign renderer."""

import cv2
import numpy as np

from indus.data import template_path
from indus.render import OUT_SIZE, _thicken, render_carved


def template(sign: int) -> np.ndarray:
    return cv2.imread(str(template_path(sign)), cv2.IMREAD_GRAYSCALE)


def test_output_is_a_colour_image_of_the_right_size():
    image = render_carved(template(342), np.random.default_rng(0))
    assert image.shape == (OUT_SIZE, OUT_SIZE, 3) and image.dtype == np.uint8


def test_same_seed_gives_the_same_image():
    a = render_carved(template(342), np.random.default_rng(7))
    b = render_carved(template(342), np.random.default_rng(7))
    assert (a == b).all()


def test_sign_is_visible_not_blank():
    # the carving must make a real difference: compare a sign with a blank drawing
    rng = np.random.default_rng(3)
    sign = render_carved(template(380), rng, sunken=True).astype(float)
    blank = render_carved(np.zeros((150, 150), np.uint8), np.random.default_rng(3), sunken=True).astype(float)
    assert np.abs(sign - blank).mean() > 5


def test_thickening_at_most_doubles_the_ink():
    # 380 is a hatched oval: unlimited thickening would fill it in
    strokes = cv2.resize(template(380), (OUT_SIZE, OUT_SIZE))
    _, strokes = cv2.threshold(strokes, 127, 255, cv2.THRESH_BINARY)
    for seed in range(10):
        thick = _thicken(strokes, np.random.default_rng(seed))
        assert (thick > 127).sum() <= 2.0 * (strokes > 127).sum()
