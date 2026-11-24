"""Sign-box helpers (#23)."""
from types import SimpleNamespace

import pytest

from indus.data import photo_signs_ltr, read_yolo_boxes


@pytest.mark.parametrize(
    ("image_file", "catalogue_order", "photo_order"),
    [
        ("L-58_imp_a.jpg", [740, 550], [550, 740]),
        ("M-657_imp_a.jpg", [520, 240], [240, 520]),
        ("M-657_alternate.jpg", [520, 240], [520, 240]),
    ],
)
def test_photo_sign_order_is_specific_to_the_image(image_file, catalogue_order, photo_order):
    seal = SimpleNamespace(image_file=image_file, signs_ltr=catalogue_order.copy())
    assert photo_signs_ltr(seal) == photo_order
    assert seal.signs_ltr == catalogue_order


def test_yolo_boxes_become_pixels_sorted_left_to_right(tmp_path):
    label = tmp_path / "seal.txt"
    # right-hand box first in the file; YOLO = class, x centre, y centre, width, height (fractions)
    label.write_text("0 0.75 0.5 0.2 0.4\n0 0.25 0.5 0.2 0.4\n")
    boxes = read_yolo_boxes(label, width=200, height=100)
    assert boxes == [(30, 30, 70, 70), (130, 30, 170, 70)]


def test_blank_lines_are_ignored(tmp_path):
    label = tmp_path / "seal.txt"
    label.write_text("\n0 0.5 0.5 1.0 1.0\n\n")
    assert read_yolo_boxes(label, 10, 10) == [(0, 0, 10, 10)]
