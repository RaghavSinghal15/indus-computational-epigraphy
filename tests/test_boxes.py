"""Sign-box helpers (#23)."""
from indus.data import read_yolo_boxes


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
