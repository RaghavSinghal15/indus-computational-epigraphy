"""Sign-box helpers (#23)."""
from types import SimpleNamespace

import pytest

from indus.data import photo_signs_ltr, read_yolo_boxes


@pytest.mark.parametrize(
    ("image_file", "catalogue_order", "photo_order"),
    [
        ("L-16_imp_a.jpg", [154, 4], [4, 154]),
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


def test_box_loader_uses_alternate_photo_dimensions(tmp_path, monkeypatch):
    import pandas as pd
    from PIL import Image

    from indus import data

    batch = tmp_path / "batch_02"
    batch.mkdir()
    alternate = tmp_path / "seal_images" / "alternate.png"
    alternate.parent.mkdir()
    Image.new("L", (200, 100)).save(alternate)
    (tmp_path / "batches.csv").write_text("batch,cisi,image_file\n2,A,original.jpg\n")
    (batch / "photo_overrides.csv").write_text("cisi,image_file\nA,alternate.png\n")
    (batch / "original.txt").write_text("0 0.8 0.8 0.1 0.1\n")
    (batch / "alternate.txt").write_text("0 0.25 0.4 0.2 0.2\n")
    monkeypatch.setattr(data, "BOXES_DIR", tmp_path)
    monkeypatch.setattr(data, "LIPI_DIR", tmp_path)
    monkeypatch.setattr(data, "load_lipi_seals", lambda: pd.DataFrame([
        {"cisi": "A", "image": "original.jpg", "image_file": "original.jpg", "signs_ltr": [740]},
    ]))

    row = data.load_sign_boxes().iloc[0]
    assert row.image == str(alternate)
    assert list(row[["x0", "y0", "x1", "y1"]]) == [30, 30, 70, 50]
    assert row.wells == 740
