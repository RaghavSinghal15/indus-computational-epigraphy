"""Assignment of sign IDs to the stacked pair in the L-29 photo."""
from indus.data import read_yolo_boxes


def test_l29_reads_three_strokes_before_the_caret(tmp_path):
    label = tmp_path / "L-29_imp_a.txt"
    label.write_text(
        "0 0.095578 0.211773 0.191156 0.396482\n"
        "0 0.256786 0.211773 0.109639 0.158322\n"
        "0 0.267595 0.083897 0.165231 0.105548\n"
        "0 0.418929 0.162382 0.211558 0.305819\n"
        "0 0.689167 0.175237 0.307299 0.336942\n"
    )
    boxes = read_yolo_boxes(label, 339, 388)
    assert boxes[1][1] < boxes[2][1]

    other = label.rename(tmp_path / "other.txt")
    assert read_yolo_boxes(other, 339, 388) == [boxes[0], boxes[2], boxes[1], *boxes[3:]]
