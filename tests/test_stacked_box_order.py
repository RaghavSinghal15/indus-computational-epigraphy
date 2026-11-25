"""Box order for signs whose horizontal centers nearly coincide."""
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


def test_m416_places_fork_before_the_stacked_strokes(tmp_path):
    label = tmp_path / "M-416_imp_a.txt"
    label.write_text(
        "0 0.718902 0.398560 0.372989 0.168848\n"
        "0 0.719666 0.243455 0.417320 0.204188\n"
        "0 0.246551 0.375654 0.369932 0.201571\n"
        "0 0.429988 0.217277 0.272098 0.269634\n"
        "0 0.596610 0.107330 0.076432 0.149215\n"
    )
    boxes = read_yolo_boxes(label, 274, 320)
    assert boxes[3][1] < boxes[4][1]

    other = label.rename(tmp_path / "other.txt")
    assert read_yolo_boxes(other, 274, 320) == [*boxes[:3], boxes[4], boxes[3]]
