"""Checks for A2 template matching. These test that the matcher behaves
sensibly; accuracy on real photos needs the D2 ground truth."""

import cv2
import numpy as np

from indus.classify import classify_crop, classify_drawing, template_matrix
from indus.data import REPO_ROOT, template_path
from indus.segment import find_sign_boxes, load_image

SAMPLES = REPO_ROOT / "data" / "raw" / "seal_samples_ocr2017"


def template(sign: int) -> np.ndarray:
    return cv2.imread(str(template_path(sign)), cv2.IMREAD_GRAYSCALE)


def test_every_template_matches_itself_first():
    # Correlation of a normalized template with itself is exactly 1, and no
    # other template is identical, so each one must be its own best match.
    matrix = template_matrix()
    best = np.argmax(matrix @ matrix.T, axis=1) + 1
    assert (best == np.arange(1, 418)).all()


def test_returns_k_matches_best_first_with_valid_ids():
    matches = classify_drawing(template(342), k=5)
    assert len(matches) == 5
    assert matches[0].mahadevan_id == 342
    assert abs(matches[0].score - 1.0) < 1e-4
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)
    assert all(1 <= m.mahadevan_id <= 417 for m in matches)


def test_slightly_rotated_jar_is_still_in_top_5():
    img = template(342)
    turn = cv2.getRotationMatrix2D((75, 75), 8, 1.0)   # 8 degrees about the centre
    rotated = cv2.warpAffine(img, turn, (150, 150))
    assert 342 in [m.mahadevan_id for m in classify_drawing(rotated)]


def test_empty_image_does_not_crash():
    matches = classify_drawing(np.zeros((150, 150), np.uint8))
    assert len(matches) == 5


def test_mirrored_flag_undoes_a_left_right_flip():
    image = load_image(SAMPLES / "text_row_28.png")
    crop = find_sign_boxes(image)[0].crop(image)
    flipped = crop[:, ::-1].copy()
    normal = classify_crop(crop)[0]
    unflipped = classify_crop(flipped, mirrored=True)[0]
    # Resizing is not perfectly mirror-symmetric, so close runners-up can swap
    # places; the best match and its score must stay the same.
    assert normal.mahadevan_id == unflipped.mahadevan_id
    assert abs(normal.score - unflipped.score) < 0.02
