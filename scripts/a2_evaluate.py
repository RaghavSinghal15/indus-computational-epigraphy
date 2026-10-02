"""Measure A2 template matching and draw its top-5 guesses on the sample photos.

Usage:
    python scripts/make_synthetic.py      # once, creates data/processed/synthetic/
    python scripts/a2_evaluate.py

1. Synthetic accuracy: classifies every image in data/processed/synthetic/
   (50 distorted copies of each of the 417 templates) and prints top-1 and
   top-5 accuracy plus the most common confusions. This is a BEST CASE: the
   test images are made from the templates themselves.
2. Real photos: no ground truth yet (needs D2), so it saves a figure for
   checking by eye: reports/figures/a2_top5_samples.png. Each row is one
   cut-out sign, its stroke mask, then the 5 best templates with their numbers.
"""

from collections import Counter

import cv2
import numpy as np

from indus.classify import classify_crop, classify_drawing, strokes_from_photo
from indus.data import REPO_ROOT, template_path
from indus.segment import find_sign_boxes, load_image

SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
SAMPLES = REPO_ROOT / "data" / "raw" / "seal_samples_ocr2017"
FIGURE = REPO_ROOT / "reports" / "figures" / "a2_top5_samples.png"
# Rows where A1's boxes looked clean by eye, so the figure shows A2 and not A1 errors.
SAMPLE_IDS = [28, 49, 69, 826, 888]
TILE = 60


def synthetic_accuracy() -> None:
    if not SYNTHETIC.exists():
        print("No synthetic data. Run: python scripts/make_synthetic.py")
        return
    top1 = top5 = total = 0
    confusions = Counter()
    for folder in sorted(SYNTHETIC.iterdir(), key=lambda p: int(p.name)):
        true_id = int(folder.name)
        for path in sorted(folder.glob("*.png")):
            guesses = [m.mahadevan_id for m in classify_drawing(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))]
            total += 1
            top1 += guesses[0] == true_id
            top5 += true_id in guesses
            if guesses[0] != true_id:
                confusions[(true_id, guesses[0])] += 1
    print(f"Synthetic images: {total}")
    print(f"  top-1 accuracy: {top1 / total:.1%}")
    print(f"  top-5 accuracy: {top5 / total:.1%}")
    print("  most common confusions (true -> guessed: count):")
    for (t, g), n in confusions.most_common(10):
        print(f"    {t} -> {g}: {n}")


def tile(gray: np.ndarray) -> np.ndarray:
    return np.pad(cv2.resize(gray, (TILE, TILE)), 2, constant_values=128)


def sample_figure() -> None:
    rows = []
    for sample in SAMPLE_IDS:
        image = load_image(SAMPLES / f"text_row_{sample}.png")
        for n, box in enumerate(find_sign_boxes(image), start=1):
            crop = box.crop(image)
            matches = classify_crop(crop)
            pieces = [tile(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), tile(strokes_from_photo(crop))]
            pieces += [tile(cv2.imread(str(template_path(m.mahadevan_id)), cv2.IMREAD_GRAYSCALE)) for m in matches]
            row = np.hstack(pieces)
            cv2.putText(row, f"#{sample}.{n}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
            for j, m in enumerate(matches):
                x = (j + 2) * (TILE + 4) + 3
                cv2.putText(row, f"{m.mahadevan_id} {m.score:.2f}", (x, TILE - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 200, 1)
            rows.append(row)

    # two columns so the figure fits on a slide
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    right += [np.full_like(rows[0], 255)] * (len(left) - len(right))
    gap = np.full((half * rows[0].shape[0], 12), 255, np.uint8)
    figure = np.hstack([np.vstack(left), gap, np.vstack(right)])

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), figure)
    print(f"Saved {FIGURE.relative_to(REPO_ROOT)} ({len(rows)} signs)")


if __name__ == "__main__":
    synthetic_accuracy()
    sample_figure()
