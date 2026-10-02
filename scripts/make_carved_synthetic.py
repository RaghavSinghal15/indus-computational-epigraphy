"""Make photo-like carved renders of every template: training data for the CNN.

Usage:
    python scripts/make_carved_synthetic.py

Writes 50 renders per sign to data/processed/synthetic_carved/<sign_id>/
(about 20,850 images, under a minute), and an example grid with real photo
crops underneath for comparison to reports/figures/carved_synthetic_examples.png.

Complements scripts/make_synthetic.py (P4), which distorts the clean drawings.
See src/indus/render.py for how a render is made. Fixed seed: rerunning gives
identical images.
"""

import cv2
import numpy as np

from indus.data import REPO_ROOT, template_path
from indus.render import OUT_SIZE, render_carved
from indus.segment import find_sign_boxes, load_image

OUT = REPO_ROOT / "data" / "processed" / "synthetic_carved"
FIGURE = REPO_ROOT / "reports" / "figures" / "carved_synthetic_examples.png"
SAMPLES = REPO_ROOT / "data" / "raw" / "seal_samples_ocr2017"
COPIES_PER_SIGN = 50
SEED = 0
EXAMPLE_SIGNS = [342, 51, 380, 99, 267, 17, 391]


def make_dataset() -> None:
    rng = np.random.default_rng(SEED)        # one generator for the whole run: reproducible
    for sign in range(1, 418):
        folder = OUT / str(sign)
        folder.mkdir(parents=True, exist_ok=True)
        template = cv2.imread(str(template_path(sign)), cv2.IMREAD_GRAYSCALE)
        for c in range(COPIES_PER_SIGN):
            cv2.imwrite(str(folder / f"{sign}_carved_{c:02d}.png"), render_carved(template, rng))
        if sign % 100 == 0:
            print(f"  {sign}/417 signs")
    print(f"{417 * COPIES_PER_SIGN} images -> {OUT.relative_to(REPO_ROOT)}")


def tile(image: np.ndarray) -> np.ndarray:
    return cv2.copyMakeBorder(cv2.resize(image, (OUT_SIZE, OUT_SIZE)), 2, 2, 2, 2,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def real_crop(image: np.ndarray) -> np.ndarray:
    """Pad a crop to a square by repeating its edge pixels, so it isn't squashed."""
    h, w = image.shape[:2]
    s = max(h, w)
    return cv2.copyMakeBorder(image, (s - h) // 2, (s - h + 1) // 2, (s - w) // 2, (s - w + 1) // 2,
                              cv2.BORDER_REPLICATE)


def make_figure() -> None:
    rng = np.random.default_rng(SEED + 1)    # separate seed: the figure doesn't change the dataset
    rows = []
    for sign in EXAMPLE_SIGNS:
        template = cv2.imread(str(template_path(sign)), cv2.IMREAD_GRAYSCALE)
        tiles = [tile(cv2.cvtColor(template, cv2.COLOR_GRAY2BGR))]
        tiles += [tile(render_carved(template, rng)) for _ in range(7)]
        rows.append(np.hstack(tiles))

    # last row: real crops from the sample photos, for comparison
    real = []
    for sample in [28, 49, 69, 826, 888]:
        image = load_image(SAMPLES / f"text_row_{sample}.png")
        real += [tile(real_crop(b.crop(image))) for b in find_sign_boxes(image)[:2]]
    gap = np.full((12, rows[0].shape[1], 3), 255, np.uint8)
    figure = np.vstack(rows + [gap, np.hstack(real[:8])])

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), figure)
    print(f"example grid -> {FIGURE.relative_to(REPO_ROOT)} (left: template; bottom row: real crops)")


if __name__ == "__main__":
    make_dataset()
    make_figure()
