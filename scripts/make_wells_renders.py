"""Carved-look renders of every Wells sign (M1 v2 pretraining data, issue #22).

Same renderer as v1 (src/indus/render.py), but the drawings come from Lipi's sign
font (indus.glyphs) instead of Mahadevan's 417 templates, so all 712 Wells signs
are covered. Writes data/processed/synthetic_carved_wells_64.npz:
    images  uint8 (N, 64, 64) grayscale
    labels  int   (N,)        Wells code
and an example grid to reports/figures/wells_renders_examples.png.
Fixed seed: rerunning gives identical images.
"""
import cv2
import numpy as np

from indus.data import REPO_ROOT
from indus.glyphs import glyph
from indus.render import render_carved
from indus.wells import wells_codes

OUT = REPO_ROOT / "data" / "processed" / "synthetic_carved_wells_64.npz"
FIGURE = REPO_ROOT / "reports" / "figures" / "wells_renders_examples.png"
COPIES = 50
SEED = 0
EXAMPLES = [740, 817, 820, 861, 390, 342, 2]


def template(code: int) -> np.ndarray:
    """Font drawing in the v1 template convention: white strokes on black, 150 x 150."""
    return 255 - glyph(code, 150, pad=0.08)


def small(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)


def main() -> None:
    rng = np.random.default_rng(SEED)
    images, labels = [], []
    for k, code in enumerate(wells_codes()):
        t = template(code)
        for _ in range(COPIES):
            images.append(small(render_carved(t, rng)))
            labels.append(code)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(wells_codes())} signs")
    np.savez_compressed(OUT, images=np.stack(images), labels=np.array(labels))
    print(f"{len(images)} renders -> {OUT.relative_to(REPO_ROOT)}")

    rng = np.random.default_rng(SEED + 1)     # separate seed: the figure doesn't change the data
    rows = []
    for code in EXAMPLES:
        t = template(code)
        tiles = [cv2.resize(255 - t, (64, 64))] + [small(render_carved(t, rng)) for _ in range(7)]
        rows.append(np.hstack([cv2.copyMakeBorder(x, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255) for x in tiles]))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), np.vstack(rows))
    print(f"example grid -> {FIGURE.relative_to(REPO_ROOT)} (left: font drawing)")


if __name__ == "__main__":
    main()
