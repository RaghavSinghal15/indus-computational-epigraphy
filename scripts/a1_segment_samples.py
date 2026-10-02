"""Run A1 segmentation on every sample text row and save one overview figure.

Usage:
    python scripts/a1_segment_samples.py

Writes reports/figures/a1_segmentation_samples.png: each photo with numbered
boxes (1 = first sign read) and, underneath, the carving mask the boxes came from.
"""

from pathlib import Path

import cv2
import numpy as np

from indus.data import REPO_ROOT
from indus.segment import WORKING_HEIGHT, carving_mask, draw_boxes, find_sign_boxes, load_image

SAMPLES = REPO_ROOT / "data" / "raw" / "seal_samples_ocr2017"
OUT = REPO_ROOT / "reports" / "figures" / "a1_segmentation_samples.png"
TILE_H = 120
PAGE_W = 1900


def tile_for(path: Path) -> np.ndarray:
    image = load_image(path)
    boxes = find_sign_boxes(image)
    shown = draw_boxes(image, boxes)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = WORKING_HEIGHT / gray.shape[0]
    mask = carving_mask(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))

    shown = cv2.resize(shown, (int(image.shape[1] * TILE_H / image.shape[0]), TILE_H))
    mask = cv2.cvtColor(cv2.resize(mask, (shown.shape[1], TILE_H)), cv2.COLOR_GRAY2BGR)
    tile = np.vstack([shown, mask])
    label = f"{path.stem.replace('text_row_', '#')}  {len(boxes)} signs"
    cv2.putText(tile, label, (3, tile.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return tile


def main() -> None:
    paths = sorted(SAMPLES.glob("text_row_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    tiles = [tile_for(p) for p in paths]

    # pack tiles into rows no wider than PAGE_W
    rows, current, width = [], [], 0
    for t in tiles:
        if current and width + t.shape[1] > PAGE_W:
            rows.append(current)
            current, width = [], 0
        current.append(t)
        width += t.shape[1] + 6
    rows.append(current)

    h = tiles[0].shape[0]
    page = np.full((len(rows) * (h + 6), PAGE_W, 3), 255, np.uint8)
    for r, row in enumerate(rows):
        x = 0
        for t in row:
            page[r * (h + 6):r * (h + 6) + h, x:x + t.shape[1]] = t
            x += t.shape[1] + 6

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), page)
    print(f"{len(paths)} photos -> {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
