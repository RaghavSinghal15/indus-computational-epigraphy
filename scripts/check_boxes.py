"""Check a batch of hand-drawn sign boxes (issue #23). See docs/ANNOTATION.md.

    python scripts/check_boxes.py 3          # report problems in batch 3
    python scripts/check_boxes.py 3 --show   # also draw boxes + expected signs to data/processed/box_check/

Every seal in the batch needs a label file whose box count equals its number of signs.
Also flags boxes that overlap a lot (likely duplicates) or touch in a way that makes
left-to-right order ambiguous (two signs stacked above each other).
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from indus.data import BOXES_DIR, REPO_ROOT, load_lipi_seals, photo_signs_ltr, read_yolo_boxes
from indus.glyphs import glyph


def overlap(a, b) -> float:
    """Intersection area divided by the smaller box's area (1.0 = one box inside the other)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / max(smaller, 1)


def check(batch: int, show: bool) -> int:
    seals = load_lipi_seals().set_index("cisi")
    plan = pd.read_csv(BOXES_DIR / "batches.csv")
    folder = BOXES_DIR / f"batch_{batch:02d}"
    out = REPO_ROOT / "data" / "processed" / "box_check" / f"batch_{batch:02d}"
    problems = ok = 0
    for _, row in plan[plan.batch == batch].iterrows():
        seal = seals.loc[row.cisi]
        label = folder / (row.image_file.rsplit(".", 1)[0] + ".txt")
        if not label.exists():
            print(f"{row.cisi}: no label file {label.name}"); problems += 1; continue
        img = Image.open(seal.image).convert("RGB")
        boxes = read_yolo_boxes(label, *img.size)
        issues = []
        if len(boxes) != seal.n_signs:
            issues.append(f"{len(boxes)} boxes, expected {seal.n_signs} signs")
        for a, b in zip(boxes, boxes[1:]):
            if overlap(a, b) > 0.5:
                issues.append("two boxes overlap a lot (duplicate, or signs stacked: check order)")
        if issues:
            print(f"{row.cisi}: " + "; ".join(issues)); problems += 1
        else:
            ok += 1
        if show:
            draw_check(img, boxes, photo_signs_ltr(seal), out / f"{row.cisi}.png")
    print(f"batch {batch}: {ok} seals fine, {problems} to fix" + (f"; pictures in {out}" if show else ""))
    return problems


def draw_check(img, boxes, signs, path) -> None:
    """Photo with numbered boxes, and under each box the sign it will be labelled as."""
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        draw.rectangle(b, outline=(220, 40, 40), width=2)
        draw.text((b[0] + 2, b[1] + 2), str(i + 1), fill=(220, 40, 40))
    sheet = Image.new("RGB", (img.width, img.height + 40), "white")
    sheet.paste(img, (0, 0))
    for b, s in zip(boxes, signs):
        g = Image.fromarray(glyph(s, 36)).convert("RGB")
        sheet.paste(g, (int(np.clip((b[0] + b[2]) / 2 - 18, 0, img.width - 36)), img.height + 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=int)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    raise SystemExit(1 if check(a.batch, a.show) else 0)
