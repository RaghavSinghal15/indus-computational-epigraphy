"""Sign-box annotation batches (issue #23). See docs/ANNOTATION.md.

    python scripts/make_box_batch.py --plan     # (once) write data/raw/sign_boxes/batches.csv
    python scripts/make_box_batch.py 3          # copy batch 3's photos to data/processed/box_batches/batch_03/

Batch 1 holds the first-phase test and validation seals (needed first, for evaluation),
then other non-Harappa seals in a fixed random order, Harappa seals last.
"""
import argparse
import random
import shutil

import pandas as pd

from PIL import Image

from indus.data import CISI_DIR, REPO_ROOT, load_lipi_seals
from indus.glyphs import strip

BATCHES = REPO_ROOT / "data" / "raw" / "sign_boxes" / "batches.csv"
OUT = REPO_ROOT / "data" / "processed" / "box_batches"
SIZE = 100


def plan() -> None:
    """Assign every usable Lipi seal to a batch of 100 and save the list."""
    seals = load_lipi_seals()
    eval_ids = set((CISI_DIR / "test_seals.txt").read_text().split()) | set((CISI_DIR / "val_seals.txt").read_text().split())
    first = sorted(c for c in seals.cisi if c in eval_ids)
    rest = sorted(set(seals.cisi) - set(first))
    random.Random(0).shuffle(rest)
    harappa = set(seals.cisi[seals.site == "Harappa"])
    order = first + [c for c in rest if c not in harappa] + [c for c in rest if c in harappa]
    out = seals.set_index("cisi").loc[order, ["image_file", "site", "n_signs"]].reset_index()
    out.insert(0, "batch", [i // SIZE + 1 for i in range(len(out))])
    out.to_csv(BATCHES, index=False)
    print(out.groupby("batch").agg(seals=("cisi", "size"), signs=("n_signs", "sum")).to_string())


def copy_batch(n: int) -> None:
    """Copy batch n's impression photos into one folder, ready to upload to makesense.ai."""
    batches = pd.read_csv(BATCHES)
    seals = load_lipi_seals().set_index("cisi")
    dest = OUT / f"batch_{n:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    rows = batches[batches.batch == n]
    (dest / "guide").mkdir(exist_ok=True)
    for cisi in rows.cisi:
        seal = seals.loc[cisi]
        shutil.copy(seal.image, dest)
        save_guide(seal, dest / "guide" / f"{cisi}_guide.png")
    print(f"{len(rows)} photos ({rows.n_signs.sum()} signs) in {dest}; expected signs per seal in {dest / 'guide'}")


def save_guide(seal, path) -> None:
    """Photo above the expected signs, left to right, so annotators know how many boxes to draw."""
    photo = Image.open(seal.image).convert("L")
    photo.thumbnail((480, 480))
    glyphs = Image.fromarray(strip(seal.signs_ltr, 56))
    glyphs.thumbnail((photo.width, 56))
    sheet = Image.new("L", (max(photo.width, glyphs.width), photo.height + glyphs.height + 10), 255)
    sheet.paste(photo, (0, 0))
    sheet.paste(glyphs, (0, photo.height + 10))
    sheet.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", nargs="?", type=int)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    plan() if args.plan else copy_batch(args.batch)
