"""Cut every labelled seal into its signs and fix the train/test split (once).

Usage:
    python scripts/make_real_crops.py

Writes data/processed/real_crops/crops.npz with:
    images   uint8 (N, 64, 64)  grayscale crops
    labels   int   (N,)          Mahadevan sign number
    seal     str   (N,)          CISI label of the seal the crop came from
    split    str   (N,)          "train", "val" or "test"
    view     str   (N,)          "impression" or "seal" (the seal photo, flipped back)

Split (by seal, so no seal is in two parts), chosen once with fixed seeds and
saved so every later run uses the same seals (AGENTS.md rule 7):
  * test: 25% of the VERIFIED seals (data/raw/cisi_seals/test_seals.txt).
    Measured once, at the very end. Never used to choose settings.
  * val: 15% of the remaining verified seals (val_seals.txt). Used to choose
    settings (epochs, learning rate, ...) and to watch for overfitting.
  * train: everything else, including all ChatGPT-read seals.
Val and test crops come from impression photos only; illegible (0) and
doubtful (*N) signs are left out of them (doubtful ones stay in training).
"""

import random

import numpy as np

from indus.data import REPO_ROOT, load_labelled_seals
from indus.real_crops import sign_crops
from indus.segment import load_image

SPLIT_FILE = REPO_ROOT / "data" / "raw" / "cisi_seals" / "test_seals.txt"
VAL_FILE = REPO_ROOT / "data" / "raw" / "cisi_seals" / "val_seals.txt"
OUT = REPO_ROOT / "data" / "processed" / "real_crops" / "crops.npz"
TEST_FRACTION = 0.25
VAL_FRACTION = 0.15
SEED = 0


def test_seals(df) -> set[str]:
    if SPLIT_FILE.exists():
        return set(SPLIT_FILE.read_text().split())
    verified = sorted(df.loc[df["verified"] == "1", "cisi_label"])
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(verified, round(TEST_FRACTION * len(verified))))
    SPLIT_FILE.write_text("\n".join(chosen) + "\n")
    print(f"fixed test split: {len(chosen)} seals -> {SPLIT_FILE.relative_to(REPO_ROOT)}")
    return set(chosen)


def val_seals(df, test: set[str]) -> set[str]:
    if VAL_FILE.exists():
        return set(VAL_FILE.read_text().split())
    rest = sorted(set(df.loc[df["verified"] == "1", "cisi_label"]) - test)
    rng = random.Random(SEED + 1)
    chosen = sorted(rng.sample(rest, round(VAL_FRACTION * len(rest))))
    VAL_FILE.write_text("\n".join(chosen) + "\n")
    print(f"fixed validation split: {len(chosen)} seals -> {VAL_FILE.relative_to(REPO_ROOT)}")
    return set(chosen)


def split_of(label: str, test: set[str], val: set[str]) -> str:
    return "test" if label in test else "val" if label in val else "train"


def main() -> None:
    df = load_labelled_seals()
    test = test_seals(df)
    val = val_seals(df, test)
    images, labels, seals, splits, views = [], [], [], [], []
    for _, row in df.iterrows():
        split = split_of(row["cisi_label"], test, val)
        signs = row["signs_reading"]
        sources = [("impression", row["impression_photo"], False)]
        if split == "train" and row["seal_photo"]:
            sources.append(("seal", row["seal_photo"], True))     # mirrored photo, flipped back
        for view, path, mirrored in sources:
            if not path:
                continue
            crops, _, _ = sign_crops(load_image(path), len(signs), mirrored=mirrored)
            for crop, sign in zip(crops, signs):
                doubtful = sign.startswith("*")
                number = int(sign.lstrip("*"))
                if number == 0 or (doubtful and split != "train"):
                    continue
                images.append(crop); labels.append(number); seals.append(row["cisi_label"])
                splits.append(split); views.append(view)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, images=np.stack(images), labels=np.array(labels), seal=np.array(seals),
                        split=np.array(splits), view=np.array(views))
    splits = np.array(splits)
    print(f"{len(labels)} crops: {(splits == 'train').sum()} train, {(splits == 'val').sum()} val, "
          f"{(splits == 'test').sum()} test ({len(val)} val, {len(test)} test seals) -> {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
