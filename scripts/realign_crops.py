"""3a: re-cut every labelled seal with the trained CNN (recognition-driven segmentation).

Usage:
    python scripts/realign_crops.py [--model data/processed/models/sign_cnn.pt]

Same seals and the same fixed split as scripts/make_real_crops.py, but the cuts
come from indus.align instead of equal slices:
  * training seals: aligned WITH their labels (cuts go where the right signs are),
  * validation and test seals: aligned WITHOUT labels, using only the sign
    count, so they stay honest.

Writes data/processed/real_crops/crops_aligned.npz (same fields as crops.npz)
and reports/figures/aligned_cuts.png (equal slices vs aligned cuts on a few
training seals). Then retrain:
    python scripts/train_sign_cnn.py --crops crops_aligned.npz --init data/processed/models/sign_cnn.pt
"""

import argparse

import cv2
import numpy as np
import torch

from indus.align import aligned_crops
from indus.cnn import load_model
from indus.data import REPO_ROOT, load_labelled_seals
from indus.real_crops import sign_crops
from indus.segment import load_image
from make_real_crops import VAL_FILE, split_of     # same folder: scripts/

CROPS = REPO_ROOT / "data" / "processed" / "real_crops"
SPLIT_FILE = REPO_ROOT / "data" / "raw" / "cisi_seals" / "test_seals.txt"
FIGURE = REPO_ROOT / "reports" / "figures" / "aligned_cuts.png"
FIGURE_SEALS = 6


def draw(image, band, edges, color):
    out = image.copy()
    for a, b in edges:
        cv2.rectangle(out, (band.x0 + a, band.y0), (band.x0 + b, band.y1), color, 2)
    return cv2.resize(out, (300, int(300 * out.shape[0] / out.shape[1])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO_ROOT / "data" / "processed" / "models" / "sign_cnn.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model = load_model(args.model, args.device)

    df = load_labelled_seals()
    test = set(SPLIT_FILE.read_text().split())
    val = set(VAL_FILE.read_text().split())
    images, labels, seals, splits, views, figure = [], [], [], [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        split = split_of(row["cisi_label"], test, val)
        signs = row["signs_reading"]
        sources = [("impression", row["impression_photo"], False)]
        if split == "train" and row["seal_photo"]:
            sources.append(("seal", row["seal_photo"], True))
        for view, path, mirrored in sources:
            if not path:
                continue
            image = load_image(path)
            known = signs if split == "train" else None          # never use val/test labels to cut
            crops, band, edges, _ = aligned_crops(model, image, len(signs), known, mirrored)
            if split == "train" and view == "impression" and len(figure) < FIGURE_SEALS:
                _, b0, cuts = sign_crops(image, len(signs))
                equal = list(zip(cuts[:-1], cuts[1:]))
                figure.append(np.hstack([draw(image, b0, equal, (0, 0, 255)), draw(image, band, edges, (0, 160, 0))]))
            for crop, sign in zip(crops, signs):
                doubtful = sign.startswith("*")
                number = int(sign.lstrip("*"))
                if number == 0 or (doubtful and split != "train"):
                    continue
                images.append(crop); labels.append(number); seals.append(row["cisi_label"])
                splits.append(split); views.append(view)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)} seals")

    np.savez_compressed(CROPS / "crops_aligned.npz", images=np.stack(images), labels=np.array(labels),
                        seal=np.array(seals), split=np.array(splits), view=np.array(views))
    print(f"{len(labels)} crops -> {(CROPS / 'crops_aligned.npz').relative_to(REPO_ROOT)}")

    # figure: left equal slices (red), right aligned (green); same width rows stacked
    width = max(f.shape[1] for f in figure)
    rows = [cv2.copyMakeBorder(f, 0, 6, 0, width - f.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for f in figure]
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), np.vstack(rows))
    print(f"figure -> {FIGURE.relative_to(REPO_ROOT)} (red: equal slices, green: aligned)")


if __name__ == "__main__":
    main()
