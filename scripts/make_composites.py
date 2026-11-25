"""Make synthetic inscriptions from real sign crops (issue #32). See src/indus/composite.py.

    python scripts/make_composites.py            # 5,000 composites (about 2 minutes)

Writes data/processed/composites/ with images/*.png and manifest.jsonl, one row per
composite: image and label (YOLO boxes) as relative paths, signs_reading (Wells IDs, reading order),
source_seals (every training seal used), boxes (x0, y0, x1, y1 in visual order).
This is the format scripts/train_crnn.py --composites expects.
Also writes reports/figures/composites_vs_real.png.

Sequences: real ICIT texts whose signs all have real training crops, minus every text
that occurs on a validation, test or Harappa seal (so no held-out sequence leaks in).
Fixed seed.
"""
import argparse
import json

import cv2
import numpy as np

from indus.composite import build_bank, compose, covered
from indus.data import REPO_ROOT, load_icit_full, load_sign_boxes
from indus.segment import load_image, trim_caption
from indus.wells import load_v2_seals

OUT = REPO_ROOT / "data" / "processed" / "composites"
FIGURE = REPO_ROOT / "reports" / "figures" / "composites_vs_real.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    seals = load_v2_seals()
    split = dict(zip(seals.cisi, seals.split))
    boxes = load_sign_boxes()
    train_boxes = boxes[boxes.cisi.map(split) == "train"]
    bank = build_bank(train_boxes)
    held_out = {tuple(s) for s, sp in zip(seals.signs_reading, seals.split) if sp != "train"}
    texts = [t for t in load_icit_full("reading") if tuple(t) not in held_out]
    usable = [t for t in texts if covered(t, bank) and len(t) <= 10]
    print(f"bank: {sum(map(len, bank.crops.values()))} crops of {len(bank.crops)} signs from "
          f"{train_boxes.cisi.nunique()} training seals, {len(bank.backgrounds)} backgrounds")
    print(f"sequences: {len(usable)} of {len(texts)} non-held-out ICIT texts are fully covered")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.jsonl").open("w") as f:
        for i in range(args.n):
            seq = usable[rng.integers(len(usable))]
            img, bxs, sources = compose(seq, bank, rng)
            name = f"images/c_{i:05d}.png"
            cv2.imwrite(str(OUT / name), img)
            label = f"labels/c_{i:05d}.txt"          # YOLO format for the sign detector (#36)
            h, w = img.shape[:2]
            (OUT / label).write_text("\n".join(
                f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
                for x0, y0, x1, y1 in bxs) + "\n")
            f.write(json.dumps({"image": name, "label": label, "signs_reading": [int(s) for s in seq],
                                "source_seals": sources, "boxes": bxs}) + "\n")
    print(f"{args.n} composites -> {OUT.relative_to(REPO_ROOT)}")

    # figure: 6 composites (left) next to 6 real training bands (right), all 64 px tall
    def strip(img, h=64, w=420):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (max(8, round(g.shape[1] * h / g.shape[0])), h))
        out = np.full((h, w), 255, np.uint8)
        out[:, :min(w, g.shape[1])] = g[:, :w]
        return cv2.copyMakeBorder(out, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)
    fig_rng = np.random.default_rng(1)
    comp = [strip(compose(usable[fig_rng.integers(len(usable))], bank, fig_rng)[0]) for _ in range(6)]
    real = []
    for cisi, rows in list(train_boxes.groupby("cisi"))[:60:10]:
        img = load_image(rows.image.iloc[0])
        real.append(strip(img[max(0, rows.y0.min() - 3):rows.y1.max() + 3, max(0, rows.x0.min() - 3):rows.x1.max() + 3]))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), np.hstack([np.vstack(comp), np.full((len(comp) * 70, 20), 255, np.uint8), np.vstack(real)]))
    print(f"figure -> {FIGURE.relative_to(REPO_ROOT)} (left: composites, right: real training bands)")


if __name__ == "__main__":
    main()
