"""Does training the sign CNN on hand-drawn box crops help? (issues #31/#32 follow-up)

Devam's segmentation audit (#64) showed the frozen v2 CNN reads hand-box crops far
better than our forced cuts (same CNN, same test signs). Here we also TRAIN on them.

Both runs start from the same v2 checkpoint (wells_all.pt, trained on forced cuts) and
get the same extra fine-tune (renders + real crops, 15 epochs, best validation epoch):
  * control: real crops = forced cuts (crops_v2.npz), validation on forced cuts
  * hand-box: real crops = hand-box crops, validation on hand-box crops
so any gap is the crops, not the extra training. Each model is then scored on the
test seals with BOTH kinds of crop (and on Harappa, never trained on).

    python scripts/m1_handbox_cnn.py      # about 10 minutes on CPU
Writes data/processed/real_crops/crops_handbox.npz and reports/m1_handbox_cnn.md.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_sign_cnn import scores                                      # noqa: E402
from train_wells_cnn import class_numbers, finetune, load_weights, read_renders, save_weights   # noqa: E402
from indus.cnn import SignCNN, predict, to_input                       # noqa: E402
from indus.data import REPO_ROOT, load_sign_boxes                      # noqa: E402
from indus.real_crops import square_crop                               # noqa: E402
from indus.segment import load_image                                   # noqa: E402
from indus.wells import load_v2_seals, wells_codes                     # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
FORCED = PROCESSED / "real_crops" / "crops_v2.npz"
HAND = PROCESSED / "real_crops" / "crops_handbox.npz"
INIT = PROCESSED / "models" / "wells_all.pt"
REPORT = REPO_ROOT / "reports" / "m1_handbox_cnn.md"


def build_hand_crops() -> dict:
    """64x64 crops cut along the reviewed boxes, in reading order, with the v2 split."""
    split = dict(zip(*load_v2_seals()[["cisi", "split"]].T.values))
    boxes = load_sign_boxes()
    images, labels, seals, splits, pos = [], [], [], [], []
    for (cisi, path), rows in boxes.groupby(["cisi", "image"], sort=True):
        gray = cv2.cvtColor(load_image(path), cv2.COLOR_BGR2GRAY)
        rows = rows.sort_values("pos_ltr", ascending=False)          # right to left = reading order
        for i, r in enumerate(rows.itertuples()):
            images.append(square_crop(gray[r.y0:r.y1], r.x0, r.x1))   # same crop shape as the forced cuts
            labels.append(r.wells); seals.append(cisi); splits.append(split[cisi]); pos.append(i)
    data = dict(images=np.stack(images), labels=np.array(labels), seal=np.array(seals),
                split=np.array(splits), pos=np.array(pos))
    np.savez_compressed(HAND, **data)
    return data


def load(path) -> dict:
    with np.load(path) as a:
        return {k: a[k] for k in a.files}


def evaluate(model, crops, codes, split, seals=None):
    """(top-1, top-5, n) on one split, optionally restricted to a set of seals."""
    keep = crops["split"] == split
    if seals is not None:
        keep &= np.isin(crops["seal"], list(seals))
    y = class_numbers(crops["labels"][keep], codes)
    t1, t5 = scores(predict(model, to_input(crops["images"][keep]))[0], y)
    return t1.mean(), t5.mean(), int(keep.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    codes = wells_codes()
    forced = load(FORCED)
    hand = build_hand_crops()
    renders = read_renders(PROCESSED / "synthetic_carved_wells_64.npz", codes)
    # compare on seals that have hand boxes, so both crop types cover the same signs
    boxed = {s: set(hand["seal"][hand["split"] == s]) for s in ("test", "harappa")}

    models = {}
    base = SignCNN(len(codes)); load_weights(INIT, base, codes, "real_finetuned"); base.eval()
    models["v2 CNN (forced cuts), as released"] = base
    for name, crops in (("+15 epochs on forced cuts (control)", forced), ("+15 epochs on hand-box crops", hand)):
        torch.manual_seed(args.seed)
        gen = torch.Generator().manual_seed(args.seed)
        m = SignCNN(len(codes)); load_weights(INIT, m, codes, "real_finetuned")
        finetune(m, renders, crops, crops["split"] == "train", crops["split"] == "val", codes, gen, 128, args.epochs, "cpu")
        m.eval(); models[name] = m
        if "hand-box" in name:      # keep it for detect-then-classify (#36)
            save_weights(m, PROCESSED / "models" / "wells_handbox.pt", codes, "real_finetuned",
                         scope="all", crops="hand boxes", init=INIT.name, epochs=args.epochs, seed=args.seed)

    lines = ["# M1 · Training the sign CNN on hand-drawn box crops", "",
             "Same start (v2 `wells_all.pt`), same extra fine-tune; only the real training crops differ. "
             "Scored on test and Harappa seals that have reviewed boxes, with each kind of crop. "
             "Sign count given in every row (crops come from known boxes or forced cuts). Seed "
             f"{args.seed}, CPU, {args.epochs} epochs, best validation epoch kept.", "",
             f"Training crops: forced cuts {int((forced['split'] == 'train').sum())}, "
             f"hand boxes {int((hand['split'] == 'train').sum())} (from {len(set(hand['seal'][hand['split'] == 'train']))} training seals).", "",
             "| Model | Test, forced-cut crops | Test, hand-box crops | Harappa, forced-cut crops | Harappa, hand-box crops |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, m in models.items():
        cells = []
        for split in ("test", "harappa"):
            for crops in (forced, hand):
                t1, t5, n = evaluate(m, crops, codes, split, boxed[split])
                cells.append(f"{t1:.1%} / {t5:.1%} (n={n})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        print(lines[-1], flush=True)
    lines += ["", "Cells: top-1 / top-5. Hand-box crops of test and Harappa seals are used only for scoring.", "",
              "Reproduce: `python scripts/m1_handbox_cnn.py` after `scripts/make_real_crops_v2.py` and with "
              "`wells_all.pt` from the `wells-v2-cnn-pr60` release in `data/processed/models/`."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"report -> {REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
