"""Compare seal-level and crop-level splits on the v2 crop pool and Dixit label subset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_sign_cnn import run_epochs
from indus.cnn import SignCNN, predict, to_input
from indus.data import REPO_ROOT, wells_to_mahadevan
from indus.wells import SPLIT_FILE, wells_codes

# The paper's § III.B.1 names these 38 codes and Other, while saying there are 40 classes.
NAMED = (8, 12, 15, 17, 19, 28, 48, 51, 53, 59, 102, 104, 141, 162, 173, 174, 176, 204, 205,
         211, 216, 245, 249, 267, 287, 294, 296, 302, 307, 326, 327, 328, 330, 336, 342, 387, 389, 391)
OTHER = 0
OUT = REPO_ROOT / "reports" / "m1_dixit_protocol.md"


def split_indices(seals, groups):
    """Fixed seal-level split and a random crop split of the same sizes."""
    fixed = tuple(np.flatnonzero(groups == key) for key in ("train", "val", "test"))
    if any(len(i) == 0 for i in fixed):
        raise ValueError("The mappable crops need nonempty train, val and test parts")
    sets = [set(seals[i]) for i in fixed]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("A seal occurs in more than one fixed split")
    order = np.random.default_rng(0).permutation(len(seals))
    random_split = np.split(order, np.cumsum([len(i) for i in fixed])[:2])
    return fixed, tuple(random_split)


def run(images, labels, split, codes, epochs, device, backbone=None):
    """Fit each protocol independently and select a checkpoint using validation crops."""
    train, val, test = split
    code_index = {code: i for i, code in enumerate(codes)}
    target = np.array([code_index[code] for code in labels])
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = SignCNN(n_classes=len(codes)).to(device=device, memory_format=torch.channels_last)
    if backbone is not None:
        model.features.load_state_dict(backbone)

    def batch():
        ix = rng.choice(train, size=min(64, len(train)), replace=False)
        return torch.from_numpy(images[ix]), torch.from_numpy(target[ix])

    run_epochs(model, batch, max(1, (len(train) + 63) // 64), epochs, 0.001, device,
                          "protocol pilot", val=(images[val], target[val] + 1))
    predicted, _ = predict(model, images[test], batch=64)
    top1 = np.mean(predicted[:, 0] == target[test] + 1)
    top5 = np.mean((predicted == target[test, None] + 1).any(axis=1))
    return f"{top1:.1%}", f"{top5:.1%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crops", type=Path)
    parser.add_argument("--epochs", type=int, default=0, help="0 audits label/split coverage; positive runs a from-scratch pilot")
    parser.add_argument("--init", type=Path, help="small v2 CNN checkpoint with training_phase=renders_only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.epochs < 0:
        parser.error("--epochs must be nonnegative")
    if args.init and not args.epochs:
        parser.error("--init requires positive --epochs")
    backbone = None
    if args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=True)
        if checkpoint.get("arch") != "small" or checkpoint.get("training_phase") != "renders_only":
            raise ValueError("Use a small CNN that has only trained on renders")
        if tuple(checkpoint.get("wells_codes", ())) != wells_codes():
            raise ValueError("Render checkpoint must state the pinned Wells vocabulary")
        backbone = SignCNN(n_classes=len(wells_codes())).features
        backbone.load_state_dict({
            key.removeprefix("features."): value for key, value in checkpoint["state_dict"].items()
            if key.startswith("features.")
        })
        backbone = backbone.state_dict()
    torch.set_num_threads(2)
    plan = pd.read_csv(SPLIT_FILE).set_index("cisi")
    if not plan.index.is_unique:
        raise ValueError("Duplicate seals in the fixed split")
    m = wells_to_mahadevan()
    with np.load(args.crops, allow_pickle=False) as data:
        seals, wells, splits = data["seal"], data["labels"], data["split"]
        site_pool = np.isin(seals, plan.index[plan.site == "Mohenjo-daro"]) & np.isin(splits, ["train", "val", "test"])
        eligible = np.flatnonzero(site_pool & np.array([len(m.get(code, ())) == 1 for code in wells]))
        if not np.array_equal(plan.loc[seals[eligible], "split"].to_numpy(), splits[eligible]):
            raise ValueError("Crop archive does not match the fixed v2 split")
        pool_seals, wells, splits = seals[eligible], wells[eligible], splits[eligible]
        images = to_input(data["images"][eligible]) if args.epochs else None
    collapsed = np.array([label if label in NAMED else OTHER for label in (next(iter(m[w])) for w in wells)])
    fixed, random_split = split_indices(pool_seals, splits)
    options = (("Fixed seal split", fixed), ("Random crop split", random_split))
    rows = []
    for split_name, split in options:
        train, val, test = split
        overlap = np.mean(np.isin(pool_seals[test], pool_seals[train]))
        for task, labels, codes in (
            ("Full Wells IDs", wells, wells_codes()),
            ("38 listed codes + Other", collapsed, (*NAMED, OTHER)),
        ):
            seen = np.mean(np.isin(labels[test], labels[train]))
            candidates, counts = np.unique(labels[train], return_counts=True)
            majority = np.mean(labels[test] == candidates[np.argmax(counts)])
            top1, top5 = run(images, labels, split, codes, args.epochs, args.device, backbone) if args.epochs else ("—", "—")
            rows.append(f"| {split_name} | {task} | {len(train)} / {len(val)} / {len(test)} | {overlap:.1%} | "
                        f"{len(set(labels[test]))} | {seen:.1%} | {majority:.1%} | {top1} | {top5} |")
    lines = [
        "# M1 · Dixit et al. protocol audit", "",
        "The paper [Dixit et al. (2025), § III.B](https://doi.org/10.5334/jcaa.175) reports 94.5% for "
        "MobileNet and describes a reserved, balanced set of 200 unaugmented crops. Its separate cross-validation "
        "result is 85.28% mean accuracy. The published list contains 38 named Mahadevan codes plus Other, "
        "though the text says 40 classes. This pilot uses the listed codes and leaves the unidentified class unresolved.",
        "",
        f"Both protocols use the same {len(eligible):,} eligible Mohenjo-daro crops "
        f"(of {int(site_pool.sum()):,} before excluding Wells codes without a single Mahadevan mapping). "
        f"{int((collapsed == OTHER).sum()):,} eligible crops map to Other. "
        "Using the same pool for both tasks isolates the split and label grouping from the choice of images.",
        "",
        "| Split | Target labels | Train / val / test crops | Test crops from train seals | Test classes | "
        "Test targets seen in train | Frequency top 1 | CNN top 1 | CNN top 5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows, "",
        "Random crop splits deliberately allow different positions on one seal to appear in train and test. "
        "Both splits use the same partition sizes. Frequency top 1 always picks the most common training label. A change in these scores measures a protocol effect on "
        "our forced-cut crop pool. The published study used different images, class balance, cropping, and a "
        "different model, so matching its published accuracy is a separate question.",
        "",
        f"{'Render-pretrained backbone with fresh classification heads' if args.init else 'From-scratch CNN pilot'}, "
        f"{args.epochs} epochs, seed 0 and device {args.device}."
        if args.epochs else "Coverage audit only. No recognition model was fitted.",
        "Use --init with a render-only v2 checkpoint to repeat the comparison with pretrained features. "
        "Separate fits per split keep crop-level test examples out of their respective training runs.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
