"""Compare known-count v2 recognition on held-out Mohenjo-daro and Harappa crops."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from indus.cnn import ResNetSign, SignCNN, predict, to_input
from indus.data import REPO_ROOT
from indus.wells import SPLIT_FILE, wells_codes

OUT = REPO_ROOT / "reports" / "m1_harappa.md"
CROPS = REPO_ROOT / "data" / "processed" / "real_crops" / "crops_v2.npz"
SITES = {"test": "Mohenjo-daro", "harappa": "Harappa"}
FIELDS = ("cisi", "position", "split", "site", "target", "predicted", "top5")


def load_checkpoint(path, device, split):
    """Require evidence that the checkpoint used only the specified source-site seals."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    needed = {"state_dict", "arch", "wells_codes", "train_seals", "val_seals"}
    if not needed <= checkpoint.keys():
        raise ValueError(f"Checkpoint needs model weights plus provenance fields: {sorted(needed)}")
    codes = tuple(checkpoint["wells_codes"])
    if codes != wells_codes():
        raise ValueError("Checkpoint class order does not match the pinned v2 Wells vocabulary")
    training = set(checkpoint["train_seals"])
    validation = set(checkpoint["val_seals"])
    source = split["site"] == "Mohenjo-daro"
    allowed_train = set(split.index[source & (split["split"] == "train")])
    allowed_val = set(split.index[source & (split["split"] == "val")])
    if not training or not training <= allowed_train or not validation <= allowed_val:
        raise ValueError("Use a checkpoint trained and tuned only on the Mohenjo-daro train/val seals")
    if checkpoint["arch"] == "small":
        model = SignCNN(n_classes=len(codes))
    elif checkpoint["arch"] == "resnet18":
        model = ResNetSign(n_classes=len(codes), imagenet=False)
    else:
        raise ValueError(f'Unsupported architecture: {checkpoint["arch"]}')
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), np.array(codes), training


def rows_for_site(data, split, split_name, site):
    want = set(split.index[(split["site"] == site) & (split["split"] == split_name)])
    indices = np.flatnonzero((data["split"] == split_name) & np.isin(data["seal"], list(want)))
    if not want or set(data["seal"][indices]) != want:
        raise ValueError(f"Crop archive must cover all {site} {split_name} seals")
    keys = list(zip(data["seal"][indices], data["pos"][indices]))
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate crop positions in {site}")
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, default=CROPS)
    parser.add_argument("--checkpoint", type=Path, help="Mohenjo-daro-only v2 model, with train/val metadata")
    parser.add_argument("--predictions", type=Path, help="optional CSV export of held-out CNN predictions")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.predictions and not args.checkpoint:
        parser.error("--predictions requires a checkpoint")
    split = pd.read_csv(SPLIT_FILE).set_index("cisi", verify_integrity=True)
    with np.load(args.crops, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("images", "labels", "seal", "split", "pos")}
    if args.checkpoint:
        model, codes, training = load_checkpoint(args.checkpoint, args.device, split)
    else:
        training = set(split.index[(split["site"] == "Mohenjo-daro") & (split["split"] == "train")])
    train_labels = arrays["labels"][(arrays["split"] == "train") & np.isin(arrays["seal"], list(training))]
    if not len(train_labels):
        raise ValueError("No source-site training crops")
    signs, counts = np.unique(train_labels, return_counts=True)
    common = signs[np.argsort(-counts, kind="stable")[:5]]
    output_rows, records = [], []
    for split_name, site in SITES.items():
        idx = rows_for_site(arrays, split, split_name, site)
        truth = arrays["labels"][idx]
        baseline = np.tile(common, (len(idx), 1))
        predictions = [(baseline, "Training frequency")]
        if args.checkpoint:
            top_ids, _ = predict(model, to_input(arrays["images"][idx]), batch=64)
            guess = codes[top_ids - 1]
            predictions.append((guess, "v2 CNN"))
            for i, choices in zip(idx, guess):
                records.append([arrays["seal"][i], arrays["pos"][i], split_name, site, arrays["labels"][i],
                                choices[0], " ".join(map(str, choices))])
        for guess, name in predictions:
            top1 = np.mean(guess[:, 0] == truth)
            top5 = np.mean((guess == truth[:, None]).any(axis=1))
            seen = np.mean(np.isin(truth, signs))
            output_rows.append(f"| {site} | {name} | {len(set(arrays['seal'][idx]))} | {len(idx)} | "
                               f"{seen:.1%} | {top1:.1%} | {top5:.1%} |")
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows(records)
    model_note = ("Frequency baseline only. To score the CNN, supply the source-site-only #30 checkpoint "
                  "with class order and split provenance.") if not args.checkpoint else (
        f"Checkpoint SHA-256: `{hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}`. "
        f"Training provenance lists {len(training)} Mohenjo-daro training seals."
    )
    lines = [
        "# M1 · Mohenjo-daro to Harappa", "", model_note, "",
        "| Site | Method | Seals | Signs | Seen in source training | Top 1 | Top 5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *output_rows, "",
        "Known sign count, forced-cut impressions, Wells categories, fixed object splits. "
        "The training frequency baseline ranks signs using Mohenjo-daro training crops only. "
        "Harappa is excluded from training and model selection. The cross-site gap can also reflect "
        "different sign frequencies or recognition difficulty. CRNN requires a separate whole-band evaluation.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
