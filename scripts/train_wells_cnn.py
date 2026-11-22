"""Train the Wells sign CNN and export predictions for the fixed v2 split."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from train_sign_cnn import BUCKETS, run_epochs, sampler
from indus.cnn import SignCNN, predict, to_input
from indus.data import LIPI_COMMIT, REPO_ROOT
from indus.wells import SPLIT_FILE, wells_codes

PROCESSED = REPO_ROOT / "data" / "processed"
RENDERS = PROCESSED / "synthetic_carved_wells_64.npz"
CROPS = PROCESSED / "real_crops" / "crops_v2.npz"
OUTPUT = PROCESSED / "wells_runs"
FIELDS = ("cisi", "position", "split", "site", "target", "predicted", "top5")


def read_crops(path, plan):
    """Load the known-count crops and verify complete, disjoint fixed split coverage."""
    with np.load(path, allow_pickle=False) as archive:
        crops = {key: archive[key] for key in ("images", "labels", "seal", "split", "pos")}
    if crops["images"].shape != (len(crops["labels"]), 64, 64) or crops["images"].dtype != np.uint8:
        raise ValueError("Expected uint8 (N, 64, 64) crops")
    if set(crops["seal"]) != set(plan.index) or not np.array_equal(plan.loc[crops["seal"], "split"], crops["split"]):
        raise ValueError("Crop archive does not match the fixed v2 seal split")
    positions = {}
    for cisi, pos in zip(crops["seal"], crops["pos"]):
        positions.setdefault(cisi, []).append(int(pos))
    if any(sorted(row) != list(range(len(row))) for row in positions.values()):
        raise ValueError("Crop positions must be unique and contiguous within each seal")
    return crops


def training_rows(crops, plan, scope):
    """The original train/val partitions, optionally filtered to Mohenjo-daro only."""
    site = plan.loc[crops["seal"], "site"].to_numpy()
    allowed = site == "Mohenjo-daro" if scope == "mohenjo" else np.ones(len(site), dtype=bool)
    train = allowed & (crops["split"] == "train")
    val = allowed & (crops["split"] == "val")
    if not train.any() or not val.any():
        raise ValueError("Both the training and validation partitions need crops")
    return train, val


def class_numbers(labels, codes):
    """Map Wells codes to the trainer's one-based class numbers."""
    indices = {code: i for i, code in enumerate(codes, start=1)}
    try:
        return np.array([indices[int(label)] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown Wells code in labels: {error.args[0]}") from error


def read_renders(path, codes):
    with np.load(path, allow_pickle=False) as archive:
        images, labels = archive["images"], archive["labels"]
    if images.shape != (len(labels), 64, 64) or images.dtype != np.uint8 or set(labels) != set(codes):
        raise ValueError("Render archive does not match the pinned Wells vocabulary")
    return to_input(images), class_numbers(labels, codes)


def load_weights(path, model, codes, phase):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("arch") != "small" or data.get("training_phase") != phase
            or tuple(data.get("wells_codes", ())) != codes or data.get("lipi_commit") != LIPI_COMMIT):
        raise ValueError(f"Expected {phase} Wells small-CNN weights from the pinned corpus")
    model.load_state_dict(data["state_dict"])
    return data


def save_weights(model, path, codes, phase, **metadata):
    """Write a checkpoint atomically once a stage is complete."""
    checkpoint = dict(arch="small", state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                      training_phase=phase, wells_codes=list(codes), lipi_commit=LIPI_COMMIT, **metadata)
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)
    print(f"Checkpoint saved: {path}", flush=True)


def pretrain(model, renders, gen, batch, epochs, device):
    x, y = renders
    steps = math.ceil(len(x) / batch)
    print(f"Pretrain: {len(x):,} renders, {steps} steps per epoch", flush=True)
    start = time.perf_counter()
    run_epochs(model, sampler(x, y, gen, batch), steps, epochs, 0.002, device, "Wells pretrain")
    return time.perf_counter() - start


def finetune(model, renders, crops, train, val, codes, gen, batch, epochs, device):
    real_x = to_input(crops["images"])
    real_y = class_numbers(crops["labels"], codes)
    real_batch = sampler(real_x[train], real_y[train], gen, batch // 2)
    render_batch = sampler(*renders, gen, batch // 2)

    def mixed():
        real, synthetic = real_batch(), render_batch()
        return torch.cat([real[0], synthetic[0]]), torch.cat([real[1], synthetic[1]])

    steps = math.ceil(int(train.sum()) / (batch // 2))
    print(f"Fine-tune: {train.sum():,} training crops, {val.sum():,} validation crops, "
          f"{steps} steps per epoch", flush=True)
    start = time.perf_counter()
    best, _ = run_epochs(model, mixed, steps, epochs, 0.0005, device, "Wells fine-tune",
                         val=(real_x[val], real_y[val]))
    return best, time.perf_counter() - start


def export_predictions(model, path, data, crops, plan, codes, split):
    train, val = training_rows(crops, plan, data["scope"])
    if (set(data["train_seals"]) != set(crops["seal"][train])
            or set(data["val_seals"]) != set(crops["seal"][val])):
        raise ValueError("Checkpoint seal provenance no longer matches the fixed training plan")
    if split == "harappa" and data["scope"] != "mohenjo":
        raise ValueError("Use the Mohenjo-daro-only model for the Harappa comparison")
    held = crops["split"] == split
    labels = crops["labels"][held]
    guesses, _ = predict(model, to_input(crops["images"][held]), batch=64)
    # cnn.predict returns one-based class numbers. Translate back to Wells codes.
    top = np.asarray(codes)[guesses - 1]
    seal = crops["seal"][held]
    pos = crops["pos"][held]
    sites = plan.loc[seal, "site"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for cisi, p, site, true, ranked in zip(seal, pos, sites, labels, top):
            writer.writerow([cisi, p, split, site, true, ranked[0], " ".join(map(str, ranked))])

    ranking = top == labels[:, None]
    frequency = Counter(crops["labels"][train])
    support = np.array([frequency[s] for s in labels])
    old_test = set((REPO_ROOT / "data/raw/cisi_seals/test_seals.txt").read_text().split())
    old_overlap = np.isin(seal, sorted(old_test))
    output = {
        "split": split, "scope": data["scope"], "seals": len(set(seal)), "crops": len(labels),
        "top1": float(ranking[:, 0].mean()), "top5": float(ranking.any(axis=1).mean()),
        "old_test_seals_covered": len(set(seal[old_overlap])), "old_test_seals_total": len(old_test),
        "by_training_frequency": [],
    }
    if old_overlap.any():
        output["old_test_subset"] = {"crops": int(old_overlap.sum()),
                                     "top1": float(ranking[old_overlap, 0].mean()),
                                     "top5": float(ranking[old_overlap].any(axis=1).mean())}
    for lo, hi, name in BUCKETS:
        mask = (support >= lo) & (support <= hi)
        if mask.any():
            output["by_training_frequency"].append({
                "bin": name, "crops": int(mask.sum()), "top1": float(ranking[mask, 0].mean()),
                "top5": float(ranking[mask].any(axis=1).mean()),
            })
    path.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, default=CROPS)
    parser.add_argument("--renders", type=Path, default=RENDERS)
    parser.add_argument("--run-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="stage", required=True)
    measure = sub.add_parser("time", help="time one complete render epoch and one fine-tune epoch; no checkpoints")
    measure.add_argument("--scope", choices=("all", "mohenjo"), default="all")
    render = sub.add_parser("pretrain")
    render.add_argument("--epochs", type=int, default=8)
    real = sub.add_parser("finetune")
    real.add_argument("--epochs", type=int, default=15)
    real.add_argument("--init", type=Path, help="reuse a render-only checkpoint (defaults to the run directory)")
    real.add_argument("--scope", choices=("all", "mohenjo"), default="all")
    export = sub.add_parser("export")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--split", choices=("val", "test", "harappa"), default="test")
    args = parser.parse_args()
    if args.batch < 4 or args.batch % 2 or getattr(args, "epochs", 1) < 1:
        parser.error("batch must be even and at least 4, and epochs must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Select a GPU runtime and verify driver/PyTorch compatibility.")
    print(f"PyTorch {torch.__version__}, device {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    codes = wells_codes()
    model = SignCNN(n_classes=len(codes)).to(device)
    pre_path = args.run_dir / "wells_renders.pt"
    if args.stage == "pretrain":
        if pre_path.exists():
            parser.error(f"{pre_path} already exists; reuse it or select a fresh --run-dir")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        seconds = pretrain(model, read_renders(args.renders, codes), gen, args.batch, args.epochs, device)
        save_weights(model, pre_path, codes, "renders_only", seed=args.seed, epochs=args.epochs,
                     batch=args.batch, seconds=seconds)
        return

    plan = pd.read_csv(SPLIT_FILE).set_index("cisi")
    if not plan.index.is_unique:
        raise ValueError("Duplicate seal in the fixed split")
    crops = read_crops(args.crops, plan)
    if args.stage == "export":
        data = load_weights(args.checkpoint, model, codes, "real_finetuned")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        path = args.run_dir / f"predictions_{data['scope']}_{args.split}.csv"
        export_predictions(model, path, data, crops, plan, codes, args.split)
        return

    train, val = training_rows(crops, plan, args.scope)
    renders = read_renders(args.renders, codes)
    if args.stage == "time":
        a = pretrain(model, renders, gen, args.batch, 1, device)
        _, b = finetune(model, renders, crops, train, val, codes, gen, args.batch, 1, device)
        print(f"Render epoch: {a:.1f}s. Fine-tune epoch: {b:.1f}s. "
              f"Estimate for 8 render + 2 × 15 fine-tune epochs: {(8 * a + 30 * b) / 60:.0f} min "
              "before setup and other experiments. Use this as a planning estimate.", flush=True)
        return

    init = args.init or pre_path
    path = args.run_dir / f"wells_{args.scope}.pt"
    if path.exists():
        parser.error(f"{path} already exists; choose a fresh --run-dir")
    load_weights(init, model, codes, "renders_only")
    best, seconds = finetune(model, renders, crops, train, val, codes, gen, args.batch, args.epochs, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_weights(model, path, codes, "real_finetuned", scope=args.scope, seed=args.seed, epochs=args.epochs,
                 batch=args.batch, seconds=seconds, best_epoch=best,
                 train_seals=sorted(set(crops["seal"][train].tolist())),
                 val_seals=sorted(set(crops["seal"][val].tolist())))


if __name__ == "__main__":
    main()
