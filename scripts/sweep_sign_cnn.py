"""Try several fine-tuning settings and rank them on the VALIDATION seals.

Usage:
    python scripts/sweep_sign_cnn.py                       # single renders and rows, about 40 min on a GPU
    python scripts/sweep_sign_cnn.py --synthetic rows      # one kind of synthetic data only

For each kind of synthetic data: pretrain once (skipped if
data/processed/models/sign_cnn_pretrained_<kind>.pt exists), then fine-tune
from that same starting point with each setting below, using --no-test so the
test seals are never looked at. Prints the ranking and the command for the
final run with the winning setting (the only run that scores the test seals).

Careful reading the ranking: the validation set has about 170 crops, so a
difference under about 5 points top-1 can be luck. When results are that close,
prefer the simpler setting.
"""

import argparse
import csv
import subprocess
import sys

from indus.data import REPO_ROOT

TRAIN = [sys.executable, str(REPO_ROOT / "scripts" / "train_sign_cnn.py")]
MODELS = REPO_ROOT / "data" / "processed" / "models"
RUNS = REPO_ROOT / "reports" / "a5_runs.csv"

# each entry: extra command-line settings for one fine-tuning run
SETTINGS = [
    [],                                                         # defaults, for reference
    ["--freeze", "2"],
    ["--freeze", "3"],
    ["--weight-decay", "1e-3"],
    ["--weight-decay", "5e-3", "--freeze", "2"],
    ["--aug", "1.5"],
    ["--aug", "1.5", "--freeze", "2", "--weight-decay", "1e-3"],
    ["--finetune-lr", "2e-4"],
    ["--finetune-lr", "1e-3"],
]


def n_runs() -> int:
    return sum(1 for _ in open(RUNS)) - 1 if RUNS.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", nargs="+", default=["single", "both"], choices=["single", "rows", "both"])
    args = ap.parse_args()

    first = None
    for kind in args.synthetic:
        pretrained = MODELS / f"sign_cnn_pretrained_{kind}.pt"
        if not pretrained.exists():
            print(f"pretraining once on '{kind}' synthetic data ...")
            subprocess.run(TRAIN + ["--synthetic", kind, "--no-test", "--finetune-epochs", "1"], check=True)
        if first is None:
            first = n_runs()        # runs logged from here on belong to this sweep
        for extra in SETTINGS:
            print(f"\n=== {kind} synthetic, fine-tune with", " ".join(extra) or "defaults")
            subprocess.run(TRAIN + ["--synthetic", kind, "--init", str(pretrained), "--no-test", *extra], check=True)

    with open(RUNS) as f:
        rows = list(csv.DictReader(f))[first:]
    rows.sort(key=lambda r: float(r["val_top1"]), reverse=True)
    print("\nRanking on validation (higher is better):")
    print(f"{'val top-1':>9} {'val top-5':>9} {'train acc':>9} {'epoch':>5}  setting")
    for r in rows:
        setting = (f"{r['synthetic']} synthetic, freeze {r['freeze']}, wd {r['weight_decay']}, "
                   f"lr {r['finetune_lr']}, aug {r['aug']}")
        print(f"{float(r['val_top1']):>9.1%} {float(r['val_top5']):>9.1%} {float(r['train_acc_at_best']):>9.1%} "
              f"{r['best_epoch']:>5}  {setting}")
    b = rows[0]
    print("\nFinal run with the best setting (scores the test seals once):")
    print(f"python scripts/train_sign_cnn.py --synthetic {b['synthetic']} "
          f"--init data/processed/models/sign_cnn_pretrained_{b['synthetic']}.pt --freeze {b['freeze']} "
          f"--weight-decay {b['weight_decay']} --finetune-lr {b['finetune_lr']} --aug {b['aug']}")


if __name__ == "__main__":
    main()
