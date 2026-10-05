"""Try several fine-tuning settings and rank them on the VALIDATION seals.

Usage:
    python scripts/sweep_sign_cnn.py            # about 15 minutes on a laptop GPU

Pretrains once (if data/processed/models/sign_cnn_pretrained.pt is missing),
then fine-tunes from that same starting point with each setting below, using
--no-test so the test seals are never looked at. Prints the ranking and the
command for the final run with the winning setting (the only run that scores
the test seals).

Careful reading the ranking: the validation set has about 170 crops, so a
difference under about 5 points top-1 can be luck. When results are that close,
prefer the simpler setting.
"""

import csv
import subprocess
import sys

from indus.data import REPO_ROOT

TRAIN = [sys.executable, str(REPO_ROOT / "scripts" / "train_sign_cnn.py")]
PRETRAINED = REPO_ROOT / "data" / "processed" / "models" / "sign_cnn_pretrained.pt"
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


def main() -> None:
    if not PRETRAINED.exists():
        print("pretraining once ...")
        subprocess.run(TRAIN + ["--no-test", "--finetune-epochs", "1"], check=True)
    before = sum(1 for _ in open(RUNS)) if RUNS.exists() else 0
    for extra in SETTINGS:
        print("\n=== fine-tune with", " ".join(extra) or "defaults")
        subprocess.run(TRAIN + ["--init", str(PRETRAINED), "--no-test", *extra], check=True)

    with open(RUNS) as f:
        rows = list(csv.DictReader(f))[max(0, before - 1):]          # this sweep's runs only
    rows.sort(key=lambda r: float(r["val_top1"]), reverse=True)
    print("\nRanking on validation (higher is better):")
    print(f"{'val top-1':>9} {'val top-5':>9} {'train acc':>9} {'epoch':>5}  setting")
    for r in rows:
        setting = f"freeze {r['freeze']}, wd {r['weight_decay']}, lr {r['finetune_lr']}, aug {r['aug']}"
        print(f"{float(r['val_top1']):>9.1%} {float(r['val_top5']):>9.1%} {float(r['train_acc_at_best']):>9.1%} "
              f"{r['best_epoch']:>5}  {setting}")
    b = rows[0]
    print("\nFinal run with the best setting (scores the test seals once):")
    print(f"python scripts/train_sign_cnn.py --init {PRETRAINED.relative_to(REPO_ROOT)} --freeze {b['freeze']} "
          f"--weight-decay {b['weight_decay']} --finetune-lr {b['finetune_lr']} --aug {b['aug']}")


if __name__ == "__main__":
    main()
