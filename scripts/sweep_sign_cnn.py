"""Compare training variants on the VALIDATION seals, each with several seeds.

Usage:
    python scripts/sweep_sign_cnn.py                                    # single renders vs rows
    python scripts/sweep_sign_cnn.py --synthetic both --crops crops_aligned.npz

For each kind of synthetic data: pretrain once (skipped if
data/processed/models/sign_cnn_pretrained_<kind>.pt exists), then fine-tune
from that starting point with each setting below and each seed, using
--no-test so the test seals are never looked at. Prints the mean and spread
over seeds for each variant, and the command for the final run.

Why seeds: the validation set has about 170 crops, so one run's score moves
by a few points just from luck (which batches came first, which random
distortions). The first sweep showed 9 different settings all within 2 points
of each other, i.e. settings matter less than that noise. Averaging 3 seeds
makes differences between DATA variants (renders vs rows, cuts) visible.
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict

import numpy as np

from indus.data import REPO_ROOT

TRAIN = [sys.executable, str(REPO_ROOT / "scripts" / "train_sign_cnn.py")]
MODELS = REPO_ROOT / "data" / "processed" / "models"
RUNS = REPO_ROOT / "reports" / "a5_runs.csv"
SETTINGS = [[], ["--aug", "1.5"]]          # the two best of the first sweep
SEEDS = [0, 1, 2]


def n_runs() -> int:
    return sum(1 for _ in open(RUNS)) - 1 if RUNS.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", nargs="+", default=["single", "both"], choices=["single", "rows", "both"])
    ap.add_argument("--crops", default="crops.npz")
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
            for seed in SEEDS:
                print(f"\n=== {kind} synthetic, {args.crops}, {' '.join(extra) or 'defaults'}, seed {seed}")
                subprocess.run(TRAIN + ["--synthetic", kind, "--init", str(pretrained), "--crops", args.crops,
                                        "--seed", str(seed), "--no-test", *extra], check=True)

    with open(RUNS) as f:
        rows = list(csv.DictReader(f))[first:]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["synthetic"], r["crops"], r["aug"])].append((float(r["val_top1"]), float(r["val_top5"])))
    ranked = sorted(groups.items(), key=lambda kv: -np.mean([v[0] for v in kv[1]]))
    print("\nValidation, mean over seeds (+- spread):")
    print(f"{'top-1':>14} {'top-5':>14}  variant")
    for (kind, crops, aug), vals in ranked:
        t1, t5 = np.array(vals).T
        print(f"{t1.mean():>7.1%} +-{t1.std():.1%} {t5.mean():>7.1%} +-{t5.std():.1%}  "
              f"{kind} synthetic, {crops}, aug {aug} ({len(vals)} seeds)")
    kind, crops, aug = ranked[0][0]
    print("\nFinal run with the best variant (scores the test seals once):")
    print(f"python scripts/train_sign_cnn.py --synthetic {kind} --crops {crops} --aug {aug} "
          f"--init data/processed/models/sign_cnn_pretrained_{kind}.pt")


if __name__ == "__main__":
    main()
