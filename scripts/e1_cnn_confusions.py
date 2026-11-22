"""Compare frequency-weighted and measured Wells CNN substitution errors (E1, #41)."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np
import pandas as pd

from e1_error_propagation import STATS
from indus.data import REPO_ROOT, load_sequences
from indus.guardrails import shuffle_within
from indus.propagation import corrupt, sign_weights
from indus.wells import SPLIT_FILE


def load_confusions(path, crop_archive, split_name):
    """Count actual errors from unique held-out seal positions; keep support by true sign."""
    plan = pd.read_csv(SPLIT_FILE).set_index("cisi")["split"].to_dict()
    with np.load(crop_archive, allow_pickle=False) as crops:
        in_split = crops["split"] == split_name
        keys = list(zip(crops["seal"][in_split], crops["pos"][in_split]))
        truth = dict(zip(keys, crops["labels"][in_split]))
    if not truth or len(truth) != len(keys) or {cisi for cisi, _ in truth} != {
        cisi for cisi, name in plan.items() if name == split_name
    }:
        raise ValueError("The crop archive must completely cover the requested fixed split")
    right = total = 0
    seen = set()
    errors = defaultdict(Counter)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split_name:
                continue
            if plan.get(row["cisi"]) != split_name:
                raise ValueError(f"{row['cisi']} is outside {split_name} in the fixed v2 split")
            pos = int(row["position"])
            target, predicted = int(row["target"]), int(row["predicted"])
            key = row["cisi"], pos
            if key in seen or pos < 0 or min(target, predicted) <= 0 or max(target, predicted) >= 999:
                raise ValueError(f"Invalid or duplicate prediction row: {key}")
            if key not in truth or truth[key] != target:
                raise ValueError(f"Prediction truth does not match the v2 crop archive: {key}")
            seen.add(key)
            total += 1
            if target == predicted:
                right += 1
            else:
                errors[target][predicted] += 1
    if seen != truth.keys():
        raise ValueError(f"Missing predictions for {len(truth.keys() - seen)} fixed split positions")
    if not total or not errors:
        raise ValueError("Provide held-out v2 predictions containing measured errors")
    return right, total, len({cisi for cisi, _ in seen}), errors


def measured_corruption(texts, accuracy, rng, errors, prior):
    """Hold overall error risk fixed, then draw errors from each true sign's observed confusion row."""
    global_errors = Counter()
    for row in errors.values():
        global_errors.update(row)
    signs, weights = prior
    draws = {}
    for source in signs:
        counts = errors.get(source, global_errors)
        choices = [(sign, count) for sign, count in counts.items() if sign != source]
        if not choices:
            choices = [(sign, weight) for sign, weight in zip(signs, weights) if sign != source]
        draws[source] = tuple(zip(*choices))
    out, misses = [], 0
    for text in texts:
        noisy = []
        for sign in text:
            if rng.random() < accuracy:
                noisy.append(sign)
            else:
                choices, weights = draws[sign]
                noisy.append(rng.choices(choices, weights)[0])
                misses += sign not in errors
        out.append(noisy)
    return out, misses


def study(texts, accuracy, errors, seeds):
    prior = sign_weights(texts)
    rows = []
    for method in ("Frequency-weighted", "Measured CNN confusions"):
        realizations = []
        for seed in seeds:
            rng = random.Random(seed)
            if method == "Frequency-weighted":
                noisy = corrupt(texts, accuracy, rng, prior)
                fallback = None
            else:
                noisy, fallback = measured_corruption(texts, accuracy, rng, errors, prior)
            shuffled = shuffle_within(noisy, random.Random(seed + 100))
            observed = [measure(noisy, random.Random(seed + 200)) for _, measure in STATS.values()]
            null = [measure(shuffled, random.Random(seed + 200)) if key != "nb" else 0.5
                    for key, (_, measure) in STATS.items()]
            changed = sum(a != b for real, prediction in zip(texts, noisy) for a, b in zip(real, prediction))
            actual = 1 - changed / sum(map(len, texts))
            realizations.append([actual, fallback / changed if fallback is not None and changed else 0,
                                 *[x for pair in zip(observed, null) for x in pair]])
        rows.append((method, np.mean(realizations, axis=0)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--crops", type=Path, required=True, help="the v2 crop archive used to verify truth and coverage")
    parser.add_argument("--split", choices=("test", "val", "harappa"), default="test")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "e1_cnn_confusions.md")
    args = parser.parse_args()
    right, total, seals, errors = load_confusions(args.predictions, args.crops, args.split)
    results = study(load_sequences(), right / total, errors, range(5))
    lines = [
        "# E1 · Error propagation with measured CNN confusions", "",
        f"Wells v2 CNN top-1 was {right}/{total} ({right / total:.1%}) "
        f"on the supplied predictions for {seals} seals from the fixed {args.split} split. "
        f"At least one error was observed for {len(errors)} distinct truth signs.",
        "",
        "| Error draw | Realized sign accuracy | Fallback error share | Jar last | Entropy (bits) | "
        "NB real vs shuffle | Masked top 5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in results:
        a, fallback, j, j0, e, e0, n, n0, m, m0 = row
        fallback_cell = f"{fallback:.1%}" if method == "Measured CNN confusions" else "—"
        lines.append(f"| {method} | {a:.1%} | {fallback_cell} | {j:.1%} / {j0:.1%} | "
                     f"{e:.3f} / {e0:.3f} | {n:.1%} / {n0:.1%} | {m:.1%} / {m0:.1%} |")
    lines += [
        "",
        "Five draws on the 1,837-text C1 benchmark. Each metric cell shows noisy readings / within-text "
        "shuffle of the same noisy readings, with the NB chance control fixed at 50%. "
        "Both simulations share the target global accuracy. Only the distribution of wrong signs changes.",
        "",
        "For a sign without measured wrong examples, draw from pooled measured errors, omitting the true "
        "sign. Frequency-weighted wrong labels are the final fallback if the pool is empty after omission. "
        "This conditions errors on the true sign and does not model errors correlated across positions, "
        "missing signs, or extra signs. Sparse confusion rows and coverage affect the comparison.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
