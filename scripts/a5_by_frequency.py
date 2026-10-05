"""Test accuracy of one model split by how many real training examples each sign had.

Usage:
    python scripts/a5_by_frequency.py --model data/processed/models/cnn_equal.pt

Same test seals, cuts (equal slices, sign count known) and sign statistics as
A6 (pairs + ends, lambda 0.5, the setting chosen on validation there), so the
"all signs" row must match reports/a6_context_equal.md (sanity check).

Also lists every test seal with no doubtful or illegible sign, with which
signs were read right, to pick a fair example for the slides.

Writes reports/a5_by_frequency.md and data/processed/a5_test_readings.pkl
(per seal: label, true signs and top-5 guesses left to right) for figures.
"""

import argparse
import pickle
from collections import Counter

import numpy as np
import torch

from a6_context_decode import CISI, seal_rows            # same cutting and CNN scoring as A6
from indus.cnn import load_model
from indus.data import REPO_ROOT, load_labelled_seals
from indus.decode import bigram_model, edge_model, posteriors

OUT = REPO_ROOT / "reports" / "a5_by_frequency.md"
READINGS = REPO_ROOT / "data" / "processed" / "a5_test_readings.pkl"
LAM = 0.5
BUCKETS = [(0, 0, "never"), (1, 4, "1 to 4"), (5, 19, "5 to 19"), (20, 10**9, "20 or more")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO_ROOT / "data" / "processed" / "models" / "cnn_equal.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model = [load_model(args.model, args.device)]

    df = load_labelled_seals()
    df = df[df["impression_photo"] != ""]
    test = set((CISI / "test_seals.txt").read_text().split())
    val = set((CISI / "val_seals.txt").read_text().split())
    held_out = df[df["cisi_label"].isin(test | val)]["signs_reading"]
    log_trans, log_uni = bigram_model([[int(s.lstrip("*")) for s in seq][::-1] for seq in held_out])
    train_texts = [[int(s.lstrip("*")) for s in seq][::-1]
                   for seq in df[~df["cisi_label"].isin(test | val)]["signs_reading"]]
    log_left, log_right = edge_model(train_texts, log_uni)

    # how many real training crops each sign had (the CNN's real examples)
    crops = np.load(REPO_ROOT / "data" / "processed" / "real_crops" / "crops.npz")
    train_count = Counter(crops["labels"][crops["split"] == "train"].tolist())

    labels = [l for l in df["cisi_label"] if l in test]   # seal_rows keeps the data frame's order
    rows = seal_rows(model, df, test, "equal", False)
    records, seals = [], []                               # one record per scorable sign
    for label, (lp, truth) in zip(labels, rows):
        post = posteriors(lp, log_trans, log_left, LAM, log_right)
        top_cnn = np.argsort(-lp, axis=1)[:, :5] + 1      # class index -> sign number
        top_ctx = np.argsort(-post, axis=1)[:, :5] + 1
        seals.append({"label": label, "truth": truth, "top_cnn": top_cnn, "top_ctx": top_ctx})
        for t, a, b in zip(truth, top_cnn, top_ctx):
            if t.startswith("*") or t == "0":
                continue
            s = int(t)
            records.append((train_count.get(s, 0), a[0] == s, s in a, b[0] == s, s in b))
    r = np.array(records, dtype=float)

    def row(name, m):
        return (f"| {name} | {int(m.sum())} | {r[m, 1].mean():.1%} | {r[m, 2].mean():.1%} "
                f"| {r[m, 3].mean():.1%} | {r[m, 4].mean():.1%} |")
    table = [row("all signs", np.ones(len(r), bool))]
    table += [row(name, (r[:, 0] >= lo) & (r[:, 0] <= hi)) for lo, hi, name in BUCKETS]

    clean = []
    for s in seals:
        if any(t.startswith("*") or t == "0" for t in s["truth"]):
            continue
        right = sum(int(t) == g[0] for t, g in zip(s["truth"], s["top_ctx"]))
        top3 = sum(int(t) in g[:3] for t, g in zip(s["truth"], s["top_ctx"]))
        clean.append(f"| {s['label']} | {len(s['truth'])} | {right} | {top3} | "
                     f"{' '.join(s['truth'][::-1])} |")

    lines = [
        "# A5: test accuracy by how many real training examples a sign had",
        "",
        f"Model: {args.model}. Test: {len(r):.0f} scorable signs on {len(rows)} seals, equal cuts with the "
        f"sign count known. Statistics: pairs + ends, lambda {LAM} (chosen on validation in A6).",
        "",
        "| Real training crops of the sign | Test signs | CNN top-1 | CNN top-5 | CNN + statistics top-1 | "
        "CNN + statistics top-5 |",
        "| --- | --- | --- | --- | --- | --- |",
        *table,
        "",
        "Sanity check: the 'all signs' row should match reports/a6_context_equal.md.",
        "",
        "## Test seals with no doubtful or illegible sign (CNN + statistics)",
        "",
        "| Seal | Signs | Right first guess | Right in top 3 | True signs (reading order) |",
        "| --- | --- | --- | --- | --- |",
        *clean,
    ]
    OUT.write_text("\n".join(lines) + "\n")
    READINGS.write_bytes(pickle.dumps(seals))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
