"""Evaluate a Wells CNN with an ICIT bigram decoder fitted without held-out texts."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from indus.cnn import SignCNN, log_probs, to_input
from indus.data import LIPI_COMMIT, load_lipi_corpus
from indus.decode import posteriors
from indus.masked import NGramModel
from indus.wells import SPLIT_FILE, wells_codes
from train_wells_cnn import CROPS, load_weights, read_crops, training_rows

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def training_texts(plan, crops, scope):
    """Clean ICIT lines after dropping held-out objects and exact sequence duplicates."""
    heldout = set(plan.index[plan["split"] != "train"])
    ordered = defaultdict(list)
    for cisi, position, label in zip(crops["seal"], crops["pos"], crops["labels"]):
        if cisi in heldout:
            ordered[cisi].append((int(position), int(label)))
    blocked = {tuple(sign for _, sign in sorted(row)) for row in ordered.values()}
    found = {}
    df = load_lipi_corpus()
    valid = df[df.signs_ltr.notna() & (df.complete == "Y") & df.direction.isin(["R/L", "L/R"])]
    for row in valid.itertuples(index=False):
        if row.cisi in heldout or row.site == "Harappa":
            continue
        if scope == "mohenjo" and row.site != "Mohenjo-daro":
            continue
        text = tuple(reversed(row.signs_ltr))
        if len(text) >= 2 and text not in blocked:
            found[text] = None
    return list(found), len(heldout), len(blocked)


def emission_prior(texts, codes):
    """Start, transition and end logs in classifier order, splitting unknown mass across unseen signs."""
    model = NGramModel().fit(texts)
    ids = np.array([model.index.get(sign, model.UNK) for sign in codes])
    unseen = ids == model.UNK
    correction = np.zeros(len(codes))
    if unseen.any():
        correction[unseen] = np.log(unseen.sum())
    return (model.log_bi[np.ix_(ids, ids)] - correction[None, :],
            model.log_bi[model.START, ids] - correction,
            model.log_bi[ids, model.END])


def seal_rows(model, crops, selected, codes):
    ix = np.flatnonzero(selected)
    emissions = log_probs(model, to_input(crops["images"][ix]), batch=64)
    code_index = {sign: i for i, sign in enumerate(codes)}
    seal = crops["seal"][ix]
    groups = defaultdict(list)
    for index, label in enumerate(seal):
        groups[label].append(index)
    rows = {}
    for label, locations in groups.items():
        order = sorted(locations, key=lambda i: crops["pos"][ix[i]])
        score = np.asarray(emissions[order], dtype=np.float64)
        truth = np.array([code_index[int(crops["labels"][ix[i]])] for i in order])
        rows[label] = (score, truth)
    return rows


def evaluate(rows, prior, weight):
    """Count exact sequences and correct sign predictions over full photos."""
    transition, start, end = prior
    n = t1 = t5 = exact = 0
    for emissions, truth in rows.values():
        posterior = posteriors(emissions, transition, start, weight, end) if weight else emissions
        # Partial sort finds the five largest entries without sorting all 712 classes.
        best = np.argmax(posterior, axis=1)
        top = np.argpartition(posterior, -5, axis=1)[:, -5:]
        right = best == truth
        t1 += int(right.sum())
        t5 += int((top == truth[:, None]).any(axis=1).sum())
        exact += int(right.all())
        n += len(truth)
    if not n:
        raise ValueError("No seals to evaluate")
    return n, t1, t5, exact


def frequency_baseline(rows, common):
    """Score the same constant guesses for every crop in a test subset."""
    n = top1 = top5 = exact = 0
    for _, truth in rows.values():
        correct = truth == common[0]
        n += len(truth)
        top1 += int(correct.sum())
        top5 += int(np.isin(truth, common).sum())
        exact += int(correct.all())
    if not n:
        raise ValueError("No seals to evaluate")
    return n, top1, top5, exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--crops", type=Path, default=CROPS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = pd.read_csv(SPLIT_FILE).set_index("cisi")
    crops = read_crops(args.crops, plan)
    codes = wells_codes()
    model = SignCNN(len(codes)).to(args.device).eval()
    checkpoint = load_weights(args.checkpoint, model, codes, "real_finetuned")
    scope = checkpoint["scope"]
    train, val = training_rows(crops, plan, scope)
    if (set(checkpoint["train_seals"]) != set(crops["seal"][train])
            or set(checkpoint["val_seals"]) != set(crops["seal"][val])):
        raise ValueError("Checkpoint training provenance does not match the fixed split")

    texts, excluded_seals, excluded_sequences = training_texts(plan, crops, scope)
    if sum(text[-1] == 740 for text in texts) <= sum(text[0] == 740 for text in texts):
        raise ValueError("Check corpus orientation: jar sign should favour the end")
    prior = emission_prior(texts, codes)
    print(f"Fitted the bigram decoder on {len(texts):,} unique eligible ICIT texts", flush=True)
    val_rows = seal_rows(model, crops, val, codes)
    trials = [(weight, evaluate(val_rows, prior, weight)) for weight in WEIGHTS]
    chosen, _ = max(trials, key=lambda trial: (trial[1][1], -trial[0]))
    print(f"Selected weight {chosen:g} on {len(val_rows)} validation seals", flush=True)

    site = plan.loc[crops["seal"], "site"].to_numpy()
    test = crops["split"] == "test"
    if scope == "mohenjo":
        test &= site == "Mohenjo-daro"
    parts = [("In-site test" if scope == "mohenjo" else "Full v2 test", seal_rows(model, crops, test, codes))]
    if scope == "mohenjo":
        parts.append(("Held-out Harappa", seal_rows(model, crops, crops["split"] == "harappa", codes)))
    else:
        old = set((SPLIT_FILE.parent.parent / "cisi_seals/test_seals.txt").read_text().split())
        parts.append(("v1 test IDs present", {k: v for k, v in parts[0][1].items() if k in old}))
    lines = [
        "# M1 v2 · Wells bigram decoding", "",
        f"Checkpoint `{args.checkpoint.name}`, scope `{scope}`, Lipi `{LIPI_COMMIT}`. "
        f"The prior used {len(texts):,} distinct clean ICIT sequences. It excludes the "
        f"{excluded_seals} held-out seal IDs, their {excluded_sequences} distinct sequences wherever repeated, "
        "and Harappa rows. The Mohenjo-daro model also restricts the prior to that site.",
        "",
        f"Validation picked the smallest weight tied for best sign top-1 among {', '.join(map(str, WEIGHTS))}. "
        f"Selected weight: **{chosen:g}** on {len(val_rows)} seals.", "",
        "| Test subset | Method | Seals | Sign crops | Top 1 | Top 5 | Exact sequence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    index = {code: i for i, code in enumerate(codes)}
    counts = np.bincount([index[int(code)] for code in crops["labels"][train]], minlength=len(codes))
    common = np.argsort(-counts, kind="stable")[:5]
    for label, rows in parts:
        n, top1, top5, exact = frequency_baseline(rows, common)
        lines.append(f"| {label} | Training frequency | {len(rows)} | {n} | {top1/n:.1%} | {top5/n:.1%} | {exact/len(rows):.1%} |")
        for method, weight in (("CNN", 0), ("CNN + bigram", chosen)):
            n, top1, top5, exact = evaluate(rows, prior, weight)
            lines.append(f"| {label} | {method} | {len(rows)} | {n} | {top1/n:.1%} | {top5/n:.1%} | {exact/len(rows):.1%} |")
    lines += ["", "Known sign counts and forced-cut impressions. The old-ID subset has Wells reference labels. "
                  "A matched v1 comparison needs the same seal IDs and positions scorable under both inventories."]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
