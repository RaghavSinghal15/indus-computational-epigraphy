"""Compare the Mahadevan and Wells seal readers at compatible old test positions."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from a6_context_decode import view_log_probs
from eval_wells_context import emission_prior, evaluate, seal_rows, training_texts, WEIGHTS
from indus.cnn import SignCNN, build_model
from indus.data import CISI_DIR, load_labelled_seals, wells_to_mahadevan
from indus.decode import bigram_model, edge_model, posteriors
from indus.wells import SPLIT_FILE, load_v2_seals, wells_codes
from train_wells_cnn import CROPS, load_weights, read_crops, training_rows


def shared_positions(legacy, wells, old_test):
    """Align whole equal-length readings and keep only compatible reference signs."""
    crosswalk = wells_to_mahadevan()
    coverage = Counter()
    aligned = {}
    for code in sorted(old_test):
        if code not in wells.index:
            coverage["absent"] += 1
            continue
        old = legacy.at[code, "signs_reading"]
        new = wells.at[code, "signs_reading"]
        if len(old) != len(new):
            coverage["different lengths"] += 1
            continue
        coverage["equal-length seals"] += 1
        positions = []
        for pos, (maha, well) in enumerate(zip(old, new)):
            if maha == "0" or maha.startswith("*"):
                continue
            coverage["scorable old positions"] += 1
            choices = crosswalk.get(well, set())
            if choices:
                coverage["mapped positions"] += 1
            if int(maha) in choices:
                positions.append(pos)
        if positions:
            aligned[code] = np.array(positions)
            coverage["paired positions"] += len(positions)
    return aligned, coverage


def legacy_prior(labels, old_val, old_test):
    """The legacy A6 pairs-and-ends prior with old held-out readings removed."""
    held_out = labels.index.isin(old_val | old_test)
    texts = labels.signs_reading.map(lambda signs: [int(s.lstrip("*")) for s in signs][::-1])
    transition, unigram = bigram_model(texts[held_out].tolist())
    start, end = edge_model(texts[~held_out].tolist(), unigram)
    return transition, start, end


def count_hits(emissions, target):
    top = np.argsort(-emissions, axis=1, kind="stable")[:, :5]
    return np.array([np.sum(top[:, 0] == target), np.sum((top == target[:, None]).any(axis=1))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, help="the v1 417-class CNN checkpoint")
    parser.add_argument("--wells", type=Path, help="the all-site v2 Wells CNN checkpoint")
    parser.add_argument("--crops", type=Path, default=CROPS)
    parser.add_argument("--output", type=Path, help="write the matched comparison as Markdown")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    if not args.check_data and not all((args.legacy, args.wells, args.output)):
        parser.error("provide both checkpoints and --output, or run --check-data")

    plan = pd.read_csv(SPLIT_FILE).set_index("cisi")
    labels = load_labelled_seals()
    legacy = labels[labels.impression_photo != ""].set_index("cisi_label")
    old_test = set((CISI_DIR / "test_seals.txt").read_text().split())
    old_val = set((CISI_DIR / "val_seals.txt").read_text().split())
    wells = load_v2_seals().set_index("cisi")
    assert all(plan.at[code, "split"] == "test" for code in old_test & set(wells.index))
    aligned, coverage = shared_positions(legacy, wells, old_test)
    print(f"{len(aligned)} paired seals, coverage {dict(coverage)}", flush=True)
    if args.check_data:
        return

    torch.set_num_threads(2)
    saved = torch.load(args.legacy, map_location="cpu", weights_only=True)
    original = build_model(saved.get("arch", "small"), imagenet=False).eval()
    original.load_state_dict(saved.get("state_dict", saved))
    v1_transition, v1_start, v1_end = legacy_prior(legacy, old_val, old_test)
    crops = read_crops(args.crops, plan)
    codes = wells_codes()
    updated = SignCNN(len(codes)).eval()
    checkpoint = load_weights(args.wells, updated, codes, "real_finetuned")
    if checkpoint["scope"] != "all":
        raise ValueError("Use the all-site v2 checkpoint")
    train, val = training_rows(crops, plan, "all")
    for flag, name in ((train, "train"), (val, "val")):
        if set(checkpoint[f"{name}_seals"]) != set(crops["seal"][flag]):
            raise ValueError(f"V2 {name} split does not match its checkpoint")
    texts, _, _ = training_texts(plan, crops, "all")
    v2_prior = emission_prior(texts, codes)
    val_rows = seal_rows(updated, crops, val, codes)
    best_weight = max(WEIGHTS, key=lambda w: (evaluate(val_rows, v2_prior, w)[1], -w))
    v2_rows = seal_rows(updated, crops, np.isin(crops["seal"], list(aligned)), codes)

    totals = np.zeros((4, 2), dtype=int)
    full_legacy = np.zeros((2, 2), dtype=int)
    full_count = 0
    for code in sorted(old_test):
        row = legacy.loc[code]
        signs = row.signs_reading
        raw_v1 = view_log_probs([original], row.impression_photo, len(signs), "equal", False, False)
        # Legacy A6 applies its Markov model in visual order.
        context_v1 = posteriors(raw_v1[::-1], v1_transition, v1_start, 0.5, v1_end)[::-1]
        scorable = [i for i, sign in enumerate(signs) if sign != "0" and not sign.startswith("*")]
        truth = np.array([int(signs[i]) - 1 for i in scorable])
        full_count += len(scorable)
        full_legacy[0] += count_hits(raw_v1[scorable], truth)
        full_legacy[1] += count_hits(context_v1[scorable], truth)

        positions = aligned.get(code)
        if positions is None:
            continue
        old_target = np.array([int(signs[i]) - 1 for i in positions])
        raw_v2, new_target = v2_rows[code]
        assert tuple(codes[i] for i in new_target) == tuple(wells.at[code, "signs_reading"])
        transition, start, end = v2_prior
        context_v2 = posteriors(raw_v2, transition, start, best_weight, end) if best_weight else raw_v2
        for i, (emissions, target) in enumerate(((raw_v1, old_target), (context_v1, old_target),
                                                (raw_v2, new_target[positions]), (context_v2, new_target[positions]))):
            totals[i] += count_hits(emissions[positions], target)

    n = coverage["paired positions"]
    lines = ["## Matched v1/v2 test positions", "",
             f"{len(aligned)} seals and {n} paired sign positions, using equal-length readings and a "
             "compatible Wells-to-Mahadevan reference mapping. Each reader uses its own crop pipeline "
             "and sign inventory on the same catalogue objects. V1 uses the fixed A6 context weight "
             f"0.5, and v2 uses weight {best_weight:g} selected on v2 validation.", "",
             f"Checkpoints: `{args.legacy.name}` and `{args.wells.name}`.", "",
             "| Reader | Top-1 | Top-5 |", "| --- | ---: | ---: |"]
    for name, (top1, top5) in zip(("V1 CNN", "V1 CNN + IM77 prior", "V2 CNN", "V2 CNN + ICIT prior"), totals):
        lines.append(f"| {name} | {top1}/{n} ({top1/n:.1%}) | {top5}/{n} ({top5/n:.1%}) |")
    lines += ["", "### Full legacy test split", "",
              f"The legacy-only evaluation covers {len(old_test)} seals and {full_count} scorable signs.",
              "", "| Reader | Top-1 | Top-5 |", "| --- | ---: | ---: |"]
    for name, (top1, top5) in zip(("V1 CNN", "V1 CNN + IM77 prior"), full_legacy):
        lines.append(f"| {name} | {top1}/{full_count} ({top1/full_count:.1%}) | "
                     f"{top5}/{full_count} ({top5/full_count:.1%}) |")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
