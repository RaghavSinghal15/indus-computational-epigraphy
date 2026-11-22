"""Sweep Witten–Bell orders for single-mask sign restoration on the fixed C1 split."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

import numpy as np


@dataclass
class Row:
    log_backoff: float
    log_boost: dict[int, float]


class MaskedOrders:
    """Sum only the factors changed by the blank, for successive n-gram orders."""

    def __init__(self, texts, max_order, *, repeat_start=False):
        self.index = {s: i for i, s in enumerate(sorted({s for t in texts for s in t}))}
        self.unknown = len(self.index)
        self.end = self.unknown + 1
        self.start = self.unknown + 2
        self.max_order = max_order
        padded = [self.padded(t) for t in texts]
        uni = np.ones(self.end + 1)
        for text in padded:
            for token in text[max_order - 1:]:
                uni[token] += 1
        self.log_uni = np.log(uni / uni.sum())
        self.rows = [{}]
        self.projections = [[]]
        self.singletons = [0.0]
        for width in range(1, max_order):
            counts = defaultdict(Counter)
            for text in padded:
                for pos in range(max_order - 1, len(text)):
                    history = text[pos - width:pos]
                    # Keep #55's trigram padding. Orders above three add a
                    # component only if the longer history adds information.
                    if not repeat_start and width > 2 and history[1] == self.start:
                        continue
                    counts[history][text[pos]] += 1
            level = {}
            projections = [defaultdict(list) for _ in range(width)]
            for history, followers in counts.items():
                kinds = len(followers)
                backoff = kinds / (followers.total() + kinds)
                boost = {
                    token: math.log1p(count / (kinds * math.exp(self.log_prob(history[1:], token))))
                    for token, count in followers.items()
                }
                row = Row(math.log(backoff), boost)
                level[history] = row
                for slot, candidate in enumerate(history):
                    if candidate <= self.unknown:
                        rest = history[:slot] + history[slot + 1:]
                        projections[slot][rest].append((candidate, row))
            self.rows.append(level)
            self.projections.append(projections)
            self.singletons.append(
                sum(followers.total() == 1 for followers in counts.values()) / len(counts)
                if counts else 0.0
            )

    def padded(self, text):
        ids = tuple(self.index.get(s, self.unknown) for s in text)
        return (self.start,) * (self.max_order - 1) + ids + (self.end,)

    def log_prob(self, history, target):
        probability = self.log_uni[target]
        for width in range(1, len(history) + 1):
            row = self.rows[width].get(history[-width:])
            if row is not None:
                probability += row.log_backoff + row.log_boost.get(target, 0.0)
        return probability

    def scores(self, padded, mask):
        """For each order and candidate, return log scores up to a shared constant."""
        scores = np.empty((self.max_order, self.unknown + 1))
        scores[0] = self.log_uni[:self.unknown + 1]
        for width in range(1, self.max_order):
            scores[width] = scores[width - 1]
            row = self.rows[width].get(padded[mask - width:mask])
            if row is not None:
                for candidate, boost in row.log_boost.items():
                    if candidate <= self.unknown:
                        scores[width, candidate] += boost
            for offset in range(1, min(width + 1, len(padded) - mask)):
                pos = mask + offset
                slot = width - offset
                rest = padded[pos - width:mask] + padded[mask + 1:pos]
                for candidate, row in self.projections[width][slot].get(rest, ()):
                    scores[width, candidate] += row.log_backoff + row.log_boost.get(padded[pos], 0.0)
        return scores


def evaluate(model, texts):
    # Per-order totals for top-1 hits, top-5 hits, log loss and gold history coverage.
    totals = np.zeros((model.max_order, 4))
    positions = 0
    for text in texts:
        ids = model.padded(text)
        for mask in range(model.max_order - 1, len(ids) - 1):
            positions += 1
            target = ids[mask]
            logits = model.scores(ids, mask)
            truth = logits[:, target]
            if target != model.unknown:
                # Equal scores come from sparse counts. Ascending sign ID breaks ties.
                delta = logits[:, :model.unknown] - truth[:, None]
                rank = (delta > 1e-12).sum(axis=1) + (np.abs(delta[:, :target]) <= 1e-12).sum(axis=1)
                totals[:, 0] += rank == 0
                totals[:, 1] += rank < 5
            totals[:, 2] += np.logaddexp.reduce(logits, axis=1) - truth
            totals[:, 3] += [True] + [
                ids[mask - w:mask] in model.rows[w] for w in range(1, model.max_order)
            ]
    return totals, positions


def summarize(totals, size, singletons):
    return [
        dict(order=n, top1=hits1 / size, top5=hits5 / size, pseudo_ppl=math.exp(loss / size),
             nll=loss / size, seen_gold_history=seen / size, singleton_histories=singleton)
        for n, ((hits1, hits5, loss, seen), singleton) in enumerate(zip(totals, singletons), 1)
    ]


def check_scores(model, texts):
    """Compare sparse score updates with recomputing each full candidate sequence."""
    for text in texts:
        padded = model.padded(text)
        for mask in range(model.max_order - 1, len(padded) - 1):
            logits = model.scores(padded, mask)
            for n in range(1, model.max_order + 1):
                brute = []
                for s in range(model.unknown + 1):
                    candidate = padded[:mask] + (s,) + padded[mask + 1:]
                    brute.append(sum(
                        model.log_prob(candidate[i - n + 1:i], candidate[i])
                        for i in range(model.max_order - 1, len(candidate))
                    ))
                brute = np.array(brute)
                np.testing.assert_allclose(logits[n - 1] - logits[n - 1, 0], brute - brute[0],
                                           atol=1e-11, rtol=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/icit_1837_sequences.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-start", action="store_true",
                        help="allow multiple start markers in orders above three")
    args = parser.parse_args()
    with args.data.open(newline="") as file:
        texts = [tuple(int(token) for token in row if token.strip())[::-1] for row in csv.reader(file)]
    assert len(texts) == len(set(texts)) == 1837
    max_order = max(map(len, texts)) + 1  # All signs in the longest inscription plus a boundary.
    order = list(range(len(texts)))
    random.Random(0).shuffle(order)
    cutoff = int(len(texts) * 0.8)
    train, test = ([texts[i] for i in part] for part in (order[:cutoff], order[cutoff:]))
    fold_order = list(range(len(train)))
    random.Random(2026).shuffle(fold_order)
    folds = [set(fold_order[i::5]) for i in range(5)]

    fixture = [(1, 2, 1), (1, 2, 3), (3, 2), (3, 1, 2, 3)]
    check_scores(MaskedOrders(fixture, 5, repeat_start=args.repeat_start), [*fixture, (1, 9, 2)])
    print("Sparse scores match full candidate scores on a fixture.", flush=True)
    validation_total = np.zeros((max_order, 4))
    fit_total = np.zeros_like(validation_total)
    valid_positions = fit_positions = 0
    fold_summaries = []
    for i, fold in enumerate(folds, 1):
        fit = [t for j, t in enumerate(train) if j not in fold]
        validation = [t for j, t in enumerate(train) if j in fold]
        model = MaskedOrders(fit, max_order, repeat_start=args.repeat_start)
        val, n_val = evaluate(model, validation)
        resub, n_fit = evaluate(model, fit)
        validation_total += val
        fit_total += resub
        valid_positions += n_val
        fit_positions += n_fit
        fold_summaries.append(summarize(val, n_val, model.singletons))
        print(f"Finished fold {i} with {n_val} validation masks.", flush=True)
        del model
    singleton_average = np.mean([[r["singleton_histories"] for r in fold] for fold in fold_summaries], axis=0)
    cv = summarize(validation_total, valid_positions, singleton_average)
    resub = summarize(fit_total, fit_positions, singleton_average)
    winners = {
        "top1": max(cv, key=lambda r: r["top1"])["order"],
        "top5": max(cv, key=lambda r: r["top5"])["order"],
        "pseudo_ppl": min(cv, key=lambda r: r["pseudo_ppl"])["order"],
    }
    # Fix the shortlist before looking at the legacy benchmark test scores.
    shortlisted = {2, 3, 4, *winners.values()}
    final_model = MaskedOrders(train, max(shortlisted), repeat_start=args.repeat_start)
    test_total, test_positions = evaluate(final_model, test)
    assert test_positions == 1693 and sum(s not in final_model.index for t in test for s in t) == 66
    report = dict(
        train_texts=len(train), test_texts=len(test), median_length=float(np.median(list(map(len, texts)))),
        max_order=max_order, repeat_start=args.repeat_start,
        validation_positions=valid_positions, fit_positions=fit_positions,
        test_positions=test_positions, selected=winners, cv=cv, fit=resub, folds=fold_summaries,
        test=[r for r in summarize(test_total, test_positions, final_model.singletons) if r["order"] in shortlisted],
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}. Validation choices: {winners}", flush=True)


if __name__ == "__main__":
    main()
