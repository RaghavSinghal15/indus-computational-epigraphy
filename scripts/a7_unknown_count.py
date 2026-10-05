"""A7: read a seal photo with NO labels at all, not even the number of signs.

Usage:
    python scripts/a7_unknown_count.py --model data/processed/models/cnn_equal.pt

The full M1 pipeline as it would run on a new seal:
  1. find the text band (pipeline.find_text_row);
  2. for every possible count n = 1..13, find the best n pieces with the CNN
     (align.count_candidates);
  3. choose n: the candidate's score + alpha * n + beta * log P(n)
     + gamma * (how well the CNN's guesses fit the sign statistics,
     decode.log_evidence), where P(n) is how often texts of length n occur
     among the training seals; alpha, beta and gamma are chosen on the
     VALIDATION seals (count accuracy);
  4. read the chosen pieces with the CNN, optionally rescored with the
     sign-pair and start/end statistics (indus.decode, lambda 0.5, the value
     chosen on validation in A6).

Measured on the test seals (impression photos) against the old pipeline
(A1 boxes + A2 template matching, scripts/a4_evaluate_real.py):
  * how often the sign count is exactly right,
  * character error rate (CER): edit distance between the read sequence and
    the true one (insertions, deletions, substitutions each count 1), divided
    by the true length. Standard for OCR; 0 is perfect, 1 means as many
    errors as signs. Illegible signs (0) match anything; doubtful (*N) count as N.

Writes reports/a7_unknown_count.md.
"""

import argparse
import pathlib
import pickle
from collections import Counter

import numpy as np
import torch

from indus.align import count_candidates
from indus.cnn import load_model
from indus.data import REPO_ROOT, load_labelled_seals
from indus.decode import bigram_model, edge_model, log_evidence, posteriors
from indus.pipeline import read_seal
from indus.segment import load_image

CISI = REPO_ROOT / "data" / "raw" / "cisi_seals"
OUT = REPO_ROOT / "reports" / "a7_unknown_count.md"
N_MAX = 13
ALPHAS = np.arange(-3.0, 6.01, 0.25)
BETAS = [0.0, 1.0, 2.0]
GAMMAS = [0.0, 0.25, 0.5, 1.0]     # weight of the sign statistics' opinion when choosing the count
LAMBDA = 0.5


def edit_distance(guess: list[int], truth: list[int]) -> int:
    """Fewest insertions, deletions and substitutions turning guess into truth (0 in truth = any sign)."""
    d = np.arange(len(truth) + 1)
    for i, g in enumerate(guess, 1):
        prev, d[0] = d.copy(), i
        for j, t in enumerate(truth, 1):
            same = t == 0 or g == t
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (0 if same else 1))
    return int(d[-1])


def to_ints(signs: list[str]) -> list[int]:
    return [int(s.lstrip("*")) for s in signs]


def choose_n(cands: dict, alpha: float, beta: float, log_len: np.ndarray, gamma: float = 0.0, ev=None) -> int:
    """The count with the best total; ev[n] = log_evidence of the n-piece reading."""
    return max(cands, key=lambda n: cands[n][0] + alpha * n + beta * log_len[n] + (gamma * ev[n] if gamma else 0))


def read(cands, n, lm, lam):
    """Sign numbers in READING order for the chosen count."""
    lp = cands[n][2]                                   # left to right
    if lam > 0:
        lp = posteriors(lp, lm[0], lm[1], lam, lm[2])
    return [int(i) + 1 for i in lp.argmax(axis=1)][::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO_ROOT / "data" / "processed" / "models" / "sign_cnn.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model = load_model(args.model, args.device)

    df = load_labelled_seals()
    df = df[df["impression_photo"] != ""]
    test = set((CISI / "test_seals.txt").read_text().split())
    val = set((CISI / "val_seals.txt").read_text().split())
    train = df[~df["cisi_label"].isin(test | val)]

    # length prior from the training seals (add-one smoothed), as log P(n)
    lengths = Counter(train["n_signs"].astype(int))
    log_len = np.log(np.array([lengths.get(n, 0) + 1 for n in range(N_MAX + 1)], float) /
                     (sum(lengths.values()) + N_MAX + 1))

    held = df[df["cisi_label"].isin(test | val)]["signs_reading"]
    log_trans, log_uni = bigram_model([to_ints(s)[::-1] for s in held])
    log_left, log_right = edge_model([to_ints(s)[::-1] for s in train["signs_reading"]], log_uni)
    lm = (log_trans, log_left, log_right)

    # the cutting step is slow (minutes); cache it per model so reruns take seconds
    cache_file = REPO_ROOT / "data" / "processed" / f"a7_candidates_{pathlib.Path(args.model).stem}.pkl"
    cache = pickle.loads(cache_file.read_bytes()) if cache_file.exists() else {}

    def candidates(labels):
        rows = []
        for _, row in df[df["cisi_label"].isin(labels)].iterrows():
            if row["cisi_label"] not in cache:
                cache[row["cisi_label"]] = count_candidates(model, load_image(row["impression_photo"]), N_MAX)[2]
            cands = cache[row["cisi_label"]]
            ev = {n: log_evidence(c[2], *lm[:2], LAMBDA, lm[2]) for n, c in cands.items()}
            rows.append((row, cands, ev))
        cache_file.write_bytes(pickle.dumps(cache))
        return rows

    # ---- choose alpha, beta on validation (count exactly right)
    val_rows = candidates(val)
    best = None
    for gamma in GAMMAS:
        for beta in BETAS:
            for alpha in ALPHAS:
                right = sum(choose_n(c, alpha, beta, log_len, gamma, ev) == len(r["signs_reading"])
                            for r, c, ev in val_rows)
                key = (right, -gamma, -abs(alpha), -beta)        # ties: the simpler, smaller correction
                if best is None or key > best[0]:
                    best = (key, alpha, beta, gamma)
    _, alpha, beta, gamma = best
    without = max(sum(choose_n(c, a, b, log_len) == len(r["signs_reading"]) for r, c, ev in val_rows)
                  for a in ALPHAS for b in BETAS)
    print(f"chosen on validation: alpha {alpha}, beta {beta}, gamma {gamma}: count right on "
          f"{best[0][0]}/{len(val_rows)} seals (best without statistics: {without})")

    # ---- test
    test_rows = candidates(test)
    n_right = cer_cnn = cer_lm = cer_old = 0
    old_right = n_close = old_close = 0
    total = 0
    diffs = Counter()
    for row, cands, ev in test_rows:
        truth = to_ints(row["signs_reading"])
        image = load_image(row["impression_photo"])
        n = choose_n(cands, alpha, beta, log_len, gamma, ev)
        diffs[n - len(truth)] += 1
        n_right += n == len(truth)
        n_close += abs(n - len(truth)) <= 1
        cer_cnn += edit_distance(read(cands, n, lm, 0.0), truth)
        cer_lm += edit_distance(read(cands, n, lm, LAMBDA), truth)
        _, readings = read_seal(image)                          # the old pipeline: A1 boxes + A2 templates
        old = [r.best for r in readings]
        old_right += len(old) == len(truth)
        old_close += abs(len(old) - len(truth)) <= 1
        cer_old += edit_distance(old, truth)
        total += len(truth)

    k = len(test_rows)
    lines = [
        "# A7: reading seals with no labels (sign count unknown)",
        "",
        f"Model: {args.model}. Test: {k} seals, {total} signs (impression photos). "
        f"Count chosen as best candidate score + {alpha} x n + {beta} x log P(n) + {gamma} x fit to the sign "
        f"statistics; weights chosen on "
        f"{len(val_rows)} validation seals. Sign statistics: pairs + ends, lambda {LAMBDA} (chosen on validation in A6).",
        "",
        "| Pipeline | Count exactly right | Count within 1 | Character error rate |",
        "| --- | --- | --- | --- |",
        f"| Old: A1 boxes + A2 template matching | {old_right / k:.0%} | {old_close / k:.0%} | {cer_old / total:.2f} |",
        f"| New: CNN-guided cutting + CNN | {n_right / k:.0%} | {n_close / k:.0%} | {cer_cnn / total:.2f} |",
        f"| New + sign statistics | {n_right / k:.0%} | {n_close / k:.0%} | {cer_lm / total:.2f} |",
        "",
        "Character error rate = edit distance / true length (0 = perfect). "
        "Chosen count minus true count: " + ", ".join(f"{d:+d}: {c}" for d, c in sorted(diffs.items())),
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
