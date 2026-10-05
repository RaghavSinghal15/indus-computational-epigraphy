"""A6: does reading the whole seal (CNN + sign-pair statistics) beat the CNN alone?

Usage:
    python scripts/a6_context_decode.py [--model data/processed/models/sign_cnn.pt] [--cut equal|aligned]

For every validation and test seal: cut the impression into its n signs
(equal slices, or CNN-aligned using the count only), get the CNN's
probabilities for every position, then combine with the IM77 bigram model
(indus.decode) at several weights lambda. Pairs from the validation and test
texts are removed from the IM77 counts first.

Settings (lambda, test-time augmentation on/off) are chosen on VALIDATION;
the test seals are scored with the chosen settings only. Illegible (0) and
doubtful (*N) signs are kept as positions (they still have neighbours) but not
scored. Writes reports/a6_context.md.
"""

import argparse

import numpy as np
import torch

from indus.align import aligned_crops
from indus.cnn import load_model, log_probs, to_input
from indus.data import REPO_ROOT, load_labelled_seals
from indus.decode import bigram_model, posteriors
from indus.real_crops import sign_crops
from indus.segment import load_image

CISI = REPO_ROOT / "data" / "raw" / "cisi_seals"
OUT = REPO_ROOT / "reports" / "a6_context.md"
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def seal_rows(model, df, labels_set, cut, tta):
    """Per seal: (CNN log-probs left to right (n,417), true signs left to right as strings)."""
    rows = []
    for _, row in df[df["cisi_label"].isin(labels_set)].iterrows():
        signs = row["signs_reading"]
        image = load_image(row["impression_photo"])
        if cut == "aligned":
            crops = aligned_crops(model, image, len(signs))[0]
        else:
            crops = sign_crops(image, len(signs))[0]
        lp = log_probs(model, to_input(np.stack(crops)), tta=tta)
        rows.append((lp[::-1], signs[::-1]))           # reading order -> left to right
    return rows


def score(rows, log_trans, log_uni, lam):
    t1 = t5 = n = 0
    for lp, truth in rows:
        post = lp if lam == 0 else posteriors(lp, log_trans, log_uni, lam)
        top5 = np.argsort(-post, axis=1)[:, :5] + 1
        for guesses, t in zip(top5, truth):
            if t.startswith("*") or t == "0":
                continue
            n += 1; t1 += guesses[0] == int(t); t5 += int(t) in guesses
    return t1 / n, t5 / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(REPO_ROOT / "data" / "processed" / "models" / "sign_cnn.pt"))
    ap.add_argument("--cut", choices=["equal", "aligned"], default="equal")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model = load_model(args.model, args.device)

    df = load_labelled_seals()
    df = df[df["impression_photo"] != ""]
    test = set((CISI / "test_seals.txt").read_text().split())
    val = set((CISI / "val_seals.txt").read_text().split())
    held_out = df[df["cisi_label"].isin(test | val)]["signs_reading"]
    exclude = [[int(s.lstrip("*")) for s in seq][::-1] for seq in held_out]
    log_trans, log_uni = bigram_model(exclude)

    # ---- choose settings on validation
    results = []
    for tta in (False, True):
        rows = seal_rows(model, df, val, args.cut, tta)
        for lam in LAMBDAS:
            t1, t5, n = score(rows, log_trans, log_uni, lam)
            results.append((t1, t5, tta, lam))
            print(f"val  tta={tta!s:5} lambda={lam:<4}: top-1 {t1:.1%}  top-5 {t5:.1%}  ({n} signs)")
    best = max(results)
    _, _, tta, lam = best
    print(f"chosen on validation: tta={tta}, lambda={lam}")

    # ---- test, once with the CNN alone and once with the chosen settings
    rows_plain = seal_rows(model, df, test, args.cut, False)
    rows_tta = seal_rows(model, df, test, args.cut, True) if tta else rows_plain
    a1, a5, n = score(rows_plain, log_trans, log_uni, 0.0)
    b1, b5, _ = score(rows_tta, log_trans, log_uni, 0.0)
    c1, c5, _ = score(rows_tta, log_trans, log_uni, lam)
    lines = [
        "# A6: reading the whole seal (CNN + sign-pair statistics)",
        "",
        f"Model: {args.model}. Cuts: {args.cut} (sign count known, labels not used). "
        f"Test: {n} scorable signs on {len(rows_plain)} seals. Settings chosen on {len(val)} validation seals: "
        f"test-time augmentation {'on' if tta else 'off'}, lambda {lam}. "
        "IM77 pairs of all validation and test texts removed before building the bigram model.",
        "",
        "| Method | Top-1 | Top-5 |",
        "| --- | --- | --- |",
        f"| CNN alone | {a1:.1%} | {a5:.1%} |",
        *([f"| CNN + test-time augmentation | {b1:.1%} | {b5:.1%} |"] if tta else []),
        f"| CNN + sign-pair statistics (lambda {lam}) | {c1:.1%} | {c5:.1%} |",
        "",
        "Validation, all settings tried:",
        "",
        "| Test-time augmentation | Lambda | Top-1 | Top-5 |",
        "| --- | --- | --- | --- |",
        *[f"| {'on' if t else 'off'} | {l} | {v1:.1%} | {v5:.1%} |" for v1, v5, t, l in results],
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
