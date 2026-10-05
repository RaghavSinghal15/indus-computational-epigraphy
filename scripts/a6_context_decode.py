"""A6: does reading the whole seal (CNN + sign-pair statistics) beat the CNN alone?

Usage:
    python scripts/a6_context_decode.py [--model data/processed/models/sign_cnn.pt] [--cut equal|aligned]

For every validation and test seal: cut the impression (and, if chosen, the
photo of the seal itself, flipped back) into its n signs
(equal slices, or CNN-aligned using the count only), get the CNN's
probabilities for every position, then combine with the IM77 bigram model
(indus.decode) at several weights lambda, with and without the start/end
statistics of the training texts (e.g. the jar sign ends texts). Pairs from the validation and test
texts are removed from the IM77 counts first.

Settings (both photos or not, test-time augmentation, lambda) are chosen on VALIDATION;
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
from indus.decode import bigram_model, edge_model, posteriors
from indus.real_crops import sign_crops
from indus.segment import load_image

CISI = REPO_ROOT / "data" / "raw" / "cisi_seals"
OUT = REPO_ROOT / "reports" / "a6_context.md"
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def view_log_probs(model, path, n, cut, tta, mirrored):
    """CNN log-probs (n, 417) in reading order for one photo of a seal."""
    image = load_image(path)
    if cut == "aligned":
        crops = aligned_crops(model, image, n, mirrored=mirrored)[0]
    else:
        crops = sign_crops(image, n, mirrored=mirrored)[0]
    return log_probs(model, to_input(np.stack(crops)), tta=tta)


def seal_rows(model, df, labels_set, cut, tta, both_views=False):
    """Per seal: (CNN log-probs left to right (n,417), true signs left to right as strings).

    both_views: if the seal also has a photo of the seal itself (mirrored),
    cut that too (flipped back) and average the two views' probabilities.
    """
    rows = []
    for _, row in df[df["cisi_label"].isin(labels_set)].iterrows():
        signs = row["signs_reading"]
        lp = view_log_probs(model, row["impression_photo"], len(signs), cut, tta, False)
        if both_views and row["seal_photo"]:
            other = view_log_probs(model, row["seal_photo"], len(signs), cut, tta, True)
            lp = np.logaddexp(lp, other) - np.log(2)     # log of the average probability
        rows.append((lp[::-1], signs[::-1]))           # reading order -> left to right
    return rows


def score(rows, lm, lam):
    """Top-1 and top-5 over scorable signs; lm = (log_trans, log_start, log_end or None)."""
    t1 = t5 = n = 0
    for lp, truth in rows:
        post = lp if lam == 0 else posteriors(lp, lm[0], lm[1], lam, lm[2])
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
    train_texts = [[int(s.lstrip("*")) for s in seq][::-1]
                   for seq in df[~df["cisi_label"].isin(test | val)]["signs_reading"]]
    log_left, log_right = edge_model(train_texts, log_uni)
    models = {"pairs": (log_trans, log_uni, None), "pairs + ends": (log_trans, log_left, log_right)}

    # ---- choose settings on validation
    results = []
    for views in (False, True):
        for tta in (False, True):
            rows = seal_rows(model, df, val, args.cut, tta, views)
            for name, lm in models.items():
                for lam in LAMBDAS:
                    t1, t5, n = score(rows, lm, lam)
                    results.append((t1, t5, views, tta, lam, name))
                    print(f"val  both views={views!s:5} tta={tta!s:5} {name:12} lambda={lam:<4}: "
                          f"top-1 {t1:.1%}  top-5 {t5:.1%}  ({n} signs)")
    best = max(results)
    _, _, views, tta, lam, name = best
    print(f"chosen on validation: both views={views}, tta={tta}, {name}, lambda={lam}")

    # ---- test, step by step, each step adding one chosen setting
    rows_plain = seal_rows(model, df, test, args.cut, False)
    rows_chosen = seal_rows(model, df, test, args.cut, tta, views) if (tta or views) else rows_plain
    a1, a5, n = score(rows_plain, models[name], 0.0)
    b1, b5, _ = score(rows_chosen, models[name], 0.0)
    c1, c5, _ = score(rows_chosen, models[name], lam)
    lines = [
        "# A6: reading the whole seal (CNN + sign-pair statistics)",
        "",
        f"Model: {args.model}. Cuts: {args.cut} (sign count known, labels not used). "
        f"Test: {n} scorable signs on {len(rows_plain)} seals. Settings chosen on {len(val)} validation seals: "
        f"both photos {'yes' if views else 'no'}, test-time augmentation {'on' if tta else 'off'}, "
        f"{name}, lambda {lam}. "
        "IM77 pairs of all validation and test texts removed before building the bigram model.",
        "",
        "| Method | Top-1 | Top-5 |",
        "| --- | --- | --- |",
        f"| CNN alone | {a1:.1%} | {a5:.1%} |",
        *([f"| CNN{' + both photos' if views else ''}{' + test-time augmentation' if tta else ''} "
           f"| {b1:.1%} | {b5:.1%} |"] if (tta or views) else []),
        f"| CNN + sign statistics ({name}, lambda {lam}) | {c1:.1%} | {c5:.1%} |",
        "",
        "Validation, all settings tried:",
        "",
        "| Both photos | Test-time augmentation | Statistics | Lambda | Top-1 | Top-5 |",
        "| --- | --- | --- | --- | --- | --- |",
        *[f"| {'yes' if v else 'no'} | {'on' if t else 'off'} | {nm} | {l} | {v1:.1%} | {v5:.1%} |"
          for v1, v5, v, t, l, nm in results],
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
