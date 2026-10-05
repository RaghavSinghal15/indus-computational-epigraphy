"""M4 baseline: can order statistics tell real Indus texts from shuffled ones?

Usage:
    python scripts/d1_guardrails.py

Data: the 1,837 ICIT texts (Wells numbers, reading order). Same fixed 80/20
split of texts as C1 (seed 0). Each text also gets one shuffled copy (its own
signs in a random order, seed 1), so classes are balanced.

Writes reports/d1_guardrails.md and reports/figures/d1_permutation.png:
  * Naive Bayes real vs shuffled on the held-out texts, with sign PAIRS as
    features and, as a control, single signs (must be about 50%), overall
    and by text length;
  * permutation tests (1,000 within-text shuffles of the whole corpus) for
    two statistics: conditional entropy of the next sign, and the share of
    texts that end with the jar sign (Wells 740).
"""

import random

import matplotlib
import numpy as np

matplotlib.use("Agg")                     # draw to a file, no window
import matplotlib.pyplot as plt

from indus.data import REPO_ROOT, load_sequences
from indus.guardrails import (NaiveBayes, conditional_entropy, last_sign_share, pair_features,
                              permutation_test, shuffle_within, sign_features)

OUT = REPO_ROOT / "reports" / "d1_guardrails.md"
FIGURE = REPO_ROOT / "reports" / "figures" / "d1_permutation.png"
JAR = 740                                 # Wells number of the jar sign (Mahadevan 342)
SEED = 0


def split(texts):
    order = list(range(len(texts)))
    random.Random(SEED).shuffle(order)
    cut = int(len(texts) * 0.8)
    return [texts[i] for i in order[:cut]], [texts[i] for i in order[cut:]]


def labelled(texts, rng):
    """Real texts (label 1) followed by one shuffled copy of each (label 0)."""
    return texts + shuffle_within(texts, rng), [1] * len(texts) + [0] * len(texts)


def main() -> None:
    texts = load_sequences()
    train, test = split(texts)
    rng = random.Random(SEED + 1)
    tr_x, tr_y = labelled(train, rng)
    te_x, te_y = labelled(test, rng)
    te_y = np.array(te_y)

    results = {}
    for name, feat in [("pairs", pair_features), ("single signs (control)", sign_features)]:
        nb = NaiveBayes().fit([feat(t) for t in tr_x], tr_y)
        pred = nb.predict([feat(t) for t in te_x])
        results[name] = pred == te_y

    # by length (pairs model); length 1 cannot differ from its shuffle at all
    lengths = np.array([len(t) for t in te_x])
    by_len = []
    for lo, hi, label in [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 7, "5-7"), (8, 99, "8+")]:
        m = (lengths >= lo) & (lengths <= hi)
        if m.any():
            by_len.append(f"| {label} | {m.sum() // 2} | {results['pairs'][m].mean():.1%} |")

    ce_real, ce_null, ce_p = permutation_test(texts, conditional_entropy, larger_is_extreme=False)
    jar_real, jar_null, jar_p = permutation_test(texts, lambda t: last_sign_share(t, JAR))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, null, real, title, unit in [
        (axes[0], ce_null, ce_real, "Next-sign uncertainty (bits)", "bits"),
        (axes[1], jar_null * 100, jar_real * 100, "Texts ending with the jar sign (%)", "%"),
    ]:
        ax.hist(null, bins=30, color="#9aa7b8")
        ax.axvline(real, color="#c0392b", linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.text(real, ax.get_ylim()[1] * 0.9, "  real corpus", color="#c0392b", fontsize=9)
        ax.set_yticks([])
    axes[0].set_xlabel("grey: 1,000 shuffled corpora")
    axes[1].set_xlabel("grey: 1,000 shuffled corpora")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)

    lines = [
        "# D1 (M4 baseline): real texts vs the same signs in random order",
        "",
        f"Data: ICIT corpus, {len(texts)} texts, reading order. Train/test split of texts as in C1 "
        f"({len(train)} / {len(test)}); every text plus one within-text shuffle (seed {SEED + 1}).",
        "",
        "## Naive Bayes: real or shuffled?",
        "",
        "| Features | Accuracy on held-out texts (chance 50%) |",
        "| --- | --- |",
        *[f"| {name} | {ok.mean():.1%} |" for name, ok in results.items()],
        "",
        "The control has to be at chance: a shuffle keeps exactly the same signs, only their order changes. "
        "Sign pairs carry the order, so anything above 50% is order structure.",
        "",
        "Pairs model by text length (texts of length 1 are identical to their shuffle):",
        "",
        "| Length | Held-out texts | Accuracy |",
        "| --- | --- | --- |",
        *by_len,
        "",
        "## Permutation tests (1,000 within-text shuffles of the whole corpus)",
        "",
        "| Statistic | Real corpus | Shuffled corpora (mean, 95% range) | p-value |",
        "| --- | --- | --- | --- |",
        f"| Next-sign uncertainty H(next given current), bits | {ce_real:.3f} | {ce_null.mean():.3f} "
        f"({np.percentile(ce_null, 2.5):.3f} to {np.percentile(ce_null, 97.5):.3f}) | {ce_p:.4f} |",
        f"| Texts ending with the jar sign | {jar_real:.1%} | {jar_null.mean():.1%} "
        f"({np.percentile(jar_null, 2.5):.1%} to {np.percentile(jar_null, 97.5):.1%}) | {jar_p:.4f} |",
        "",
        "p = (1 + shuffles at least as extreme) / (1 + 1000); 0.001 is the smallest possible value, meaning no "
        "shuffle came close. Figure: reports/figures/d1_permutation.png.",
        "",
        f"Cross-check with B4: without the start/end markers the same estimator gives "
        f"{conditional_entropy(texts, markers=False):.4f} bits, B4's value (3.3082). With markers it is higher "
        "because it also counts the uncertainty about how texts begin and end.",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
