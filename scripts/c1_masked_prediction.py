"""M3 baseline: hide a sign, predict it from its neighbours (n-gram models).

Usage:
    python scripts/c1_masked_prediction.py

Data: the 1,837 ICIT texts (Wells numbers, reading order, indus.data.load_sequences).
A fixed 80/20 split of TEXTS (seed 0): models are trained on 80% and
measured on the other 20%, so no test text was seen in training.

Reports (reports/c1_masked_prediction.md):
  * perplexity of the held-out texts under the unigram and bigram models,
    next to the published numbers (Yadav et al. 2010, Mahadevan corpus);
  * masked prediction top-1 / top-5 / top-10 for: unigram, bigram using the
    left neighbour, bigram using both neighbours;
  * the same in the reversed direction (left to right on the seal). A bigram
    model is nearly direction-blind by construction, so this is a check of
    the code, not evidence about reading direction (see the report).
"""

import random

from indus.data import REPO_ROOT, load_sequences
from indus.masked import NGramModel, masked_accuracy

OUT = REPO_ROOT / "reports" / "c1_masked_prediction.md"
SEED = 0
TEST_FRACTION = 0.2


def split(texts):
    order = list(range(len(texts)))
    random.Random(SEED).shuffle(order)
    cut = int(len(texts) * (1 - TEST_FRACTION))
    return [texts[i] for i in order[:cut]], [texts[i] for i in order[cut:]]


def main() -> None:
    texts = load_sequences()                          # reading order
    train, test = split(texts)
    m = NGramModel().fit(train)
    unseen = sum(s not in m.index for t in test for s in t)
    n_test = sum(len(t) for t in test)

    rows = []
    for kind, name in [("unigram", "Unigram (overall frequency)"),
                       ("left", "Bigram, previous sign"),
                       ("both", "Bigram, both neighbours")]:
        acc = masked_accuracy(m, test, kind)
        rows.append(f"| {name} | {acc[1]:.1%} | {acc[5]:.1%} | {acc[10]:.1%} |")

    # reversed direction: same split, texts read the other way
    rev_train, rev_test = [t[::-1] for t in train], [t[::-1] for t in test]
    mr = NGramModel().fit(rev_train)
    ppl_rev = mr.perplexity(rev_test, "bigram")
    acc_rev = masked_accuracy(mr, rev_test, "left")

    lines = [
        "# C1 (M3 baseline): masked sign prediction with n-gram models",
        "",
        f"Data: ICIT corpus, {len(texts)} texts in reading order (Wells numbers). Fixed split of texts, "
        f"seed {SEED}: {len(train)} for training, {len(test)} held out ({n_test} signs, "
        f"{unseen} of them never seen in training). Witten-Bell smoothing; start and end markers.",
        "",
        "## Perplexity of held-out texts (lower is better)",
        "",
        "| Model | Ours (ICIT, held out) | Published (Yadav et al. 2010, Mahadevan corpus) |",
        "| --- | --- | --- |",
        f"| Unigram | {m.perplexity(test, 'unigram'):.1f} | about 68.8 |",
        f"| Bigram | {m.perplexity(test, 'bigram'):.1f} | about 26.7 |",
        "",
        "Not the same corpus or sign list, so close but not identical numbers are expected.",
        "",
        "## Hide one sign, predict it",
        "",
        "| Model | Top-1 | Top-5 | Top-10 |",
        "| --- | --- | --- | --- |",
        f"| Chance (568 signs) | {1 / 568:.1%} | {5 / 568:.1%} | {10 / 568:.1%} |",
        *rows,
        "",
        "## Direction check",
        "",
        f"Bigram perplexity reading right to left (the accepted direction): {m.perplexity(test, 'bigram'):.1f}; "
        f"read the other way: {ppl_rev:.1f}. Predicting a sign from the PREVIOUS one, top-1: "
        f"{masked_accuracy(m, test, 'left')[1]:.1%} in reading order vs {acc_rev[1]:.1%} reversed.",
        "",
        "Caution: this is not a test of reading direction. With start and end markers, every sign "
        "occurrence is counted once as the left and once as the right member of a pair, so an unsmoothed "
        "bigram model gives its training corpus exactly the same total probability in both directions. "
        "Near-equal perplexities on held-out texts are therefore expected. "
        "Direction evidence comes from where signs sit (the jar sign ends texts, B2), not from bigram "
        "perplexity (B4 makes the same point with conditional entropy).",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
