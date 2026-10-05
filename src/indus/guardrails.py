"""M4 baseline: guardrails. Is the sign ORDER in real texts different from chance?

Any claim about the script ("this sign is a suffix", "texts read right to
left") should beat a null model: the same signs in a random order. Here the
null model shuffles the signs WITHIN each text, so every text keeps its
length and its exact signs; only the order is destroyed.

Two tools:

1. Naive Bayes classifier, real vs shuffled. Each text is turned into counts
   of its sign pairs (with start and end markers). Multinomial Naive Bayes
   learns, for each class, how likely each pair is (add-one smoothed), and
   scores a text as
       log P(class) + sum over pairs  count(pair) * log P(pair | class)
   ("naive" = it treats the pairs as independent). If real order carries
   structure, it should beat 50%. As a control, the same classifier on
   single-sign counts must sit at 50%: shuffling never changes which signs
   a text has.

2. Permutation tests. Compute a statistic on the real corpus, then on many
   shuffled corpora. The p-value is the share of shuffles at least as
   extreme as the real value (plus one, so it is never exactly 0):
       p = (1 + #{shuffles with statistic >= real}) / (1 + number of shuffles)
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

START, END = "<s>", "</s>"


def shuffle_within(texts: list[list[int]], rng: random.Random) -> list[list[int]]:
    """Each text with its signs in a random order (same signs, same length)."""
    out = []
    for t in texts:
        t = list(t)
        rng.shuffle(t)
        out.append(t)
    return out


def pair_features(text: list[int], markers: bool = True) -> Counter:
    """Counts of adjacent pairs, including (start, first) and (last, end) when markers=True."""
    ids = [START] + list(text) + [END] if markers else list(text)
    return Counter(zip(ids, ids[1:]))


def sign_features(text: list[int]) -> Counter:
    """Counts of single signs (the control: shuffling never changes these)."""
    return Counter(text)


class NaiveBayes:
    """Multinomial Naive Bayes over count features, add-one smoothing, two classes."""

    def fit(self, feats: list[Counter], labels: list[int]) -> "NaiveBayes":
        self.vocab = sorted({f for c in feats for f in c}, key=str)
        index = {f: i for i, f in enumerate(self.vocab)}
        counts = np.ones((2, len(self.vocab) + 1))          # +1 column for features never seen in training
        for c, y in zip(feats, labels):
            for f, k in c.items():
                counts[y, index[f]] += k
        self.index = index
        self.log_p = np.log(counts / counts.sum(axis=1, keepdims=True))
        prior = np.bincount(labels, minlength=2) / len(labels)
        self.log_prior = np.log(prior)
        return self

    def predict(self, feats: list[Counter]) -> np.ndarray:
        unseen = len(self.vocab)
        out = []
        for c in feats:
            score = self.log_prior.copy()
            for f, k in c.items():
                score += k * self.log_p[:, self.index.get(f, unseen)]
            out.append(int(np.argmax(score)))
        return np.array(out)


def conditional_entropy(texts: list[list[int]], markers: bool = True) -> float:
    """H(next sign | current sign) in bits from pair counts.
    Lower = the next sign is more predictable = more order structure.
    markers=True also counts (start, first sign) and (last sign, end), so where
    texts begin and end is part of the statistic; markers=False uses pairs
    inside texts only, which is B4's estimator (3.3082 bits in reading order)."""
    pairs = Counter()
    for t in texts:
        pairs.update(pair_features(t, markers))
    by_first = Counter()
    for (a, _), k in pairs.items():
        by_first[a] += k
    total = sum(pairs.values())
    return float(-sum(k / total * np.log2(k / by_first[a]) for (a, _), k in pairs.items()))


def last_sign_share(texts: list[list[int]], sign: int) -> float:
    """Share of texts (length >= 2) whose last sign in reading order is `sign`."""
    long = [t for t in texts if len(t) >= 2]
    return sum(t[-1] == sign for t in long) / len(long)


def permutation_test(texts, statistic, n_shuffles: int = 1000, seed: int = 0, larger_is_extreme: bool = True):
    """(real value, shuffled values, p-value) for statistic(texts) against within-text shuffles."""
    rng = random.Random(seed)
    real = statistic(texts)
    null = np.array([statistic(shuffle_within(texts, rng)) for _ in range(n_shuffles)])
    extreme = (null >= real) if larger_is_extreme else (null <= real)
    p = (1 + extreme.sum()) / (1 + n_shuffles)
    return real, null, float(p)
