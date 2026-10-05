"""Extension: error propagation. How do M1's reading mistakes distort M2-M4 results?

If the corpus had been read by our vision pipeline instead of by experts,
every statistic would be computed on partly wrong sequences. Here we simulate
that: each sign is kept with probability `accuracy` and otherwise replaced by
a wrong sign drawn in proportion to how common signs are (the CNN's mistakes
lean towards common signs, which it has seen most). Then the same statistics
are recomputed and compared with the true corpus and with shuffled corpora.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


def sign_weights(texts: list[list[int]]) -> tuple[list[int], list[float]]:
    """(signs, probabilities) proportional to how often each sign occurs."""
    counts = Counter(s for t in texts for s in t)
    signs = sorted(counts)
    total = sum(counts.values())
    return signs, [counts[s] / total for s in signs]


def corrupt(texts: list[list[int]], accuracy: float, rng: random.Random, weights) -> list[list[int]]:
    """Copy of the texts where each sign is right with probability `accuracy`,
    otherwise replaced by a DIFFERENT sign drawn by frequency."""
    signs, probs = weights
    out = []
    for t in texts:
        new = []
        for s in t:
            if rng.random() < accuracy:
                new.append(s)
            else:
                while True:                               # redraw until it is actually wrong
                    w = rng.choices(signs, probs)[0]
                    if w != s:
                        break
                new.append(w)
        out.append(new)
    return out
