"""Extension: null-model benchmark. Which statistics could a process with no language reproduce?

Claims that the Indus script encodes language often rest on corpus statistics
(Zipf's law, next-sign entropy "in the linguistic range", signs fixed at text
ends). A fair benchmark asks: does a simple random process with NO language
behind it reproduce the same numbers? If it does, those numbers cannot be
evidence for language.

Two generators, both fitted to the real corpus:
  * unigram process: each text has a real length; its signs are drawn
    independently by overall frequency. Knows which signs are common, nothing
    about order.
  * bigram process (a Markov chain): start at a start marker and keep drawing
    the next sign from the real P(next | current) until the end marker. Knows
    which sign follows which, nothing more: no words, no grammar, no meaning.
    Texts outside the real length range (2-13) are redrawn.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

START, END = "<s>", "</s>"


class BigramProcess:
    """Markov chain over signs with start/end markers, fitted by counting (no smoothing)."""

    def __init__(self, texts: list[list[int]], min_len: int = 2, max_len: int = 13):
        self.next = {}
        pairs = Counter()
        for t in texts:
            ids = [START] + list(t) + [END]
            pairs.update(zip(ids, ids[1:]))
        for (a, b), k in pairs.items():
            self.next.setdefault(a, ([], []))
            self.next[a][0].append(b)
            self.next[a][1].append(k)
        self.min_len, self.max_len = min_len, max_len

    def text(self, rng: random.Random) -> list[int]:
        while True:                                       # redraw until the length is in range
            out, cur = [], START
            while True:
                options, weights = self.next[cur]
                cur = rng.choices(options, weights)[0]
                if cur == END or len(out) > self.max_len:
                    break
                out.append(cur)
            if self.min_len <= len(out) <= self.max_len:
                return out

    def corpus(self, n: int, rng: random.Random) -> list[list[int]]:
        return [self.text(rng) for _ in range(n)]


def unigram_corpus(texts: list[list[int]], rng: random.Random) -> list[list[int]]:
    """Same lengths as the real texts; signs drawn independently by frequency."""
    counts = Counter(s for t in texts for s in t)
    signs, weights = zip(*counts.items())
    return [list(rng.choices(signs, weights, k=len(t))) for t in texts]


def zipf_slope(texts: list[list[int]], top: int = 100) -> float:
    """Slope of log(frequency) against log(rank) for the `top` commonest signs (Zipf's law: about -1)."""
    freq = np.array(sorted(Counter(s for t in texts for s in t).values(), reverse=True)[:top], float)
    ranks = np.arange(1, len(freq) + 1)
    return float(np.polyfit(np.log(ranks), np.log(freq), 1)[0])


def triple_features(text: list[int]) -> Counter:
    """Counts of three-sign windows, with start/end markers."""
    ids = [START] + list(text) + [END]
    return Counter(zip(ids, ids[1:], ids[2:]))
