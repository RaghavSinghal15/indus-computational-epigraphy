"""M3 baseline: predict a hidden sign from its neighbours with smoothed n-gram models.

The task ("masked prediction"): take a real text, hide one sign, and ask the
model which sign was there. If the script has structure, the neighbours should
narrow the choice a lot; a random guess among 568 signs is right 0.2% of the time.

Models (all trained on whole texts in reading order, with boundary markers):
  * unigram: ignore the neighbours, guess by overall frequency.
  * bigram, left only: P(sign | previous sign).
  * bigram, both sides: P(sign | previous) * P(next | sign). This uses the
    sign on the right too, like filling a gap in a damaged inscription.

Boundary markers: every text gets a start marker before its first sign and
an end marker after its last, so "this sign starts texts" and "this sign ends
texts" (the jar sign) are learned like any other pair.

Smoothing (Witten-Bell, as in B5): a pair never seen in training must still
get some probability. For a context u followed by T(u) different signs in
c(u) pairs:
    P(w | u) = (c(u, w) + T(u) * P_unigram(w)) / (c(u) + T(u))
Contexts with many different followers keep more probability for unseen ones.
P_unigram is add-one smoothed and includes an "unknown sign" slot for signs
that never appear in training.

Perplexity: exp of the average negative log-probability per predicted token
(each sign and the end marker). Read it as "the model is as unsure as if it
were choosing uniformly among this many signs". Published reference (Yadav
et al. 2010, Mahadevan corpus): unigram about 68.8, bigram about 26.7.
"""

from __future__ import annotations

import numpy as np

UNK = "unk"


class NGramModel:
    """Unigram and Witten-Bell bigram over signs, with start/end markers."""

    def fit(self, texts: list[list[int]]) -> "NGramModel":
        signs = sorted({s for t in texts for s in t})
        # index layout: 0..S-1 signs, S unknown, S+1 end marker, S+2 start marker
        self.signs = signs
        self.index = {s: i for i, s in enumerate(signs)}
        S = len(signs)
        self.UNK, self.END, self.START = S, S + 1, S + 2
        V = S + 2                                        # what can be PREDICTED: signs, unknown, end
        uni = np.zeros(V)
        pairs = np.zeros((S + 3, V))                     # rows: any context incl. start; cols: predictable
        for t in texts:
            ids = [self.START] + [self.index[s] for s in t] + [self.END]
            for a, b in zip(ids, ids[1:]):
                pairs[a, b] += 1
                uni[b] += 1
        self.log_uni = np.log((uni + 1) / (uni.sum() + V))
        p_uni = np.exp(self.log_uni)
        c = pairs.sum(axis=1, keepdims=True)
        T = (pairs > 0).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            prob = (pairs + T * p_uni[None, :]) / (c + T)
        prob[c[:, 0] == 0] = p_uni                       # contexts never seen: fall back to the unigram
        self.log_bi = np.log(prob)
        return self

    def ids(self, text: list[int]) -> list[int]:
        """Model indices for a text (unseen signs -> unknown)."""
        return [self.index.get(s, self.UNK) for s in text]

    def perplexity(self, texts: list[list[int]], model: str = "bigram") -> float:
        """Perplexity per predicted token (signs + end marker) on held-out texts."""
        total, n = 0.0, 0
        for t in texts:
            ids = [self.START] + self.ids(t) + [self.END]
            for a, b in zip(ids, ids[1:]):
                total += self.log_uni[b] if model == "unigram" else self.log_bi[a, b]
                n += 1
        return float(np.exp(-total / n))

    def masked_scores(self, text: list[int], pos: int, model: str = "both") -> np.ndarray:
        """Log-score of every known sign (array of length len(self.signs)) at hidden position pos."""
        ids = [self.START] + self.ids(text) + [self.END]
        left, right = ids[pos], ids[pos + 2]             # neighbours of position pos (shifted by the start marker)
        S = len(self.signs)
        if model == "unigram":
            return self.log_uni[:S].copy()
        score = self.log_bi[left, :S].copy()             # P(sign | left neighbour)
        if model == "both":
            score += self.log_bi[:S, right]              # times P(right neighbour | sign)
        return score


def masked_accuracy(model: NGramModel, texts: list[list[int]], kind: str, ks=(1, 5, 10)) -> dict:
    """Hide each sign of each text in turn; share of positions where the true sign is in the top k."""
    hits = {k: 0 for k in ks}
    n = 0
    for t in texts:
        for pos, sign in enumerate(t):
            scores = model.masked_scores(t, pos, kind)
            n += 1
            if sign not in model.index:                  # never seen in training: cannot be guessed
                continue
            rank = int((scores > scores[model.index[sign]]).sum())   # how many signs score higher
            for k in ks:
                hits[k] += rank < k
    return {k: hits[k] / n for k in ks} | {"n": n}
