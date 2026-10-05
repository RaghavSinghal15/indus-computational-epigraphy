"""A6: read a whole seal at once, combining the CNN with sign-pair statistics (M2).

The CNN looks at one sign at a time. But signs are not independent: 99 is very
often followed by 267, the jar sign (342) ends texts, and so on. A phone
keyboard's autocorrect uses the same trick with words. Here:

  * the CNN gives, for every position i, a probability for each of the 417 signs;
  * a BIGRAM model (M2) gives P(next sign | this sign), from the IM77 pair counts;
  * we combine them over the whole row and get, for every position, a new
    probability for each sign that takes its neighbours into account.

The combination is the forward-backward algorithm of a hidden Markov model
(HMM). Hidden states = the true signs, observations = the crops. The score of
a whole guessed sequence s_1..s_n (left to right on the impression) is

    sum_i log p_cnn(s_i | crop_i)  +  lam * [ log P(s_1) + sum_i log P(s_i | s_{i-1}) ]

lam (lambda) says how much to trust the pair statistics compared with the
CNN; 0 means CNN alone. It is chosen on the validation seals.
Forward-backward adds up the scores of ALL sequences that put sign s at
position i (not just the single best sequence), which gives the probability
of s at i; that is what top-1 / top-5 per sign are measured on.

Leakage: IM77 is the corpus our seals' sequences come from, so its pair counts
include the validation and test texts themselves. bigram_model() subtracts
the pairs of any texts passed in `exclude` before building the model.

Smoothing (Witten-Bell): a pair never seen in IM77 must still get a small
probability. For a sign a that was followed by T different signs in N pairs,
    P(b | a) = (N / (N + T)) * count(a, b) / N  +  (T / (N + T)) * P_unigram(b)
i.e. signs with many different followers lend more weight to "anything can
follow". P_unigram is add-one smoothed sign frequency.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp   # log(sum(exp(x))) without overflow

from indus.data import load_im77_ngrams

N_SIGNS = 417


def bigram_model(exclude: list[list[int]] = ()) -> tuple[np.ndarray, np.ndarray]:
    """(log P(right | left) as a 417 x 417 array, log P_unigram as a 417 array).

    Index k is Mahadevan sign k + 1. Order is left to right on the impression.
    exclude: sign sequences (left to right, 0 = illegible) whose pairs are
    removed from the IM77 counts first.
    """
    counts = np.zeros((N_SIGNS, N_SIGNS))
    b = load_im77_ngrams(2)
    np.add.at(counts, (b["left_sign"].to_numpy() - 1, b["right_sign"].to_numpy() - 1), b["count"].to_numpy())
    for seq in exclude:
        for a, c in zip(seq, seq[1:]):
            if a > 0 and c > 0:
                counts[a - 1, c - 1] = max(0.0, counts[a - 1, c - 1] - 1)

    # unigram: how often each sign appears in a pair (either side), add-one smoothed
    uni = counts.sum(axis=0) + counts.sum(axis=1) + 1.0
    uni /= uni.sum()

    n = counts.sum(axis=1, keepdims=True)                    # N: pairs starting with sign a
    t = (counts > 0).sum(axis=1, keepdims=True)              # T: distinct followers of a
    with np.errstate(invalid="ignore", divide="ignore"):
        seen = np.where(n > 0, counts / n, 0.0)
        w = np.where(n > 0, n / (n + t), 0.0)                # weight on the observed pairs
    trans = w * seen + (1 - w) * uni[None, :]
    return np.log(trans), np.log(uni)


def edge_model(sequences: list[list[int]], log_uni: np.ndarray, strength: float = 20.0):
    """(log P(leftmost sign), log P(rightmost sign)) on the impression, from training texts.

    The leftmost sign on an impression is the LAST one read (the jar sign 342
    is very often there). Few texts, so each estimate is blended with the
    unigram: (count + strength * P_unigram) / (texts + strength).
    """
    uni = np.exp(log_uni)
    out = []
    for pick in (0, -1):
        counts = np.zeros(N_SIGNS)
        for seq in sequences:
            if seq and seq[pick] > 0:
                counts[seq[pick] - 1] += 1
        out.append(np.log((counts + strength * uni) / (counts.sum() + strength)))
    return out[0], out[1]


def posteriors(log_emit: np.ndarray, log_trans: np.ndarray, log_start: np.ndarray, lam: float,
               log_end: np.ndarray | None = None) -> np.ndarray:
    """Forward-backward. log_emit: (n, 417) CNN log-probabilities, left to right.
    log_start / log_end: log P of each sign at the left / right end (log_end optional).
    Returns (n, 417) log-probabilities of each sign at each position given the whole row."""
    n = len(log_emit)
    T = lam * log_trans
    alpha = np.zeros_like(log_emit)
    alpha[0] = log_emit[0] + lam * log_start
    for i in range(1, n):
        # alpha[i, s] = emit[i, s] + log sum_t exp(alpha[i-1, t] + T[t, s])
        alpha[i] = log_emit[i] + logsumexp(alpha[i - 1][:, None] + T, axis=0)
    beta = np.zeros_like(log_emit)
    if log_end is not None:
        beta[n - 1] = lam * log_end
    for i in range(n - 2, -1, -1):
        # beta[i, t] = log sum_s exp(T[t, s] + emit[i+1, s] + beta[i+1, s])
        beta[i] = logsumexp(T + (log_emit[i + 1] + beta[i + 1])[None, :], axis=1)
    post = alpha + beta
    return post - logsumexp(post, axis=1, keepdims=True)


def log_evidence(log_emit: np.ndarray, log_trans: np.ndarray, log_start: np.ndarray, lam: float,
                 log_end: np.ndarray | None = None) -> float:
    """How well a row of CNN guesses fits the sign statistics: the log of the
    total score summed over all possible sign sequences (the forward pass of
    posteriors() without the backward pass).

    0 would mean the statistics fully agree with the CNN; very negative means
    whatever the CNN sees forms unlikely sequences (for example one sign cut
    into two halves that never occur next to each other).
    """
    T = lam * log_trans
    alpha = log_emit[0] + lam * log_start
    for i in range(1, len(log_emit)):
        alpha = log_emit[i] + logsumexp(alpha[:, None] + T, axis=0)
    if log_end is not None:
        alpha = alpha + lam * log_end
    return float(logsumexp(alpha))
