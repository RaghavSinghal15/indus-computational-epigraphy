"""M4 baseline (guardrails.py): shuffles, Naive Bayes, entropy, permutation tests."""

import random

import numpy as np

from indus.data import load_sequences
from indus.guardrails import (NaiveBayes, conditional_entropy, last_sign_share, pair_features,
                              permutation_test, shuffle_within, sign_features)


def test_shuffle_keeps_signs_and_lengths():
    texts = load_sequences()[:200]
    shuffled = shuffle_within(texts, random.Random(0))
    assert all(sorted(a) == sorted(b) for a, b in zip(texts, shuffled))


def test_entropy_matches_b4_without_markers():
    # B4 (reports/b4_results.md): 3.3081803036 bits in reading order
    assert abs(conditional_entropy(load_sequences(), markers=False) - 3.3081803036) < 1e-6


def test_naive_bayes_separates_a_fixed_order():
    real = [[1, 2, 3]] * 30
    fake = [[3, 2, 1]] * 30
    nb = NaiveBayes().fit([pair_features(t) for t in real + fake], [1] * 30 + [0] * 30)
    assert list(nb.predict([pair_features([1, 2, 3]), pair_features([3, 2, 1])])) == [1, 0]
    # single-sign counts cannot tell them apart
    assert sign_features([1, 2, 3]) == sign_features([3, 2, 1])


def test_jar_sign_ends_texts_far_more_than_chance():
    real, null, p = permutation_test(load_sequences(), lambda t: last_sign_share(t, 740), n_shuffles=50)
    assert real > null.max() and np.isclose(p, 1 / 51)
