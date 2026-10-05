"""Error propagation (propagation.py): corruption keeps lengths and hits the requested accuracy."""

import random

from indus.data import load_sequences
from indus.propagation import corrupt, sign_weights


def test_corrupt_accuracy_and_lengths():
    texts = load_sequences()
    noisy = corrupt(texts, 0.3, random.Random(0), sign_weights(texts))
    assert [len(t) for t in noisy] == [len(t) for t in texts]
    pairs = [(a, b) for t, n in zip(texts, noisy) for a, b in zip(t, n)]
    right = sum(a == b for a, b in pairs) / len(pairs)
    assert abs(right - 0.3) < 0.02                       # 8,359 signs: well within 2 points


def test_extremes():
    texts = load_sequences()[:100]
    w = sign_weights(texts)
    assert corrupt(texts, 1.0, random.Random(0), w) == texts
    assert all(a != b for t, n in zip(texts, corrupt(texts, 0.0, random.Random(0), w)) for a, b in zip(t, n))
