"""M3 baseline (masked.py): smoothed n-gram probabilities and masked prediction."""

import numpy as np

from indus.data import load_sequences
from indus.masked import NGramModel, masked_accuracy


def test_every_context_is_a_probability_distribution():
    m = NGramModel().fit([[1, 2, 3], [1, 2], [3, 3, 1]])
    assert np.allclose(np.exp(m.log_bi).sum(axis=1), 1)
    assert np.isclose(np.exp(m.log_uni).sum(), 1)


def test_neighbours_recover_a_fixed_pattern():
    # every text is 5 6 7: with both neighbours known, the middle sign is certain
    m = NGramModel().fit([[5, 6, 7]] * 20 + [[8, 9]] * 5)
    assert masked_accuracy(m, [[5, 6, 7]], "both")[1] == 1.0
    assert m.perplexity([[5, 6, 7]]) < m.perplexity([[5, 6, 7]], "unigram")


def test_bigram_beats_unigram_on_the_real_corpus():
    texts = load_sequences()
    m = NGramModel().fit(texts[:1500])
    held = texts[1500:]
    assert m.perplexity(held, "bigram") < m.perplexity(held, "unigram")
    assert masked_accuracy(m, held, "both")[1] > masked_accuracy(m, held, "unigram")[1]
