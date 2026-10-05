"""Null models (nullmodels.py): generated corpora have the right shape."""

import random

from indus.data import load_sequences
from indus.nullmodels import BigramProcess, triple_features, unigram_corpus, zipf_slope


def test_bigram_process_only_makes_seen_pairs_and_valid_lengths():
    texts = load_sequences()
    seen = {p for t in texts for p in zip(t, t[1:])}
    fake = BigramProcess(texts).corpus(300, random.Random(0))
    assert all(2 <= len(t) <= 13 for t in fake)
    assert all(p in seen for t in fake for p in zip(t, t[1:]))


def test_unigram_corpus_keeps_lengths():
    texts = load_sequences()
    fake = unigram_corpus(texts, random.Random(0))
    assert [len(t) for t in fake] == [len(t) for t in texts]


def test_zipf_slope_of_real_corpus_is_near_minus_one():
    assert -1.2 < zipf_slope(load_sequences()) < -0.7


def test_triples_include_markers():
    assert len(triple_features([1, 2])) == 2          # (<s>,1,2) and (1,2,</s>)
