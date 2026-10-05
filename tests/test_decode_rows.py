"""Context decoding (decode.py) and rendered rows (render_rows.py)."""

import numpy as np

from indus.decode import bigram_model, posteriors
from indus.render_rows import render_row, row_crops


def test_bigram_rows_are_probabilities_and_know_99_267():
    log_trans, log_uni = bigram_model()
    p = np.exp(log_trans)
    assert np.allclose(p.sum(axis=1), 1) and np.isclose(np.exp(log_uni).sum(), 1)
    assert p[99 - 1].argmax() + 1 == 267          # the commonest IM77 pair: 99 then 267


def test_excluding_texts_removes_their_pairs():
    with_all, _ = bigram_model()
    without, _ = bigram_model(exclude=[[99, 267]] * 1000)   # remove far more than exist
    assert np.exp(without[98, 266]) < np.exp(with_all[98, 266])


def test_context_tips_an_unsure_sign():
    log_trans, log_uni = bigram_model()
    emit = np.full((2, 417), -20.0)
    emit[0, 99 - 1] = 0.0                          # sure: 99
    emit[1, 267 - 1], emit[1, 390 - 1] = np.log(0.4), np.log(0.6)   # unsure: 390 slightly ahead of 267
    assert posteriors(emit, log_trans, log_uni, 0.0)[1].argmax() + 1 == 390
    assert posteriors(emit, log_trans, log_uni, 1.0)[1].argmax() + 1 == 267


def test_rendered_row_gives_one_crop_per_sign():
    rng = np.random.default_rng(0)
    gray, bounds = render_row([342, 99, 267], rng)
    assert gray.ndim == 2 and len(bounds) == 3 and all(a < b for a, b in bounds)
    crops, labels = row_crops([342, 99, 267], rng)
    assert len(crops) == 3 and labels == [267, 99, 342]      # reading order
    assert all(c.shape == (64, 64) for c in crops)
