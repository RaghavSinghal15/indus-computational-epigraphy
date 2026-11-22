"""Sanity checks on the Lipi data (full ICIT corpus + seal photos).

Skipped until the data is downloaded:  python scripts/fetch_lipi.py
"""
import re

import pytest

from indus.data import LIPI_DIR, load_icit_full, load_labelled_seals, load_lipi_seals, load_sequences, wells_to_mahadevan

pytestmark = pytest.mark.skipif(not (LIPI_DIR / "inscriptions.csv").exists(),
                                reason="run python scripts/fetch_lipi.py first")


def test_usable_seal_counts():
    seals = load_lipi_seals()
    assert len(seals) == 855
    assert (~seals.in_v1).sum() == 571               # new to us
    assert (seals.site == "Harappa").sum() == 181


def test_m58_reads_like_our_label():
    m58 = load_lipi_seals().set_index("cisi").loc["M-58"]
    assert m58.signs_reading == [817, 2, 503, 460, 740]   # jar sign (740) read last


def test_jar_sign_ends_texts_in_full_corpus():
    texts = load_icit_full()
    ends = sum(t[-1] == 740 for t in texts) / len(texts)
    starts = sum(t[0] == 740 for t in texts) / len(texts)
    assert 0.30 < ends < 0.40 and starts < 0.02


def test_full_corpus_contains_most_of_the_benchmark():
    full = {tuple(t) for t in load_icit_full()}
    bench = load_sequences()
    assert sum(tuple(t) in full for t in bench) / len(bench) > 0.9


def test_lipi_labels_agree_with_our_im77_labels():
    """Sign by sign, mapped through wells_to_mahadevan, on seals of equal length."""
    lipi = load_lipi_seals().set_index("cisi")
    ours = load_labelled_seals().set_index("cisi_label")
    w2m = wells_to_mahadevan()
    agree = total = 0
    for cisi in lipi.index.intersection(ours.index):
        ltr = [int(re.sub(r"\D", "", s) or 0) for s in ours.loc[cisi, "signs_ltr"]]
        wells = lipi.loc[cisi, "signs_ltr"]
        if len(ltr) != len(wells):
            continue
        for w, m in zip(wells, ltr):
            if w2m.get(w):
                total += 1
                agree += m in w2m[w]
    assert total > 1000 and agree / total > 0.95
