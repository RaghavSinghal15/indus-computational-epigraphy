"""Sanity checks on the raw data. Run with:  pytest

If any of these fail, something upstream changed and every result built on
the data is suspect. Fix the data before trusting any analysis.
"""

from collections import Counter

from indus.data import (
    load_im77_ngrams,
    load_sequences,
    load_sign_mapping,
    template_path,
    wells_to_mahadevan,
)


def test_sequence_counts_match_paper():
    seqs = load_sequences()
    assert len(seqs) == 1837
    assert sum(len(s) for s in seqs) == 8359
    assert len({sign for s in seqs for sign in s}) == 568
    assert min(len(s) for s in seqs) == 2


def test_reading_order_is_reverse_of_visual():
    visual = load_sequences(order="visual")
    reading = load_sequences(order="reading")
    assert all(r == v[::-1] for r, v in zip(reading, visual))


def test_jar_sign_ends_texts_in_reading_order():
    # Wells 740 = Mahadevan 342, the "jar", a known text-ending sign.
    last = Counter(s[-1] for s in load_sequences())
    first = Counter(s[0] for s in load_sequences())
    assert last.most_common(1)[0][0] == 740
    assert last[740] > 10 * first[740]


def test_im77_tables():
    bi, tri = load_im77_ngrams(2), load_im77_ngrams(3)
    assert len(bi) == 2730 and len(tri) == 3681
    assert bi.iloc[0].tolist() == [99, 267, 290]
    assert bi["count"].is_monotonic_decreasing
    sign_cols = ["left_sign", "right_sign"]
    assert bi[sign_cols].min().min() >= 1 and bi[sign_cols].max().max() <= 417


def test_mapping_links_jar_across_schemes():
    assert len(load_sign_mapping()) == 397
    assert wells_to_mahadevan()[740] == {342}


def test_all_417_templates_present():
    assert all(template_path(i).exists() for i in range(1, 418))


def test_labelled_seals_reading_order_and_photos():
    from indus.data import load_labelled_seals
    df = load_labelled_seals()
    assert len(df) >= 200
    # reading order is the reverse of the impression's left-to-right order
    assert all(r == l[::-1] for r, l in zip(df["signs_reading"], df["signs_ltr"]))
    # jar sign (Mahadevan 342) ends texts: often last in reading order, never first
    last = sum(s[-1] == "342" for s in df["signs_reading"])
    first = sum(s[0] == "342" for s in df["signs_reading"])
    assert last > 50 and first == 0
    # every listed photo exists, and almost every seal has an impression photo
    import os
    assert all(os.path.exists(p) for p in df["impression_photo"] if p)
    assert (df["impression_photo"] != "").mean() > 0.95
