"""Forced segmentation (real_crops.py), the text-only band rule and the CNN's shapes."""

import numpy as np
import torch

from indus.cnn import SignCNN, augment, to_input
from indus.data import load_labelled_seals
from indus.pipeline import find_text_row
from indus.real_crops import CROP_SIZE, forced_cuts, sign_crops
from indus.segment import load_image


def test_forced_cuts_land_in_gaps():
    # three blocks of ink separated by clear gaps at columns 30-39 and 70-79
    profile = np.ones(110)
    profile[30:40] = 0
    profile[70:80] = 0
    cuts = forced_cuts(profile, 3)
    assert len(cuts) == 2
    assert 30 <= cuts[0] < 40 and 70 <= cuts[1] < 80


def test_forced_cuts_always_increasing():
    cuts = forced_cuts(np.zeros(20), 12)        # flat profile, more pieces than is comfortable
    assert len(cuts) == 11 and all(b > a for a, b in zip(cuts, cuts[1:]))


def test_sign_crops_count_and_size():
    row = load_labelled_seals().iloc[0]
    n = len(row["signs_reading"])
    crops, band, edges = sign_crops(load_image(row["impression_photo"]), n)
    assert len(crops) == n and len(edges) == n + 1
    assert all(c.shape == (CROP_SIZE, CROP_SIZE) for c in crops)


def test_text_only_seal_band_is_whole_photo():
    # M-371 is a long text-only seal: the band must cover most of the photo
    df = load_labelled_seals().set_index("cisi_label")
    image = load_image(df.loc["M-371", "impression_photo"])
    band = find_text_row(image)
    assert band.x1 - band.x0 > 0.8 * image.shape[1]
    assert band.y1 - band.y0 > 0.8 * image.shape[0]


def test_cnn_output_shape_and_augment_keeps_shape():
    x = torch.from_numpy(to_input(np.random.default_rng(0).integers(0, 255, (4, 64, 64), dtype=np.uint8)))
    assert augment(x).shape == x.shape
    assert SignCNN()(x).shape == (4, 417)


def test_alignment_picks_best_pieces_and_skips_cheap_junk():
    from indus.align import best_pieces
    positions = np.arange(11)
    pieces = [(i, j) for i in range(11) for j in range(i + 1, 11) if j - i <= 4]
    # slot 0 wants piece (1, 3), slot 1 wants piece (5, 8); everything else scores badly
    scores = np.full((len(pieces), 2), -10.0)
    scores[pieces.index((1, 3)), 0] = 0.0
    scores[pieces.index((5, 8)), 1] = 0.0
    no_ink = np.zeros(11)                       # nothing to lose by skipping columns
    chosen = best_pieces(positions, pieces, scores, 2, no_ink)
    assert [pieces[p] for p in chosen] == [(1, 3), (5, 8)]
