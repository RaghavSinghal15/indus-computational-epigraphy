"""3a: recognition-driven segmentation. Let the CNN decide where the cuts go.

Equal-width slices (real_crops.forced_cuts) go wrong when something that is not
a sign sits in the text band (the unicorn's horn, the seal's frame) or when
signs differ a lot in width. Here we try MANY possible pieces of the band, ask
the CNN how confident it is about each, and pick the set of n pieces that
together score best. This is the same idea as "forced alignment" in speech
recognition.

Two uses:
  * labels known (building TRAINING crops): a piece's score is the CNN's
    log-probability of the sign that should be there, so the cuts move to
    where the right signs are. Better crops -> retrain -> better CNN.
  * labels unknown (TEST crops, and reading new seals): a piece's score is the
    log-probability of the CNN's best guess. Only the sign COUNT is used.
    Test crops must be cut this way, never with their labels, or the test
    would be rigged in the CNN's favour.

The search ("dynamic programming"): candidate cut positions every 8% of the
band height; a piece is any span between two positions whose width is
0.2-1.6 band heights; pieces may not overlap; columns left out between or
around pieces cost a little in proportion to the carving they contain (so
the horn can be dropped cheaply but a real sign cannot); pieces much narrower
or wider than a typical sign pay a width penalty (no slivers, no two-sign boxes). best[k][j] = best
total score with k pieces placed, the last ending at position j; each step
extends it by one piece. Total work: about n x (number of pieces).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from indus.cnn import to_input
from indus.pipeline import find_text_row
from indus.real_crops import _band_profile, square_crop

STEP_FRAC = 0.08                 # candidate cut every 8% of the band height
MIN_W, MAX_W = 0.2, 1.6          # piece width limits, in band heights
SKIP_COST = 3.0                  # cost of leaving out one average sign's worth of carving
WIDTH_PRIOR = 1.0                # penalty for pieces far from a typical width (see width_penalty)
TYPICAL_W = 0.75                 # typical sign width, in band heights


def band_gray(image_bgr: np.ndarray, mirrored: bool = False):
    """(text band box, grayscale band in impression orientation)."""
    band = find_text_row(image_bgr)
    gray = cv2.cvtColor(band.crop(image_bgr), cv2.COLOR_BGR2GRAY)
    return band, (gray[:, ::-1] if mirrored else gray)


@torch.no_grad()
def piece_log_probs(model, gray: np.ndarray, n: int):
    """Candidate positions, pieces (start index, end index) and the CNN's
    log-probabilities for every piece, shape (pieces, 417)."""
    h, w = gray.shape
    step = max(1, round(STEP_FRAC * h))
    positions = np.array(sorted(set(range(0, w, step)) | {w}))
    pieces = [(i, j) for i in range(len(positions)) for j in range(i + 1, len(positions))
              if MIN_W * h <= positions[j] - positions[i] <= MAX_W * h]
    if len(pieces) < n:                                  # band too short for the limits: allow anything
        pieces = [(i, j) for i in range(len(positions)) for j in range(i + 1, len(positions))]
    crops = np.stack([square_crop(gray, positions[i], positions[j]) for i, j in pieces])
    model.eval()
    dev = next(model.parameters()).device
    x = torch.from_numpy(to_input(crops))
    logp = torch.cat([F.log_softmax(model(x[k:k + 512].to(dev)), dim=1).cpu() for k in range(0, len(x), 512)])
    return positions, pieces, logp.numpy()


def width_penalty(widths: np.ndarray, height: int) -> np.ndarray:
    """Cost of each piece width: WIDTH_PRIOR * (log(width / typical width))^2.

    Zero at the typical width (0.75 band heights), 0.5 for half or 1.5x that
    (with WIDTH_PRIOR = 1), growing fast for slivers and for pieces that
    swallow two signs. Mild enough that a confident CNN can still pick a
    narrow stroke sign.
    """
    return WIDTH_PRIOR * np.log(widths / (TYPICAL_W * height)) ** 2


def best_pieces(positions, pieces, piece_scores, n, ink_cum):
    """Dynamic programme: the n non-overlapping pieces (left to right) with the
    highest total score minus the cost of carving left out.

    piece_scores: (pieces, n) array, score of each piece in slot k (slot 0 = leftmost).
    ink_cum[j]: carving to the left of position j, in units of one average sign.
    Returns a list of n piece indices.
    """
    P = len(positions)
    NEG = -1e18
    best = np.full((n + 1, P), NEG)
    back = np.full((n + 1, P), -1, int)          # which piece ended slot k at position j
    best[0] = -SKIP_COST * ink_cum               # everything left of j skipped
    starts = np.array([i for i, _ in pieces])
    ends = np.array([j for _, j in pieces])
    for k in range(n):
        # G[s] = best way to have k pieces done and be standing at position s,
        # with the columns between the last piece's end and s skipped:
        # max over j <= s of best[k][j] + cost*ink_cum[j], minus cost*ink_cum[s]
        G = np.maximum.accumulate(best[k] + SKIP_COST * ink_cum) - SKIP_COST * ink_cum
        cand = G[starts] + piece_scores[:, k]
        for p in np.argsort(cand):               # ascending, so the best candidate per end wins last
            if cand[p] > best[k + 1, ends[p]]:
                best[k + 1, ends[p]] = cand[p]
                back[k + 1, ends[p]] = p
    final = best[n] - SKIP_COST * (ink_cum[-1] - ink_cum)   # columns right of the last piece skipped
    j = int(np.argmax(final))
    chosen = []
    for k in range(n, 0, -1):
        p = back[k, j]
        chosen.append(p)
        # the previous piece ended at the best position at or before this piece's start
        s = starts[p]
        prev = best[k - 1, :s + 1] + SKIP_COST * ink_cum[:s + 1]
        j = int(np.argmax(prev))
    return chosen[::-1]


def aligned_crops(model, image_bgr: np.ndarray, n: int, labels_reading=None, mirrored: bool = False):
    """n crops in reading order, cut where the CNN is most confident.

    labels_reading: the known signs in reading order (training), or None (test / new seals).
    Returns (crops, band, piece edges in band columns left to right, top guess per crop).
    """
    band, gray = band_gray(image_bgr, mirrored)
    positions, pieces, logp = piece_log_probs(model, gray, n)
    if labels_reading is not None:
        ltr = [int(s.lstrip("*")) for s in labels_reading][::-1]      # left to right on the impression
        # illegible (0) signs: no label to check, so score them by the best guess
        cols = [logp[:, s - 1] if s > 0 else logp.max(axis=1) for s in ltr]
        scores = np.stack(cols, axis=1)
    else:
        scores = np.repeat(logp.max(axis=1, keepdims=True), n, axis=1)
    widths = np.array([positions[j] - positions[i] for i, j in pieces], float)
    scores = scores - width_penalty(widths, gray.shape[0])[:, None]

    # carving per position, in units of one average sign, for the skip cost
    scale, _, profile = _band_profile(gray)
    cum = np.concatenate([[0.0], np.cumsum(profile)])
    per_sign = max(cum[-1] / n, 1e-6)
    ink_cum = np.array([cum[min(int(x * scale), len(profile))] for x in positions]) / per_sign

    chosen = best_pieces(positions, pieces, scores, n, ink_cum)
    edges = [(int(positions[pieces[p][0]]), int(positions[pieces[p][1]])) for p in chosen]
    crops = [square_crop(gray, a, b) for a, b in edges][::-1]        # reading order
    guesses = [int(logp[p].argmax()) + 1 for p in chosen][::-1]
    return crops, band, edges, guesses
