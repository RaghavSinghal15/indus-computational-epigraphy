"""Small masked sign model and a smoothed trigram comparison."""

from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn


class MaskedTransformer(nn.Module):
    """Encode the inscription in both directions and classify its one masked position."""

    def __init__(self, n_signs, max_length, width=64, layers=2, heads=4):
        super().__init__()
        self.unknown = n_signs
        self.mask = n_signs + 1
        self.pad = n_signs + 2
        self.tokens = nn.Embedding(n_signs + 3, width, padding_idx=self.pad)
        self.positions = nn.Embedding(max_length, width)
        layer = nn.TransformerEncoderLayer(width, nhead=heads, dim_feedforward=2 * width, dropout=0.15,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(width),
                                             enable_nested_tensor=False)
        self.classifier = nn.Linear(width, n_signs + 1)

    def forward(self, tokens, masked_positions):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        embedded = self.tokens(tokens) + self.positions(positions)
        encoded = self.encoder(embedded, src_key_padding_mask=tokens == self.pad)
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        return self.classifier(encoded[rows, masked_positions])


class MaskedNGram:
    """Score a masked sign by multiplying all nearby factors it affects."""

    def __init__(self, bigram, train):
        self.bigram = bigram
        self.pairs = defaultdict(Counter)
        for text in train:
            ids = [bigram.START, bigram.START, *bigram.ids(text), bigram.END]
            for a, b, c in zip(ids, ids[1:], ids[2:]):
                self.pairs[a, b][c] += 1
        self.totals = {context: sum(row.values()) for context, row in self.pairs.items()}
        self.followers = {context: len(row) for context, row in self.pairs.items()}
        self.backoff = np.exp(bigram.log_bi)

    def trigram(self, a, b, c):
        lower = self.backoff[b, c]
        context = (a, b)
        followers = self.followers.get(context)
        if followers is None:
            return np.log(lower)
        count = self.pairs[context][c]
        return np.log((count + followers * lower) / (self.totals[context] + followers))

    def scores(self, text, pos, order):
        m = self.bigram
        ids = [m.START, m.START, *m.ids(text), m.END]
        prev2, prev = ids[pos:pos + 2]
        next_sign = ids[pos + 3]
        if order == 2:
            return m.log_bi[prev, :m.UNK + 1] + m.log_bi[:m.UNK + 1, next_sign]
        next2 = ids[pos + 4] if pos + 4 < len(ids) else None
        return np.array([
            self.trigram(prev2, prev, sign) + self.trigram(prev, sign, next_sign)
            + (self.trigram(sign, next_sign, next2) if next2 is not None else 0)
            for sign in range(m.UNK + 1)
        ])


def metrics(logits, targets, unknown):
    """Masked top-k and pseudo-perplexity over the same positions for every model."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets)
    known = targets != unknown
    # Break ties by ascending sign ID. Unknown targets are misses in the ranking.
    ranked = np.argsort(-logits[:, :unknown], axis=1, kind="stable")
    hits = ranked == targets[:, None]
    log_normalizer = np.logaddexp.reduce(logits, axis=1)
    loss = log_normalizer - logits[np.arange(len(targets)), targets]
    return {
        "top1": float(hits[:, :1].any(axis=1).mean()),
        "top5": float(hits[:, :5].any(axis=1).mean()),
        "pseudo_ppl": float(np.exp(loss.mean())),
        "known_pseudo_ppl": float(np.exp(loss[known].mean())),
        "unknown": int((~known).sum()),
        "positions": len(targets),
    }
