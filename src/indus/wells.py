"""M1 v2 label space and split: Wells (ICIT) sign IDs on the Lipi seals. See docs/ROADMAP.md.

Classes are every Wells code used anywhere in the full ICIT corpus (712 signs),
in increasing order; class index i <-> Wells code WELLS_CODES[i].
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from indus.data import RAW, load_lipi_corpus, load_lipi_seals

SPLIT_FILE = RAW / "splits" / "v2_seals.csv"


@lru_cache(maxsize=1)
def wells_codes() -> tuple[int, ...]:
    """All Wells codes in the corpus (legible signs only), sorted. Index = class number."""
    import re
    texts = load_lipi_corpus()["text"].dropna()
    codes = {int(c) for t in texts for c in re.findall(r"\d{3}", t)} - {0, 999}
    return tuple(sorted(codes))


def class_index() -> dict[int, int]:
    """Wells code -> class number (0 .. len(wells_codes()) - 1)."""
    return {code: i for i, code in enumerate(wells_codes())}


def load_v2_seals() -> pd.DataFrame:
    """load_lipi_seals() plus the fixed v2 'split': train, val, test or harappa (cross-site test)."""
    split = pd.read_csv(SPLIT_FILE).set_index("cisi")["split"]
    seals = load_lipi_seals()
    seals["split"] = seals.cisi.map(split)
    return seals
