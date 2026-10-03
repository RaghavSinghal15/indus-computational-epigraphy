"""Loaders for every dataset in data/raw/.

Orientation convention (read this before using any sequence):

    Every file in data/raw/ stores signs in VISUAL order, left to right,
    exactly as they appear on the seal impression. The Indus script is read
    RIGHT TO LEFT, so reading order is the reverse of the stored order.

    load_sequences() returns READING order by default, so the first element
    of each sequence is the sign a reader meets first. Pass order="visual"
    to get the raw file order.

    Check: Wells sign 740 (the "jar", Mahadevan 342) is the best-known
    text-ending sign. In reading order it should be the most common LAST sign.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"

SEQUENCES_FILE = RAW / "icit_1837_sequences.csv"
BIGRAMS_FILE = RAW / "im77_bigrams.csv"
TRIGRAMS_FILE = RAW / "im77_trigrams.csv"
MAPPING_FILE = RAW / "sign_mapping_parpola_wells_mahadevan.csv"
TEMPLATES_DIR = RAW / "sign_templates"
CISI_DIR = RAW / "cisi_seals"


def load_sequences(order: str = "reading") -> list[list[int]]:
    """The 1,837 ICIT inscriptions as lists of Wells sign numbers.

    order="reading": first element is read first (right-most on the seal).
    order="visual":  as stored in the file, left to right on the seal.
    """
    if order not in ("reading", "visual"):
        raise ValueError("order must be 'reading' or 'visual'")
    with open(SEQUENCES_FILE, newline="") as f:
        rows = [[int(x) for x in row if x.strip()] for row in csv.reader(f)]
    if order == "reading":
        rows = [row[::-1] for row in rows]
    return rows


def load_im77_ngrams(n: int) -> pd.DataFrame:
    """IM77 pair (n=2) or triple (n=3) counts from indusscript.in, Mahadevan numbering.

    Columns are in visual order (left_sign ... right_sign) plus 'count'.
    Useful as a validation target, not as training data: these are counts
    over Mahadevan's corpus, which is a different corpus from the 1,837 ICIT texts.
    """
    if n == 2:
        return pd.read_csv(BIGRAMS_FILE)
    if n == 3:
        return pd.read_csv(TRIGRAMS_FILE)
    raise ValueError("n must be 2 or 3")


def load_sign_mapping() -> pd.DataFrame:
    """Cross-scheme sign table: one row per Parpola sign, with its Wells and Mahadevan codes.

    ID lists are ';'-separated strings like 'W390;W392' and 'M161;M162'.
    Many-to-many: one Wells sign can correspond to several Mahadevan signs.
    Covers 359 of Mahadevan's 417 signs.
    """
    return pd.read_csv(MAPPING_FILE).fillna("")


def wells_to_mahadevan() -> dict[int, set[int]]:
    """Wells number -> set of possible Mahadevan numbers."""
    out: dict[int, set[int]] = {}
    for _, row in load_sign_mapping().iterrows():
        wells = [int(w[1:]) for w in row["wells_ids"].split(";") if w]
        mahad = {int(m[1:]) for m in row["mahadevan_ids"].split(";") if m}
        for w in wells:
            out.setdefault(w, set()).update(mahad)
    return out


def template_path(mahadevan_id: int) -> Path:
    """Path to the 150x150 grayscale reference drawing for a Mahadevan sign (1..417)."""
    return TEMPLATES_DIR / f"{mahadevan_id}.png"


def load_labelled_seals() -> pd.DataFrame:
    """Real seal photos with known sign sequences (Mohenjo-daro, CISI Vol. 1 + IM77).

    One row per seal. Adds two list columns:
      signs_reading: Mahadevan IDs in reading order (right to left on the impression)
      signs_ltr:     the same signs left to right, as they appear on the impression photo
    IDs are strings because the source marks doubtful signs with "*" (e.g. "*342")
    and illegible ones as "0"; see data/raw/cisi_seals/README.md.
    Photo paths are made absolute.
    """
    df = pd.read_csv(CISI_DIR / "labelled_seals_M.csv", dtype=str).fillna("")
    df["signs_reading"] = df["signs_reading_order"].str.split()
    df["signs_ltr"] = df["signs_impression_left_to_right"].str.split()
    for col in ("impression_photo", "seal_photo"):
        df[col] = df[col].map(lambda p: str(CISI_DIR / p) if p else "")
    return df
