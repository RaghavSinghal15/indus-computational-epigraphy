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

# Lipi: full ICIT corpus + CISI photos, downloaded by scripts/fetch_lipi.py (never committed)
LIPI_COMMIT = "fc8813a52e183d19c441631dc26c33a4dccbea39"
LIPI_DIR = REPO_ROOT / "data" / "external" / "lipi"


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


# ---------------------------------------------------------------- Lipi / full ICIT corpus
#
# Lipi's inscriptions.csv is Fuls's Corpus of Indus Inscriptions (ICIT database).
# Text codes look like "+740-460-503-002-817+": Wells numbers in VISUAL order, left
# to right, normalised so that every text reads right to left (Fuls 2022, ch. 3).
# "000" = illegible sign, "999" = blank space, "/" = next line, "[" or "]" = the
# text may continue past that end. Its Sanskrit/translation columns are one
# author's proposed decipherment; we never use them.

_LIPI_TEXT = r"\+\d{3}(?:-\d{3})*\+"   # complete single-line text, nothing doubtful


def _parse_lipi_text(text) -> list[int] | None:
    """Wells IDs, left to right, for a clean complete single-line text; None otherwise."""
    import re
    if not isinstance(text, str) or not re.fullmatch(_LIPI_TEXT, text.strip()):
        return None
    signs = [int(s) for s in text.strip().strip("+").split("-")]
    if 0 in signs or 999 in signs:
        return None
    return signs


def load_lipi_corpus() -> pd.DataFrame:
    """Every row of Lipi's inscriptions.csv, plus 'signs_ltr' (list of Wells IDs, or None if not clean)."""
    path = LIPI_DIR / "inscriptions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python scripts/fetch_lipi.py")
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["direction"] = df["dir."].fillna("").str.strip()
    df["signs_ltr"] = df["text"].map(_parse_lipi_text)
    return df


def load_icit_full(order: str = "reading", unique: bool = True) -> list[list[int]]:
    """All clean ICIT texts (complete, legible, one line, 2+ signs, known direction), Wells IDs.

    A superset of the 1,837-text benchmark from load_sequences(), for robustness checks.
    unique=True keeps one copy of each distinct sequence, like the benchmark.
    """
    df = load_lipi_corpus()
    ok = df[df.signs_ltr.notna() & (df.complete == "Y") & df.direction.isin(["R/L", "L/R"])]
    texts = [s for s in ok.signs_ltr if len(s) >= 2]
    if unique:
        texts = [list(t) for t in dict.fromkeys(tuple(t) for t in texts)]
    if order == "reading":
        texts = [t[::-1] for t in texts]
    elif order != "visual":
        raise ValueError("order must be 'reading' or 'visual'")
    return texts


def lipi_usable_seals(check_images: bool = True) -> pd.DataFrame:
    """Seals usable for M1: one object, one complete legible right-to-left text, an impression photo.

    Columns: cisi, icit_id, site, type, image_file (first impression photo),
    signs_ltr / signs_reading (Wells IDs), n_signs.
    """
    import json
    df = load_lipi_corpus()
    images = json.loads((LIPI_DIR / "seal_id_and_image_mapping.json").read_text())
    df["cisi"] = df["cisi"].fillna("").str.strip()
    one_row = df["cisi"].map(df["cisi"].value_counts()) == 1     # no multi-text objects
    ok = df[df["type"].fillna("").str.startswith("SEAL") & one_row & (df.cisi != "-")
            & df.signs_ltr.notna() & (df.complete == "Y") & (df.sides == "1") & (df.direction == "R/L")].copy()
    ok["image_file"] = ok.cisi.map(lambda c: next((f for f in images.get(c, []) if "imp" in f.lower()), None))
    ok = ok[ok.image_file.notna()]
    if check_images:
        ok = ok[ok.image_file.map(lambda f: (LIPI_DIR / "seal_images" / f).exists())]
    ok["signs_reading"] = ok.signs_ltr.map(lambda s: s[::-1])
    ok["n_signs"] = ok.signs_ltr.map(len)
    ok = ok.rename(columns={"id": "icit_id"})
    return ok[["cisi", "icit_id", "site", "type", "image_file", "signs_ltr", "signs_reading", "n_signs"]].reset_index(drop=True)


def load_lipi_seals() -> pd.DataFrame:
    """lipi_usable_seals() with absolute image paths and 'in_v1' (True for our 328 first-phase seals)."""
    df = lipi_usable_seals()
    df["image"] = df.image_file.map(lambda f: str(LIPI_DIR / "seal_images" / f))
    df["in_v1"] = df.cisi.isin(set(load_labelled_seals().cisi_label))
    return df


# ---------------------------------------------------------------- hand-drawn sign boxes (#23)

BOXES_DIR = RAW / "sign_boxes"


def photo_signs_ltr(seal) -> list[int]:
    """Return Wells IDs from left to right in this specific impression photograph."""
    # Lipi transcribes L-58 as +740-550+, but this scan shows 550 on the left
    # and the distinctive 740 jar on the right. Keep the catalogue order intact.
    if seal.image_file == "L-58_imp_a.jpg":
        return seal.signs_ltr[::-1]
    return seal.signs_ltr


def read_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Read YOLO rectangles as pixels, ordered left to right with a stacked pair on L-29.

    YOLO lines are 'class x_centre y_centre box_width box_height', all as fractions of the image size.
    """
    boxes = []
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:])
        boxes.append((round((cx - w / 2) * width), round((cy - h / 2) * height),
                      round((cx + w / 2) * width), round((cy + h / 2) * height)))
    boxes.sort(key=lambda b: b[0] + b[2])
    if path.name == "L-29_imp_a.txt" and len(boxes) == 5:
        # The three strokes sit above the caret and come before it in the guide.
        boxes[1:3] = sorted(boxes[1:3], key=lambda b: b[1] + b[3])
    return boxes


def load_sign_boxes() -> pd.DataFrame:
    """One row per hand-boxed sign: cisi, batch, image, pos_ltr, wells, x0, y0, x1, y1 (pixels).

    Labels come from the seal's known sequence: boxes sorted left to right get signs_ltr in order.
    Seals whose box count differs from their sequence length are left out (check_boxes.py lists them).
    """
    from PIL import Image
    seals = load_lipi_seals().set_index("cisi")
    batches = pd.read_csv(BOXES_DIR / "batches.csv")
    rows = []
    for _, b in batches.iterrows():
        label = BOXES_DIR / f"batch_{b.batch:02d}" / (Path(b.image_file).stem + ".txt")
        if not label.exists() or b.cisi not in seals.index:
            continue
        seal = seals.loc[b.cisi]
        w, h = Image.open(seal.image).size
        boxes = read_yolo_boxes(label, w, h)
        if len(boxes) != len(seal.signs_ltr):
            continue
        for i, (box, sign) in enumerate(zip(boxes, photo_signs_ltr(seal))):
            rows.append({"cisi": b.cisi, "batch": b.batch, "image": seal.image, "pos_ltr": i, "wells": sign,
                         "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]})
    return pd.DataFrame(rows)
