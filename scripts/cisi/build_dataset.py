"""Build data/raw/cisi_seals/labelled_seals_M.csv from the per-source sequence files.

Usage:
    python scripts/cisi/build_dataset.py PHOTO_DIR

PHOTO_DIR holds the photos cut from the CISI PDF (photos/M/<label>_<side>.png,
made by scripts/cisi/photo_extraction). Photos of newly labelled seals are
copied into data/raw/cisi_seals/photos/ as JPEG.

Sources, in order of trust (the first one that has a text wins):
    hover.csv     every sign read from the site's own tooltip
    reader.csv    scripts/cisi/read_card2.py on a card screenshot (14/14 vs hover)
    manual.csv    read by hand from the tooltips (Raghav)
    chatgpt_*.csv read by ChatGPT's agent from the tooltips (3 of 24 overlapping
                  texts had one sign wrong, so these are NOT used for testing)
`verified` = 1 for the first three. Multi-line texts (multiline.txt) are left out.
"""
import csv
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "raw" / "cisi_seals"
SOURCES = [("hover.csv", "hover", 1), ("reader.csv", "reader", 1), ("manual.csv", "manual", 1),
           ("chatgpt_a.csv", "chatgpt", 0), ("chatgpt_b.csv", "chatgpt", 0)]


def main(photo_dir: Path) -> None:
    multiline = set((D / "sequences" / "multiline.txt").read_text().split())
    seqs = {}
    for fname, name, verified in SOURCES:
        for r in csv.DictReader(open(D / "sequences" / fname)):
            s = r["signs_left_to_right"].strip()
            if s and r["im77"] not in seqs and "MULTILINE" not in r.get("notes", ""):
                seqs[r["im77"]] = (s, name, verified)
    labels = {}
    for r in csv.DictReader(open(D / "cisi_concordance.csv")):
        labels.setdefault(r["cisi_label"], r["im77"])
    out = []
    for label, im77 in labels.items():
        if not label.startswith("M-") or im77 not in seqs or im77 in multiline:
            continue
        seq, src, ver = seqs[im77]
        row = dict(cisi_label=label, im77=im77)
        for side, col in (("a", "impression_photo"), ("A", "seal_photo")):
            src_png = photo_dir / "M" / f"{label}_{side}.png"
            dst = D / "photos" / f"{label}_{side}.jpg"
            if not dst.exists() and src_png.exists():
                cv2.imwrite(str(dst), cv2.imread(str(src_png)), [cv2.IMWRITE_JPEG_QUALITY, 92])
            row[col] = f"photos/{dst.name}" if dst.exists() else ""
        signs = seq.split()
        row.update(signs_impression_left_to_right=seq, signs_reading_order=" ".join(signs[::-1]),
                   n_signs=len(signs), source=src, verified=ver)
        out.append(row)
    out.sort(key=lambda r: int("".join(c for c in r["cisi_label"][2:] if c.isdigit())))
    with open(D / "labelled_seals_M.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    n_ver = sum(r["verified"] for r in out)
    print(f"{len(out)} seals ({n_ver} verified), {sum(1 for r in out if r['impression_photo'])} with impression photo,",
          f"{sum(r['n_signs'] for r in out)} signs")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
