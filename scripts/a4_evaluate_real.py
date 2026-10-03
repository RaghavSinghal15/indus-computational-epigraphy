"""A4: accuracy of the M1 pipeline on real labelled seal impressions.

Usage:
    python scripts/a4_evaluate_real.py

For every seal in indus.data.load_labelled_seals() with an impression photo:
  1. run the whole pipeline (find text band -> A1 boxes -> A2 top-5),
  2. compare the number of boxes with the true number of signs (A1),
  3. where the counts match, compare sign by sign in reading order (A2):
     top-1 = best guess is right, top-5 = right answer among the 5 guesses.
     Doubtful (*N) and illegible (0) signs are skipped.
Also scores A2 alone with PERFECT boxes ("oracle" segmentation): the band is
cut into n equal-width slices, n = true sign count. This separates A2's errors
from A1's.

Writes reports/a4_real_seals.md with the numbers.
"""

from collections import Counter

import cv2
import numpy as np

from indus.classify import classify_crop
from indus.data import REPO_ROOT, load_labelled_seals
from indus.pipeline import find_text_row, read_seal
from indus.segment import load_image

OUT = REPO_ROOT / "reports" / "a4_real_seals.md"


def scorable(sign: str) -> bool:
    return not sign.startswith("*") and sign != "0"


def main() -> None:
    df = load_labelled_seals()
    df = df[df["impression_photo"] != ""]

    count_diff = Counter()
    p_t1 = p_t5 = p_n = 0          # pipeline, on seals where A1 found the right count
    o_t1 = o_t5 = o_n = 0          # A2 with oracle (equal-width) boxes
    for _, row in df.iterrows():
        image = load_image(row["impression_photo"])
        truth = row["signs_reading"]
        n = len(truth)

        band, readings = read_seal(image)
        count_diff[len(readings) - n] += 1
        if len(readings) == n:
            for r, t in zip(readings, truth):
                if scorable(t):
                    ids = [str(m.mahadevan_id) for m in r.matches]
                    p_n += 1; p_t1 += ids[0] == t; p_t5 += t in ids

        # oracle segmentation: equal slices of the text band, read right to left
        crop = band.crop(image)
        w = crop.shape[1]
        edges = np.linspace(w, 0, n + 1).astype(int)          # right to left = reading order
        for k, t in enumerate(truth):
            if not scorable(t):
                continue
            piece = crop[:, edges[k + 1]:edges[k]]
            ids = [str(m.mahadevan_id) for m in classify_crop(piece)]
            o_n += 1; o_t1 += ids[0] == t; o_t5 += t in ids

    total = len(df)
    exact = count_diff[0]
    lines = [
        "# A4: M1 pipeline on real seal impressions",
        "",
        f"Seals: {total} Mohenjo-daro impressions with verified sequences (data/raw/cisi_seals).",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| A1 finds the right number of signs | {exact}/{total} ({exact / total:.0%}) |",
        f"| A1 finds too few (merges) | {sum(v for k, v in count_diff.items() if k < 0)} |",
        f"| A1 finds too many (splits / junk) | {sum(v for k, v in count_diff.items() if k > 0)} |",
        f"| Full pipeline, top-1 per sign (seals with right count) | {p_t1}/{p_n} ({p_t1 / max(1, p_n):.1%}) |",
        f"| Full pipeline, top-5 per sign | {p_t5}/{p_n} ({p_t5 / max(1, p_n):.1%}) |",
        f"| A2 with equal-width boxes, top-1 | {o_t1}/{o_n} ({o_t1 / max(1, o_n):.1%}) |",
        f"| A2 with equal-width boxes, top-5 | {o_t5}/{o_n} ({o_t5 / max(1, o_n):.1%}) |",
        f"| Chance level, top-1 / top-5 | {1 / 417:.1%} / {5 / 417:.1%} |",
        "",
        "Box count minus true count: " + ", ".join(f"{k:+d}: {v}" for k, v in sorted(count_diff.items())),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
