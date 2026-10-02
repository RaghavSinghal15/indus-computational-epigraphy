"""One command: seal photo in, sign sequence out (A3).

Usage:
    python scripts/read_seal.py PHOTO [PHOTO ...] [--row] [--mirrored]

    --row        the photo shows only the row of signs (skip finding the text band)
    --mirrored   the photo is of the seal itself (carved stone), not an impression

Example:
    python scripts/read_seal.py data/raw/seal_samples_ocr2017/text_row_28.png --row

Prints the sequence in reading order (Mahadevan IDs, best guess per sign) and
the top-5 guesses for each sign, and saves the photo with boxes and guesses
drawn on it to reports/figures/read_<photo name>.png.
"""

import argparse
from pathlib import Path

import cv2

from indus.data import REPO_ROOT
from indus.pipeline import draw_readings, read_seal, read_text_row
from indus.segment import load_image

OUT_DIR = REPO_ROOT / "reports" / "figures"


def main() -> None:
    # argparse turns the words typed after the script name into Python values
    parser = argparse.ArgumentParser(description="Read the signs on an Indus seal photo.")
    parser.add_argument("photos", nargs="+", type=Path, help="one or more image files")
    parser.add_argument("--row", action="store_true", help="photo is just the text row")
    parser.add_argument("--mirrored", action="store_true", help="photo of the seal itself, not an impression")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in args.photos:
        image = load_image(path)
        if args.row:
            band, readings = None, read_text_row(image, mirrored=args.mirrored)
        else:
            band, readings = read_seal(image, mirrored=args.mirrored)

        kind = "seal (mirrored)" if args.mirrored else "impression"
        print(f"\n{path.name}  [{kind}, {len(readings)} signs found]")
        print("sequence (reading order):", " ".join(str(r.best) for r in readings))
        for i, r in enumerate(readings, start=1):
            guesses = "  ".join(f"{m.mahadevan_id} ({m.score:.2f})" for m in r.matches)
            print(f"  sign {i}: {guesses}")

        out = OUT_DIR / f"read_{path.stem}.png"
        cv2.imwrite(str(out), draw_readings(image, readings, band))
        print("saved:", out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
