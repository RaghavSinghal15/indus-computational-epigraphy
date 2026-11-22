"""Download the Lipi corpus, image index, sign font and usable seal images into data/external/lipi/.

Lipi (github.com/yajnadevam/lipi) is a digitised copy of Fuls's Corpus of Indus
Inscriptions (the ICIT database, Wells sign numbers) with CISI photos. It has no
licence, so these files are downloaded here and never committed (data/external/
is gitignored). Everything is pinned to one Lipi commit so results are reproducible.

Usage:
    python scripts/fetch_lipi.py                 # download from GitHub
    python scripts/fetch_lipi.py --clone PATH    # copy from a local clone instead
"""
import argparse
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from indus.data import LIPI_COMMIT, LIPI_DIR, lipi_usable_seals

FILES = {   # repo path -> local name
    "src/assets/data/inscriptions.csv": "inscriptions.csv",
    "src/assets/data/seal_id_and_image_mapping.json": "seal_id_and_image_mapping.json",
    "src/assets/fonts/indus-font.woff2": "indus-font.woff2",
}
RAW_URL = f"https://raw.githubusercontent.com/yajnadevam/lipi/{LIPI_COMMIT}/"


def get(repo_path: str, dest: Path, clone: Path | None) -> None:
    """Copy one file from the local clone or download it from GitHub (skips existing files)."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if clone:
        shutil.copy(clone / repo_path, dest)
    else:
        urllib.request.urlretrieve(RAW_URL + repo_path, dest)


def to_ttf(woff2: Path) -> Path:
    """Convert the web font to a plain .ttf so Pillow can draw with it."""
    from fontTools.ttLib import TTFont   # fonttools reads woff2 (needs the brotli package)
    font = TTFont(woff2)
    font.flavor = None                   # drop the woff2 compression wrapper
    out = woff2.with_suffix(".ttf")
    font.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", type=Path, help="local clone of yajnadevam/lipi at the pinned commit")
    ap.add_argument("--corpus-only", action="store_true", help="download only the inscription CSV")
    args = ap.parse_args()
    if args.corpus_only:
        repo_path = "src/assets/data/inscriptions.csv"
        get(repo_path, LIPI_DIR / FILES[repo_path], args.clone)
        print(LIPI_DIR / FILES[repo_path])
        return

    for repo_path, name in FILES.items():
        get(repo_path, LIPI_DIR / name, args.clone)
    to_ttf(LIPI_DIR / "indus-font.woff2")

    seals = lipi_usable_seals(check_images=False)
    jobs = [(f"public/seal_images/{f}", LIPI_DIR / "seal_images" / f) for f in seals.image_file]
    with ThreadPoolExecutor(8) as pool:   # images are small (~20 KB); download a few at a time
        list(pool.map(lambda j: get(j[0], j[1], args.clone), jobs))
    print(f"{len(seals)} usable seals, images in {LIPI_DIR / 'seal_images'}")


if __name__ == "__main__":
    main()
