"""Fetch the exact alternate annotation views into the ignored Lipi photo cache."""
import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from PIL import Image, ImageOps

from indus.data import BOXES_DIR, LIPI_COMMIT, LIPI_DIR, box_photo_overrides

LIPI_IMAGES = f"https://raw.githubusercontent.com/yajnadevam/lipi/{LIPI_COMMIT}/public/seal_images/"
CISI_V1 = "https://ignca.gov.in/Asi_data/76884.pdf"
FRENEZ_FIG14 = (
    "https://figures.academia-assets.com/57225728/figure_014.jpg"
    "?response-content-disposition=attachment%3B+filename%3Dfigure-14-standard-indus-seals-made-of-terracotta-ivory"
)


def render_photo(name: str, target: Path, scratch: Path) -> None:
    """Recreate the photos used in Make Sense, including their orientation and crop."""
    if name == "K-39_A_CISI_V1_p305_mirrored.png":
        book = scratch / "cisi-v1.pdf"
        page = scratch / "page"
        urlretrieve(CISI_V1, book)
        subprocess.run(
            ["pdftoppm", "-f", "341", "-l", "341", "-r", "235", "-singlefile", "-png", str(book), str(page)],
            check=True,
        )
        source = page.with_suffix(".png")
        frame = (999, 1400)
        region = (46, 963, 237, 1165)
    elif name == "M-1199_Frenez_fig14_mirrored.png":
        source = scratch / "frenez-figure14.jpg"
        urlretrieve(FRENEZ_FIG14, source)
        frame = (1887, 515)
        region = (1445, 0, 1886, 434)
    else:
        urlretrieve(LIPI_IMAGES + name, target)
        return

    with Image.open(source) as image:
        bounds = tuple(round(v * image.size[i % 2] / frame[i % 2]) for i, v in enumerate(region))
        ImageOps.mirror(image.crop(bounds).convert("RGB")).save(target, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", type=int, nargs="*", help="Batch numbers. Default: all registered batches.")
    args = parser.parse_args()
    batches = args.batches or [int(p.parent.name.removeprefix("batch_"))
                               for p in BOXES_DIR.glob("batch_*/photo_overrides.csv")]
    names = {name for batch in batches for name in box_photo_overrides(batch).values()}
    dest = LIPI_DIR / "seal_images"
    dest.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        target = dest / name
        if not target.exists():
            with TemporaryDirectory(dir=dest) as folder:
                scratch = Path(folder)
                download = scratch / name
                render_photo(name, download, scratch)
                download.replace(target)
        print(name)


if __name__ == "__main__":
    main()
