"""Clean drawings of Wells (ICIT) signs, from Lipi's sign font.

The font has one glyph per Wells code at Unicode codepoint U+E000 + code
(e.g. the jar sign, Wells 740, is chr(0xE000 + 740)). It covers all 712 codes
used in the corpus, and the glyphs match Fuls's sign list (Corpus of Indus
Inscriptions, Tables 2.6-2.8). Download it first: python scripts/fetch_lipi.py
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from indus.data import LIPI_DIR

FONT_FILE = LIPI_DIR / "indus-font.ttf"


@lru_cache(maxsize=4)
def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_FILE), px)


def wells_codes() -> list[int]:
    """Every Wells code that has a glyph in the font."""
    from fontTools.ttLib import TTFont
    cmap = TTFont(FONT_FILE).getBestCmap()          # codepoint -> glyph name
    return sorted(cp - 0xE000 for cp in cmap if 0xE000 <= cp < 0xE000 + 1000)


def glyph(wells: int, size: int = 64, pad: float = 0.1) -> np.ndarray:
    """Grayscale image (size x size, uint8): black sign on white, centred, scaled to fit."""
    big = 4 * size                                    # draw large, crop to ink, then shrink
    canvas = Image.new("L", (big, big), 255)
    ImageDraw.Draw(canvas).text((big // 8, big // 8), chr(0xE000 + wells), font=_font(big // 2), fill=0)
    ink = np.array(canvas) < 128
    if not ink.any():
        raise KeyError(f"no glyph for Wells {wells}")
    ys, xs = np.where(ink)
    crop = canvas.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    inner = int(size * (1 - 2 * pad))
    crop.thumbnail((inner, inner), Image.LANCZOS)     # keeps the aspect ratio
    out = Image.new("L", (size, size), 255)
    out.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return np.array(out)


def strip(signs: list[int], size: int = 48) -> np.ndarray:
    """Glyphs side by side in the order given (pass left-to-right order to match a photo)."""
    return np.hstack([glyph(s, size) for s in signs])
