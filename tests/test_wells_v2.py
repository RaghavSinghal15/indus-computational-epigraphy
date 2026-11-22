"""M1 v2: Wells label space and the fixed seal-level split (#22)."""
import numpy as np
import pytest

from indus.data import CISI_DIR, LIPI_DIR
from indus.segment import trim_caption

needs_lipi = pytest.mark.skipif(not (LIPI_DIR / "inscriptions.csv").exists(),
                                reason="run python scripts/fetch_lipi.py first")


def test_trim_caption_drops_white_strip_with_text():
    img = np.full((100, 80), 250, np.uint8)
    img[:70] = 120                    # the seal: grey
    img[85:90, 30:50] = 0             # caption text on white
    assert trim_caption(img).shape[0] == 70


def test_trim_caption_keeps_image_without_caption():
    img = np.full((50, 50), 120, np.uint8)
    assert trim_caption(img).shape == (50, 50)


@needs_lipi
def test_wells_classes():
    from indus.wells import class_index, wells_codes
    codes = wells_codes()
    assert len(codes) == 712 and 740 in codes and 0 not in codes and 999 not in codes
    assert class_index()[codes[5]] == 5


@needs_lipi
def test_v2_split_is_fixed_and_clean():
    from indus.wells import load_v2_seals
    s = load_v2_seals()
    assert s.split.notna().all()
    assert set(s.split) == {"train", "val", "test", "harappa"}
    assert (s[s.site == "Harappa"].split == "harappa").all()
    assert (s[s.split == "harappa"].site == "Harappa").all()
    old_test = set((CISI_DIR / "test_seals.txt").read_text().split())
    old_val = set((CISI_DIR / "val_seals.txt").read_text().split())
    assert (s[s.cisi.isin(old_test)].split == "test").all()     # v1 and v2 compared on the same seals
    assert (s[s.cisi.isin(old_val)].split == "val").all()
    counts = s.split.value_counts()
    assert counts["test"] == 141 and counts["val"] == 67 and counts["train"] == 466
