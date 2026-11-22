"""Fix the v2 seal-level split once (issue #22). Writes data/raw/splits/v2_seals.csv.

  * harappa: all Harappa seals, never trained on (cross-site test, issue #37)
  * test / val: first-phase test and val seals keep their roles (v1 vs v2 on the same seals),
    plus 20% / 10% of the new seals (fixed seed)
  * train: everything else
Refuses to overwrite an existing split: the split must never change once results exist.
"""
import random

import pandas as pd

from indus.data import CISI_DIR, load_lipi_seals
from indus.wells import SPLIT_FILE

SEED = 0

if SPLIT_FILE.exists():
    raise SystemExit(f"{SPLIT_FILE} exists; the split is fixed. Delete it only if no results use it.")

seals = load_lipi_seals()
old_test = set((CISI_DIR / "test_seals.txt").read_text().split())
old_val = set((CISI_DIR / "val_seals.txt").read_text().split())
split = {}
for c, site, v1 in zip(seals.cisi, seals.site, seals.in_v1):
    if site == "Harappa":
        split[c] = "harappa"
    elif c in old_test:
        split[c] = "test"
    elif c in old_val:
        split[c] = "val"
    elif v1:
        split[c] = "train"
new = sorted(c for c in seals.cisi if c not in split)
random.Random(SEED).shuffle(new)
n_test, n_val = round(0.2 * len(new)), round(0.1 * len(new))
for i, c in enumerate(new):
    split[c] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

out = seals[["cisi", "site", "in_v1"]].assign(split=seals.cisi.map(split)).sort_values("cisi")
SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(SPLIT_FILE, index=False)
print(out.groupby(["split", "in_v1"]).size().unstack(fill_value=0).to_string())
