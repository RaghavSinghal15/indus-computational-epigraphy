"""Export the hand-box CNN's predictions on hand-box test crops, for E1 with measured confusions.

    python scripts/e1_handbox_predictions.py
    python scripts/e1_cnn_confusions.py data/processed/e1/handbox_test_predictions.csv \
        --crops data/processed/real_crops/crops_handbox.npz --output reports/e1_handbox_confusions.md

Needs wells_handbox.pt (from m1_handbox_cnn.py) and crops_handbox.npz.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_wells_cnn import load_weights                      # noqa: E402
from indus.cnn import SignCNN, predict, to_input              # noqa: E402
from indus.data import REPO_ROOT                              # noqa: E402
from indus.wells import wells_codes                           # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
OUT = PROCESSED / "e1" / "handbox_test_predictions.csv"

codes = wells_codes()
model = SignCNN(len(codes))
load_weights(PROCESSED / "models" / "wells_handbox.pt", model, codes, "real_finetuned")
model.eval()
with np.load(PROCESSED / "real_crops" / "crops_handbox.npz") as a:
    keep = a["split"] == "test"
    images, labels, seals, pos = a["images"][keep], a["labels"][keep], a["seal"][keep], a["pos"][keep]
top = predict(model, to_input(images))[0]          # class numbers, 1-based, best first
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cisi", "split", "position", "target", "predicted"])
    for s, p, y, t in zip(seals, pos, labels, top):
        w.writerow([s, "test", int(p), int(y), codes[t[0] - 1]])
right = sum(codes[t[0] - 1] == y for t, y in zip(top, labels))
print(f"{right}/{len(labels)} right ({right / len(labels):.1%}) -> {OUT.relative_to(REPO_ROOT)}")
