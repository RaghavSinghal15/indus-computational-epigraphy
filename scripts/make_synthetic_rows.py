"""Make carved-row training crops (indus.render_rows): synthetic data cut like real seals.

Usage:
    python scripts/make_synthetic_rows.py           # about 3 minutes

Writes data/processed/synthetic_rows_64.npz (images uint8 (N,64,64), labels =
Mahadevan sign numbers) and an example figure, reports/figures/synthetic_rows.png.

Sign sequences: half the rows follow the IM77 sign-pair statistics (realistic
neighbours), half are uniformly random signs (so all 417 signs get examples).
Row lengths follow the labelled training seals. The pairs of the validation
and test texts are removed from the IM77 counts first, so those texts don't
leak into training. Fixed seed.
"""

import cv2
import numpy as np

from indus.data import REPO_ROOT, load_labelled_seals
from indus.decode import N_SIGNS, bigram_model
from indus.render_rows import render_row, row_crops

OUT = REPO_ROOT / "data" / "processed" / "synthetic_rows_64.npz"
FIGURE = REPO_ROOT / "reports" / "figures" / "synthetic_rows.png"
CISI = REPO_ROOT / "data" / "raw" / "cisi_seals"
N_ROWS = 8000
SEED = 0


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load_labelled_seals()
    held = set((CISI / "test_seals.txt").read_text().split()) | set((CISI / "val_seals.txt").read_text().split())
    exclude = [[int(s.lstrip("*")) for s in seq][::-1] for seq in df[df["cisi_label"].isin(held)]["signs_reading"]]
    log_trans, log_uni = bigram_model(exclude)
    trans, uni = np.exp(log_trans), np.exp(log_uni)
    lengths = df[~df["cisi_label"].isin(held)]["n_signs"].astype(int).to_numpy()

    images, labels = [], []
    for r in range(N_ROWS):
        n = int(rng.choice(lengths))
        if r % 2 == 0:                                   # follow the pair statistics
            seq = [int(rng.choice(N_SIGNS, p=uni)) + 1]
            for _ in range(n - 1):
                seq.append(int(rng.choice(N_SIGNS, p=trans[seq[-1] - 1])) + 1)
        else:                                            # any signs
            seq = [int(s) for s in rng.integers(1, N_SIGNS + 1, n)]
        crops, labs = row_crops(seq, rng)
        images += crops
        labels += labs
        if (r + 1) % 2000 == 0:
            print(f"  {r + 1}/{N_ROWS} rows")
    np.savez_compressed(OUT, images=np.stack(images), labels=np.array(labels))
    counts = np.bincount(labels, minlength=N_SIGNS + 1)[1:]
    print(f"{len(labels)} crops -> {OUT.relative_to(REPO_ROOT)}; every sign has at least {counts.min()} crops")

    # figure: four rows with their true sign borders
    fig_rng = np.random.default_rng(SEED + 1)
    tiles = []
    for seq in [[342, 99, 267, 59, 17], [89, 336, 391, 1], [342, 176, 48, 65, 99, 86, 134]]:
        gray, bounds = render_row(seq, fig_rng)
        v = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x0, x1 in bounds:
            cv2.rectangle(v, (x0, 2), (x1, gray.shape[0] - 3), (0, 160, 0), 1)
        tiles.append(cv2.resize(v, (int(v.shape[1] * 120 / v.shape[0]), 120)))
    width = max(t.shape[1] for t in tiles)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURE), np.vstack([cv2.copyMakeBorder(t, 0, 6, 0, width - t.shape[1], cv2.BORDER_CONSTANT,
                                                           value=(255, 255, 255)) for t in tiles]))


if __name__ == "__main__":
    main()
