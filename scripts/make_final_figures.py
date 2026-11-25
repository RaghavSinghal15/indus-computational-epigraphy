"""Figures for the final report and video (issue #46). Writes reports/figures/final/*.png."""
import cv2
import numpy as np

from indus.data import REPO_ROOT, load_sign_boxes
from indus.real_crops import sign_crops
from indus.segment import load_image, trim_caption
from indus.wells import load_v2_seals

OUT = REPO_ROOT / "reports" / "figures" / "final"
RED, GREEN = (40, 40, 220), (60, 170, 60)


def segmentation_figure(seals=("M-58", "M-1114", "M-1152")) -> None:
    """Forced cuts (red, sign count given) vs reviewed hand boxes (green) on test seals."""
    boxes = load_sign_boxes()
    tiles = []
    for cisi in seals:
        rows = boxes[boxes.cisi == cisi]
        if rows.empty:
            continue
        img = trim_caption(load_image(rows.image.iloc[0]))
        _, band, edges = sign_crops(img, len(rows))
        vis = img.copy()
        for e in edges:
            x = band.x0 + int(e)
            cv2.line(vis, (x, band.y0), (x, band.y1), RED, 3)
        cv2.rectangle(vis, (band.x0, band.y0), (band.x1, band.y1), RED, 2)
        for r in rows.itertuples():
            cv2.rectangle(vis, (r.x0, r.y0), (r.x1, r.y1), GREEN, 3)
        h = 300
        tiles.append(cv2.resize(vis, (round(vis.shape[1] * h / vis.shape[0]), h)))
    gap = np.full((300, 16, 3), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])
    cv2.imwrite(str(OUT / "segmentation_cuts_vs_boxes.png"), row)




def guardrail_figure() -> None:
    """Order tests on the Indus corpus vs two non-linguistic symbol systems (numbers from reports/)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    systems = ["Indus script", "Kudurru deity symbols", "Barn-star motifs"]
    # D1 real vs within-text shuffle, sign pairs (d1_guardrails.md, m4_symbol_controls.md)
    shuffle = [81.4, 84.6, 50.0]
    # E2 real vs fitted bigram process, sign triples (e2_null_benchmark.md, m4_symbol_controls.md)
    triples = [61.0, 50.8, 58.1]
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    y = np.arange(len(systems))
    h = 0.36
    ax.barh(y - h / 2 - 0.01, shuffle, h, color=blue, label="Real vs shuffled (pairs)")
    ax.barh(y + h / 2 + 0.01, triples, h, color=orange, label="Real vs bigram process (triples)")
    for yi, a, b in zip(y, shuffle, triples):
        ax.text(a + 0.8, yi - h / 2, f"{a:.1f}%", va="center", fontsize=8, color=ink)
        ax.text(b + 0.8, yi + h / 2, f"{b:.1f}%", va="center", fontsize=8, color=ink)
    ax.axvline(50, color=muted, lw=1, ls="--")
    ax.text(50.5, -0.75, "chance", fontsize=7, color=muted)
    ax.set_yticks(y, systems, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(40, 95)
    ax.set_xlabel("Held-out Naive Bayes accuracy (%)", fontsize=9, color=ink)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "guardrails_controls.png")



def composite_figure(n: int = 4, width: int = 520, height: int = 90) -> None:
    """Composites (left) next to real training bands (right), each strip scaled to the same width."""
    import json
    from indus.wells import load_v2_seals
    manifest = [json.loads(l) for l in (REPO_ROOT / "data/processed/composites/manifest.jsonl").open()]
    comp = [load_image(REPO_ROOT / "data/processed/composites" / m["image"]) for m in manifest[:40]]
    comp = [c for c in comp if c.shape[1] / c.shape[0] > 3.2][:n]
    split = dict(zip(*load_v2_seals()[["cisi", "split"]].T.values))
    boxes = load_sign_boxes()
    real = []
    for cisi, rows in boxes[boxes.cisi.map(split) == "train"].groupby("cisi"):
        if len(rows) < 5:
            continue
        img = load_image(rows.image.iloc[0])
        band = img[max(0, rows.y0.min() - 3):rows.y1.max() + 3, max(0, rows.x0.min() - 3):rows.x1.max() + 3]
        if band.shape[1] / band.shape[0] > 3.2:
            real.append(band)
        if len(real) == n:
            break

    def fit(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return cv2.copyMakeBorder(cv2.resize(g, (width, height), interpolation=cv2.INTER_AREA), 4, 4, 4, 4,
                                  cv2.BORDER_CONSTANT, value=255)
    left, right = np.vstack([fit(c) for c in comp]), np.vstack([fit(r) for r in real])
    cv2.imwrite(str(OUT / "composites_vs_real.png"), np.hstack([left, np.full((left.shape[0], 24), 255, np.uint8), right]))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    segmentation_figure()
    guardrail_figure()
    composite_figure()
    print("figures ->", OUT.relative_to(REPO_ROOT))
