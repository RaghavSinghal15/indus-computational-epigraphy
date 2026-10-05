"""Picture of the pipeline reading one seal: for slides and demos.

Usage:
    python scripts/demo_read_seal.py M-26                 # a CISI label from the labelled set
    python scripts/demo_read_seal.py M-26 --lam 0.75 --ends

Shows, top to bottom: the impression photo with the cuts, then for every sign
(in reading order, right to left like the seal) the crop, the model's top-3
guesses as template drawings, and the true sign. A guess matching the truth
has a green frame. With --lam > 0 the guesses use the sign-pair statistics
(indus.decode), with the pairs of all validation and test texts removed.

Writes reports/figures/demo_<label>.png.
"""

import argparse

import cv2
import numpy as np
import torch

from indus.align import aligned_crops
from indus.cnn import load_model, log_probs, to_input
from indus.data import REPO_ROOT, load_labelled_seals, template_path
from indus.decode import bigram_model, edge_model, posteriors
from indus.real_crops import sign_crops
from indus.segment import load_image

CISI = REPO_ROOT / "data" / "raw" / "cisi_seals"
TILE = 80
GREEN, GREY, WHITE = (60, 170, 60), (200, 200, 200), (255, 255, 255)


def tile(gray, frame=GREY):
    t = cv2.resize(gray, (TILE - 6, TILE - 6), interpolation=cv2.INTER_AREA)
    t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) if t.ndim == 2 else t
    return cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=frame)


def template(sign: int) -> np.ndarray:
    """Template drawn dark on white, like a printed sign list."""
    return 255 - cv2.imread(str(template_path(sign)), cv2.IMREAD_GRAYSCALE)


def label(img, text):
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seal", help="CISI label, e.g. M-26")
    ap.add_argument("--model", default=str(REPO_ROOT / "data" / "processed" / "models" / "sign_cnn.pt"))
    ap.add_argument("--cut", choices=["equal", "aligned"], default="equal")
    ap.add_argument("--lam", type=float, default=0.0, help="weight of the sign-pair statistics (0 = CNN alone)")
    ap.add_argument("--ends", action="store_true", help="also use start/end statistics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_model(args.model, args.device)
    df = load_labelled_seals()
    row = df.set_index("cisi_label").loc[args.seal]
    signs = row["signs_reading"]
    image = load_image(row["impression_photo"])
    if args.cut == "aligned":
        crops, band, edges, _ = aligned_crops(model, image, len(signs))
        cuts = sorted({e for pair in edges for e in pair})
    else:
        crops, band, cuts = sign_crops(image, len(signs))
    lp = log_probs(model, to_input(np.stack(crops)))
    if args.lam > 0:
        held = set((CISI / "test_seals.txt").read_text().split()) | set((CISI / "val_seals.txt").read_text().split())
        to_ltr = lambda seq: [int(s.lstrip("*")) for s in seq][::-1]
        log_trans, log_uni = bigram_model([to_ltr(s) for s in df[df["cisi_label"].isin(held)]["signs_reading"]])
        start, end = log_uni, None
        if args.ends:
            start, end = edge_model([to_ltr(s) for s in df[~df["cisi_label"].isin(held)]["signs_reading"]], log_uni)
        lp = posteriors(lp[::-1], log_trans, start, args.lam, end)[::-1]    # decode works left to right
    top3 = np.argsort(-lp, axis=1)[:, :3] + 1

    # photo with cuts
    photo = image.copy()
    cv2.rectangle(photo, (band.x0, band.y0), (band.x1, band.y1), (0, 0, 255), 2)
    for c in cuts:
        cv2.line(photo, (band.x0 + c, band.y0), (band.x0 + c, band.y1), (0, 200, 0), 2)

    # one column per sign, right to left so it lines up with the seal
    columns = []
    for crop, guesses, truth in list(zip(crops, top3, signs))[::-1]:
        t = int(truth.lstrip("*"))
        cells = [tile(crop)]
        cells += [tile(template(g), GREEN if g == t else GREY) for g in guesses]
        cells.append(label(tile(template(t) if t > 0 else np.full((64, 64), 255, np.uint8), WHITE), f"true {truth}"))
        columns.append(np.vstack(cells))
    grid = np.hstack([cv2.copyMakeBorder(c, 0, 0, 4, 4, cv2.BORDER_CONSTANT, value=WHITE) for c in columns])
    head = np.full((22, grid.shape[1], 3), 255, np.uint8)
    what = "CNN" + (f" + sign statistics (lambda {args.lam})" if args.lam > 0 else "")
    label(head, f"{args.seal}: rows = crop, guesses 1-3 ({what}), true sign. Green = right.")
    photo = cv2.resize(photo, (grid.shape[1], int(photo.shape[0] * grid.shape[1] / photo.shape[1])))
    if photo.shape[0] > 360:                       # keep the photo from dominating
        w = int(photo.shape[1] * 360 / photo.shape[0])
        photo = cv2.copyMakeBorder(cv2.resize(photo, (w, 360)), 0, 0, (grid.shape[1] - w) // 2,
                                   grid.shape[1] - w - (grid.shape[1] - w) // 2, cv2.BORDER_CONSTANT, value=WHITE)
    out = REPO_ROOT / "reports" / "figures" / f"demo_{args.seal}.png"
    cv2.imwrite(str(out), np.vstack([photo, head, grid]))
    right = sum(int(t.lstrip("*")) == g[0] for t, g in zip(signs, top3) if t != "0")
    print(f"{args.seal}: {right}/{len(signs)} signs right at top-1 -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
