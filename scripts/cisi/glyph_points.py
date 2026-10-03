"""Where to hover: sign positions on a saved indusscript.in Browse screenshot.

Usage: python glyph_points.py SCREENSHOT.jpg
Prints JSON with, in the 1536x770 coordinate frame the browser tools use:
  card: [top, bottom] of the centred text card (null if not found)
  footer_zoom: region to zoom to read "Length:N"
  points: [x, y] hover points, one per ink cluster, LEFT to RIGHT
  runs: [x0, x1] raw ink column runs (for when clusters hold more than one sign)
"""
import json, sys

import cv2
import numpy as np


def main(path):
    img = cv2.imread(path, 0)
    h, w = img.shape
    s = w / 1536                        # screenshot may be scaled; report full-frame coordinates
    band = img[:, int(0.15 * w):int(0.85 * w)]
    dark_frac = (band < 235).mean(axis=1)
    rows = [y for y in range(h) if dark_frac[y] > 0.9]
    groups = []
    for y in rows:
        if groups and y - groups[-1][-1] <= 2:
            groups[-1].append(y)
        else:
            groups.append([y])
    ys = [g[0] for g in groups]
    card = None
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            if 250 * s <= ys[j] - ys[i] <= 420 * s:
                card = (ys[i], ys[j])
    if card is None:
        print(json.dumps({"card": None}))
        return
    top, bottom = card
    y0, y1 = top + int(80 * s), bottom - int(55 * s)
    x0, x1 = int(330 * s), int(1430 * s)
    ink = img[y0:y1, x0:x1] < 140
    cols = ink.any(axis=0)
    runs, x = [], 0
    while x < len(cols):
        if cols[x]:
            a = x
            while x < len(cols) and cols[x]:
                x += 1
            runs.append([a, x])
        x += 1
    # clusters: merge runs closer than 12 px (full scale)
    clusters = []
    for r in runs:
        if clusters and r[0] - clusters[-1][1] < 12 * s:
            clusters[-1][1] = r[1]
        else:
            clusters.append(list(r))
    rows_with_ink = np.nonzero(ink.any(axis=1))[0]
    cy = (y0 + (rows_with_ink.min() + rows_with_ink.max()) / 2) if len(rows_with_ink) else (y0 + y1) / 2
    f = lambda v: int(round(v / s))
    out = {
        "card": [f(top), f(bottom)],
        "footer_zoom": [f(140 * s), f(bottom - 45 * s), f(700 * s), f(bottom - 5 * s)],
        "glyph_zoom": [f(x0 + clusters[0][0] - 40 * s) if clusters else f(x0), f(y0 - 10 * s), f(1440 * s), f(y1 + 50 * s)],
        "points": [[f(x0 + (a + b) / 2), f(cy)] for a, b in clusters],
        "runs": [[f(x0 + a), f(x0 + b)] for a, b in runs],
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
