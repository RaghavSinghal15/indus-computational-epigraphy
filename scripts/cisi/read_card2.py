"""Read a sign sequence off a saved indusscript.in Browse screenshot, using the
site's own sign drawings (scratchpad/site_glyphs) as the reference.

Each candidate sign (a group of neighbouring ink pieces) is scored against all
417 site drawings on shape, size (the site draws at one fixed scale) and,
for short signs, height on the line (97/98, 99/100 differ only there). The row
is split into exactly Length signs (OCR'd from the card footer) by dynamic
programming, maximizing the total score.

Usage: python read_card2.py SCREENSHOT [SCREENSHOT ...]
"""
import json, re, subprocess, sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

LIB = Path(__file__).resolve().parents[2] / "data" / "raw" / "cisi_seals" / "site_glyphs"
META = json.load(open(LIB / "meta.json"))
SIZE = 40


def norm(binary):
    ys, xs = np.nonzero(binary)
    b = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = b.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = b
    v = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = cv2.GaussianBlur(v, (0, 0), 1.2).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


@lru_cache(maxsize=1)
def library():
    ids, vecs, hs, ws, tops = [], [], [], [], []
    for s in range(0, 418):                        # 0 = the site's hatched "illegible sign"
        g = cv2.imread(str(LIB / f"{s}.png"), 0)
        m = META.get(str(s))
        ids.append(s); vecs.append(norm(g > 127)); hs.append(g.shape[0]); ws.append(g.shape[1])
        tops.append(m["top"] if m and s not in (327, 346) else np.nan)
    return np.array(ids), np.stack(vecs), np.array(hs, float), np.array(ws, float), np.array(tops, float)


@lru_cache(maxsize=None)
def lib_mask(sign):
    return cv2.imread(str(LIB / f"{sign}.png"), 0) > 127


def stroke_f1(piece, sign):
    """Overlap of the card piece and a library drawing resized to the piece's exact size.

    Fraction of each one's stroke pixels that land on (1 px tolerance) the other's
    strokes, combined as an F1 score. Small extra or missing strokes count fully,
    unlike in the blurred shape score."""
    ys, xs = np.nonzero(piece)
    a = piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    b = cv2.resize(lib_mask(sign).astype(np.uint8) * 255, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) > 100
    k = np.ones((3, 3), np.uint8)
    da, db = cv2.dilate(a, k) > 0, cv2.dilate(b.astype(np.uint8), k) > 0
    a = a > 0
    p_ = (a & db).sum() / max(1, a.sum())
    r_ = (b & da).sum() / max(1, b.sum())
    return 2 * p_ * r_ / max(1e-6, p_ + r_)


def find_card(gray):
    h, w = gray.shape
    band = gray[:, int(0.15 * w):int(0.85 * w)]
    dark = (band < 235).mean(axis=1)
    rows = [y for y in range(h) if dark[y] > 0.9]
    groups = []
    for y in rows:
        if groups and y - groups[-1][-1] <= 2:
            groups[-1].append(y)
        else:
            groups.append([y])
    ys = [g[0] for g in groups]
    s = w / 1536
    best = None
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            if 250 * s <= ys[j] - ys[i] <= 420 * s:
                best = (ys[i], ys[j])
    return best


def ocr_length(gray, top, bottom, left):
    """Number of signs from the card footer "Length:N" (None if it can't be read)."""
    fix = str.maketrans({"s": "5", "S": "5", "l": "1", "I": "1", "i": "1", "O": "0", "o": "0",
                         "z": "2", "Z": "2", "g": "9", "B": "8", ";": "3"})
    txt_all = []
    for dy0, dy1, scale, binarize in ((45, 5, 4, False), (50, 2, 5, True), (40, 8, 3, True)):
        foot = gray[bottom - dy0:bottom - dy1, left + 3:left + 560]
        foot = cv2.resize(foot, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if binarize:
            _, foot = cv2.threshold(foot, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "7"],
                             input=cv2.imencode(".png", foot)[1].tobytes(), capture_output=True).stdout.decode()
        txt_all.append(txt.strip())
        m = re.search(r"Leng\w*\s*[:;.,]?\s*([0-9sSlIiOozZgB;]{1,2})", txt)
        if m:
            d = m.group(1).translate(fix)
            if d.isdigit():
                v = int(d)
                if v > 14:                      # texts have at most 14 signs: "33" is "3" plus noise
                    v = int(d[0])
                if 1 <= v <= 14:
                    return v, txt.strip()
    return None, " | ".join(txt_all)


def card_sides(gray, top, bottom):
    """x of the centred card's left and right frame lines (columns dark along the card's height)."""
    h, w = gray.shape
    mid = gray[top + 20:bottom - 20]
    dark = (mid < 235).mean(axis=0)
    xs = [x for x in range(w) if dark[x] > 0.85]
    left = [x for x in xs if x < w * 0.3]
    right = [x for x in xs if x > w * 0.7]
    return (max(left) if left else int(137 * w / 1536)), (min(right) if right else int(1437 * w / 1536))


def _split(runs, L, cand):
    """Best split of the runs into exactly L consecutive groups: (total score, groups) or None."""
    m = len(runs)
    NEG = -1e9
    dp = [[NEG] * (m + 1) for _ in range(L + 1)]
    back = [[None] * (m + 1) for _ in range(L + 1)]
    dp[0][0] = 0.0
    for k in range(1, L + 1):
        for i in range(1, m + 1):
            for j in range(max(0, i - 12), i):
                if dp[k - 1][j] == NEG:
                    continue
                v = dp[k - 1][j] + cand(j, i)[0].max()
                if v > dp[k][i]:
                    dp[k][i], back[k][i] = v, j
    if dp[L][m] == NEG:
        return None
    groups, i = [], m
    for k in range(L, 0, -1):
        j = back[k][i]; groups.append((j, i)); i = j
    groups.reverse()
    return dp[L][m], groups


def read(path, scale_hint=None):
    img = cv2.imread(str(path), 0)
    H, W = img.shape
    s = 1.0                                        # glyphs are drawn at a fixed size whatever the window width
    card = find_card(img)
    if card is None:
        return dict(ok=False, why="no card")
    top, bottom = card
    cl, cr = card_sides(img, top, bottom)
    L, foot = ocr_length(img, top, bottom, cl)
    y0, y1 = top + 80, bottom - 55
    x0, x1 = cl + 190, cr - 7
    gray_row = img[y0:y1, x0:x1]
    row = gray_row < 190
    cols = row.any(axis=0)
    runs, x = [], 0
    while x < len(cols):
        if cols[x]:
            a = x
            while x < len(cols) and cols[x]:
                x += 1
            runs.append((a, x))
        x += 1
    # signs form one right-aligned group; drop anything left of a wide gap
    # (long field-symbol captions such as "COMPOSITE MOTIF (ANIMALS)" reach into this area)
    keep = len(runs)
    for i in range(len(runs) - 1, 0, -1):
        if runs[i][0] - runs[i - 1][1] > 90:
            break
        keep = i
    runs = runs[keep - 1:] if keep > 0 else runs
    if not runs:
        return dict(ok=False, why="no ink")
    ids, V, LH, LW, LT = library()
    # card draws at a fixed fraction of the library size; estimate it from tall pieces
    k_scale = 0.525 * s if scale_hint is None else scale_hint   # measured: card glyphs are 0.525x the library size
    em = 110 * k_scale

    cache = {}

    def cand(i, j):
        if (i, j) in cache:
            return cache[(i, j)]
        a, b = runs[i][0], runs[j - 1][1]
        piece = row[:, a:b]
        ys = np.nonzero(piece.any(axis=1))[0]
        hpx, wpx = ys.max() - ys.min() + 1, b - a
        shape = V @ norm(piece)
        size_pen = np.abs(np.log((hpx / k_scale) / LH)) + np.abs(np.log((wpx / k_scale) / LW))
        sc = shape - 0.6 * size_pen
        cache[(i, j)] = (sc, ys.min(), hpx)
        return cache[(i, j)]

    m = len(runs)
    guessed = False
    if L is None or L > m:
        guessed = True
        best = None
        for Ltry in range(1, min(m, 14) + 1):
            r = _split(runs, Ltry, cand)
            if r is not None:
                mean = r[0] / Ltry
                if best is None or mean > best[0]:
                    best = (mean, Ltry)
        if best is None:
            return dict(ok=False, why=f"length {L} vs {m} runs", foot=foot)
        L = best[1]
    res = _split(runs, L, cand)
    if res is None:
        return dict(ok=False, why="no split", foot=foot)
    groups = res[1]

    # line position: estimate the em-box top from confidently matched tall signs
    first = [cand(a, b) for a, b in groups]
    tops = []
    for sc, ytop, hpx in first:
        t = int(np.argmax(sc))
        if sc[t] > 0.85 and hpx > 0.45 * em and not np.isnan(LT[t]):
            tops.append(ytop - LT[t] * em)
    em_top = float(np.median(tops)) if tops else None

    out, scores, margins, runners = [], [], [], []
    for (ga, gb), (sc, ytop, hpx) in zip(groups, first):
        sc = sc.copy()
        piece = row[:, runs[ga][0]:runs[gb - 1][1]]
        for t in np.argsort(-sc)[:10]:                    # re-rank the 10 best by exact stroke overlap
            sc[t] += 2.0 * stroke_f1(piece, int(ids[t]))
        if em_top is not None:
            rel = (ytop - em_top) / em
            pos_pen = np.nan_to_num(np.abs(rel - LT), nan=0.0)
            sc = sc - 1.5 * pos_pen * (LH < 50)          # position matters only for short signs
        # broken vs continuous strokes: 87/105 and 86/101 differ only by a small gap
        # in the middle of the strokes, which the shape scores barely see
        top2 = {int(ids[t]) for t in np.argsort(-sc)[:2]}
        for whole, broken in ((87, 105), (86, 101)):
            if top2 == {whole, broken}:
                ys = np.nonzero(piece.any(axis=1))[0]
                seg = piece[ys.min():ys.max() + 1]
                mid = seg[int(0.3 * len(seg)):int(0.7 * len(seg))]
                gap = (~mid.any(axis=1)).sum() >= 2              # 2+ empty rows mid-stroke
                winner = broken if gap else whole
                sc[np.where(ids == winner)[0][0]] = sc.max() + 0.1
        order = np.argsort(-sc)
        g = gray_row[:, runs[ga][0]:runs[gb - 1][1]]
        dark = np.percentile(g[g < 190], 10) if (g < 190).any() else 255
        out.append(("*" if dark > 100 else "") + str(int(ids[order[0]]))); scores.append(round(float(sc[order[0]]), 3))
        margins.append(round(float(sc[order[0]] - sc[order[1]]), 3)); runners.append(int(ids[order[1]]))
    return dict(ok=True, length=L, length_guessed=guessed, signs_left_to_right=out, scores=scores, margins=margins, runners=runners)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(Path(p).name, read(p))
