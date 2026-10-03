"""Photo and caption-box detection on CISI plate pages (100 dpi grayscale)."""
import cv2, numpy as np

HEADER_H = 48  # header text band at top of page

def load(n):
    return cv2.imread(f'pages/p-{n:03d}.pgm', 0)

def flatten(im):
    """Divide out slow paper shading (gutter shadow); returns uint8 with paper ~250.
    Paper level = 92nd percentile of each column within a 400-row sliding window."""
    f = im.astype(np.float32)
    h, w = f.shape
    centers = list(range(0, h, 25))
    bg = np.zeros((len(centers), w), np.float32)
    for i, c in enumerate(centers):
        y0, y1 = max(0, c - 200), min(h, c + 200)
        bg[i] = np.percentile(f[y0:y1], 92, axis=0)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    f = f / np.maximum(bg, 80) * 250
    return np.clip(f, 0, 255).astype(np.uint8)

def text_lines(f):
    """Caption-like text lines on white paper. f = flattened page. Returns [x,y,w,h]."""
    h, w = f.shape
    m = (f < 175).astype(np.uint8)
    m[:HEADER_H] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if 3 <= bh <= 13 and bw <= 45 and a >= 3:
            keep[i] = True
    chars = keep[lab].astype(np.uint8)
    d = cv2.dilate(chars, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d)
    lines = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if bh < 5 or bh > 17:
            continue
        # undo dilation margins
        x, y, bw, bh = x + 4, y + 1, max(1, bw - 8), bh - 2
        # caption must sit on paper, not inside a photo: either the padded box is mostly
        # white, or the pixels between the letter strokes are (caption squeezed by photos)
        crop2 = f[max(0, y - 2):y + bh + 2, max(0, x - 2):x + bw + 2]
        crop = f[y:y + bh, x:x + bw]
        bgpx = crop[crop >= 175]
        ok1 = (crop2 > 215).mean() >= 0.45
        ok2 = bgpx.size > 0 and (bgpx > 205).mean() >= 0.6
        if not (ok1 or ok2):
            continue
        lines.append([x, y, bw, bh])
    out = []
    for x, y, bw, bh in merge_lines(lines):
        if bw < 12:
            continue
        # faint trailing side letter (light grey 'a') may be missed by the char threshold:
        # extend right over faint ink within 14 px
        reg = f[y:y + bh, x + bw:min(w, x + bw + 14)]
        cols = np.where((reg < 210).any(axis=0))[0] if reg.size else []
        if len(cols) and cols[0] <= 8:
            bw += int(cols[-1]) + 1
        out.append([x, y, bw, bh])
    return out

def words(f, box):
    """Split a caption line into words; returns list of words, each a list of char boxes."""
    x, y, w, h = box
    c = f[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
    thr = (int(c.min()) + 250) // 2
    m = (c < thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = sorted([list(st[i][:4]) for i in range(1, n) if st[i][4] >= 2], key=lambda b: b[0])
    # merge components overlapping in x (e.g. dot of i)
    chars = []
    for b in comps:
        if chars and b[0] <= chars[-1][0] + chars[-1][2] - 1:
            p = chars[-1]; x0 = min(p[0], b[0]); y0 = min(p[1], b[1])
            x1 = max(p[0] + p[2], b[0] + b[2]); y1 = max(p[1] + p[3], b[1] + b[3])
            chars[-1] = [x0, y0, x1 - x0, y1 - y0]
        else:
            chars.append(b)
    ws = []
    for b in chars:
        if ws and b[0] - (ws[-1][-1][0] + ws[-1][-1][2]) < 3:
            ws[-1].append(b)
        else:
            ws.append([b])
    return ws

def side_ratio(f, box):
    r = side_info(f, box)
    return None if r is None else r[0]

def side_info(f, box):
    """Height of the side letter (glyph after the space) relative to the digit height,
    both measured up from the shared baseline. ~1.0 for capitals, ~0.6-0.7 for x-height
    letters (a, c, e). None if no space found."""
    x, y, w, h = box
    c = f[max(0, y - 3):y + h + 3, max(0, x - 1):x + w + 1].astype(int)
    ink = c < 218
    cols = ink.any(axis=0)
    runs = []                                  # [start, end) of inked column runs
    i = 0
    while i < len(cols):
        if cols[i]:
            j = i
            while j < len(cols) and cols[j]: j += 1
            runs.append([i, j]); i = j
        else:
            i += 1
    if len(runs) < 2:
        return None
    gaps = [(runs[k + 1][0] - runs[k][1], k) for k in range(len(runs) - 1)]
    big = max(g for g, _ in gaps)
    if big < 2:
        return None
    k = next((k for g, k in gaps if g >= max(2, 0.75 * big) and runs[k][1] > 0.4 * len(cols)), None)
    if k is None:
        return None
    if runs[k + 1][1] - runs[k + 1][0] < 4:      # too narrow for a letter (digit 1 / fragment)
        return None
    left = ink[:, runs[0][0] + 6:runs[k][1]]   # skip the 'M'
    side = ink[:, runs[k + 1][0]:runs[k + 1][1]]
    if not left.any() or not side.any():
        return None
    rl = np.where(left.any(axis=1))[0]; rs = np.where(side.any(axis=1))[0]
    base = rl.max()
    ratio = round((base - rs.min() + 1) / max(1, base - rl.min() + 1), 2)
    x0 = max(0, x - 1) + runs[k + 1][0]; x1 = max(0, x - 1) + runs[k + 1][1]
    return ratio, (int(x0), int(x1))

def photo_boxes(f, lines, thr=222):
    """Photo bounding boxes [x,y,w,h]; caption lines are erased from the mask first."""
    h, w = f.shape
    m = (cv2.GaussianBlur(f, (3, 3), 0) < thr).astype(np.uint8)
    m[:HEADER_H] = 0
    for x, y, bw, bh in lines:
        m[max(0, y - 1):y + bh + 1, max(0, x - 1):x + bw + 1] = 0
    # kill thin structures (text strokes, page rules) before filling photos
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    opened = m.copy()
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    comps = []
    for i in range(1, n):
        comps += unbridge(opened, st[i])
    boxes = []
    for x, y, bw, bh, a in comps:
        if a < 150 or bw < 12 or bh < 10:
            continue
        if bw < 30 and bh > 5 * bw and bh > 120:   # long thin rules / shadows
            continue
        crop = f[y:y + bh, x:x + bw]
        fill = (crop < thr).mean()
        if bh <= 18:
            blank = (crop.min(axis=0) > 200).mean()
            if blank > 0.08 or fill < 0.55:  # leftover bold text
                continue
        boxes.append([x, y, bw, bh])
    # merge overlapping boxes (fragments of one photo)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]) + 2
                oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]) + 2
                small = min(a[2] * a[3], b[2] * b[3]); big = max(a[2] * a[3], b[2] * b[3])
                # merge fragments: touching/overlapping boxes, unless they are two big photos
                # whose boxes merely clip each other (irregular objects laid out close together)
                if ox > 0 and oy > 0 and (ox * oy >= 0.3 * small or small < 0.2 * big) \
                        and embedded(m, a if a[2] * a[3] <= b[2] * b[3] else b):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2]); y1 = max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]; changed = True; break
            if changed: break
    f2 = f.copy()                            # captions painted white for split/trim
    for x, y, bw, bh in lines:
        f2[max(0, y - 1):y + bh + 1, max(0, x - 1):x + bw + 1] = 255
    out = []
    for b in boxes:
        out += split_box(f2, b)
    return out

def embedded(m, b, r=12):
    """True if the ring of width r around box b holds photo pixels (piece of a bigger photo);
    False if it is surrounded by white paper (a separate small object)."""
    H, W = m.shape
    x, y, w, h = b[:4]
    X0, Y0, X1, Y1 = max(0, x - r), max(0, y - r), min(W, x + w + r), min(H, y + h + r)
    outer = m[Y0:Y1, X0:X1].astype(bool).copy()
    ix0, iy0 = max(0, x - 2) - X0, max(0, y - 2) - Y0
    outer[iy0:iy0 + h + 4, ix0:ix0 + w + 4] = False
    ring = (X1 - X0) * (Y1 - Y0) - min(h + 4, Y1 - Y0) * min(w + 4, X1 - X0)
    return outer.sum() > 0.1 * max(ring, 1)

def unbridge(opened, s):
    """The closing can bridge two abutting photos. If the opened mask inside this component
    holds >= 2 big pieces whose boxes sit side by side (no overlap), return them separately."""
    x, y, w, h, a = [int(v) for v in s]
    sub = opened[y:y + h, x:x + w]
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub)
    big = [list(map(int, st[i])) for i in range(1, n) if st[i][4] >= max(1500, 0.15 * a)]
    if len(big) < 2:
        return [[x, y, w, h, a]]
    for i in range(len(big)):
        for j in range(i + 1, len(big)):
            p, q = big[i], big[j]
            ox = min(p[0] + p[2], q[0] + q[2]) - max(p[0], q[0])
            oy = min(p[1] + p[3], q[1] + q[3]) - max(p[1], q[1])
            if ox > 3 and oy > 3:
                return [[x, y, w, h, a]]      # interlocking pieces: one photo
    out = []
    for p in big:
        out.append([x + p[0], y + p[1], p[2], p[3], p[4]])
    return out

def split_box(f, b, depth=0):
    """Split a box that contains two abutting photos separated by a thin white stripe
    (the 7x7 closing bridges gaps < 7 px). Returns a list of boxes trimmed to content."""
    x, y, w, h = b
    crop = f[y:y + h, x:x + w]
    if depth > 6 or w < 40 and h < 40:
        return [b]
    for axis in (0, 1):                      # 0: vertical stripes (split in x), 1: horizontal
        white = (crop > 225).mean(axis=axis)  # fraction of white pixels per column / row
        n = len(white)
        cand = np.where(white >= 0.97)[0]
        cand = cand[(cand > 15) & (cand < n - 15)]
        if len(cand) >= 2:
            # take the middle of the longest run of white stripes
            runs = np.split(cand, np.where(np.diff(cand) > 1)[0] + 1)
            run = max(runs, key=len)
            if len(run) < 2:
                continue
            c = int(run[len(run) // 2])
            if axis == 0:
                parts = [[x, y, c, h], [x + c, y, w - c, h]]
            else:
                parts = [[x, y, w, c], [x, y + c, w, h - c]]
            res = []
            for p in parts:
                p = trim(f, p)
                if p: res += split_box(f, p, depth + 1)
            return res
    return [b]

def trim(f, b, thr=222):
    """Shrink a box to its non-white content; None if empty."""
    x, y, w, h = b
    m = f[y:y + h, x:x + w] < thr
    rows = np.where(m.mean(axis=1) > 0.02)[0]; cols = np.where(m.mean(axis=0) > 0.02)[0]
    if len(rows) < 10 or len(cols) < 10:
        return None
    return [x + int(cols[0]), y + int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)]

def merge_lines(lines):
    """Join pieces of one caption split by a wide space (e.g. 'M-51' + 'A')."""
    lines = sorted(lines, key=lambda b: b[0])
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(lines):
            for j, b in enumerate(lines):
                if i == j or b[0] < a[0]:
                    continue
                ov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if ov < 0.5 * min(a[3], b[3]):
                    continue
                gap = b[0] - (a[0] + a[2])
                if gap <= 12 or (b[2] <= 22 and gap <= 28):
                    x0, y0 = a[0], min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2]); y1 = max(a[1] + a[3], b[1] + b[3])
                    lines[i] = [x0, y0, x1 - x0, y1 - y0]
                    del lines[j]; changed = True; break
            if changed: break
    return lines
