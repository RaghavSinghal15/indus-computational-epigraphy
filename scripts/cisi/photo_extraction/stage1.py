"""Stage 1: per page, detect photos + caption lines, OCR captions, link captions to photos.
Writes work/raw/<page>.json"""
import sys, json, os
from concurrent.futures import ProcessPoolExecutor
from detect import load, flatten, photo_boxes, text_lines, side_ratio
from ocr import read_caption

def link(photos, caps):
    """For each photo, index of the caption directly below it (or None)."""
    out = []
    for (px, py, pw, ph) in photos:
        best = None
        for j, (cx, cy, cw, chh) in enumerate(caps):
            gap = cy - (py + ph)
            if gap < -4 or gap > 45:
                continue
            ovx = min(px + pw + 10, cx + cw) - max(px - 10, cx)
            if ovx <= 0:
                continue
            ccx = cx + cw / 2; pcx = px + pw / 2
            score = gap + 0.3 * abs(ccx - pcx)
            if best is None or score < best[0]:
                best = (score, j)
        out.append(None if best is None else best[1])
    return out

def page(n):
    im = load(n); f = flatten(im)
    caps = [list(map(int, b)) for b in text_lines(f)]
    photos = [list(map(int, b)) for b in photo_boxes(f, caps)]
    # caption lines fully inside a photo box are inscriptions/labels, drop them
    caps = [c for c in caps if not any(p[0] <= c[0] and p[1] <= c[1] and c[0] + c[2] <= p[0] + p[2]
                                       and c[1] + c[3] <= p[1] + p[3] for p in photos)]
    rc = [read_caption(f, c) for c in caps]
    txt = [r[0] for r in rc]; side_glyph = [r[1] for r in rc]
    lk = link(photos, caps)
    res = dict(page=n, photos=photos, caps=caps, cap_text=txt, link=lk,
               side_ratio=[side_ratio(f, c) for c in caps], side_glyph=side_glyph)
    json.dump(res, open(f'work/raw/{n:03d}.json', 'w'))
    return n, len(photos), len(caps)

if __name__ == '__main__':
    os.makedirs('work/raw', exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[2])
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(page, range(a, b + 1)):
            print(*r, flush=True)
