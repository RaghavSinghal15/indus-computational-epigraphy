"""Stage 2: turn raw OCR captions into labels using the page range, the valid label list
and layout; write crops + photos/index.csv rows for the given site."""
import json, re, csv, os, sys, glob
import cv2, numpy as np

VALID = {r['cisi_label'] for r in csv.DictReader(open('cisi_im77.csv'))}
RANGES = {int(k): v for k, v in json.load(open('page_ranges.json')).items()}
MAIN = {'M', 'H', 'L', 'K', 'C', 'B'}
HEADERS = {int(l.split(' ', 1)[0]): l.split(' ', 1)[1] for l in open('headers.txt')}
SMALL_SITES = {'ALAMG': 'Agr', 'AMRI': 'Amri', 'CHANDIG': 'Ch', 'DAIMA': 'Dmd', 'DESAL': 'Dlp', 'DHOLA': 'Dlv',
               'HULAS': 'Hls', 'JHUK': 'Jk', 'KHIRS': 'Krs', 'LOHUM': 'Lh', 'MASK': 'Msk', 'MEHI': 'Mehi',
               'PABU': 'Pbm', 'PRABH': 'Phs', 'RAKHI': 'Rgr', 'RANGP': 'Rgp', 'ROHIR': 'Rhr', 'ROJD': 'Rjd',
               'RUPA': 'Rpr', 'SHAHI': 'Sht', 'SURKO': 'Sktd', 'TARKH': 'Tkwd'}
for _p in range(386, 400):
    RANGES.setdefault(_p, ['X', None, None])
RANGES.setdefault(400, ['ALL', None, None])

# cheap OCR confusions (ocr char -> true char)
CONF = {('S', '5'), ('s', '5'), ('I', '1'), ('l', '1'), ('L', '1'), ('i', '1'), ('|', '1'), ('t', '1'),
        ('O', '0'), ('o', '0'), ('D', '0'), ('A', '4'), ('Z', '2'), ('z', '2'), ('B', '8'), ('G', '6'),
        ('b', '6'), ('T', '7'), ('g', '9'), ('q', '9'), ('?', '7'), ('E', '3'), ('a', '4'), ('h', '4'),
        ('M', '4'), ('H', 'M'), ('N', 'M'), ('W', 'M'), ('.', '-'), ('—', '-'), ('–', '-'), ('~', '-'),
        ('_', '-'), ('S', '3'), ('8', '3'), ('3', '8'), ('6', '8'), ('8', '6'), ('5', '6'), ('0', '9'), ('9', '0'),
        ('0', '6'), ('6', '0'), ('0', '8'), ('8', '0'), ('1', '7'), ('7', '1'), ('2', '7'), ('3', '5'), ('5', '3')}

def sub_cost(o, t):
    if o == t: return 0.0
    if (o, t) in CONF: return 0.35
    return 1.0

def align(ocr, target, nodigit_after=None):
    """Min cost of matching target against a prefix of ocr; returns (cost, consumed_len)."""
    n, m = len(ocr), len(target)
    INF = 1e9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0: continue
            best = INF
            if i > 0 and j > 0: best = D[i - 1][j - 1] + sub_cost(ocr[i - 1], target[j - 1])
            if i > 0: best = min(best, D[i - 1][j] + (0.4 if ocr[i - 1] in '.-,\'' else 1.0))  # extra ocr char
            if j > 0: best = min(best, D[i][j - 1] + (0.4 if target[j - 1] == '-' else 1.0))   # missing char
            if i > 0 and j > 1 and target[j - 2:j] == '11' and ocr[i - 1] in 'HUnuN':
                best = min(best, D[i - 1][j - 2] + 0.3)        # '11' read as one glyph 'H'
            if i > 1 and j > 0 and target[j - 1] == '1' and ocr[i - 2:i] in ('ll', 'ii', 'Il', 'lI', 'il', 'li', 'II'):
                best = min(best, D[i - 2][j - 1] + 0.5)        # '1' read as two strokes
            D[i][j] = best
    costs = [(D[i][m] + (0.8 if nodigit_after is not None and i < n and ocr[i].isdigit()
                         and i not in nodigit_after else 0), i)
             for i in range(max(0, min(n, m - 3)), min(n, m + 3) + 1)]
    return min(costs)

SIDE_RE = re.compile(r'^([A-Ha-h])(?![a-z])(?:\s*([-+])\s*([A-Ha-h])(?![a-z]))?\s*(bis|ter)?\s*(\(\s*\d\s*\)|\d(?:\s*-\s*\d)?(?![\d%]))?\s*(bis|ter)?')
SIDE_FIX = {'4': 'A', '8': 'B', '0': 'D', '6': 'G', 'R': 'B', 'n': 'a', 'u': 'a', 'o': 'a', 's': 'a',
            'e': 'c', 'P': 'B', 'k': 'b', 'I': 'b'}

def parse_side(rest):
    r = rest.strip().lstrip('.,-_~ ')
    if not r:
        return None, 0
    fixed = 0
    if not SIDE_RE.match(r) and len(r) > 1 and r[0] in "LIl|i1.,:;')" and SIDE_RE.match(r[1:].strip()):
        r = r[1:].strip(); fixed = 1
    if len(r) > 1 and r[0] in '0123456789' and re.match(r'^[A-Ha-h](?![a-z])', r[1:].strip()):
        r = r[1:].strip(); fixed = 1          # stray digit before the letter ('4D')
    elif r[0] in SIDE_FIX and not SIDE_RE.match(r):
        r = SIDE_FIX[r[0]] + r[1:]; fixed = 1
    r = re.sub(r'\b(tei|tet|ler|tcr|te)\b', 'ter', r)
    r = re.sub(r'\b(bls|bes|brs|bjs|his|Dis|bis\.)\b', 'bis', r)
    r = r.replace('bls', 'bis').replace('his', 'bis').replace('Dis', 'bis').replace('bjs', 'bis')
    m = SIDE_RE.match(r)
    if not m:
        if re.match(r'^(bis|ter)\b', r):
            return r[:3], 'nolet'
        return None, 0
    s = m.group(1)
    if m.group(3): s += m.group(2) + m.group(3)
    if m.group(4): s += '_' + m.group(4)
    if m.group(5): s += '_' + re.sub(r'[^\d-]', '', m.group(5))
    if m.group(6) and not m.group(4): s += '_' + m.group(6)
    return s, (fixed if not fixed else rest.strip()[0])

def interpret(text, site, candidates):
    """Score every candidate number against an OCR string.
    Returns dict(best num, side, cost, margin, costs{num: cost})."""
    if not text:
        return None
    flat = text.replace(' ', '')
    # flat positions that were followed by a space in the OCR text: a digit after a space is
    # a side letter misread ('M-22 4'), not more of the number
    spaced = set(); cnt = 0
    for ch in text:
        if ch == ' ': spaced.add(cnt)
        else: cnt += 1
    scored = []
    for num in candidates:
        # stopping right before another digit means the number was cut short ('M-4' in 'M-435')
        c, k = align(flat, num, spaced)
        scored.append((round(c, 3), -k, num))
    scored.sort()
    c, negk, num = scored[0]
    k = -negk
    margin = (scored[1][0] - c) if len(scored) > 1 else 9
    cnt = 0; pos = 0                      # map consumed non-space chars back into text
    while pos < len(text) and cnt < k:
        if text[pos] != ' ': cnt += 1
        pos += 1
    side, fixed = parse_side(text[pos:])
    return dict(num=num, side=side, cost=c + 0.3 * bool(fixed), margin=margin,
                fixchar=fixed if fixed else None,
                costs={n_: c_ for c_, _, n_ in scored})

def faint_caption(f, photo, others, caps, pm=None):
    """Search a band at the bottom of / below a photo for a light-grey caption line."""
    from detect import merge_lines
    from ocr import read_caption
    H, W = f.shape
    px, py, pw, ph = photo
    x0, x1 = max(0, px - 20), min(W, px + pw + 20)
    y0, y1 = max(0, py + int(0.6 * ph)), min(H, py + ph + 60)
    band = f[y0:y1, x0:x1].copy()
    if pm is not None:                         # hide photo pixels (not whole boxes: irregular
        band[pm[y0:y1, x0:x1] > 0] = 255         # objects leave white corners holding captions)
    else:
        for qx, qy, qw, qh in others:
            band[max(0, qy - y0 - 2):max(0, qy + qh - y0 + 2), max(0, qx - x0 - 2):max(0, qx + qw - x0 + 2)] = 255
    for qx, qy, qw, qh in caps:                # hide captions already used
        band[max(0, qy - y0 - 2):max(0, qy + qh - y0 + 2), max(0, qx - x0 - 2):max(0, qx + qw - x0 + 2)] = 255
    m = (band < 205).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        if 3 <= st[k][3] <= 13 and st[k][2] <= 45: keep[k] = True
    chars = keep[lab].astype(np.uint8)
    d = cv2.dilate(chars, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d)
    lines = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if not (5 <= h <= 17 and w >= 20): continue
        x, y, w, h = int(x) + 4 + x0, int(y) + 1 + y0, int(w) - 8, int(h) - 2
        x -= 2; y -= 1; w += 4; h += 2               # faint glyph edges
        crop = f[y:y + h, x:x + w]
        bg = crop[crop >= 200]
        if bg.size == 0 or (bg > 212).mean() < 0.5: continue
        lines.append([x, y, w, h])
    for box in sorted(merge_lines(lines), key=lambda b: abs(b[1] - (py + ph))):
        ts, _ = read_caption(f, box)
        if any(re.search(r'[A-Za-z]{1,4}\s*[-.]?\s*\d', t) for t in ts if t):
            return box, ts
    return None

def rows_of(boxes):
    """Group boxes into rows; returns row index per box."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1] + boxes[i][3] / 2)
    rows = [-1] * len(boxes); r = -1; last = None
    for i in order:
        cy = boxes[i][1] + boxes[i][3] / 2
        if last is None or cy - last > 25:
            r += 1
        rows[i] = r; last = cy
    return rows

def process_page(n, site_prefix):
    raw = json.load(open(f'work/raw/{n:03d}.json'))
    site, a, b = RANGES[n]
    from detect import flatten, load, photo_boxes
    fpage = flatten(load(n))
    caps, texts = list(raw['caps']), list(raw['cap_text'])
    sglyph = list(raw.get('side_glyph') or [''] * len(caps))
    # a line holding two captions ('M-330 D M-330 F'): split it at the widest inner gap
    for j in range(len(caps)):
        t = texts[j][0]
        if len(re.findall(r'[A-Z]\s*-\s*\d', t)) >= 2:
            x, y, w, h = caps[j]
            col = (fpage[y:y + h, x:x + w] < 200).any(axis=0)
            mid = range(w // 3, 2 * w // 3)
            gaps = [k for k in mid if not col[k]]
            if gaps:
                c = max(gaps, key=lambda k: min(abs(k - w // 2), 99) * -1)
                parts = re.split(r'(?=[A-Z]\s*-\s*\d)', t)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) == 2:
                    caps[j] = [x, y, c, h]; texts[j] = [parts[0]]
                    caps.append([x + c, y, w - c, h]); texts.append([parts[1]]); sglyph.append('')
                    sglyph[j] = ''
    photos = [list(map(int, b)) for b in photo_boxes(fpage, caps)]
    # caption lines fully inside a photo box are inscriptions/labels, not captions
    # a caption line is 'inside' a photo (an inscription / museum number) only if it lies in a
    # photo box AND is surrounded by photo pixels; captions in the white corner of an
    # irregular object's box are not inside
    from detect import embedded
    pm = (cv2.GaussianBlur(fpage, (3, 3), 0) < 222).astype(np.uint8)
    pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    def is_inside(c):
        inbox = any(p[0] <= c[0] and p[1] <= c[1] and c[0] + c[2] <= p[0] + p[2] and c[1] + c[3] <= p[1] + p[3]
                    for p in photos)
        return inbox and embedded(pm, c, r=6)
    inside = [is_inside(c) for c in caps]
    if site == 'X':          # pages of small sites: labels of the sites named in the header
        hdr = HEADERS.get(n, '').upper()
        pref = {p for key, p in SMALL_SITES.items() if key in hdr}
        cands = sorted(l for l in VALID if l.split('-')[0] in pref) or \
            sorted(l for l in VALID if l.split('-')[0] not in MAIN)
    elif site == 'ALL':      # addenda page: anything
        cands = sorted(VALID)
    else:
        cands = [f'{site}-{k}' for k in range(a, b + 1)]
    # drop caption lines that are not captions (stamps, stray marks)
    def captionish(ts):
        if any(re.search(r'\d', t) or re.match(r'^\W*[A-Za-z]{1,4}\s*[-.]', t) for t in ts if t):
            return True
        # garbled but close to a label of this page (e.g. 'CMA' for 'C-34 A')
        return any(min(align(t.replace(' ', ''), k)[0] for k in cands) <= 2.0 for t in ts if t and len(t) >= 3)
    okcap = [captionish(texts[j]) and not inside[j] for j in range(len(caps))]
    link = [None] * len(photos)
    # pass 1: each photo -> nearest caption just below it, overlapping in x
    for i, (px, py, pw, ph) in enumerate(photos):
        best = None
        for j, (cx, cy, cw, chh) in enumerate(caps):
            if not okcap[j]: continue
            gap = cy - (py + ph)
            if gap < -4 or gap > 45: continue
            if min(px + pw + 10, cx + cw) - max(px - 10, cx) <= 0: continue
            score = gap + 0.3 * abs(cx + cw / 2 - px - pw / 2)
            if best is None or score < best[0]: best = (score, j)
        link[i] = None if best is None else best[1]
    # pass 2: captions still without a photo -> nearest free photo above (gap up to 120 px,
    # nothing in between) or beside the photo's lower corner
    used = {j for j in link if j is not None}
    for j, (cx, cy, cw, ch) in enumerate(caps):
        if j in used or not okcap[j]: continue
        best = None
        for i, (px, py, pw, ph) in enumerate(photos):
            if link[i] is not None: continue
            ovx = min(px + pw + 15, cx + cw) - max(px - 15, cx)
            gap = cy - (py + ph)
            if ovx > 0 and -25 <= gap <= 120:
                between = any(k != i and min(qx + qw, cx + cw) > max(qx, cx) and qy > py + ph - 4 and qy + qh < cy + 4
                              for k, (qx, qy, qw, qh) in enumerate(photos))
                if not between:
                    d = gap
                else:
                    continue
            elif py + ph - 60 <= cy <= py + ph + 30 and (-15 <= px - (cx + cw) <= 40 or -15 <= cx - (px + pw) <= 40):
                d = 50 + min(abs(px - cx - cw), abs(cx - px - pw))     # caption beside the photo
            else:
                continue
            if best is None or d < best[0]: best = (d, i)
        if best:
            link[best[1]] = j
    # pass 2b: silhouetted objects on white can have their caption inside their bounding box
    used = {j for j in link if j is not None}
    for j, (cx, cy, cw, ch) in enumerate(caps):
        if j in used or not inside[j] or not captionish(texts[j]): continue
        if not any(re.search(r'[A-Za-z]{1,4}\s*[-.]?\s*\d', t) for t in texts[j] if t): continue
        for i, (px, py, pw, ph) in enumerate(photos):
            if link[i] is None and px <= cx and cx + cw <= px + pw and py + 0.7 * ph <= cy <= py + ph:
                link[i] = j; used.add(j); break
    # pass 2c: still uncaptioned photo -> look for a faint caption just below / at its bottom
    for i, (px, py, pw, ph) in enumerate(photos):
        if link[i] is not None: continue
        fc = faint_caption(fpage, photos[i], [p for k, p in enumerate(photos) if k != i],
                           [caps[j] for j in set(link) if j is not None], pm)
        if fc is not None:
            box, ts = fc
            caps.append(box); texts.append(ts); sglyph.append(''); okcap.append(True); inside.append(False)
            link[i] = len(caps) - 1
    # pass 3: an uncaptioned piece touching a captioned photo (object split by unbridge)
    absorbed = set()
    for i, (px, py, pw, ph) in enumerate(photos):
        if link[i] is not None: continue
        for k, (qx, qy, qw, qh) in enumerate(photos):
            if k == i or link[k] is None or k in absorbed: continue
            gx = max(qx - (px + pw), px - (qx + qw)); gy = max(qy - (py + ph), py - (qy + qh))
            if max(gx, gy) <= 8 and min(gx, gy) < 0 and pw * ph < 0.5 * qw * qh:
                link[i] = link[k]; absorbed.add(i); break
    # interpret each linked caption
    from detect import side_ratio
    ratios = [side_ratio(fpage, c) for c in caps]
    items = []
    for j in sorted(set(x for x in link if x is not None)):
        tv = [t for t in texts[j] if t]
        cand = [c for c in (interpret(t, site, cands) for t in tv) if c]
        raw_txt = tv[0] if tv else ''
        sg = sglyph[j]
        if not any(re.search(r'\d\s*[A-Ha-h]', t) for t in tv):
            # side letter unread: OCR again on a box widened to the right (cut-off line)
            from ocr import read_caption
            x_, y_, w_, h_ = caps[j]
            wide, _ = read_caption(fpage, [x_, y_ - 1, w_ + 25, h_ + 2])
            tv = tv + [t for t in wide if t]
            cand = [c for c in (interpret(t, site, cands) for t in tv) if c]
        if not cand:
            items.append(dict(cap=j, num=None, side=None, conf=0, cost=9, costs={}, raw=raw_txt)); continue
        best = min(cand, key=lambda c: (c['cost'], c['side'] is None))
        side_note = ''
        if best['side'] is None:
            for c in cand:
                if c['side'] and c['num'] == best['num']:
                    best = dict(best, side=c['side'], fixchar=c.get('fixchar')); break
        if best['side'] is None and sg and len(sg) == 1:
            best = dict(best, side=sg, fixchar='glyph'); side_note = 'side from single-glyph OCR'
        if best['side'] is None and ratios[j] is not None and ratios[j] < 0.8:
            best = dict(best, side='a', fixchar='glyph'); side_note = 'side a inferred from x-height glyph'
        # vote on side among variants that agree on the number
        votes = [c['side'] for c in cand if c['side'] and c['num'] == best['num']]
        if votes and best.get('fixchar') is not None:
            top = max(set(votes), key=votes.count)
            if votes.count(top) >= 2: best = dict(best, side=top, fixchar=None)
        if best['side']:
            for c in cand:
                if c['side'] and c['num'] == best['num'] and c['side'][0] == best['side'][0] \
                        and len(c['side']) > len(best['side']) and c['side'].startswith(best['side']):
                    best = dict(best, side=c['side'])
        costs = {}
        for c in cand:
            for k_, v in c['costs'].items(): costs[k_] = min(costs.get(k_, 9), v)
        good = [c['num'] for c in cand if c['cost'] <= 1.0]
        agree = len(good) >= 2 and good.count(best['num']) >= 2 and len(set(good)) == 1
        conf = int(best['cost'] <= 0.01 or (best['cost'] <= 0.8 and best['margin'] >= 0.5) or agree)
        it = dict(cap=j, num=best['num'] if best['cost'] <= 2.5 else None, side=best['side'],
                  conf=conf, cost=round(best['cost'], 2), costs=costs, raw=raw_txt, note=side_note)
        if side_note: it['conf'] = 0
        it['perfect'] = sum(1 for c in cand if c['cost'] == 0 and c['num'] == best['num'])
        it['ratio'] = ratios[j]
        # upper/lower case of x-height letters: trust explicit, consistent OCR letters;
        # otherwise ('4', unread, conflicting) use glyph height (ratio ~1 caps, ~0.65 lower)
        r = ratios[j]
        sd = it['side']
        fx = best.get('fixchar')
        if sd and fx is not None and fx not in ('4', 'glyph') and sd[0] in 'BDG':
            # side letter guessed from a digit-like glyph ('8'->B, '0'->D, '6'->G): check height
            if r is not None and r < 0.8:
                it['side'] = 'a' + sd[1:]; sd = it['side']; it['note'] += f' {fx}->a by glyph height {r}'
            it['conf'] = 0
        if sd and sd[0] == 'D' and not any(re.search(r'[\d\s]D', t) for t in tv):
            if r is not None and r < 0.83:
                it['side'] = 'a' + sd[1:]; sd = it['side']; it['note'] += f' 0->a by glyph height {r}'
            elif r is None:          # '0' could be D or a: decide later from the page layout
                it['side'] = None; sd = None; it['conf'] = 0; it['note'] += ' side 0 ambiguous'
        if sd and sd[0] in 'aAcCeE':
            explicit = {c['side'][0] for c in cand if c['side'] and c['num'] == best['num']
                        and c.get('fixchar') is None}
            if len(explicit) == 1 and sd[0] in explicit:
                pass
            elif r is not None:
                want = sd[0].lower() if r < 0.83 else sd[0].upper()
                if want != sd[0]:
                    it['side'] = want + sd[1:]
                it['note'] += f' case from glyph height {r}'
                if 0.76 <= r <= 0.88:
                    it['conf'] = 0
            else:
                it['conf'] = 0; it['note'] += ' letter case uncertain'
        if it['side'] is None:
            it['conf'] = 0
        elif it['side'][:3] in ('bis', 'ter'):
            it['conf'] = 0; it['note'] += ' no side letter read'
        items.append(it)
    items = [it for it in items if not (it['num'] is None and not re.search(r'[\d-]', it['raw']))]
    cbox = {it['cap']: caps[it['cap']] for it in items}
    rows = dict(zip([it['cap'] for it in items], rows_of([cbox[it['cap']] for it in items])))
    # (1) caption side unread: same-row neighbour with same number gives the A/a pairing
    for it in items:
        if it['side'] is None and it['num'] is not None:
            mates = [o for o in items if o is not it and o['num'] == it['num'] and rows[o['cap']] == rows[it['cap']]]
            if len(mates) == 1 and mates[0]['side'] in ('A', 'a'):
                left = cbox[it['cap']][0] < cbox[mates[0]['cap']][0]
                it['side'] = 'A' if left else 'a'; it['conf'] = 0; it['note'] += ' side from layout'
    # (1b) still no side: the number's only other photo is 'A' -> this one is the impression
    for it in items:
        if it['side'] is None and it['num'] is not None:
            others = [o['side'] for o in items if o is not it and o['num'] == it['num']]
            if others and 'A' in others and 'a' not in others:
                it['side'] = 'a'; it['conf'] = 0; it['note'] += ' side a by elimination'
    # (2) A vs a collisions: if two captions claim (num, A) and (num, a) is free, the
    #     less certain one flips case (and vice versa)
    taken = {}
    for it in sorted(items, key=lambda o: (-o['conf'], o['cost'])):
        if it['num'] is None or it['side'] is None: continue
        key = (it['num'], it['side'])
        r_ = it.get('ratio')
        geom_ok = r_ is None or (r_ < 0.83) == (it['side'][0] == 'A')   # glyph height agrees with flip
        if key in taken and it['side'][0] in 'Aa' and it.get('perfect', 0) < 2 and geom_ok:
            other = it['side'].swapcase()[0] + it['side'][1:]
            if (it['num'], other) not in taken:
                it['side'] = other; it['conf'] = 0; it['note'] += ' A/a flipped (slot taken)'
                key = (it['num'], other)
        taken.setdefault(key, it)
    # (3) page-level matching: within each side label, captions get distinct numbers from
    #     the page range, minimising total OCR alignment cost (Hungarian algorithm)
    from scipy.optimize import linear_sum_assignment
    groups = {}
    for it in items:
        if it['side'] is not None and it['costs']:
            groups.setdefault(it['side'], []).append(it)
    nums = [k for k in cands if k in VALID] or cands
    for side_, grp in groups.items():
        C = np.full((len(grp), len(nums) + len(grp)), 3.0)     # extra columns = 'leave as is'
        for r_, it in enumerate(grp):
            for c_, k in enumerate(nums):
                C[r_, c_] = it['costs'].get(k, 9)
            if it.get('perfect', 0) >= 2:      # read perfectly twice: keep even if duplicated
                C[r_, len(nums):] = 0.9
        rr, cc = linear_sum_assignment(C)
        for r_, c_ in zip(rr, cc):
            it = grp[r_]
            if c_ < len(nums):
                k = nums[c_]
                if k != it['num']:
                    it['note'] += f" matched {it['num']}->{k}"
                    it['num'] = k; it['conf'] = 0
            else:
                it['contested'] = True
    # (4) a single unreadable caption takes the single number not seen on the page
    present = {it['num'] for it in items if it['num'] is not None}
    missing = [k for k in cands if k not in present and k in VALID]
    unres = [it for it in items if it['num'] is None]
    if not unres:
        unres = [it for it in items if it['cost'] >= 1.5 and it['side'] is None]   # badly read
    if len(missing) == 1 and len(unres) == 1:
        unres[0]['num'] = missing[0]; unres[0]['conf'] = 0; unres[0]['note'] += ' only missing number on page'
    for it in items:
        if it.get('contested'):
            it['conf'] = 0; it['note'] += ' duplicate label on page'
        it['note'] = it['note'].strip()
    used = {j for j in link if j is not None}
    orphan = [(caps[j], texts[j][0]) for j in range(len(caps)) if okcap[j] and j not in used]
    return dict(page=n, site=site, rng=(a, b), cands=cands, absorbed=sorted(absorbed), photos=photos, caps=caps, link=link, items=items,
                orphan_caps=orphan)

def crop_and_index(res, out_rows, pad=3):
    n = res['page']; site = res['site']
    im = cv2.imread(f'pages_hi/p-{n:03d}.png') if os.path.exists(f'pages_hi/p-{n:03d}.png') else None
    if im is None:
        im = cv2.imread(f'pages/p-{n:03d}.pgm', 0)
    H, W = im.shape[:2]
    by_cap = {}
    for i, j in enumerate(res['link']):
        if j is not None: by_cap.setdefault(j, []).append(i)
    used_names = {}
    res['assigned'] = []
    for it in res['items']:
        pis = by_cap.get(it['cap'], [])
        if not pis: continue
        if any(i in res.get('absorbed', ()) for i in pis):
            it['note'] = (it.get('note', '') + ' includes touching uncaptioned piece').strip()
        boxes = [res['photos'][i] for i in pis]
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        label = it['num'] if it['num'] is not None else f'{site}-UNK'
        folder = label.split('-')[0]
        side = it['side'] or 'X'
        conf = it['conf']
        if it['num'] is not None and label not in VALID:
            conf = 0; it['note'] = (it.get('note', '') + ' label not in list').strip()
        name = f'{label}_{side}'
        if name in GLOBAL_NAMES or os.path.exists(f'photos/{folder}/{name}.png'):
            k = 2
            while f'{name}_dup{k}' in GLOBAL_NAMES or os.path.exists(f'photos/{folder}/{name}_dup{k}.png'): k += 1
            name = f'{name}_dup{k}'; conf = 0
            it['note'] = (it.get('note', '') + ' duplicate name').strip()
        used_names[name] = 1; GLOBAL_NAMES.add(name)
        if it['num'] is None:
            name = f'{site}-UNK_p{n}_{it["cap"]}'
        crop = im[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
        os.makedirs(f'photos/{folder}', exist_ok=True)
        cv2.imwrite(f'photos/{folder}/{name}.png', crop)
        out_rows.append(dict(label=label, side=side, page=n, x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                             caption_raw=it['raw'], confident=conf, file=f'{folder}/{name}.png',
                             note=it.get('note', '') + (' group of %d photos' % len(pis) if len(pis) > 1 else '')))
        res['assigned'].append((x0, y0, x1 - x0, y1 - y0, name.replace('_', ' ', 1), conf))
    res['unlinked_photos'] = [res['photos'][i] for i, j in enumerate(res['link']) if j is None]

GLOBAL_NAMES = set()
