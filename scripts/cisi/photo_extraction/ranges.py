"""Page -> (site prefix, first, last object number) from OCR'd running headers,
repaired with the rule that consecutive pages of one site cover contiguous ranges."""
import re, json
SITES = {'MOHEN': 'M', 'HARAPPA': 'H', 'LOTHAL': 'L', 'KALIBANGAN': 'K', 'CHANHUJO': 'C', 'BANAWALI': 'B'}
FIX = str.maketrans({'I': '1', 'l': '1', ']': '1', '[': '1', 'S': '5', 'O': '0', 'o': '0', 'J': '1', 'i': '1', 'Y': ''})

def parse(t):
    for key, p in SITES.items():
        m = re.search(key + r'[A-Z\-\.]*\s*([0-9IlS\]\[OoJY]+)(?:\s*[-~—–]+\s*([0-9IlS\]\[OoJ]+))?', t)
        if m:
            a = m.group(1).translate(FIX); b = (m.group(2) or m.group(1)).translate(FIX)
            try:
                a, b = int(a), int(b)
                if b < a or b - a > 20:
                    b = None
                return p, a, b
            except ValueError:
                return p, None, None
        if key in t:
            return p, None, None
    return None

def build():
    raw = {}
    for l in open('headers.txt'):
        n, t = l.split(' ', 1)
        raw[int(n)] = parse(t)
    pages = {}
    order = sorted(raw)
    # site of each page: parsed site, else neighbour's
    for n in order:
        if raw[n] is None and 37 <= n <= 385:
            raw[n] = (raw[n - 1][0], None, None)
    for n in order:
        r = raw[n]
        if r is None or not (37 <= n <= 385):
            continue
        pages[n] = list(r)
    # contiguity repair
    ns = sorted(pages)
    for i, n in enumerate(ns):
        site, a, b = pages[n]
        prev = pages.get(n - 1)
        if prev and prev[0] == site and prev[2] is not None:
            exp = prev[2] + 1
            if a == prev[2] and b == a:
                continue            # same object continued over pages
            if a != exp:
                # accept header only if it is consistent; header with only an end number e.g. 'I-12'
                pages[n][1] = exp
                if b is None or b < exp:
                    pages[n][2] = None
        elif prev is None or prev[0] != site:
            if a is None or a > 3:
                pages[n][1] = 1
    # fill missing ends from next page start
    for n in reversed(ns):
        site, a, b = pages[n]
        nxt = pages.get(n + 1)
        if nxt and nxt[0] == site and nxt[1] is not None:
            if nxt[1] == a:
                pages[n][2] = a      # object continues on next page
            elif b is None or b != nxt[1] - 1:
                if b is None or b >= nxt[1] or b < a:
                    pages[n][2] = nxt[1] - 1
    return pages

if __name__ == '__main__':
    p = build()
    json.dump({k: v for k, v in p.items()}, open('page_ranges.json', 'w'))
    prev = None
    for n in sorted(p):
        s, a, b = p[n]
        flag = '' if (prev is None or prev[0] != s or prev[2] + 1 == a) else '  <-- gap'
        if a is None or b is None or b < a or b - a > 20: flag += ' ??'
        print(n, s, a, b, flag)
        prev = p[n]
