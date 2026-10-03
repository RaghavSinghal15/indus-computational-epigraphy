"""Contact sheet: page with photo boxes + assigned labels. green=confident, red=flagged,
blue=photo with no caption (not exported)."""
import cv2, json, sys, numpy as np

def sheet(n, out=None):
    res = json.load(open(f'work/res_{n:03d}.json'))
    c = cv2.imread(f'pages_hi/p-{n:03d}.png')
    for x, y, w, h, name, conf in res['assigned']:
        col = (0, 170, 0) if conf else (0, 0, 255)
        cv2.rectangle(c, (x, y), (x + w, y + h), col, 2)
        txt = name.replace(' ', '_')
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y + 16
        cv2.rectangle(c, (x + 2, ty - th - 3), (x + 4 + tw, ty + 3), (255, 255, 255), -1)
        cv2.putText(c, txt, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    for x, y, w, h in res['unlinked_photos']:
        cv2.rectangle(c, (x, y), (x + w, y + h), (255, 120, 0), 2)
    for (x, y, w, h) in res['caps']:
        cv2.rectangle(c, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(c, f"p{n} {res['site']}-{res['rng'][0]}..{res['rng'][1]}", (10, 1140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if out: cv2.imwrite(out, c)
    return c

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'save':
        for n in map(int, sys.argv[2:]):
            sheet(n, f'photos/check_page_{n}.png')
    else:   # 'view out.png width pages...'
        out, W = sys.argv[2], int(sys.argv[3])
        ims = [cv2.resize(sheet(n), (W, int(W * 1152 / 810))) for n in map(int, sys.argv[4:])]
        cv2.imwrite(out, np.hstack(ims))
