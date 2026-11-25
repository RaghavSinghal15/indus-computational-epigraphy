"""Animated 3-minute project video (final submission), dark midsem style with sign-frieze borders.

Each scene is a function f(t) -> PIL image (t in seconds from the scene start). Frames are
piped to ffmpeg (one file per scene) in data/processed/video/clips/; join them with
ffmpeg's concat demuxer. Narration: report/video_script.md.
    python scripts/make_video.py 1 2 3 4 5 6 7 8      # all scenes (about 15 minutes on a CPU)
    python scripts/make_video.py still 4@20          # one frame of scene 4 at t = 20 s
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from indus.composite import build_bank                      # noqa: E402
from indus.data import load_sign_boxes                      # noqa: E402
from indus.glyphs import glyph                              # noqa: E402
from indus.real_crops import sign_crops                     # noqa: E402
from indus.segment import load_image, trim_caption          # noqa: E402
from indus.wells import load_v2_seals                       # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "data" / "processed" / "video"
CLIPS = HERE / "clips"
CLIPS.mkdir(parents=True, exist_ok=True)
W, H, FPS = 1920, 1080, 30

BG = (22, 19, 15)
PANEL = (33, 29, 24)
LINE = (70, 62, 52)
INK = (238, 230, 216)
MUTED = (168, 159, 146)
CARN = (217, 119, 63)      # carnelian: forced cuts, emphasis
TEAL = (95, 176, 168)      # teal: hand boxes, good results
GOLD = (150, 130, 100)

F = "/usr/share/fonts/truetype/google-fonts/ofl/crimsontext/CrimsonText-SemiBold.ttf"
FR = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
FS = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
_fonts = {}


def font(path, size):
    if (path, size) not in _fonts:
        _fonts[(path, size)] = ImageFont.truetype(path, size)
    return _fonts[(path, size)]


def lerp(a, b, x):
    x = max(0.0, min(1.0, x))
    return tuple(int(round(p + (q - p) * x)) for p, q in zip(a, b))


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def ramp(t, start, dur=0.6):
    return ease((t - start) / dur)


def text(d, xy, s, size=40, path=FR, color=INK, a=1.0, anchor="la"):
    if a <= 0:
        return
    d.text(xy, s, font=font(path, size), fill=lerp(BG, color, a), anchor=anchor)


def para(d, xy, s, size, width, path=FR, color=INK, a=1.0, gap=1.35):
    """Word-wrapped text; returns the y below it."""
    x, y = xy
    words, line = s.split(), ""
    f = font(path, size)
    for w in words:
        trial = (line + " " + w).strip()
        if f.getlength(trial) > width and line:
            text(d, (x, y), line, size, path, color, a)
            y += int(size * gap)
            line = w
        else:
            line = trial
    if line:
        text(d, (x, y), line, size, path, color, a)
        y += int(size * gap)
    return y


def paste(canvas, img, xy, a=1.0):
    if a <= 0:
        return
    img = img.convert("RGB")
    if a >= 1:
        canvas.paste(img, xy)
    else:
        region = canvas.crop((xy[0], xy[1], xy[0] + img.width, xy[1] + img.height))
        canvas.paste(Image.blend(region, img, a), xy)


# ------------------------------------------------------------------ frame: frieze borders
def glyph_rgb(code, size, color=INK, bg=BG):
    g = glyph(code, size).astype(np.float32) / 255.0      # 1 = white paper, 0 = ink
    rgb = np.empty((size, size, 3), np.uint8)
    for c in range(3):
        rgb[..., c] = (g * bg[c] + (1 - g) * color[c]).astype(np.uint8)
    return Image.fromarray(rgb)


FRIEZE_CODES = [740, 817, 2, 820, 861, 390, 342, 405, 32, 60, 267, 99, 220, 33, 590, 176, 240, 400, 900, 520,
                235, 741, 125, 162, 700, 840, 1, 3, 12, 14, 15, 31, 48, 59, 87, 403]


def make_frieze():
    strip = Image.new("RGB", (W, 44), BG)
    x, i = 30, 0
    while x < W - 40:
        strip.paste(glyph_rgb(FRIEZE_CODES[i % len(FRIEZE_CODES)], 34, GOLD), (x, 5))
        x += 52
        i += 1
    return strip


FRIEZE = None


def base():
    img = Image.new("RGB", (W, H), BG)
    img.paste(FRIEZE, (0, 18))
    img.paste(FRIEZE, (0, H - 62))
    d = ImageDraw.Draw(img)
    d.line([(30, 70), (W - 30, 70)], fill=LINE, width=2)
    d.line([(30, H - 72), (W - 30, H - 72)], fill=LINE, width=2)
    return img, d


def header(d, t, kicker, title):
    text(d, (120, 112), kicker.upper(), 26, FS, CARN, ramp(t, 0.0, 0.5))
    text(d, (120, 150), title, 62, F, INK, ramp(t, 0.15, 0.6))


# ------------------------------------------------------------------ data
SEALS = load_v2_seals().set_index("cisi")
BOXES = load_sign_boxes()


def seal_photo(cisi):
    s = SEALS.loc[cisi]
    return trim_caption(load_image(s.image))


def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def fit(img, h=None, w=None):
    if h:
        return img.resize((round(img.width * h / img.height), h), Image.LANCZOS)
    return img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)


def boxes_of(cisi):
    r = BOXES[BOXES.cisi == cisi].sort_values("pos_ltr")
    return [(int(a.x0), int(a.y0), int(a.x1), int(a.y1)) for a in r.itertuples()], list(r.wells)


# ------------------------------------------------------------------ scene 1: title
def s1_setup():
    img = seal_photo("M-58")
    boxes, signs = boxes_of("M-58")
    return dict(img=img, boxes=boxes, signs=signs)


def s1(t, c):
    im, d = base()
    text(d, (120, 150), "TEAM ARTIFICIALLY INTELLIGENT  ·  CS F407  ·  BITS PILANI GOA", 26, FS, CARN, ramp(t, 0.2))
    y = para(d, (120, 300), "Measuring the structure of an undeciphered script", 84, 900, F, INK, ramp(t, 0.6, 0.8), 1.1)
    para(d, (120, y + 30), "About 4,000 Indus inscriptions, 4,500 years old, never deciphered. We don't try to read "
         "them. We measure their structure, and how far each measurement can be trusted.", 34, 860, FR, MUTED,
         ramp(t, 1.6, 0.8))
    scale = 640 / c["img"].shape[0]
    photo = fit(to_pil(c["img"]), h=640)
    px, py = 1140, 170
    paste(im, photo, (px, py), ramp(t, 1.0, 1.0))
    d = ImageDraw.Draw(im)
    n = len(c["boxes"])
    for i, ((x0, y0, x1, y1), s) in enumerate(zip(c["boxes"], c["signs"])):
        a = ramp(t, 4.0 + i * 0.9, 0.5)
        if a <= 0:
            continue
        box = [px + x0 * scale, py + y0 * scale, px + x1 * scale, py + y1 * scale]
        d.rectangle(box, outline=lerp(BG, TEAL, a), width=5)
        g = glyph_rgb(s, 64, TEAL)
        paste(im, g, (int((box[0] + box[2]) / 2 - 32), py + 660), a)
    if t > 4 + n * 0.9:
        text(d, (120, 860), "Every sign boxed by hand, labelled from the corpus  →", 30, FS, TEAL, ramp(t, 4.5 + n * 0.9))
    return im


# ------------------------------------------------------------------ scene 2: data
def s2_setup():
    thumbs = []
    for cisi in ["M-1114", "M-1152", "H-46", "M-12", "M-74", "M-232"]:
        if cisi not in SEALS.index:
            continue
        b, _ = boxes_of(cisi)
        if not b:
            continue
        thumbs.append((seal_photo(cisi), b))
    return dict(thumbs=thumbs[:6])


def s2(t, c):
    im, d = base()
    header(d, t, "Data", "Labelled seal photos and hand-drawn sign boxes")
    stats = [(855, "seal photos (CISI) with their sign sequences"), (834, "seals where the team boxed every sign"),
             (4078, "signs boxed and checked")]
    for i, (v, label) in enumerate(stats):
        a = ramp(t, 0.8 + i * 1.2)
        shown = int(v * ease((t - 0.8 - i * 1.2) / 1.6))
        text(d, (120, 300 + i * 150), f"{max(0, shown):,}", 96, F, TEAL, a)
        text(d, (420, 330 + i * 150), label, 34, FR, INK, a)
    # split bar
    a = ramp(t, 5.5)
    text(d, (120, 780), "One fixed split, never changed", 30, FS, MUTED, a)
    parts = [(466, "train", TEAL), (67, "val", GOLD), (141, "test", CARN), (181, "Harappa (never trained on)", (120, 140, 200))]
    x, total, width = 120, 855, 1000
    grow = ease((t - 6.0) / 2.0)
    for n, label, col in parts:
        w = int(width * n / total * grow)
        if w > 0:
            d.rounded_rectangle([x, 840, x + w, 890], 6, fill=col)
        x += int(width * n / total * grow) + 6
    for k, (n, label, col) in enumerate(parts):
        a = ramp(t, 8.0 + k * 0.2)
        lx, ly = 120 + (k % 2) * 420, 910 + (k // 2) * 44
        d.rounded_rectangle([lx, ly + 4, lx + 26, ly + 30], 4, fill=lerp(BG, col, a))
        text(d, (lx + 40, ly), f"{n} {label}", 28, FR, INK, a)
    # thumbnails with boxes appearing
    for k, (img, boxes) in enumerate(c["thumbs"]):
        col, row = k % 2, k // 2
        h = 230
        s = h / img.shape[0]
        th = fit(to_pil(img), h=h)
        x0, y0 = 1250 + col * 300, 260 + row * 250
        th = th.crop((0, 0, min(th.width, 280), h))
        paste(im, th, (x0, y0), ramp(t, 1.0 + k * 0.4))
        dd = ImageDraw.Draw(im)
        for j, (a0, b0, a1, b1) in enumerate(boxes):
            aa = ramp(t, 3.0 + k * 0.5 + j * 0.25, 0.3)
            if aa > 0 and a0 * s < 280:
                dd.rectangle([x0 + a0 * s, y0 + b0 * s, x0 + min(a1 * s, 279), y0 + b1 * s], outline=lerp(BG, TEAL, aa), width=3)
    return im


# ------------------------------------------------------------------ scene 3: M1
def _cnn(path):
    import torch  # noqa: F401  (loaded by train_wells_cnn)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_wells_cnn import load_weights
    from indus.cnn import SignCNN
    from indus.wells import wells_codes
    codes = wells_codes()
    m = SignCNN(len(codes))
    load_weights(path, m, codes, "real_finetuned")
    m.eval()
    return m, codes


def s3_setup():
    from indus.cnn import predict, to_input
    from indus.real_crops import square_crop
    models = Path(__file__).resolve().parents[1] / "data" / "processed" / "models"
    cisi = "M-131"                                 # test seal: decent-looking cuts, but they slice signs
    seal = SEALS.loc[cisi]
    img = seal_photo(cisi)
    n = seal.n_signs
    crops, band, edges = sign_crops(img, n)        # reading order (right to left)
    boxes, signs_ltr = boxes_of(cisi)              # left to right
    gray = cv2.cvtColor(load_image(seal.image), cv2.COLOR_BGR2GRAY)
    cut_m, codes = _cnn(models / "wells_all.pt")
    hb_m, _ = _cnn(models / "wells_handbox.pt")
    pred_cut = [codes[t[0] - 1] for t in predict(cut_m, to_input(np.stack(crops)))[0]][::-1]          # -> left to right
    hb_crops = [square_crop(gray[y0:y1], x0, x1) for x0, y0, x1, y1 in boxes]
    pred_hb = [codes[t[0] - 1] for t in predict(hb_m, to_input(np.stack(hb_crops)))[0]]
    xs = [band.x0 + int(e) for e in edges]
    cut_pieces = [img[band.y0:band.y1, a:b] for a, b in zip(xs[:-1], xs[1:])]       # left to right
    box_pieces = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
    return dict(img=img, band=band, xs=xs, boxes=boxes, truth=signs_ltr, pred_cut=pred_cut, pred_hb=pred_hb,
                cut_pieces=cut_pieces, box_pieces=box_pieces)


def _row(im, d, t, start, pieces, preds, truth, y, label, colour):
    """A row of photo pieces (height 120) with the CNN's guess under each and a score."""
    text(d, (760, y - 46), label, 30, FS, colour, ramp(t, start))
    x = 760
    right = 0
    for k, (pc, p, tr) in enumerate(zip(pieces, preds, truth)):
        tile = fit(to_pil(pc), h=120)
        if tile.width > 140:
            tile = fit(to_pil(pc), w=140)
        a = ramp(t, start + 0.2 + k * 0.25, 0.4)
        paste(im, tile, (x + (140 - tile.width) // 2, y + (120 - tile.height) // 2), a)
        ok = p == tr
        right += ok
        g = ramp(t, start + 1.8 + k * 0.25, 0.4)
        paste(im, glyph_rgb(p, 72, TEAL if ok else CARN), (x + 34, y + 130), g)
        d = ImageDraw.Draw(im)
        text(d, (x + 70, y + 206), "✓" if ok else "✗", 30, FS, TEAL if ok else CARN, g, "ma")
        x += 150
    text(d, (x + 20, y + 130), f"{right}/{len(truth)}", 64, F, TEAL if right == len(truth) else CARN,
         ramp(t, start + 3.6))
    return d


def s3(t, c):
    im, d = base()
    h = 560
    s = h / c["img"].shape[0]
    px, py = 120, 300
    b = c["band"]
    if t < 17:
        header(d, t, "M1 · Vision", "Seal photo → sign sequence")
        paste(im, fit(to_pil(c["img"]), h=h), (px, py), ramp(t, 0.3))
        d = ImageDraw.Draw(im)
        a = ramp(t, 1.2)
        if a > 0 and t < 10.5:
            d.rectangle([px + b.x0 * s, py + b.y0 * s, px + b.x1 * s, py + b.y1 * s], outline=lerp(BG, CARN, a), width=4)
        if 2.2 < t < 10.5:
            for k, x in enumerate(c["xs"]):
                aa = ramp(t, 2.2 + k * 0.2, 0.3)
                d.line([(px + x * s, py + b.y0 * s), (px + x * s, py + b.y1 * s)], fill=lerp(BG, CARN, aa), width=4)
        text(d, (120, 905), "Find the text band, cut it into as many pieces as the seal has signs, let a CNN name each piece",
             26, FR, MUTED, ramp(t, 1.0))
        d = _row(im, d, t, 3.5, c["cut_pieces"], c["pred_cut"], c["truth"], 330, "Equal cuts → CNN", CARN)
        if t > 10.5:
            for k, (x0, y0, x1, y1) in enumerate(c["boxes"]):
                aa = ramp(t, 10.5 + k * 0.25, 0.3)
                d.rectangle([px + x0 * s, py + y0 * s, px + x1 * s, py + y1 * s], outline=lerp(BG, TEAL, aa), width=5)
            d = _row(im, d, t, 11.0, c["box_pieces"], c["pred_hb"], c["truth"], 645, "Hand-drawn boxes → CNN trained on them", TEAL)
        return im
    header(d, 5, "M1 · Vision", "Cutting, not recognising, was the bottleneck")
    paste(im, fit(to_pil(c["img"]), h=h), (px, py), 1)
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in c["boxes"]:
        d.rectangle([px + x0 * s, py + y0 * s, px + x1 * s, py + y1 * s], outline=TEAL, width=5)
    bars = [("Equal cuts", 33.2, CARN, 17.5), ("Hand boxes, same CNN", 55.9, GOLD, 18.4),
            ("Hand boxes, CNN trained on them", 73.9, TEAL, 19.3)]
    text(d, (760, 300), "First-guess accuracy on unseen test seals", 30, FS, MUTED, ramp(t, 17.2))
    for i, (lab, v, col, st) in enumerate(bars):
        g = ease((t - st) / 1.0)
        y = 360 + i * 120
        text(d, (760, y), lab, 30, FR, INK, ramp(t, st, 0.4))
        if g > 0:
            d.rounded_rectangle([760, y + 44, 760 + int(12 * v * g), y + 90], 6, fill=col)
            text(d, (775 + int(12 * v * g), y + 46), f"{v * g:.1f}%", 38, F, INK, 1)
    para(d, (760, 740), "So we trained a sign detector on our boxes. Detector + this CNN reads whole seals at "
         "40.8% sign error without knowing the count, against 68% for equal cuts even when given the count.",
         28, 1060, FS, TEAL, ramp(t, 21.5))
    return im


# ------------------------------------------------------------------ scene 4: compositor
def s4_setup():
    rng = np.random.default_rng(3)
    split = dict(zip(SEALS.index, SEALS.split))
    train = BOXES[BOXES.cisi.map(split) == "train"]
    bank = build_bank(train)
    donor = "M-131" if "M-131" in set(train.cisi) else None
    cands = [x for x in train.cisi.unique() if len(boxes_of(x)[0]) >= 5]
    donor = cands[3]
    donor_img, donor_boxes = seal_photo(donor), boxes_of(donor)[0]
    # a wall of real crops from many seals (the "bank")
    wall = []
    for code in list(bank.crops)[:200:7][:24]:
        p = bank.crops[code][0]
        wall.append(p.image)
    # background: a wide, clean band, and the original band before painting out
    bgs = [b for b in bank.backgrounds if 3.0 < b.image.shape[1] / b.image.shape[0] < 5.5 and b.image.shape[0] >= 60]
    def coverage(b):          # share of the band covered by sign boxes: less = more bare stone to show
        r = BOXES[BOXES.cisi == b.seal]
        area = ((r.x1 - r.x0) * (r.y1 - r.y0)).sum()
        return area / ((r.x1.max() - r.x0.min()) * (r.y1.max() - r.y0.min()))
    bg = min(bgs, key=coverage)
    brow = BOXES[BOXES.cisi == bg.seal]
    simg = seal_photo(bg.seal)
    m = 3
    band_before = simg[max(0, brow.y0.min() - m):brow.y1.max() + m, max(0, brow.x0.min() - m):brow.x1.max() + m]
    boxes_before = [(r.x0 - max(0, brow.x0.min() - m), r.y0 - max(0, brow.y0.min() - m),
                     r.x1 - max(0, brow.x0.min() - m), r.y1 - max(0, brow.y0.min() - m)) for r in brow.itertuples()]
    # a real corpus text (never on a held-out seal) whose signs all have crops
    from indus.data import load_icit_full
    held = {tuple(x) for x, sp in zip(SEALS.signs_reading, SEALS.split) if sp != "train"}
    texts = [x for x in load_icit_full("reading") if tuple(x) not in held and len(x) == 5
             and all(s in bank.crops for s in x)]
    seq = texts[11]
    visual = seq[::-1]
    bg_mean = bg.image.mean()
    picks = []
    for s_ in visual:      # the crop whose stone is closest in brightness to the background
        picks.append(min(bank.crops[s_], key=lambda p: abs(p.image.mean() - bg_mean)))
    hgt = 160
    pieces = []
    for p in picks:
        f = hgt * 0.9 / p.band_height
        ph = max(8, min(hgt - 6, round(p.image.shape[0] * f)))
        pw = max(4, round(p.image.shape[1] * ph / p.image.shape[0]))
        pieces.append(cv2.resize(p.image, (pw, ph), interpolation=cv2.INTER_AREA))
    gap = 6
    width = 24 + sum(pc.shape[1] for pc in pieces) + gap * (len(pieces) - 1) + 24
    back = cv2.resize(bg.image, (width, hgt), interpolation=cv2.INTER_CUBIC)
    slots, x = [], 24
    for pc in pieces:
        slots.append((x, (hgt - pc.shape[0]) // 2))
        x += pc.shape[1] + gap
    fig = Image.open(Path(__file__).resolve().parents[1] / "reports/figures/final/composites_vs_real.png")
    return dict(donor=donor_img, donor_boxes=donor_boxes, wall=wall, seq=seq, pieces=pieces, slots=slots,
                back=back, band_before=band_before, boxes_before=boxes_before, band_after=bg.image, fig=fig)


def blend_piece(canvas, pc, x, y):
    """Paste a crop with feathered edges and brightness matched to the stone (as in indus.composite)."""
    ph, pw = pc.shape[:2]
    region = canvas[y:y + ph, x:x + pw].astype(np.float32)
    piece = pc.astype(np.float32) + region.mean(axis=(0, 1)) - pc.astype(np.float32).mean(axis=(0, 1))
    alpha = np.zeros((ph, pw), np.float32)
    inset = max(1, min(ph, pw) // 6)
    alpha[inset:ph - inset, inset:pw - inset] = 1
    alpha = cv2.GaussianBlur(alpha, (0, 0), inset)[..., None]
    canvas[y:y + ph, x:x + pw] = np.clip(alpha * piece + (1 - alpha) * region, 0, 255).astype(np.uint8)


def s4(t, c):
    im, d = base()
    header(d, t if t < 9 else 5, "M1 · Prof. Ashwin's suggestion", "Build new inscriptions out of real signs")
    if t < 9:
        text(d, (120, 270), "1   Cut every hand-boxed sign out of the training seals", 34, FS, INK, ramp(t, 0.4))
        h = 520
        img = c["donor"]
        s = h / img.shape[0]
        px, py = 120, 340
        paste(im, fit(to_pil(img), h=h), (px, py), ramp(t, 0.6))
        d = ImageDraw.Draw(im)
        cell, cols, gx, gy = 112, 7, 980, 340
        for j, (x0, y0, x1, y1) in enumerate(c["donor_boxes"]):
            aa = ramp(t, 1.4 + j * 0.3, 0.3)
            d.rectangle([px + x0 * s, py + y0 * s, px + x1 * s, py + y1 * s], outline=lerp(BG, TEAL, aa), width=4)
            g = ease((t - 3.4 - j * 0.25) / 1.1)
            if g > 0:
                crop = fit(to_pil(img[y0:y1, x0:x1]), h=cell - 12)
                if crop.width > cell - 12:
                    crop = fit(to_pil(img[y0:y1, x0:x1]), w=cell - 12)
                tx, ty = gx + (j % cols) * cell, gy + (j // cols) * cell
                sx, sy = px + x0 * s, py + y0 * s
                paste(im, crop, (int(sx + (tx - sx) * g), int(sy + (ty - sy) * g)))
        n0 = len(c["donor_boxes"])
        for k, w_ in enumerate(c["wall"]):
            j = n0 + k
            if j >= cols * 5:
                break
            a = ramp(t, 5.0 + k * 0.08, 0.3)
            crop = fit(to_pil(w_), h=cell - 12)
            if crop.width > cell - 12:
                crop = fit(to_pil(w_), w=cell - 12)
            paste(im, crop, (gx + (j % cols) * cell, gy + (j // cols) * cell), a)
        text(d, (980, 920), "2,240 real sign crops of 336 signs, training seals only", 28, FR, MUTED, ramp(t, 6.0))
        return im
    if t < 15:
        text(d, (120, 270), "2   Paint the signs out of a real seal surface (OpenCV inpainting)", 34, FS, INK, ramp(t, 9.0))
        before = fit(to_pil(c["band_before"]), h=250)
        if before.width > 1100:
            before = fit(to_pil(c["band_before"]), w=1100)
        sc = before.width / c["band_before"].shape[1]
        after = to_pil(c["band_after"]).resize(before.size, Image.LANCZOS)
        paste(im, before, (120, 340), ramp(t, 9.2))
        dd = ImageDraw.Draw(im)
        for x0, y0, x1, y1 in c["boxes_before"]:
            dd.rectangle([120 + x0 * sc, 340 + y0 * sc, 120 + x1 * sc, 340 + y1 * sc], outline=lerp(BG, TEAL, ramp(t, 9.8)), width=4)
        y2 = 340 + before.height + 60
        paste(im, after, (120, y2), ramp(t, 11.3, 1.2))
        text(d, (160 + before.width, 340 + before.height // 2 - 18), "real seal band, signs boxed", 30, FR, INK, ramp(t, 9.8))
        text(d, (160 + before.width, y2 + before.height // 2 - 18), "signs painted out: bare stone", 30, FS, TEAL, ramp(t, 12.0))
        return im
    if t < 26:
        text(d, (120, 270), "3   Paste real signs along a real corpus text", 34, FS, INK, ramp(t, 15.0))
        scale = min(1500 / c["back"].shape[1], 340 / c["back"].shape[0])
        bx, by = 210, 530
        bx = (W - round(c["back"].shape[1] * scale)) // 2
        canvas = c["back"].copy()
        for k, (pc, (x, y)) in enumerate(zip(c["pieces"], c["slots"])):
            if t > 17.4 + k * 1.3:
                blend_piece(canvas, pc, x, y)
        band = to_pil(canvas).resize((round(canvas.shape[1] * scale), round(canvas.shape[0] * scale)), Image.LANCZOS)
        paste(im, band, (bx, by), ramp(t, 15.5))
        dd = ImageDraw.Draw(im)
        vis = c["seq"][::-1]
        for k, (s_, pc, (x, y)) in enumerate(zip(vis, c["pieces"], c["slots"])):
            cx = bx + (x + pc.shape[1] / 2) * scale
            paste(im, glyph_rgb(s_, 90, GOLD), (int(cx - 45), 360), ramp(t, 15.6 + k * 0.12))
            g = ease((t - 16.2 - k * 1.3) / 1.2)
            if 0 < g < 1:
                fly = to_pil(pc).resize((round(pc.shape[1] * scale), round(pc.shape[0] * scale)), Image.LANCZOS)
                sx, sy = cx - fly.width / 2, 360
                tx, ty = bx + x * scale, by + y * scale
                paste(im, fly, (int(sx + (tx - sx) * g), int(sy + (ty - sy) * g)))
        dd = ImageDraw.Draw(im)
        text(dd, (bx + band.width, 470), "←  read right to left", 26, FR, MUTED, ramp(t, 16.0), "ra")
        if t > 17.4 + 1.3 * len(c["pieces"]):
            for k, (pc, (x, y)) in enumerate(zip(c["pieces"], c["slots"])):
                a = ramp(t, 23.0, 0.5) * (1 - ramp(t, 25.0, 0.8))
                dd.rectangle([bx + x * scale, by + y * scale, bx + (x + pc.shape[1]) * scale, by + (y + pc.shape[0]) * scale],
                             outline=lerp(BG, TEAL, a), width=4)
            text(dd, (bx, by + band.height + 40), "A new inscription that comes with its exact sequence and sign boxes",
                 32, FS, TEAL, ramp(t, 23.0))
        return im
    if t < 31:
        text(d, (120, 270), "5,000 composites (left) next to real seal bands (right)", 34, FS, INK, ramp(t, 26.0))
        paste(im, fit(c["fig"].convert("RGB"), w=1500), (210, 340), ramp(t, 26.2))
        return im
    text(d, (120, 270), "Does it help a reader that needs no cutting (CRNN + CTC)?", 34, FS, INK, ramp(t, 31.0))
    rows = [("Test seals: sign error", 55.4, 54.4, "%"), ("Harappa, never trained on: sign error", 57.9, 53.2, "%"),
            ("Harappa: seals with the right sign count", 53, 65, " / 181")]
    for i, (lab, a0, a1, unit) in enumerate(rows):
        y = 360 + i * 170
        text(d, (120, y), lab, 32, FR, INK, ramp(t, 31.5 + i * 0.6))
        for j, (v, col, name) in enumerate([(a0, MUTED, "real bands only"), (a1, TEAL, "+ composites")]):
            g = ease((t - 32.0 - i * 0.6 - j * 0.5) / 1.0)
            k = 12 if unit == "%" else 10
            if g > 0:
                d.rounded_rectangle([120, y + 50 + j * 50, 120 + int(k * v * g), y + 88 + j * 50], 5, fill=col)
                text(d, (135 + int(k * v * g), y + 50 + j * 50), f"{v * g:.1f}{unit}" if unit == "%" else f"{int(v * g)}{unit}", 30, FS, INK, 1)
                text(d, (1500, y + 52 + j * 50), name, 26, FR, MUTED, 1)
    para(d, (120, 900), "Biggest gain on the site it never saw: composites add arrangements and backgrounds the 466 real bands lack.",
         28, 1600, FR, MUTED, ramp(t, 35.0))
    return im


# ------------------------------------------------------------------ scene 5: M2 + M3
def s5(t, c):
    im, d = base()
    header(d, t, "M2 · M3 · Statistics and prediction", "Sign order is structured and partly predictable")
    paste(im, glyph_rgb(740, 160, GOLD), (120, 300), ramp(t, 0.6))
    text(d, (320, 320), "The jar sign", 40, FS, INK, ramp(t, 0.8))
    for i, (lab, v, col) in enumerate([("ends a text", 35.2, TEAL), ("starts a text", 0.16, CARN)]):
        g = ease((t - 1.5 - i * 0.6) / 1.2)
        y = 390 + i * 60
        text(d, (320, y), lab, 28, FR, MUTED, ramp(t, 1.2))
        if g > 0:
            d.rounded_rectangle([520, y, 520 + max(4, int(9 * v * g)), y + 36], 5, fill=col)
            text(d, (535 + max(4, int(9 * v * g)), y), f"{v * g:.2f}%" if v < 1 else f"{v * g:.1f}%", 28, FS, INK, 1)
    para(d, (120, 560), "Next-sign uncertainty: 3.31 bits read right to left, 3.82 reversed. The script reads right to left.",
         30, 760, FR, INK, ramp(t, 4.0))
    # masked prediction demo, shown as on the seal (left to right = reverse of reading order)
    text(d, (1000, 300), "Hide one sign, predict it from its neighbours", 32, FS, INK, ramp(t, 6.0))
    reading = [33, 820, 2, 441, 740]
    vis = reading[::-1]
    hidden_vis = len(reading) - 1 - 2
    for i, s in enumerate(vis):
        x = 1000 + i * 150
        if i == hidden_vis:
            d.rounded_rectangle([x, 370, x + 120, 490], 10, outline=lerp(BG, CARN, ramp(t, 6.5)), width=4)
            text(d, (x + 60, 395), "?", 64, F, CARN, ramp(t, 6.5) * (1 - ramp(t, 13.0)), "ma")
            if t > 13:
                paste(im, glyph_rgb(2, 100, TEAL), (x + 10, 380), ramp(t, 13.0))
        else:
            paste(im, glyph_rgb(s, 100, INK), (x + 10, 380), ramp(t, 6.5 + i * 0.1))
    text(d, (1000, 510), "←  reading direction", 26, FR, MUTED, ramp(t, 7))
    cands = [(2, 0.36), (60, 0.13), (1, 0.13)]
    for i, (s, p) in enumerate(cands):
        y = 580 + i * 95
        a = ramp(t, 8.5 + i * 0.8)
        paste(im, glyph_rgb(s, 76, TEAL if i == 0 else INK), (1000, y), a)
        g = ease((t - 8.8 - i * 0.8) / 1.0)
        if g > 0:
            d.rounded_rectangle([1100, y + 20, 1100 + int(1200 * p * g), y + 56], 5, fill=TEAL if i == 0 else LINE)
            text(d, (1115 + int(1200 * p * g), y + 20), f"{p * g:.2f}", 28, FS, INK, 1)
    para(d, (120, 760), "Bigram statistics recover the hidden sign 35% of the time (60% in the top 5). A small transformer "
         "did worse (30%): with under 2,000 short texts, counting beats learning.", 30, 800, FR, INK, ramp(t, 14.5))
    return im


# ------------------------------------------------------------------ scene 6: M4 + E2
def s6(t, c):
    im, d = base()
    header(d, t, "M4 · Guardrails and null models", "Non-random order is not proof of language")
    groups = [("Indus script", 81.4, 61.0), ("Kudurru deity symbols (not writing)", 84.6, 50.8), ("Barn-star motifs", 50.0, 58.1)]
    x0, x1 = 760, 1780              # axis 40% .. 95%

    def X(v):
        return x0 + (v - 40) / 55 * (x1 - x0)
    a = ramp(t, 0.8)
    d.line([(X(50), 300), (X(50), 820)], fill=lerp(BG, MUTED, a), width=2)
    text(d, (X(50) + 8, 290), "chance", 24, FR, MUTED, a)
    for i, (name, pair, tri) in enumerate(groups):
        y = 340 + i * 165
        text(d, (120, y + 20), name, 32, FS, INK, ramp(t, 1.0 + i * 3.0))
        for j, (v, col, st) in enumerate([(pair, TEAL, 1.3 + i * 3.0), (tri, CARN, 8.0 + i * 1.5)]):
            if i == 0 and j == 1:
                st = 11.0
            g = ease((t - st) / 1.2)
            if g > 0:
                d.rounded_rectangle([X(40), y + j * 58, X(40 + (v - 40) * g), y + j * 58 + 44], 5, fill=col)
                text(d, (X(40 + (v - 40) * g) + 12, y + j * 58 + 6), f"{40 + (v - 40) * g:.1f}%", 28, FS, INK, 1)
    a2 = ramp(t, 0.8)
    d.rounded_rectangle([120, 860, 150, 890], 4, fill=lerp(BG, TEAL, a2))
    text(d, (165, 860), "real vs shuffled texts (sign pairs)", 26, FR, MUTED, a2)
    a3 = ramp(t, 8.0)
    d.rounded_rectangle([720, 860, 750, 890], 4, fill=lerp(BG, CARN, a3))
    text(d, (765, 860), "real vs a no-language Markov chain (sign triples)", 26, FR, MUTED, a3)
    para(d, (120, 920), "A bigram chain with no language already reproduces Zipf's law and the jar sign; only recurring "
         "groups of three signs set the Indus texts apart.", 26, 1680, FR, INK, ramp(t, 16.0))
    return im


# ------------------------------------------------------------------ scene 7: E1
def s7(t, c):
    im, d = base()
    header(d, t, "E1 · Error propagation", "Are our readings good enough now? Yes, with good boxes")
    ox, oy, w, h = 200, 860, 1000, 520     # chart: x = reading accuracy 100 -> 30, y = NB accuracy 45..85

    def P(acc, v):
        return ox + (100 - acc) / 75 * w, oy - (v - 45) / 40 * h
    a = ramp(t, 0.5)
    d.line([(ox, oy), (ox + w, oy)], fill=lerp(BG, MUTED, a), width=2)
    d.line([(ox, oy), (ox, oy - h)], fill=lerp(BG, MUTED, a), width=2)
    for acc in (100, 60, 30):
        text(d, (P(acc, 45)[0], oy + 12), f"{acc}%", 26, FR, MUTED, a, "ma")
    text(d, (ox + w / 2, oy + 50), "reading accuracy of the transcriptions", 26, FR, MUTED, a, "ma")
    text(d, (ox - 20, oy - h - 50), "real vs shuffled (Naive Bayes)", 26, FR, MUTED, a)
    cy = P(100, 50)[1]
    d.line([(ox, cy), (ox + w, cy)], fill=lerp(BG, LINE, a), width=2)
    text(d, (ox + 12, cy - 34), "chance", 24, FR, MUTED, a)
    text(d, (ox + 30, P(92, 79.6)[1] + 80), "random mistakes", 26, FR, CARN, ramp(t, 2.5))
    pts = [(100, 82.7), (60, 64.5), (30, 53.5)]
    for i, (acc, v) in enumerate(pts):
        g = ramp(t, 1.5 + i * 1.2)
        if g > 0:
            x, y = P(acc, v)
            if i > 0:
                px, py = P(*pts[i - 1])
                d.line([(px, py), (px + (x - px) * g, py + (y - py) * g)], fill=CARN, width=4)
            d.ellipse([x - 10, y - 10, x + 10, y + 10], fill=lerp(BG, CARN, g))
            text(d, (x + 18, y + 8 if acc == 30 else y - 40), f"{v}%", 30, FS, INK, g)
    g = ramp(t, 5.5)
    x, y = P(31, 61.0)
    d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=lerp(BG, GOLD, g))
    text(d, (x + 22, y - 34), "61.0% forced-", 26, FS, GOLD, g)
    text(d, (x + 22, y - 4), "cut CNN (31%)", 26, FS, GOLD, g)
    g = ramp(t, 9.0)
    x, y = P(73.9, 73.7)
    d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=lerp(BG, TEAL, g))
    text(d, (x + 24, y - 64), "73.7%  hand-box CNN", 30, FS, TEAL, g)
    text(d, (x + 24, y - 28), "(74% accurate, its real mistakes)", 26, FR, TEAL, g)
    para(d, (1430, 360), "At 31% accuracy most of what you would measure is the reader. At our new 74%, every signal "
         "survives: real vs shuffled 73.7%, the jar sign still ends texts, and masked prediction is twice its "
         "shuffled control. The readings are now good enough for the structure to be the script's.",
         28, 400, FR, INK, ramp(t, 11.0))
    return im


# ------------------------------------------------------------------ scene 8: summary
def s8(t, c):
    im, d = base()
    header(d, t, "Summary", "What we built")
    items = ["855 labelled seals and 4,078 hand-checked sign boxes",
             "A seal reader tested on unseen seals and an unseen site",
             "Inscriptions composed from real signs (Prof. Ashwin's idea)",
             "Markov statistics, masked prediction, guardrails, null models"]
    for i, s in enumerate(items):
        text(d, (120, 300 + i * 70), "·  " + s, 36, FR, INK, ramp(t, 0.5 + i * 0.5))
    text(d, (120, 640), "github.com/RaghavSinghal15/indus-computational-epigraphy", 38, FS, TEAL, ramp(t, 3.0))
    text(d, (120, 700), "Code, fixed splits, tests and every report", 28, FR, MUTED, ramp(t, 3.2))
    text(d, (120, 820), "Raghav Singhal  ·  Devam Jatin Bhavsar  ·  Krrish Bagaria  ·  Saatwik Tiwari  ·  Agrim Agarwal",
         30, FR, INK, ramp(t, 4.0))
    return im


SCENES = [(s1, s1_setup, 15), (s2, s2_setup, 18), (s3, s3_setup, 27), (s4, s4_setup, 38),
          (s5, None, 22), (s6, None, 24), (s7, None, 20), (s8, None, 12)]          # 176 s


def render(i):
    fn, setup, dur = SCENES[i - 1]
    ctx = setup() if setup else {}
    tmp, out = CLIPS / f"tmp_s{i}.mp4", CLIPS / f"s{i}.mp4"
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-vf", f"fade=t=in:st=0:d=0.3,fade=t=out:st={dur - 0.35}:d=0.35",
                          "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(tmp)],
                         stdin=subprocess.PIPE)
    for f in range(int(dur * FPS)):
        p.stdin.write(fn(f / FPS, ctx).tobytes())
    p.stdin.close()
    if p.wait() == 0:
        tmp.replace(out)
    print("scene", i, "done", flush=True)


def still(i, t, path):
    fn, setup, _ = SCENES[i - 1]
    fn(t, setup() if setup else {}).save(path)


if __name__ == "__main__":
    FRIEZE = make_frieze()
    if sys.argv[1] == "still":
        for spec in sys.argv[2:]:
            i, t = spec.split("@")
            still(int(i), float(t), HERE / f"still_{i}_{t}.png")
    else:
        for i in map(int, sys.argv[1:]):
            render(i)
