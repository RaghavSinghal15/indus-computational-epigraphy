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
def s3_setup():
    img = seal_photo("M-1114")
    crops, band, edges = sign_crops(img, 6)
    boxes, signs = boxes_of("M-1114")
    truth = list(SEALS.loc["M-1114"].signs_reading)
    pred = [920, 2, 740, 827, 32, 740]          # v2 CNN on the forced cuts (wells_all.pt), reading order
    return dict(img=img, band=band, edges=edges, crops=crops, boxes=boxes, truth=truth, pred=pred)


def s3(t, c):
    im, d = base()
    header(d, t, "M1 · Vision", "Seal photo → sign sequence")
    h = 560
    s = h / c["img"].shape[0]
    px, py = 120, 290
    paste(im, fit(to_pil(c["img"]), h=h), (px, py), ramp(t, 0.5))
    d = ImageDraw.Draw(im)
    b = c["band"]
    steps = ["1  Find the text band", "2  Cut it into n pieces", "3  A CNN names each piece",
             "4  Sign-pair statistics fix some mistakes"]
    for i, st in enumerate(steps):
        text(d, (720, 300 + i * 56), st, 34, FS if i == min(3, int(max(0, t - 1) // 2.5)) else FR, INK, ramp(t, 1.0 + i * 2.5))
    # band
    a = ramp(t, 1.5)
    if a > 0 and t < 15:
        d.rectangle([px + b.x0 * s, py + b.y0 * s, px + b.x1 * s, py + b.y1 * s], outline=lerp(BG, CARN, a), width=4)
    # cuts sweep
    if 4.0 < t < 15:
        for k, e in enumerate(c["edges"]):
            aa = ramp(t, 4.0 + k * 0.25, 0.3)
            x = px + (b.x0 + e) * s
            d.line([(x, py + b.y0 * s), (x, py + b.y1 * s)], fill=lerp(BG, CARN, aa), width=4)
    # crops + predictions (reading order: right to left, so show as on the seal: left to right = reversed)
    if t > 6.5:
        crops_vis = c["crops"][::-1]
        pred_vis, truth_vis = c["pred"][::-1], c["truth"][::-1]
        for k, cr in enumerate(crops_vis):
            aa = ramp(t, 6.5 + k * 0.3, 0.4)
            x = 720 + k * 170
            paste(im, Image.fromarray(cr).convert("RGB").resize((140, 140)), (x, 560), aa)
            ok = pred_vis[k] == truth_vis[k]
            g = glyph_rgb(pred_vis[k], 90, TEAL if ok else CARN)
            paste(im, g, (x + 25, 715), ramp(t, 8.5 + k * 0.3, 0.4))
            d = ImageDraw.Draw(im)
            text(d, (x + 70, 830), "✓" if ok else "✗", 40, FS, TEAL if ok else CARN, ramp(t, 9.0 + k * 0.3), "ma")
        text(d, (720, 885), "CNN guess under each cut piece: 2 of 6 right on this seal", 28, FR, MUTED, ramp(t, 10.5))
    # results
    if t >= 15:
        im, d = base()
        header(d, 5, "M1 · Vision", "Cutting, not recognising, is the bottleneck")
        paste(im, fit(to_pil(c["img"]), h=h), (px, py), 1)
        d = ImageDraw.Draw(im)
        if t < 16.5:
            for k, e in enumerate(c["edges"]):
                x = px + (b.x0 + e) * s
                d.line([(x, py + b.y0 * s), (x, py + b.y1 * s)], fill=CARN, width=4)
        for k, (x0, y0, x1, y1) in enumerate(c["boxes"]):
            aa = ramp(t, 16.5 + k * 0.3, 0.4)
            d.rectangle([px + x0 * s, py + y0 * s, px + x1 * s, py + y1 * s], outline=lerp(BG, TEAL, aa), width=5)
        bars = [("Forced cuts", 33.2, CARN, 18.5), ("Hand boxes, same CNN", 55.9, TEAL, 19.8),
                ("Hand boxes, CNN trained on them", 73.9, (140, 210, 190), 21.2)]
        text(d, (720, 300), "First-guess accuracy on unseen test seals", 32, FS, MUTED, ramp(t, 18))
        for i, (lab, v, col, st) in enumerate(bars):
            g = ease((t - st) / 1.2)
            y = 370 + i * 140
            text(d, (720, y), lab, 32, FR, INK, ramp(t, st, 0.4))
            if g > 0:
                d.rounded_rectangle([720, y + 48, 720 + int(12 * v * g), y + 98], 6, fill=col)
                text(d, (735 + int(12 * v * g), y + 50), f"{v * g:.1f}%", 40, F, INK, 1)
        para(d, (720, 820), "Forced-cut pipeline overall: 33.7% first guess, 55% top 5 on 141 unseen seals "
             "(guessing common signs: 11%), and 28.9% on Harappa, a site it never saw.", 28, 1050, FR, MUTED, ramp(t, 23.5))
    return im


# ------------------------------------------------------------------ scene 4: compositor
def s4_setup():
    rng = np.random.default_rng(3)
    split = dict(zip(SEALS.index, SEALS.split))
    train = BOXES[BOXES.cisi.map(split) == "train"]
    bank = build_bank(train)
    donors = ["M-12", "M-232", "M-74"]
    donors = [x for x in donors if x in set(train.cisi)] or list(train.cisi.unique()[:3])
    donor_imgs = [(seal_photo(x), boxes_of(x)[0]) for x in donors[:3]]
    seq = [740, 820, 2, 33]                       # a real ICIT text in reading order? pick one fully covered
    from indus.data import load_icit_full
    held = {tuple(s) for s, sp in zip(SEALS.signs_reading, SEALS.split) if sp != "train"}
    texts = [t for t in load_icit_full("reading") if tuple(t) not in held and len(t) == 5 and all(s in bank.crops for s in t)]
    seq = texts[7]
    visual = seq[::-1]
    picks = [bank.crops[s][rng.integers(len(bank.crops[s]))] for s in visual]
    bg = next(b for b in bank.backgrounds[10:] if b.image.shape[1] / b.image.shape[0] > 3.5)
    hgt = 150
    pieces = []
    for p in picks:
        f = hgt * 0.92 / p.band_height
        ph = max(8, min(hgt - 4, round(p.image.shape[0] * f)))
        pw = max(4, round(p.image.shape[1] * ph / p.image.shape[0]))
        pieces.append(cv2.resize(p.image, (pw, ph)))
    gap = 8
    width = 30 + sum(pc.shape[1] for pc in pieces) + gap * (len(pieces) - 1) + 30
    back = cv2.resize(bg.image, (width, hgt))
    slots, x = [], 30
    for pc in pieces:
        y = (hgt - pc.shape[0]) // 2
        slots.append((x, y))
        x += pc.shape[1] + gap
    # the source band of the background seal, before inpainting
    bgseal = bg.seal
    brow = BOXES[BOXES.cisi == bgseal]
    simg = seal_photo(bgseal)
    band_before = simg[max(0, brow.y0.min() - 3):brow.y1.max() + 3, max(0, brow.x0.min() - 3):brow.x1.max() + 3]
    fig = Image.open("/home/claude/indus-computational-epigraphy/reports/figures/final/composites_vs_real.png")
    return dict(donors=donor_imgs, seq=seq, pieces=pieces, slots=slots, back=back, band_before=band_before,
                band_after=bg.image, fig=fig)


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
    header(d, t, "M1 · Prof. Ashwin's suggestion", "Build new inscriptions out of real signs")
    if t < 9:
        text(d, (120, 270), "1  Cut every hand-boxed sign out of the training seals", 34, FS, INK, ramp(t, 0.5))
        slot = 0
        for k, (img, boxes) in enumerate(c["donors"]):
            h = 300
            s = h / img.shape[0]
            px, py = 120 + k * 560, 360
            th = fit(to_pil(img), h=h)
            paste(im, th, (px, py), ramp(t, 0.8 + k * 0.3))
            dd = ImageDraw.Draw(im)
            for j, (x0, y0, x1, y1) in enumerate(boxes):
                aa = ramp(t, 2.0 + k * 0.4 + j * 0.2, 0.3)
                dd.rectangle([px + x0 * s, py + y0 * s, px + x1 * s, py + y1 * s], outline=lerp(BG, TEAL, aa), width=3)
                # crops fly down into the bank
                g = ease((t - 4.5 - k * 0.3 - j * 0.12) / 1.2)
                if g > 0:
                    crop = to_pil(img[y0:y1, x0:x1])
                    crop = fit(crop, h=80)
                    tx, ty = 120 + ((slot + j) % 17) * 100, 700 + ((slot + j) // 17) * 95
                    sx, sy = px + x0 * s, py + y0 * s
                    paste(im, crop, (int(sx + (tx - sx) * g), int(sy + (ty - sy) * g)), min(1, g * 2))
            slot += len(boxes)
        text(d, (120, 900), "2,240 real crops of 336 signs, from training seals only", 30, FR, MUTED, ramp(t, 6.5))
        return im
    if t < 15:
        text(d, (120, 270), "2  Paint the signs out of a real seal surface (OpenCV inpainting)", 34, FS, INK, ramp(t, 9.0))
        before = fit(to_pil(c["band_before"]), h=220)
        after = fit(to_pil(c["band_after"]), h=220)
        paste(im, before, (120, 400), ramp(t, 9.3))
        paste(im, after, (120, 400), ramp(t, 11.5, 1.5))
        text(d, (120 + before.width + 40, 480), "→  bare stone to write on", 34, FR, MUTED, ramp(t, 12.5))
        return im
    if t < 26:
        text(d, (120, 270), "3  Paste real signs along a real corpus text", 34, FS, INK, ramp(t, 15.0))
        # the text, as glyphs, in reading order shown right to left
        seq = c["seq"]
        vis = seq[::-1]
        for i, s in enumerate(vis):
            paste(im, glyph_rgb(s, 80, GOLD), (120 + i * 100, 340), ramp(t, 15.3 + i * 0.15))
        text(d, (130 + len(vis) * 100, 370), "←  read right to left", 28, FR, MUTED, ramp(t, 16))
        canvas = c["back"].copy()
        n_in = 0
        for k, (pc, (x, y)) in enumerate(zip(c["pieces"], c["slots"])):
            if t > 17.5 + k * 1.3:
                blend_piece(canvas, pc, x, y)
                n_in += 1
        scale = 2.2
        band = fit(to_pil(canvas), h=int(canvas.shape[0] * scale))
        bx, by = 120, 520
        paste(im, band, (bx, by), ramp(t, 16.5))
        dd = ImageDraw.Draw(im)
        for k, (pc, (x, y)) in enumerate(zip(c["pieces"], c["slots"])):
            g = ease((t - 16.3 - k * 1.3) / 1.2)
            if 0 < g < 1:       # flying sign
                fly = fit(to_pil(pc), h=int(pc.shape[0] * scale))
                sx, sy = 120 + k * 100, 340
                tx, ty = bx + x * scale, by + y * scale
                paste(im, fly, (int(sx + (tx - sx) * g), int(sy + (ty - sy) * g)), 1)
            if g >= 1:
                dd.rectangle([bx + x * scale, by + y * scale, bx + (x + pc.shape[1]) * scale, by + (y + pc.shape[0]) * scale],
                             outline=lerp(BG, TEAL, 1 - ramp(t, 24, 1.0)), width=3)
        if n_in == len(c["pieces"]):
            text(d, (120, 900), "A new inscription with its exact sequence and sign boxes, for free", 32, FS, TEAL, ramp(t, 23.5))
        return im
    if t < 31:
        text(d, (120, 270), "5,000 composites (left) next to real seal bands (right)", 34, FS, INK, ramp(t, 26.0))
        paste(im, fit(c["fig"].convert("RGB"), w=1500), (210, 340), ramp(t, 26.2))
        return im
    # results
    text(d, (120, 270), "Does it help a reader that needs no cutting (CRNN + CTC)?", 34, FS, INK, ramp(t, 31.0))
    rows = [("Test seals: sign error", 55.4, 54.4, "%"), ("Harappa, never trained on: sign error", 57.9, 53.2, "%"),
            ("Harappa: seals with the right sign count", 53, 65, " / 181")]
    for i, (lab, a0, a1, unit) in enumerate(rows):
        y = 360 + i * 170
        text(d, (120, y), lab, 32, FR, INK, ramp(t, 31.5 + i * 0.6))
        for j, (v, col, name) in enumerate([(a0, MUTED, "real bands only"), (a1, TEAL, "+ composites")]):
            g = ease((t - 32.0 - i * 0.6 - j * 0.5) / 1.0)
            scale = 12 if unit == "%" else 10
            if g > 0:
                d.rounded_rectangle([120, y + 50 + j * 50, 120 + int(scale * v * g), y + 88 + j * 50], 5, fill=col)
                text(d, (135 + int(scale * v * g), y + 50 + j * 50), f"{v * g:.1f}{unit}" if unit == "%" else f"{int(v * g)}{unit}", 30, FS, INK, 1)
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
    header(d, t, "E1 · Error propagation", "How reading errors distort the statistics")
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
    g = ramp(t, 6.0)
    x, y = P(31, 61.0)
    d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=lerp(BG, TEAL, g))
    text(d, (x - 24, y - 74), "61.0% with our", 26, FS, TEAL, g, "ra")
    text(d, (x - 24, y - 42), "CNN's real mistakes", 26, FS, TEAL, g, "ra")
    para(d, (1380, 380), "Random errors wash the structure out fast. Our CNN confuses similar signs consistently, so more "
         "survives, but at today's 31% accuracy most of what you would measure is the reader, not the script.",
         30, 440, FR, INK, ramp(t, 8.0))
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
