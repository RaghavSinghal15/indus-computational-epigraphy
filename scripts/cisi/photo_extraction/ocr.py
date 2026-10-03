"""OCR of caption crops with tesseract (via subprocess)."""
import cv2, numpy as np, subprocess, os, tempfile
from detect import flatten

def ocr_crop(img, psm=7, wl=None):
    fd, p = tempfile.mkstemp(suffix='.png', dir='/dev/shm' if os.path.isdir('/dev/shm') else None)
    os.close(fd)
    cv2.imwrite(p, img)
    cmd = ['tesseract', p, '-', '--psm', str(psm)]
    if wl:
        cmd += ['-c', 'tessedit_char_whitelist=' + wl]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    os.remove(p)
    return ' '.join(out.split())

def caption_image(flat, box, scale=4, pad=4):
    x, y, w, h = box
    H, W = flat.shape
    c = flat[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    c = cv2.copyMakeBorder(c, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return c

WL = 'MHLKCB-0123456789ABCDEFGHabcdefghist() '

def enhance(img):
    """Contrast stretch + gamma so faint grey letters become dark."""
    f = img.astype(float)
    lo = f.min()
    e = np.clip((f - lo) / max(1, 235 - lo), 0, 1) ** 2.2 * 255
    return e.astype(np.uint8)

def read_caption(flat, box):
    """Returns list of OCR strings from several preprocessing variants, plus a single-glyph
    read of the side letter (or '')."""
    from detect import side_info
    img = caption_image(flat, box)
    e = enhance(img)
    texts = [ocr_crop(e, 7, WL), ocr_crop(e, 7), ocr_crop(img, 7, WL)]
    if not any(ch.isdigit() for ch in ''.join(texts)):
        img2 = caption_image(flat, box, scale=3)
        _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        texts.append(ocr_crop(img2, 7, WL))
    side = ''
    si = side_info(flat, box)
    if si is not None:
        x0, x1 = si[1]
        x, y, w, h = box
        g = enhance(caption_image(flat, [x0, y, x1 - x0, h], scale=5, pad=2))
        side = ocr_crop(g, 10, 'ABCDEFGHabcdefgh')
    return texts, side
