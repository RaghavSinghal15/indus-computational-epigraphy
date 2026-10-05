"""Render a template drawing so it looks like a photo of a carved sign.

Why: the templates are clean line drawings, but real crops are photos of
carved stone lit from one side. A classifier trained only on drawings fails on
stone (A2 showed this for template matching). These renders are training data
that looks like the real thing.

How one render is made:

1. **Shape jitter.** Rotate a little, stretch (carved signs are often taller
   and narrower than Mahadevan's drawings), crop with a random margin, and
   thicken the strokes (carved strokes are wider, relative to the sign, than
   drawn lines; capped so detailed signs don't fill in).
2. **Height map.** Treat the image as a surface: stroke pixels are raised
   (an impression) or sunken (the seal itself), with rounded edges from a blur.
   Small random bumps are added everywhere, like the grain of stone or clay.
3. **Lighting.** Work out which way each point of the surface faces (its
   "normal", from the slope of the height map), and make it bright if it
   faces the light, dark if it faces away (Lambert's law). The light comes
   from a random direction, as in real photos.
4. **Colour and camera.** Tint with a random stone colour (grey, beige, gold,
   brown), vary the brightness across the image, add slight blur and noise.

Everything random comes from the `rng` you pass in, so a fixed seed gives the
same images every time.
"""

from __future__ import annotations

import cv2
import numpy as np

OUT_SIZE = 96   # renders are OUT_SIZE x OUT_SIZE before the final crop

# stone colours as BGR (OpenCV's order), picked to resemble the sample photos
STONE_COLOURS = [
    (200, 200, 200),   # light grey
    (150, 150, 150),   # mid grey
    (170, 195, 215),   # beige
    (90, 160, 205),    # gold / ochre
    (80, 110, 150),    # brown
    (225, 230, 235),   # white plaster
]


def _jitter_shape(strokes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotated and stretched copy of a binary stroke image (0 or 255)."""
    h, w = strokes.shape
    angle = rng.uniform(-10, 10)
    sx = rng.uniform(0.75, 1.1)                 # carved signs: often narrower...
    sy = rng.uniform(0.95, 1.25)                # ...and taller than the drawing
    # one affine matrix that rotates about the centre and scales x and y separately
    turn = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    stretch = np.array([[sx, 0, (1 - sx) * w / 2], [0, sy, (1 - sy) * h / 2]])
    m = stretch @ np.vstack([turn, [0, 0, 1]])
    return cv2.warpAffine(strokes, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def _thicken(shape: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Wider strokes, as carved strokes are much wider (relative to the sign) than drawn lines.

    The width is random, but capped so the ink at most doubles: otherwise
    detailed signs (a hatched oval) fill in and turn into a different sign.
    """
    ink = (shape > 127).sum()
    for k in range(int(rng.integers(2, 7)), 1, -1):          # try the random width, then thinner
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thick = cv2.dilate(shape, kernel)
        if (thick > 127).sum() <= 2.0 * ink:
            return thick
    return shape


def _trim_square(binary: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Crop to the strokes plus a random margin, padded to a square, resized to OUT_SIZE."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return cv2.resize(binary, (OUT_SIZE, OUT_SIZE))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    margin = int(side * rng.uniform(0.05, 0.25))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = side // 2 + margin
    padded = cv2.copyMakeBorder(binary, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    crop = padded[cy:cy + 2 * half, cx:cx + 2 * half]          # (cy, cx) shifted by the padding
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


def _smooth_noise(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    """Random bumps about `scale` pixels across, values roughly in [-1, 1]."""
    noise = rng.standard_normal(shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), scale)
    return noise / (noise.std() + 1e-6)


def render_carved(template_gray: np.ndarray, rng: np.random.Generator, sunken: bool | None = None) -> np.ndarray:
    """One photo-like BGR image (OUT_SIZE x OUT_SIZE) of the sign in `template_gray`.

    `sunken=True` carves the strokes into the surface (photo of a seal),
    False raises them (photo of an impression), None picks at random.
    """
    if sunken is None:
        sunken = bool(rng.random() < 0.5)
    _, strokes = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)
    shape = _thicken(_trim_square(_jitter_shape(strokes, rng), rng), rng).astype(np.float32) / 255
    return render_surface(shape, rng, sunken)


def render_surface(shape: np.ndarray, rng: np.random.Generator, sunken: bool) -> np.ndarray:
    """Steps 2-4 for any stroke image: `shape` is float in [0, 1] (1 = stroke), any size.
    Returns a BGR uint8 image of the same size, lit and coloured like stone."""
    # 1. height map: rounded strokes plus stone grain
    height = cv2.GaussianBlur(shape, (0, 0), rng.uniform(0.8, 1.8))
    if sunken:
        height = -height
    height = height * rng.uniform(3, 6)                           # how deep the carving is
    height += 0.15 * _smooth_noise(height.shape, rng.uniform(1, 3), rng)

    # 2. surface normals from the slope; Sobel estimates the slope in x and y
    dx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.dstack([-dx, -dy, np.full_like(height, 4.0)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)

    # 3. light from a random side, 25-60 degrees above the surface
    azimuth = rng.uniform(0, 2 * np.pi)
    elevation = np.radians(rng.uniform(25, 60))
    light = np.array([np.cos(azimuth) * np.cos(elevation),
                      np.sin(azimuth) * np.cos(elevation),
                      np.sin(elevation)], np.float32)
    lambert = np.clip(normal @ light, 0, 1)                      # brightness = cos(angle to light)
    # rescale so a flat patch of stone gets 0.9 whatever the light's height;
    # slopes facing the light go above that (highlights), slopes facing away below (shadows)
    shade = 0.3 + 0.6 * lambert / np.sin(elevation)

    # 4. stone colour, uneven brightness across the image, camera blur and noise
    colour = np.array(STONE_COLOURS[rng.integers(len(STONE_COLOURS))], np.float32)
    colour *= rng.uniform(0.85, 1.1)
    albedo = 1 + 0.08 * _smooth_noise(height.shape, rng.uniform(4, 12), rng)
    vignette = 1 + 0.15 * _smooth_noise(height.shape, 30, rng)
    image = shade[..., None] * albedo[..., None] * vignette[..., None] * colour
    image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.0))
    image += rng.normal(0, rng.uniform(2, 6), image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)
