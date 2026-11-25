"""Background music for the project video, synthesised from scratch (no samples, no licences).

Maqam Hijaz on D (D Eb F# G A Bb C): a low drone, a plucked oud-like line (Karplus-Strong
strings), a soft frame drum in the maqsum rhythm and a simple reverb. Writes a 48 kHz stereo WAV.
    python scripts/make_music.py 171.05 data/processed/video/music.wav
"""
import sys
import wave

import numpy as np
from scipy.signal import fftconvolve, lfilter   # lfilter runs a recursive filter fast in C

SR = 48000
RNG = np.random.default_rng(7)
D3 = 146.83
HIJAZ = [0, 1, 4, 5, 7, 8, 10]          # semitones above D: the Eb-F# step is the maqam's sound


def hz(step):
    """Frequency of scale degree `step` (0 = D3; 7 = D4; negative goes down)."""
    octave, k = divmod(step, 7)
    return D3 * 2 ** (octave + HIJAZ[k] / 12)


def pluck(f, dur, bright=0.6):
    """One Karplus-Strong string: a burst of noise fed through a tuned, slightly lossy delay line."""
    n = int(dur * SR)
    period = SR / f
    N = int(period - 0.5)                                         # the 3-tap loss filter below adds 0.5 + frac
    frac = period - 0.5 - N
    burst = RNG.uniform(-1, 1, N)
    burst = lfilter([1 - bright], [1, -bright], burst)           # one-pole low-pass: a warm, oud-like pick
    burst -= burst.mean()                                         # no DC: the loop would hold it forever
    x = np.zeros(n)
    x[:N] = burst
    g = 0.996
    # y[n] = x[n] + g * ((1-frac)*0.5*(y[n-N] + y[n-N-1]) ...): a delay with a two-tap averaging loss
    a = np.zeros(N + 3)
    a[0] = 1
    a[N] -= g * 0.5 * (1 - frac)
    a[N + 1] -= g * 0.5
    a[N + 2] -= g * 0.5 * frac
    y = lfilter([1], a, x)
    y = lfilter([1, -1], [1, -0.995], y)                          # DC blocker
    y *= np.exp(-np.arange(n) / SR * 1.8)
    return y / (np.abs(y).max() + 1e-9)


def drone(total):
    """Low D and A with a few harmonics, slow swells and a touch of detune (two channels)."""
    t = np.arange(int(total * SR)) / SR
    out = np.zeros((2, t.size))
    for f, amp in [(D3 / 2, 1.0), (D3 * 0.75, 0.45), (D3, 0.35)]:
        for ch, det in enumerate((-0.25, 0.25)):
            tone = sum(np.sin(2 * np.pi * (f + det) * h * t + h) / h ** 1.6 for h in range(1, 6))
            out[ch] += amp * tone
    swell = 0.75 + 0.25 * np.sin(2 * np.pi * t / 13.0)
    return out * swell * 0.06


def drum(total, bpm=84, start=13.5):
    """Frame drum, maqsum pattern (dum tek - tek dum - tek -), soft; enters after the title."""
    out = np.zeros(int(total * SR))
    beat = 60 / bpm / 2                                             # eighth notes
    pattern = ["D", "T", "-", "T", "D", "-", "T", "-"]
    t0 = start
    k = 0
    while t0 < total - 1:
        hit = pattern[k % 8]
        i = int(t0 * SR)
        if hit == "D":
            n = int(0.45 * SR)
            tt = np.arange(n) / SR
            f = 55 + 45 * np.exp(-tt * 30)                         # pitch falls as the skin settles
            s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 7)
            amp = 0.55
        elif hit == "T":
            n = int(0.12 * SR)
            tt = np.arange(n) / SR
            s = lfilter([1, -1], [1], RNG.normal(0, 1, n)) * np.exp(-tt * 45) * 0.25
            s += np.sin(2 * np.pi * 420 * tt) * np.exp(-tt * 40) * 0.3
            amp = 0.35 if k % 16 < 8 else 0.28
        else:
            k += 1
            t0 += beat
            continue
        j = min(out.size, i + n)
        out[i:j] += amp * s[:j - i]
        k += 1
        t0 += beat * (1 + RNG.normal(0, 0.01))                     # tiny human timing drift
    return out


def melody(total, start=2.0):
    """Phrases that wander the maqam and fall back to D, with rests between them."""
    out = np.zeros(int(total * SR))
    t = start
    while t < total - 6:
        length = RNG.integers(5, 9)
        step = int(RNG.choice([4, 5, 7, 3]))
        notes = []
        for j in range(length):
            notes.append(step)
            step += int(RNG.choice([-2, -1, -1, 1, 1, 2]))
            step = int(np.clip(step, -1, 9))
        notes += [2, 1, 0]                                         # cadence: F#, Eb, D
        for j, s in enumerate(notes):
            d = float(RNG.choice([0.36, 0.36, 0.72, 0.54]))
            if j == len(notes) - 1:
                d = 1.6
            f = hz(s)
            n = pluck(f, d + 1.2, bright=0.5)
            i = int(t * SR)
            k = min(out.size, i + n.size)
            out[i:k] += 0.5 * n[:k - i]
            if d >= 0.7 and RNG.random() < 0.35:                    # tremolo (risha) on long notes
                for r in range(1, 4):
                    i2 = int((t + r * 0.09) * SR)
                    k2 = min(out.size, i2 + n.size)
                    out[i2:k2] += 0.18 * n[:k2 - i2]
            t += d
        t += float(RNG.uniform(2.5, 4.5))                          # breathe between phrases
    return out


def reverb(x, secs=2.2, wet=0.28):
    """Convolve with decaying noise: a cheap room."""
    n = int(secs * SR)
    ir = RNG.normal(0, 1, n) * np.exp(-np.arange(n) / SR * 3.2)
    ir[:int(0.02 * SR)] = 0
    y = fftconvolve(x, ir)[:x.size]
    y *= np.abs(x).max() / (np.abs(y).max() + 1e-9)
    return (1 - wet) * x + wet * y


def main(total, path):
    mel = reverb(melody(total))
    drm = reverb(drum(total), 1.2, 0.18)
    dr = drone(total)
    left = dr[0] + 0.55 * mel + 0.5 * drm
    right = dr[1] + 0.45 * mel + 0.5 * drm
    mix = np.stack([left, right])
    t = np.arange(mix.shape[1]) / SR
    env = np.clip(t / 3.0, 0, 1) * np.clip((total - t) / 4.0, 0, 1)   # fade in 3 s, out 4 s
    mix *= env
    mix /= np.abs(mix).max() / 0.9
    pcm = (mix.T * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", path, f"{total:.1f} s")


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
