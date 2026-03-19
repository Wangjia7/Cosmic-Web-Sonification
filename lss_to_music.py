#!/usr/bin/env python3
"""
lss_to_music.py — Cosmic Large-Scale Structure Image to Music (Orchestral Edition)

Designed for real cosmic web images. Three structural types map to three instrument groups:

  • Halo   (bright compact nodes)   → Harp
      Warm plucked resonance. Each halo triggers one harp note;
      large/massive halos → rich arpeggiated chords, small halos → bright high tones.

  • Filament (bright thread-like web) → Orchestral Strings (violins + violas)
      Four de-tuned string layers create the warm "ensemble" color.
      Filament direction and brightness drive the melodic contour.

  • Void   (large dark regions)      → Cello Ensemble
      Deep pizzicato entrance + sustained bowing; larger voids → lower, longer tones.

  • Background                       → Choir Pad (A major, "Ah" vowel formant simulation)
      Continuous choral atmosphere underpinning the entire piece.

All pitches are quantized to the A major scale (A/B/C#/D/E/F#/G#) so every
voice stays in harmony. An 8-tap concert-hall reverb is applied at the end.

Dependencies:
    pip install numpy Pillow scipy scikit-image

Usage:
    python lss_to_music.py <input.jpg> [output.wav] [duration_sec]
"""

import sys
import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, label, uniform_filter1d
from skimage.filters import frangi
from skimage.feature import blob_log
from skimage.morphology import skeletonize

SAMPLE_RATE    = 44100
MUSIC_DURATION = 40

# ── A major scale (A, B, C#, D, E, F#, G# across 5 octaves) ─────────────────
# A major's bright, uplifting character suits the blue-gold palette of cosmic webs.
A_MAJOR = np.array([
     55.00,  61.74,  69.30,  73.42,  82.41,  92.50,  98.00,   # A1~G#2
    110.00, 123.47, 138.59, 146.83, 164.81, 185.00, 196.00,   # A2~G#3
    220.00, 246.94, 277.18, 293.66, 329.63, 369.99, 392.00,   # A3~G#4
    440.00, 493.88, 554.37, 587.33, 659.25, 739.99, 783.99,   # A4~G#5
    880.00, 987.77,1108.73,1174.66,1318.51,1479.98,1567.98,   # A5~G#6
   1760.00,
])

HARP_SCALE   = A_MAJOR[7:]    # A2 and above  (harp range)
STRING_SCALE = A_MAJOR[7:28]  # A2~G#5        (comfortable string range)
CELLO_SCALE  = A_MAJOR[:14]   # A1~G#3        (cello low register)


def quantize(freq: float, scale: np.ndarray) -> float:
    return float(scale[np.argmin(np.abs(np.log2(scale / max(freq, 1e-9))))])


# ─────────────────────────────────────────────────────────────────────────────
#  Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str, max_dim: int = 512) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
#  Structure detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_halos(lum: np.ndarray, rgb: np.ndarray):
    h, w = lum.shape
    blobs = blob_log(np.clip(lum * 1.5, 0, 1),
                     min_sigma=2, max_sigma=min(min(h, w) // 8, 20),
                     num_sigma=8, threshold=0.02, overlap=0.5)
    halos = []
    for y, x, sigma in blobs:
        y, x   = int(y), int(x)
        r_pix  = max(1, int(sigma * 1.41))
        y0, y1 = max(0, y - r_pix), min(h, y + r_pix + 1)
        x0, x1 = max(0, x - r_pix), min(w, x + r_pix + 1)
        brightness = float(lum[y0:y1, x0:x1].mean())
        color = rgb[y0:y1, x0:x1].mean(axis=(0, 1))
        halos.append({"cx": x/w, "cy": y/h,
                      "sigma": float(sigma), "brightness": brightness,
                      "warmth": float(color[0] - color[2])})  # warmth = R-B; higher → more golden
    return sorted(halos, key=lambda h: -h["brightness"])


def detect_filaments(lum: np.ndarray):
    h, w   = lum.shape
    sigmas = np.arange(1.0, min(min(h, w) // 10, 10), 1.5)
    fmap   = normalize(frangi(lum, sigmas=sigmas, alpha=0.5, beta=0.5,
                               black_ridges=False))
    skel   = skeletonize(fmap > max(np.percentile(fmap[fmap > 0], 25), 0.04))
    labeled, n_cc = label(skel)
    segments = []
    for lab in range(1, n_cc + 1):
        coords = np.argwhere(labeled == lab)
        if len(coords) < 8:
            continue
        ys, xs = coords[:, 0], coords[:, 1]
        segments.append({
            "cx":         float(xs.mean()) / w,
            "x_min":      float(xs.min()) / w,
            "x_max":      float(xs.max()) / w,
            "pitch_norm": float(1.0 - ys.mean() / h),  # higher in image → higher pitch
            "length":     len(coords),
            "brightness": float(fmap[ys, xs].mean()),
        })
    col_act  = normalize(uniform_filter1d(fmap.sum(axis=0), size=max(1, w // 40)))
    col_y_wt = np.zeros(w)
    for x in range(w):
        s = fmap[:, x]; t = s.sum()
        if t > 1e-9:
            col_y_wt[x] = (s * np.arange(h)).sum() / t
    col_pitch = np.clip(1.0 - col_y_wt / h, 0, 1)
    return fmap, segments, col_act, col_pitch


def detect_voids(lum: np.ndarray):
    h, w   = lum.shape
    dark_t = max(np.percentile(lum, 40), float(lum.max()) * 0.08)
    dark   = lum < dark_t
    labeled, n_cc = label(dark)
    total  = h * w
    voids  = []
    for lab in range(1, n_cc + 1):
        coords = np.argwhere(labeled == lab)
        if len(coords) < total * 0.003:
            continue
        ys, xs = coords[:, 0], coords[:, 1]
        area_n = len(coords) / total
        # Larger void → lower pitch
        raw    = CELLO_SCALE[-1] - area_n * 2.5 * (CELLO_SCALE[-1] - CELLO_SCALE[0])
        freq   = quantize(float(np.clip(raw, CELLO_SCALE[0], CELLO_SCALE[-1])),
                          CELLO_SCALE)
        voids.append({
            "cx": float(xs.mean() / w),
            "x_min": float(xs.min() / w),
            "x_max": float(xs.max() / w),
            "area_norm": float(area_n),
            "freq": freq,
        })
    return sorted(voids, key=lambda v: -v["area_norm"])[:40]


# ─────────────────────────────────────────────────────────────────────────────
#  ① Harp  ── assigned to Halos
# ─────────────────────────────────────────────────────────────────────────────

def harp_note(freq: float, amp: float, dur: float) -> np.ndarray:
    """
    Plucked harp tone:
      - Harmonics 1–5, each with independent exponential decay
        (higher partials decay faster, mimicking real string behavior)
      - Very fast attack (2 ms) to simulate the pluck transient
      - Low strings sustain longer (~5 s); high strings decay quickly (~1.5 s)
      - No vibrato (plucked strings cannot sustain vibrato)
    """
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, endpoint=False)

    base_decay = max(0.8, 5.0 * (220.0 / max(freq, 110.0)) ** 0.5)  # lower pitch → longer sustain

    partials = [1,   2,   3,   4,   5  ]
    decays   = [1.0, 0.5, 0.3, 0.18, 0.10]
    weights  = [1.0, 0.50, 0.22, 0.10, 0.04]

    wave = np.zeros(n)
    for ratio, d_frac, w in zip(partials, decays, weights):
        decay = base_decay * d_frac
        env   = np.exp(-t / max(decay, 0.01))
        wave += w * np.sin(2 * np.pi * freq * ratio * t) * env

    # Fast attack envelope
    att = min(int(0.002 * SAMPLE_RATE), n)
    wave[:att] *= np.linspace(0, 1, att)

    return wave * amp * 0.42


def harp_chord(root_freq: float, amp: float, dur: float) -> np.ndarray:
    """
    Arpeggiated harp chord for large/massive halos (root + M3 + P5 + octave).
    Notes are staggered by 40 ms to create the characteristic harp roll.
    """
    intervals = [1.0, 1.2599, 1.4983, 2.0]   # root, major 3rd, perfect 5th, octave
    arp_delay = [0, int(0.04 * SAMPLE_RATE),
                    int(0.08 * SAMPLE_RATE),
                    int(0.12 * SAMPLE_RATE)]   # 40 ms stagger between notes
    n  = int(SAMPLE_RATE * dur)
    out = np.zeros(n)
    for ratio, delay in zip(intervals, arp_delay):
        f    = quantize(root_freq * ratio, HARP_SCALE)
        note = harp_note(f, amp * 0.7, dur)
        end  = min(delay + len(note), n)
        out[delay:end] += note[:end - delay]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  ② Orchestral Strings  ── assigned to Filaments
# ─────────────────────────────────────────────────────────────────────────────

def string_note(freq: float, amp: float, dur: float) -> np.ndarray:
    """
    Orchestral string section tone:
      - 4 de-tuned layers (±0–0.9%) to simulate multiple players in ensemble
      - Vibrato builds from 0 over 0.4 s (bow-induced vibrato onset)
      - Harmonic series approximated by sawtooth (1/k weights)
      - 120 ms bow attack → sustained → 180 ms release
    """
    n  = int(SAMPLE_RATE * dur)
    t  = np.linspace(0, dur, n, endpoint=False)

    vib_depth = 0.016 * np.clip(t / 0.40, 0, 1)
    vib       = 1 + vib_depth * np.sin(2 * np.pi * 5.6 * t)

    detune  = [0.000, +0.006, -0.005, +0.009]
    d_wts   = [1.000,  0.85,   0.75,   0.55]
    harms_s = np.arange(1, 9, dtype=np.float64).reshape(-1, 1)

    wave = np.zeros(n)
    for det, dw in zip(detune, d_wts):
        ph = (2 * np.pi * freq * (1 + det) * np.cumsum(vib) / SAMPLE_RATE).reshape(1, -1)
        # vectorized: (8, n) → sum over harmonics → (n,)
        wave += dw * np.sum((1.0 / harms_s) * np.sin(harms_s * ph), axis=0)

    att = min(int(0.12 * SAMPLE_RATE), n // 4)
    rel = min(int(0.18 * SAMPLE_RATE), n // 4)
    env = np.ones(n)
    env[:att]  = np.linspace(0, 1, att)
    env[-rel:] = np.linspace(1, 0, rel)

    return wave * env * amp * 0.13


def scan_strings_layer(col_act: np.ndarray,
                       col_pitch: np.ndarray,
                       duration: float) -> np.ndarray:
    """
    Continuous scanning string layer:
      - Tracks filament activity and pitch column by column
      - Pitch is quantized to A major and smoothed with uniform filter
        (prevents jarring jumps; creates portamento glide between positions)
      - Silent where filament activity drops below threshold,
        leaving space for other voices
    Vectorized cumulative-phase synthesis eliminates the inner Python loop.
    """
    n   = int(SAMPLE_RATE * duration)
    out = np.zeros(n)
    w   = len(col_act)
    spc = n / w

    # Quantize pitch per column then smooth
    raw_f = np.array([
        quantize(STRING_SCALE[0] + float(col_pitch[x]) * (STRING_SCALE[-1] - STRING_SCALE[0]),
                 STRING_SCALE)
        for x in range(w)
    ])
    raw_f = uniform_filter1d(raw_f, size=max(3, w // 25))

    phase     = 0.0
    vib_ph    = 0.0
    detune    = [0.000, +0.006, -0.005, +0.009]
    d_wts     = [1.000,  0.85,   0.75,   0.55]
    harmonics = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    h_wts     = (1.0 / harmonics).reshape(-1, 1)   # shape (6, 1)

    for x in range(w):
        act = float(col_act[x])
        if act < 0.05:
            phase = 0.0; vib_ph = 0.0
            continue
        freq = raw_f[x]
        s0   = int(x * spc); s1 = min(int((x + 1) * spc), n)
        n_c  = s1 - s0
        if n_c <= 0:
            continue

        t_c = np.arange(n_c, dtype=np.float32) / SAMPLE_RATE
        vib = 1.0 + 0.016 * np.sin(2 * np.pi * 5.6 * t_c + vib_ph)
        chunk = np.zeros(n_c, dtype=np.float32)

        for det, dw in zip(detune, d_wts):
            # Vectorized cumulative phase — no inner Python sample loop
            dp     = 2 * np.pi * freq * (1 + det) * vib / SAMPLE_RATE
            phases = phase + np.cumsum(dp)          # shape (n_c,)
            # All harmonics at once: (6, n_c) → sum axis 0
            layer  = np.sum(h_wts * np.sin(harmonics.reshape(-1, 1) * phases), axis=0)
            chunk += dw * layer

        att = min(int(0.04 * SAMPLE_RATE), n_c // 2)
        if att > 0:
            chunk[:att] *= np.linspace(0, 1, att, dtype=np.float32)

        out[s0:s1] += chunk * act * 0.10
        phase  = (2 * np.pi * freq * (s1 - s0) / SAMPLE_RATE + phase) % (2 * np.pi)
        vib_ph = (vib_ph + 2 * np.pi * 5.6 * n_c / SAMPLE_RATE) % (2 * np.pi)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  ③ Cello Ensemble  ── assigned to Voids
# ─────────────────────────────────────────────────────────────────────────────

def cello_pizzicato(freq: float, amp: float) -> np.ndarray:
    """
    Cello pizzicato — triggered at the leading edge of each void.
    Much deeper than the harp; body resonance adds thickness.
    Decay time scales with pitch: lower notes sustain longer.
    """
    dur  = 4.0 + (220.0 / max(freq, 40.0)) * 2.0   # low pitch → longer decay
    n    = int(SAMPLE_RATE * dur)
    t    = np.linspace(0, dur, n, endpoint=False)

    decay    = max(0.6, 3.0 * (110.0 / max(freq, 40.0)) ** 0.6)
    partials = [1,   2,   3,   4,  5  ]
    d_frac   = [1.0, 0.4, 0.25, 0.15, 0.08]
    weights  = [1.0, 0.55, 0.30, 0.15, 0.06]

    wave = np.zeros(n)
    for r, df, w in zip(partials, d_frac, weights):
        env   = np.exp(-t / max(decay * df, 0.01))
        wave += w * np.sin(2 * np.pi * freq * r * t) * env

    att = min(int(0.004 * SAMPLE_RATE), n)
    wave[:att] *= np.linspace(0, 1, att)

    # Body / box resonance at sub-fundamental
    body = 0.08 * np.sin(2 * np.pi * freq * 0.5 * t) * np.exp(-t / (decay * 1.5))
    wave += body

    return wave * amp * 0.50


def cello_sustain(freq: float, amp: float, dur: float) -> np.ndarray:
    """
    Sustained cello bowing — runs for the full horizontal span of a void.
    Three de-tuned layers + slow bow-pressure LFO give warmth without harshness.
    Gradual bow attack (300 ms, squared curve) and long release (400 ms).
    """
    n  = int(SAMPLE_RATE * dur)
    t  = np.linspace(0, dur, n, endpoint=False)

    vib_depth = 0.010 * np.clip(t / 0.5, 0, 1)
    vib       = 1 + vib_depth * np.sin(2 * np.pi * 4.8 * t)

    detune    = [0.000, +0.004, -0.003]
    d_wts     = [1.000,  0.70,   0.60]
    harms_c   = np.arange(1, 7, dtype=np.float64).reshape(-1, 1)

    wave = np.zeros(n)
    for det, dw in zip(detune, d_wts):
        ph   = (2 * np.pi * freq * (1 + det) * np.cumsum(vib) / SAMPLE_RATE).reshape(1, -1)
        wave += dw * np.sum((1.0 / harms_c) * np.sin(harms_c * ph), axis=0)

    # Very slow bow-pressure variation (breathing feel)
    bow_pressure = 0.85 + 0.15 * np.sin(2 * np.pi * 0.06 * t)
    wave *= bow_pressure

    att = min(int(0.30 * SAMPLE_RATE), n // 3)
    rel = min(int(0.40 * SAMPLE_RATE), n // 3)
    env = np.ones(n)
    env[:att]  = np.linspace(0, 1, att) ** 1.5
    env[-rel:] = np.linspace(1, 0, rel)

    return wave * env * amp * 0.20


# ─────────────────────────────────────────────────────────────────────────────
#  ④ Choir Pad  ── global background layer
# ─────────────────────────────────────────────────────────────────────────────

def choir_pad(duration: float, key_freq: float = 110.0) -> np.ndarray:
    """
    Choral atmosphere pad in A major ("Ah" vowel).
    Formant filtering (F1 ≈ 800 Hz, F2 ≈ 1200 Hz) simulates the "Ah" vowel
    characteristic of a choir. Multiple voice parts slowly evolve across
    the A major chord (A2, C#3, E3, A3) for the full duration.
    """
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n)

    # A major chord tones (A2, C#3, E3, A3, B3, E4)
    chord_freqs = [110.0, 138.59, 164.81, 220.00, 246.94, 293.66]
    chord_amps  = [1.00,  0.70,   0.80,   0.65,   0.45,   0.35]

    raw = np.zeros(n)
    for f, a in zip(chord_freqs, chord_amps):
        # Each voice has an independent slow LFO and slight vibrato
        lfo  = 0.65 + 0.35 * np.sin(2 * np.pi * 0.018 * t + f * 0.05)
        vib  = 1 + 0.008 * np.sin(2 * np.pi * 5.2 * t + f * 0.1)
        ph   = 2 * np.pi * f * np.cumsum(vib) / SAMPLE_RATE
        # Harmonic series with stronger odd harmonics (vocal cord characteristic)
        voice = (np.sin(ph) + 0.6 * np.sin(2*ph) + 0.8 * np.sin(3*ph)
                 + 0.3 * np.sin(4*ph) + 0.5 * np.sin(5*ph))
        raw += a * lfo * voice

    # Formant filtering to reinforce "Ah" vowel feel
    from scipy.signal import butter, lfilter

    def formant_boost(sig, center_hz, bw_hz, gain=1.6):
        nyq = SAMPLE_RATE / 2
        lo  = max(0.001, (center_hz - bw_hz / 2) / nyq)
        hi  = min(0.999, (center_hz + bw_hz / 2) / nyq)
        if lo >= hi:
            return sig
        b, a = butter(2, [lo, hi], btype='band')
        return sig + (gain - 1) * lfilter(b, a, sig)

    raw = formant_boost(raw, 800,  180, gain=1.5)   # F1
    raw = formant_boost(raw, 1200, 160, gain=1.3)   # F2

    # Global fade-in / fade-out
    fade = min(int(SAMPLE_RATE * 5.0), n // 4)
    raw[:fade]  *= np.linspace(0, 1, fade) ** 2
    raw[-fade:] *= np.linspace(1, 0, fade)

    return raw * 0.06


# ─────────────────────────────────────────────────────────────────────────────
#  ⑤ Concert Hall Reverb
# ─────────────────────────────────────────────────────────────────────────────

def concert_hall_reverb(audio: np.ndarray) -> np.ndarray:
    """
    Multi-tap comb-filter reverb simulating a large concert hall.
    Delay times are mutually co-prime (in ms) to avoid comb-filter coloration.
    """
    result = audio.copy()
    # (delay_seconds, decay_coefficient)
    taps = [
        (0.031, 0.42), (0.067, 0.34), (0.107, 0.27),
        (0.163, 0.21), (0.229, 0.16), (0.311, 0.11),
        (0.419, 0.07), (0.557, 0.04),
    ]
    for delay_s, decay in taps:
        d = int(delay_s * SAMPLE_RATE)
        if d < len(audio):
            result[d:] += audio[:len(audio) - d] * decay
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def lss_to_music(image_path: str,
                 output_path: str = "lss_music.wav",
                 duration: float  = MUSIC_DURATION) -> str:

    print(f"\n🌌  Loading image: {image_path}")
    rgb  = load_image(image_path)
    h, w = rgb.shape[:2]
    lum  = luminance(rgb)
    print(f"    Size: {w} × {h} px")

    n_samples = int(SAMPLE_RATE * duration)
    audio     = np.zeros(n_samples)

    # ── ① Halo → Harp ─────────────────────────────────────────────────────
    print("\n🔵  Detecting halos (Laplacian-of-Gaussian blob detection)...")
    halos = detect_halos(lum, rgb)
    max_h = min(len(halos), 150)
    print(f"    Found {len(halos)} halos  →  using top {max_h}")

    print("🎵  Synthesizing harp notes...")
    max_sigma = max((h["sigma"] for h in halos[:max_h]), default=1.0) + 1e-6
    for halo in halos[:max_h]:
        s_start  = int(halo["cx"] * duration * SAMPLE_RATE)

        # Small & bright → high clear harp note; large & bright → deep arpeggio chord
        s_norm   = float(np.clip(halo["sigma"] / max_sigma, 0, 1))
        b_norm   = float(np.clip(halo["brightness"], 0, 1))
        pitch_v  = np.clip(b_norm * 0.5 + (1 - s_norm) * 0.5, 0, 1)
        raw_freq = HARP_SCALE[0] + pitch_v * (HARP_SCALE[-1] - HARP_SCALE[0])
        freq     = quantize(float(raw_freq), HARP_SCALE)
        amp      = float(np.clip(0.35 + 0.65 * b_norm, 0, 1.0))

        # Large warm halos (high warmth + large sigma) → arpeggiated chord; small → single note
        use_chord = (s_norm > 0.5 and halo["warmth"] > 0.15)
        note_dur  = 4.0 + b_norm * 5.0 + s_norm * 3.0

        if use_chord:
            note = harp_chord(freq, amp, note_dur)
        else:
            note = harp_note(freq, amp, note_dur)

        end = min(s_start + len(note), n_samples)
        audio[s_start:end] += note[:end - s_start]

    # ── ② Filament → Orchestral Strings ───────────────────────────────────
    print("\n🟠  Detecting filaments (Frangi vesselness + skeletonization)...")
    fmap, segments, col_act, col_pitch = detect_filaments(lum)
    print(f"    Found {len(segments)} segments  |  coverage: {float((fmap>0.05).mean()*100):.1f}%")

    print("🎻  Synthesizing scan strings layer (A major quantized)...")
    audio += scan_strings_layer(col_act, col_pitch, duration)

    print("    Long-filament individual string notes...")
    for seg in sorted(segments, key=lambda s: -s["length"])[:60]:
        s_start  = int(seg["cx"] * duration * SAMPLE_RATE)
        raw_freq = STRING_SCALE[0] + seg["pitch_norm"] * (STRING_SCALE[-1] - STRING_SCALE[0])
        freq     = quantize(float(raw_freq), STRING_SCALE)
        amp      = float(np.clip(0.25 + 0.75 * seg["brightness"], 0, 1))
        note_dur = float(np.clip(seg["length"] / 120.0, 1.2, 7.0))
        note     = string_note(freq, amp, note_dur)
        end      = min(s_start + len(note), n_samples)
        audio[s_start:end] += note[:end - s_start]

    # ── ③ Void → Cello Ensemble ───────────────────────────────────────────
    print("\n⚫  Detecting voids (dark connected components)...")
    voids = detect_voids(lum)
    print(f"    Found {len(voids)} voids")

    print("🎸  Synthesizing cello pizzicato + sustained bowing...")
    for void in voids:
        t0      = void["x_min"] * duration
        t1      = void["x_max"] * duration
        s_start = int(t0 * SAMPLE_RATE)
        amp     = float(np.clip(0.4 + 0.6 * min(void["area_norm"] * 8, 1.0), 0, 1.0))

        # Entrance pizzicato
        pizz = cello_pizzicato(void["freq"], amp)
        end  = min(s_start + len(pizz), n_samples)
        audio[s_start:end] += pizz[:end - s_start]

        # Sustained bow for the void's full span
        sustain_dur = max(1.5, t1 - t0)
        sust = cello_sustain(void["freq"], amp * 0.8, sustain_dur)
        end  = min(s_start + len(sust), n_samples)
        audio[s_start:end] += sust[:end - s_start]

    # ── ④ Choir pad (full-length background) ─────────────────────────────
    print("\n🎤  Adding A major choir atmosphere pad...")
    audio += choir_pad(duration)

    # ── ⑤ Reverb ──────────────────────────────────────────────────────────
    print("🏛️  Applying concert hall reverb (8-tap comb filter)...")
    audio = concert_hall_reverb(audio)

    # ── Normalize & save ───────────────────────────────────────────────────
    print("\n💾  Normalizing and saving...")
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.88
    wavfile.write(output_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"✅  Saved: {output_path}")
    print(f"    Duration: {duration:.0f}s | {SAMPLE_RATE} Hz | 16-bit PCM WAV")
    print(f"\nVoice layers:")
    print(f"  🪗 Harp         ({max_h} notes, incl. arpeggiated chords for large halos)")
    print(f"  🎻 Strings      (scan layer + {min(len(segments),60)} long-filament notes)")
    print(f"  🎸 Cello        ({len(voids)} voids: pizzicato entrance + sustained bow)")
    print(f"  🎤 A major choir pad + concert hall reverb")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python lss_to_music.py <input.jpg> [output.wav] [duration_sec]")
        sys.exit(1)

    img_in  = sys.argv[1]
    wav_out = sys.argv[2] if len(sys.argv) > 2 else "lss_music.wav"
    dur     = float(sys.argv[3]) if len(sys.argv) > 3 else MUSIC_DURATION

    lss_to_music(img_in, wav_out, dur)
