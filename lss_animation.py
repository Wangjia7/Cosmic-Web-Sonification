#!/usr/bin/env python3
"""
lss_animation.py — Cosmic Large-Scale Structure Scan Animation

A scan line sweeps left-to-right, progressively revealing the three structural
types of the cosmic web, each with its own color scheme and animation effect:

  • Halo      → Gold/white pulsing glow rings (node flash), then persistent gold halo
  • Filament  → Orange/amber light trail; skeleton lights up along the scan direction
  • Void      → Deep-blue semi-transparent fill; blue border ripple when scan enters

Before the scan line: all three structures show faint "ghost" outlines.
As the scan line passes: each type bursts with its animation + label tag.
After the scan line: persistent colored markers remain.

Topology overlays run in sync:
  H0 (connected bright regions)  → gold glow blobs
  H1 (dark hole boundaries)      → cyan ring outlines

Dependencies:
    pip install numpy Pillow scipy scikit-image opencv-python imageio-ffmpeg

Usage:
    python lss_animation.py <input.jpg> <lss_music.wav> [output.mp4] [duration_sec]
"""

import sys
import os
import subprocess
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from skimage.filters import frangi
from skimage.feature import blob_log, peak_local_max
from skimage.morphology import skeletonize, binary_dilation, disk
import cv2


# ─── ffmpeg resolution ───────────────────────────────────────────────────────
def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


# ─────────────────────────────────────────────────────────────────────────────
#  Global parameters
# ─────────────────────────────────────────────────────────────────────────────
FPS          = 30
MAX_IMG_DIM  = 960

# Colors (RGB float)
COLOR_HALO   = np.array([1.00, 0.88, 0.25], dtype=np.float32)   # gold
COLOR_FILA   = np.array([1.00, 0.55, 0.10], dtype=np.float32)   # orange
COLOR_VOID   = np.array([0.12, 0.35, 0.90], dtype=np.float32)   # deep blue
COLOR_SCAN   = np.array([0.80, 0.95, 1.00], dtype=np.float32)   # white-blue scan line

GHOST_ALPHA  = 0.10   # ghost outline brightness before scan
SETTLE_ALPHA = 0.30   # settled brightness after scan passes
FLASH_PEAK   = 1.40   # peak multiplier at scan burst

HALO_FLASH_F = 0.06   # halo flash window width (fraction of image width)
FILA_GLOW    = 0.06   # filament scan glow width (fraction of image width)
VOID_PULSE_F = 0.08   # void border pulse width (fraction of image width)

MAX_HALOS    = 80
MAX_TOPO     = 20


# ─────────────────────────────────────────────────────────────────────────────
#  Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMG_DIM:
        scale = MAX_IMG_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def to_gray3(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to a 3-channel grayscale (slightly brightened)."""
    lum = np.clip(luminance(rgb) * 1.12, 0, 1)
    return np.stack([lum, lum, lum], axis=-1)


def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
#  Structure detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_halos(lum: np.ndarray, rgb: np.ndarray):
    h, w = lum.shape
    lum_eq = np.clip(lum * 1.5, 0, 1)
    blobs  = blob_log(lum_eq, min_sigma=3, max_sigma=min(h, w) // 8,
                      num_sigma=8, threshold=0.025, overlap=0.5)
    halos = []
    for y, x, sigma in blobs:
        y, x    = int(y), int(x)
        r_pix   = max(1, int(sigma * 1.41))
        y0, y1  = max(0, y - r_pix), min(h, y + r_pix + 1)
        x0, x1  = max(0, x - r_pix), min(w, x + r_pix + 1)
        brightness = float(lum[y0:y1, x0:x1].mean())
        color   = rgb[y0:y1, x0:x1].mean(axis=(0, 1))
        halos.append({
            "px": x, "py": y,
            "cx": float(x / w), "cy": float(y / h),
            "sigma": float(sigma),
            "brightness": brightness,
            "color": color,
            "radius": max(4, int(sigma * 1.8)),
        })
    return sorted(halos, key=lambda h: -h["brightness"])[:MAX_HALOS]


def detect_filaments(lum: np.ndarray):
    """Return Frangi response map, skeleton mask, and dilated skeleton (for display)."""
    h, w   = lum.shape
    sigmas = np.arange(1.5, min(h, w) // 10, 1.5)
    fmap   = normalize(frangi(lum, sigmas=sigmas, alpha=0.5, beta=0.5,
                               black_ridges=False))
    skel_t = max(np.percentile(fmap[fmap > 0], 30), 0.05)
    binary = fmap > skel_t
    skel   = skeletonize(binary)
    # Dilate skeleton for visual display (2–3 px width)
    thick_skel = binary_dilation(skel, disk(1))
    return fmap, skel, thick_skel


def detect_voids(lum: np.ndarray):
    """Return void boolean mask (H×W) and sorted list of void dicts."""
    h, w    = lum.shape
    dark_t  = max(np.percentile(lum, 42), float(lum.max()) * 0.08)
    dark    = lum < dark_t
    labeled, n_cc = label(dark)
    min_area = h * w * 0.005

    voids = []
    void_mask = np.zeros((h, w), dtype=bool)
    for lab in range(1, n_cc + 1):
        coords = np.argwhere(labeled == lab)
        if len(coords) < min_area:
            continue
        ys, xs = coords[:, 0], coords[:, 1]
        area_n = len(coords) / (h * w)
        void_mask[ys, xs] = True
        voids.append({
            "px": int(xs.mean()),
            "py": int(ys.mean()),
            "cx": float(xs.mean() / w),
            "x_min_px": int(xs.min()),
            "x_max_px": int(xs.max()),
            "area_norm": float(area_n),
            "coords": coords,
        })
    return void_mask, sorted(voids, key=lambda v: -v["area_norm"])[:30]


# ─────────────────────────────────────────────────────────────────────────────
#  Topology detection (H0 / H1, persistent homology approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _sublevel_h0(lum, n_levels=30):
    """Approximate H0 features via superlevel-set connected-component tracking."""
    thresholds = np.linspace(lum.max(), lum.min(), n_levels + 1)
    seen = {}; features = []
    h, w = lum.shape
    for t in thresholds:
        mask        = lum >= t
        lbl, _      = label(mask)
        cur         = set(np.unique(lbl[mask])) - {0}
        dead        = set(seen.keys()) - cur
        for lab in list(dead):
            birth = seen.pop(lab)
            pers  = birth - t
            if pers < 0.05: continue
            ys, xs = np.where(lbl == lab)
            if not len(ys): continue
            features.append({"cx": float(xs.mean()/w), "cy": float(ys.mean()/h),
                             "persistence": float(pers), "area": int(len(ys)),
                             "type": "H0"})
        for lab in cur:
            if lab not in seen: seen[lab] = t
    return features


def _sublevel_h1(lum, n_scales=5):
    """
    Approximate H1 features (dark loops / holes) using multi-scale local minima
    in the inverted luminance map.
    Ring radius is set by distance_transform_edt (distance to nearest bright pixel)
    so each hole gets a physically meaningful size.
    """
    h, w = lum.shape
    bright_mask = lum > np.percentile(lum, 55)
    dist_map    = distance_transform_edt(~bright_mask)
    features = []; used = []
    for scale in np.linspace(8, max(8, min(h, w) // 6), n_scales):
        blurred = gaussian_filter(lum, sigma=scale)
        inv     = 1.0 - blurred
        coords  = peak_local_max(inv, min_distance=int(scale*1.5), threshold_abs=0.15)
        for y, x in coords:
            depth = float(inv[y, x])
            pers  = depth * 0.6
            if pers < 0.08: continue
            cx, cy = float(x/w), float(y/h)
            if any(abs(cx-u[0])<0.07 and abs(cy-u[1])<0.07 for u in used): continue
            used.append((cx, cy))
            eff_r = int(np.clip(dist_map[y, x], 5, min(h, w)//4))
            features.append({"cx": cx, "cy": cy, "persistence": pers,
                             "area": 0, "scale": float(scale),
                             "eff_r": eff_r, "type": "H1"})
    return features


def extract_topology(lum, rgb, max_feats=MAX_TOPO):
    all_f = _sublevel_h0(lum) + _sublevel_h1(lum)
    all_f.sort(key=lambda f: -f["persistence"])
    h_img, w_img = lum.shape
    for i, f in enumerate(all_f[:max_feats]):
        f["idx"] = i
        px = int(np.clip(f["cx"]*w_img, 0, w_img-1))
        py = int(np.clip(f["cy"]*h_img, 0, h_img-1))
        f["px"] = px; f["py"] = py
    return all_f[:max_feats]


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-computed visual overlays
# ─────────────────────────────────────────────────────────────────────────────

def _gauss_patch(px, py, h, w, radius, color):
    """Build a Gaussian glow patch centered at (px, py) with given color."""
    y0 = max(0, py-radius); y1 = min(h, py+radius+1)
    x0 = max(0, px-radius); x1 = min(w, px+radius+1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-((yy-py)**2+(xx-px)**2)/max(1,(radius*0.4)**2))
    return g[:,:,np.newaxis]*color, y0, y1, x0, x1


def _ring_patch(px, py, h, w, ring_r, color):
    """Build a ring / annulus patch centered at (px, py)."""
    pad = ring_r+6
    y0 = max(0,py-pad); y1 = min(h,py+pad+1)
    x0 = max(0,px-pad); x1 = min(w,px+pad+1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.sqrt((yy-py)**2+(xx-px)**2)
    rw   = max(2.0, ring_r*0.15)
    ring = np.exp(-((dist-ring_r)**2)/(rw**2))
    return ring[:,:,np.newaxis]*color, y0, y1, x0, x1


def build_halo_overlays(halos, h, w):
    """Pre-compute one overlay dict per halo (mixed gold + actual halo color)."""
    ovs = []
    for i, halo in enumerate(halos):
        px, py = halo["px"], halo["py"]
        r      = halo["radius"]
        # Blend gold color with the halo's actual detected color
        star_col = np.clip(halo["color"] * 1.4, 0, 1).astype(np.float32)
        mixed    = np.clip(COLOR_HALO * 0.6 + star_col * 0.4, 0, 1)
        patch, y0, y1, x0, x1 = _gauss_patch(px, py, h, w, r, mixed)
        ovs.append({
            "idx": i, "px": px, "py": py, "cx": halo["cx"],
            "brightness": halo["brightness"],
            "patch": patch.astype(np.float32),
            "y0": y0, "y1": y1, "x0": x0, "x1": x1,
            "radius": r,
            "label": f"H{i:02d}",
            "label_color": (int(COLOR_HALO[2]*255),
                            int(COLOR_HALO[1]*255),
                            int(COLOR_HALO[0]*255)),
            "label_pos": (max(2, px-r-2), max(12, py-r-4)),
        })
    return ovs


def build_topo_overlays(features, h, w):
    """Pre-compute overlay dicts for H0 (glow blobs) and H1 (ring outlines)."""
    ovs = []
    if not features: return ovs
    max_p = max(f["persistence"] for f in features) + 1e-9
    for f in features:
        px, py = f["px"], f["py"]
        pers   = f["persistence"]
        norm_p = pers / max_p
        if f["type"] == "H0":
            r = int(np.clip(np.sqrt(max(f["area"],1)/np.pi)*1.8, 8, min(h,w)//5))
            patch, y0, y1, x0, x1 = _gauss_patch(px, py, h, w, r, COLOR_HALO*0.7)
            ring_r = None
        else:
            ring_r = int(np.clip(f.get("eff_r",12)*0.9, 6, min(h,w)//4))
            patch, y0, y1, x0, x1 = _ring_patch(px, py, h, w, ring_r,
                                                  np.array([0.1,0.85,1.0],dtype=np.float32))
            r = ring_r
        lbl_col = ((int(COLOR_HALO[2]*255), int(COLOR_HALO[1]*255), int(COLOR_HALO[0]*255))
                   if f["type"]=="H0"
                   else (255, int(0.85*255), int(0.1*255)))
        ovs.append({
            "idx": f["idx"], "type": f["type"],
            "px": px, "py": py, "cx": f["cx"],
            "norm_p": norm_p, "ring_r": ring_r,
            "patch": patch.astype(np.float32),
            "y0": y0, "y1": y1, "x0": x0, "x1": x1,
            "label_text": f"{f['type']}-{f['idx']:02d}",
            "label_color": lbl_col,
            "label_pos": (max(2, px-r-2), max(12, py-r-4)),
        })
    return ovs


# ─────────────────────────────────────────────────────────────────────────────
#  Single frame renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(scan_x: int,
                 rgb: np.ndarray,
                 gray: np.ndarray,
                 lum: np.ndarray,
                 fmap: np.ndarray,
                 thick_skel: np.ndarray,
                 void_mask: np.ndarray,
                 voids: list,
                 halo_ovs: list,
                 topo_ovs: list,
                 halo_flash_w: int,
                 void_pulse_w: int,
                 fila_glow_w: int) -> np.ndarray:

    h, w = rgb.shape[:2]

    # ── Base: color left of scan, grayscale right ─────────────────────────────
    frame = gray.copy()
    if scan_x > 0:
        frame[:, :scan_x] = rgb[:, :scan_x]
    blend_w = max(4, w // 60)
    if scan_x > 0:
        bx0, bx1 = max(0, scan_x - blend_w), min(w, scan_x)
        alpha = np.linspace(0, 1, bx1-bx0, dtype=np.float32)[np.newaxis, :, np.newaxis]
        frame[:, bx0:bx1] = alpha*rgb[:, bx0:bx1] + (1-alpha)*gray[:, bx0:bx1]

    # ── Void blue fill (revealed void pixels tinted deep-blue) ───────────────
    if scan_x > 0:
        revealed_void = void_mask[:, :scan_x]          # H × scan_x bool
        region        = frame[:, :scan_x]               # H × scan_x × 3
        region[revealed_void] = np.clip(
            region[revealed_void] * 0.55 + COLOR_VOID * 0.45, 0, 1)
        frame[:, :scan_x] = region

    # ── Void border pulse (when scan line enters / crosses a void) ────────────
    void_labels_to_draw = []
    for void in voids:
        dist = scan_x - void["x_min_px"]
        if 0 <= dist <= void_pulse_w:
            progress = dist / void_pulse_w
            # Blue border flash at the void's leading edge column
            cx = void["x_min_px"]
            if 0 <= cx < w:
                intensity = np.exp(-progress * 3.0) * 0.8
                glow_col  = COLOR_VOID * intensity
                frame[:, max(0, cx-1):min(w, cx+2)] = np.clip(
                    frame[:, max(0, cx-1):min(w, cx+2)] + glow_col, 0, 1)
            if progress < 0.3:
                void_labels_to_draw.append(
                    (f"V{voids.index(void):02d}",
                     (max(2, void["px"]-20), max(12, void["py"])),
                     (int(COLOR_VOID[2]*255), int(COLOR_VOID[1]*255), int(COLOR_VOID[0]*255))))

    # ── Filament orange glow (active skeleton columns near the scan line) ─────
    for dx in range(-fila_glow_w, fila_glow_w + 1):
        cx = scan_x + dx
        if not (0 <= cx < w): continue
        decay    = np.exp(-abs(dx) / max(1, fila_glow_w * 0.4))
        col_fmap = fmap[:, cx]
        active   = col_fmap > 0.05
        if not active.any(): continue
        glow_str = col_fmap[active] * decay * 0.9
        frame[active, cx] = np.clip(
            frame[active, cx] + glow_str[:, np.newaxis] * COLOR_FILA, 0, 1)

    # ── Filament skeleton persistent display (left of scan line) ─────────────
    if scan_x > 2:
        skel_col = thick_skel[:, :scan_x]
        frame[:, :scan_x][skel_col] = np.clip(
            frame[:, :scan_x][skel_col] * 0.4 + COLOR_FILA * 0.6, 0, 1)

    # ── Halo overlay (three states: ghost / flash / settled) ─────────────────
    halo_labels = []
    for ov in halo_ovs:
        dx   = scan_x - ov["px"]
        norm = ov["brightness"]
        if dx < -(halo_flash_w * 2):
            alpha = GHOST_ALPHA * 0.4 * norm
        elif dx < 0:
            frac  = (dx + halo_flash_w * 2) / (halo_flash_w * 2)
            alpha = GHOST_ALPHA * 0.4 * norm + frac * GHOST_ALPHA * norm
        elif dx < halo_flash_w:
            frac  = dx / halo_flash_w
            alpha = FLASH_PEAK * np.exp(-frac * 4.0) * norm
            if frac < 0.4:
                halo_labels.append((ov["label"], ov["label_pos"], ov["label_color"]))
        else:
            alpha = SETTLE_ALPHA * norm
        y0, y1, x0, x1 = ov["y0"], ov["y1"], ov["x0"], ov["x1"]
        frame[y0:y1, x0:x1] = np.clip(frame[y0:y1, x0:x1] + ov["patch"] * alpha, 0, 1)

    # ── Topology overlay (three states: ghost / flash / settled) ─────────────
    topo_labels = []
    rings_to_pulse = []
    for ov in topo_ovs:
        dx   = scan_x - ov["px"]
        np_  = ov["norm_p"]
        approach = int(w * 0.06)
        flash_w  = int(w * 0.10)
        if dx < -(approach + flash_w):
            alpha = GHOST_ALPHA * 0.4 * np_
        elif dx < 0:
            frac  = (dx + approach + flash_w) / (approach + flash_w)
            alpha = GHOST_ALPHA * np_ * (0.4 + np.clip(frac, 0, 1) * 0.6)
        elif dx < flash_w:
            frac  = dx / flash_w
            alpha = 1.0 * np.exp(-frac * 3.5) * np_
            if frac < 0.35:
                topo_labels.append((ov["label_text"], ov["label_pos"], ov["label_color"]))
            if ov["type"] == "H1":
                rings_to_pulse.append((ov, frac))
        else:
            alpha = SETTLE_ALPHA * np_
        y0, y1, x0, x1 = ov["y0"], ov["y1"], ov["x0"], ov["x1"]
        frame[y0:y1, x0:x1] = np.clip(frame[y0:y1, x0:x1] + ov["patch"] * alpha, 0, 1)

    # ── Scan line glow ────────────────────────────────────────────────────────
    if 0 < scan_x < w:
        for dx in range(-5, 6):
            lx = scan_x + dx
            if 0 <= lx < w:
                intensity  = np.exp(-abs(dx) * 0.5) * 0.8
                frame[:, lx] = np.clip(frame[:, lx] + COLOR_SCAN * intensity, 0, 1)

    # ── Convert to uint8 BGR for OpenCV ──────────────────────────────────────
    frame_u8  = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)

    # ── H1 topology pulsing rings (drawn in BGR after conversion) ────────────
    for ov, progress in rings_to_pulse:
        base_r  = ov["ring_r"] if ov["ring_r"] else 12
        pulse_r = int(base_r * (1 + progress * 1.8))
        a       = max(0.0, 1.0 - progress)
        color   = (int(0.1*255*a), int(0.85*255*a), int(255*a))
        cv2.circle(frame_bgr, (ov["px"], ov["py"]), pulse_r, color,
                   max(1, int(2*(1-progress*0.8))), lineType=cv2.LINE_AA)

    # ── Text labels ───────────────────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, pos, color in halo_labels + topo_labels + void_labels_to_draw:
        px_l, py_l = pos
        cv2.putText(frame_bgr, text, (px_l-1, py_l-1), font, 0.36,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (px_l, py_l), font, 0.36,
                    color, 1, cv2.LINE_AA)

    return frame_bgr


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_animation(image_path: str,
                       audio_path: str,
                       output_path: str = "lss_animation.mp4",
                       duration: float  = 40.0):

    print(f"\n🌌  Loading image: {image_path}")
    rgb  = load_image(image_path)
    h, w = rgb.shape[:2]
    lum  = luminance(rgb)
    gray = to_gray3(rgb)
    print(f"    Size: {w} × {h} px")

    # ── Structure detection ────────────────────────────────────────────────────
    print("🔵  Detecting halos (LoG blob detection)...")
    halos = detect_halos(lum, rgb)
    print(f"    {len(halos)} halos found")

    print("🟠  Detecting filaments (Frangi + skeletonization)...")
    fmap, skel, thick_skel = detect_filaments(lum)
    print(f"    Filament coverage: {float((fmap>0.05).mean()*100):.1f}%  |  skeleton pixels: {int(skel.sum())}")

    print("⚫  Detecting voids (dark connected components)...")
    void_mask, voids = detect_voids(lum)
    print(f"    {len(voids)} voids  |  coverage: {float(void_mask.mean()*100):.1f}%")

    print("🔬  Extracting topology (H0/H1 persistent homology)...")
    topo_features = extract_topology(lum, rgb, MAX_TOPO)
    h0_n = sum(1 for f in topo_features if f["type"]=="H0")
    h1_n = sum(1 for f in topo_features if f["type"]=="H1")
    print(f"    H0={h0_n} (gold glow blobs)   H1={h1_n} (cyan ring outlines)")

    # ── Pre-compute overlays ───────────────────────────────────────────────────
    print("🎨  Pre-computing visual overlays...")
    halo_ovs = build_halo_overlays(halos, h, w)
    topo_ovs = build_topo_overlays(topo_features, h, w)

    halo_flash_w = max(6, int(w * HALO_FLASH_F))
    void_pulse_w = max(8, int(w * VOID_PULSE_F))
    fila_glow_w  = max(4, int(w * FILA_GLOW))

    # ── Render frames ──────────────────────────────────────────────────────────
    n_frames  = int(FPS * duration)
    tmp_path  = output_path.replace(".mp4", "_noaudio.mp4")
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(tmp_path, fourcc, FPS, (w, h))

    print(f"\n🎞️  Rendering {n_frames} frames ({FPS} fps × {duration:.0f}s)...")
    print(f"    Layers: base + halos({len(halo_ovs)}) + filaments + voids({len(voids)}) + topology({len(topo_features)})")

    report = max(1, n_frames // 10)
    for fi in range(n_frames):
        progress = fi / max(1, n_frames - 1)
        scan_x   = int(progress * (w - 1))

        frame_bgr = render_frame(
            scan_x, rgb, gray, lum,
            fmap, thick_skel, void_mask, voids,
            halo_ovs, topo_ovs,
            halo_flash_w, void_pulse_w, fila_glow_w
        )
        writer.write(frame_bgr)

        if fi % report == 0:
            pct = int(100 * fi / n_frames)
            bar = "█"*(pct//10) + "░"*(10-pct//10)
            print(f"    [{bar}] {pct}%  frame {fi}/{n_frames}")

    writer.release()
    print("    [██████████] 100%  all frames rendered")

    # ── Merge audio track ──────────────────────────────────────────────────────
    if os.path.exists(audio_path):
        print(f"\n🔊  Merging audio: {audio_path}")
        cmd = [_get_ffmpeg(), "-y", "-loglevel", "error",
               "-i", tmp_path, "-i", audio_path,
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-shortest", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.remove(tmp_path)
            print(f"✅  Done: {output_path}")
        else:
            os.rename(tmp_path, output_path)
            print(f"⚠️  ffmpeg merge failed — video saved without audio")
    else:
        os.rename(tmp_path, output_path)
        print(f"⚠️  No audio file found — video saved without audio: {output_path}")

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"    {size_mb:.1f} MB  |  {w}×{h} @ {FPS}fps  |  {duration:.0f}s")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python lss_animation.py <input.jpg> <lss_music.wav> [output.mp4] [duration_sec]")
        sys.exit(1)

    img_path = sys.argv[1]
    wav_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "lss_animation.mp4"
    dur      = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0

    generate_animation(img_path, wav_path, out_path, dur)
