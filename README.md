# Cosmic Web Sonification

**Turn cosmic large-scale structure images into orchestral music with synchronized animations.**

This project maps the three structural types of the cosmic web — *halos*, *filaments*, and *voids* — to a live orchestra, then renders a scan-line animation that reveals the image in sync with the generated audio.

---

## Demo

The scan line sweeps left to right. As it passes each structure, that region bursts into color and its assigned instrument sounds. Persistent homology topology features (H0 connected components, H1 dark-loop boundaries) are overlaid as gold blobs and cyan rings throughout.

---

## Instrument Mapping

| Structure | Detection method | Instrument | Character |
|---|---|---|---|
| **Halo** (compact bright node) | Laplacian-of-Gaussian blob (`blob_log`) | **Harp** | Single plucked note; large warm halos trigger arpeggiated chords |
| **Filament** (thread-like web) | Frangi vesselness filter + skeletonization | **Orchestral Strings** | 4-layer de-tuned ensemble; filament orientation drives pitch contour |
| **Void** (large dark region) | Dark connected-component labeling | **Cello Ensemble** | Pizzicato entrance + sustained bowing; larger void → deeper, longer tone |
| **Background** | — | **Choir Pad** | A major "Ah" vowel (formant-filtered harmonics), full-length atmosphere |

All pitches are quantized to the **A major scale** (A/B/C♯/D/E/F♯/G♯) so every voice stays in harmony. An **8-tap concert-hall reverb** is applied to the final mix.

---

## Files

```
lss_to_music.py     — Image → WAV audio synthesis
lss_animation.py    — Image + WAV → MP4 scan animation
```

---

## Requirements

```bash
pip install numpy Pillow scipy scikit-image opencv-python imageio-ffmpeg
```

`imageio-ffmpeg` bundles its own `ffmpeg` binary so no system-level installation is needed.

---

## Quick Start

### 1. Generate music

```bash
python lss_to_music.py cosmic_web.jpg output.wav 40
```

Arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `<input.jpg>` | yes | — | Input image (JPG/PNG, any size) |
| `[output.wav]` | no | `lss_music.wav` | Output WAV path |
| `[duration_sec]` | no | `40` | Music duration in seconds |

### 2. Generate animation

```bash
python lss_animation.py cosmic_web.jpg output.wav animation.mp4 40
```

Arguments:

| Argument | Required | Default | Description |
|---|---|---|---|
| `<input.jpg>` | yes | — | Same input image |
| `<lss_music.wav>` | yes | — | WAV produced by step 1 |
| `[output.mp4]` | no | `lss_animation.mp4` | Output MP4 path |
| `[duration_sec]` | no | `40` | Must match the audio duration |

---

## How It Works

### Audio (`lss_to_music.py`)

**Structure detection**

- *Halos* are found with `skimage.feature.blob_log` (Laplacian-of-Gaussian). Each blob's horizontal position sets its onset time, brightness sets amplitude, and size+brightness together determine pitch. Warm (gold-tinted) large halos trigger arpeggiated major chords (root + M3 + P5 + octave) instead of single notes.
- *Filaments* are extracted with `skimage.filters.frangi` (Frangi vesselness filter) followed by skeletonization. Two layers are synthesized: a continuous scan layer tracking per-column pitch (weighted vertical centroid of the Frangi response) and individual string notes for the longest filament segments.
- *Voids* are found by thresholding dark pixels and labeling connected components. Each void triggers a cello pizzicato at its leading edge and a sustained bowing note spanning its full width. Larger voids play lower pitches.
- *Choir pad* runs for the full duration at a fixed A major chord with formant filtering to simulate an "Ah" vowel.

**Synthesis techniques**

- **Harp**: additive synthesis with harmonics 1–5, per-partial exponential decay, 2 ms attack.
- **Strings**: 4 de-tuned layers (±0–0.9%), vibrato builds over 0.4 s, sawtooth harmonic series (1/*k*), vectorized cumulative-phase computation.
- **Cello pizzicato**: 5-partial additive synthesis, frequency-dependent decay, sub-fundamental body resonance.
- **Cello sustain**: 3 de-tuned layers, slow bow-pressure LFO (0.06 Hz), 300 ms squared attack.
- **Choir pad**: per-voice slow LFO + vibrato, butter bandpass formant boost at F1 = 800 Hz and F2 = 1200 Hz.
- **Reverb**: 8-tap comb filter with co-prime delays (31, 67, 107, 163, 229, 311, 419, 557 ms).

### Animation (`lss_animation.py`)

Each frame is rendered at a given `scan_x` position:

1. **Base layer**: left of scan → full color; right of scan → grayscale. A short blend region softens the edge.
2. **Void fill**: all void pixels left of scan are tinted deep blue (overlaid 45%).
3. **Void border pulse**: when the scan enters a void, a blue flash column appears and fades over `VOID_PULSE_F * width` pixels.
4. **Filament glow**: active skeleton columns within `FILA_GLOW * width` pixels of the scan line get an orange glow proportional to their Frangi response.
5. **Filament skeleton**: skeleton pixels left of the scan are persistently colored orange.
6. **Halo overlay**: three-state animation — ghost (10% brightness, pre-scan) → flash peak (×1.4 at scan crossing, then exponential decay) → settled (30% brightness, post-scan).
7. **Topology overlays**: same three-state logic for H0 (gold Gaussian blobs) and H1 (cyan rings). H1 rings additionally pulse outward as the scan crosses them.
8. **Scan line glow**: a white-blue glow column ±5 pixels around `scan_x`.

**Topology (persistent homology approximation)**

- *H0* features are tracked via superlevel-set filtration: components are born and die as the luminance threshold descends. Persistence = birth luminance − death luminance.
- *H1* features are detected as local minima in the inverted luminance at multiple Gaussian scales. Ring radius is set by `scipy.ndimage.distance_transform_edt` — the distance from each dark minimum to the nearest bright pixel — giving each hole a physically meaningful size.

---

## Tips for Best Results

- **Image type**: the scripts work best on real cosmic web simulation snapshots or observational maps with clear halo nodes, filamentary web, and dark voids. Uniform or noisy images will produce fewer detected structures.
- **Duration**: 30–60 seconds gives the structures enough time to each be heard clearly. Very short durations (< 15 s) may compress too many halos into simultaneous chords.
- **Input size**: images are internally downsampled to 512 px (music) or 960 px (animation) on the longest axis. There is no need to resize manually.
- **Adjusting density**: edit `MAX_HALOS` in `lss_animation.py` or the `max_h = min(len(halos), 150)` line in `lss_to_music.py` to control how many halos are rendered.

---

## License

MIT
