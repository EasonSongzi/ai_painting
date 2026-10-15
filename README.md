# AI Color Match

Give it a reference photo A and it re-grades photo B to match A's color look. The tool fits a color transform on small previews of both photos, bakes it into a 3D LUT and applies that to the full-resolution image. Tone and color strength can be adjusted separately.

## Usage

```bash
pip install -r requirements.txt
```

### Web UI

Upload A and B, then adjust the sliders and watch the preview update live. You can download the full-resolution result or export the look as a `.cube` LUT. (The interface text is in Chinese.)

```bash
python3 webapp/app.py
```

Then open http://127.0.0.1:5050

### Command line

```bash
# Single photo
python3 -m colormatch match -r A.jpg -t B.jpg -o out.tif

# Batch, 70% tone strength, skin protection
python3 -m colormatch match -r A.jpg -t B1.jpg B2.jpg -o out/ --tone 0.7 --skin-protect 0.5

# Also save the fitted look as a 3D LUT
python3 -m colormatch match -r A.jpg -t B.jpg -o out.tif --save-cube look.cube

# Apply a saved LUT (.cube files also load in Lightroom, Photoshop and DaVinci Resolve)
python3 -m colormatch apply -l look.cube -t *.jpg -o out/
```

### Python

```python
from colormatch import load_image, fit_color_match, apply_color_match

cm = fit_color_match(load_image("A.jpg"), load_image("B.jpg"))  # fit once
out = apply_color_match(load_image("B.jpg"), cm.lut(amount=1, tone=0.6, color=1))
```

### Options

| Option | Meaning |
|---|---|
| `--method ot` | Default. Covariance (MKL) transfer, then sliced optimal transport to match the full 3-D color distribution |
| `--method mkl` | Matches mean and covariance only. More stable, weaker style transfer |
| `--method reinhard` | Per-channel mean and standard deviation. The gentlest option |
| `--amount` / `--tone` / `--color` | Overall strength, tone strength (Lab L channel), color strength (a/b channels), each 0–1 |
| `--skin-protect` | 0–1. Pulls skin tones back toward the original so portraits don't drift |
| `--format tif\|png\|jpg` | Output format. Defaults to the `-o` extension, otherwise the input's format |
| `--bit-depth 8\|16` | TIFF only. Defaults to 16 |
| `--quality` | JPEG quality. Defaults to 100 |

## Image quality

- **Lossless output.** TIFF is 16-bit and uncompressed, PNG is 8-bit lossless, and JPEG is written at quality 100 with no chroma subsampling (4:4:4). All three keep the source's ICC profile and EXIF.
- **16-bit input.** 16-bit TIFFs are read at full precision. Processing is float32 throughout, and values are quantized only once, on save.
- **Export 16-bit TIFF even from 8-bit sources.** Grading creates new in-between levels, and saving them at 16-bit avoids banding in smooth gradients such as skies.
- **Downscaling is for analysis only.** Colors are analyzed on 384 px previews, but the LUT is applied to the full-resolution original, so the output matches the source resolution. The 1600 px preview in the web UI is a JPEG for display only; downloads are always full resolution.
- **Not supported yet:** RAW files (would need `rawpy`) and 16-bit PNG. The current libraries can't write 16-bit RGB PNG, and 16-bit PNGs are read as 8-bit. Use TIFF for 16-bit.

## How it works

```
A, B ──downscale to 384 px──► sample colors in Lab
                               │
                               ├─ MKL affine map (weights + bias) → base 33³ LUT (a sane mapping across the whole gamut)
                               └─ sliced OT refinement → residuals splatted onto the LUT grid, Gaussian-smoothed, added to the base
                                                            │
        tone / color strength: interpolate Lab components at each LUT node (slider changes rebuild the LUT, no refit)
                                                            │
B full resolution ──trilinear LUT lookup──► skin-mask blend ──► output
```

The overall design follows a common structure for AI color matching:

1. **Analyze small previews** of the reference and the target.
2. **Produce a compact color transform**, not an image: a small per-photo mapping such as weights + bias or a LUT.
3. **Apply it per pixel** to the full-resolution original, so large exports stay fast.
4. **Control tone and color strength separately**, and use masks (for example skin) for local protection.

Having a large model predict the parameters of a small per-image transform is the idea behind work such as Neural Preset (CVPR 2023). This project uses no learned model: classical distribution matching (MKL + sliced OT) produces the transform. To switch to a learned model, replace the step in `fit_color_match` that produces the LUT.

### Code layout

| Path | Contents |
|---|---|
| `colormatch/transfer.py` | Color transfer algorithms (Reinhard, MKL, sliced OT) |
| `colormatch/lut.py` | 3D LUT construction, application, `.cube` read/write |
| `colormatch/pipeline.py` | End-to-end pipeline, strengths, skin mask |
| `colormatch/color.py` | Color-space conversion and lossless image I/O |
| `colormatch/cli.py` | Command line interface |
| `webapp/` | Flask web UI |

## Tests

```bash
python3 -m pytest -q tests
```
