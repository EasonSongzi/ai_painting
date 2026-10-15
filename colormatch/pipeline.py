"""End-to-end "AI 追色" pipeline.

1. analyse small previews of the reference (A) and the target (B)
   -> produce a compact color transform (a global affine "weights + bias"
   map refined by sliced optimal transport),
2. bake the transform into a 3D LUT that the full-resolution image is run
   through,
3. apply separate tone / color strengths and an optional protection mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from . import transfer
from .color import nlab_to_rgb, rgb_to_lab, lab_to_rgb, rgb_to_nlab, thumbnail
from .lut import apply_lut, identity_lut, lut_from_function, splat_residual

METHODS = ("ot", "mkl", "reinhard")


def _samples(rgb: np.ndarray, max_side: int, max_count: int, rng) -> np.ndarray:
    px = thumbnail(rgb, max_side).reshape(-1, 3)
    if len(px) > max_count:
        px = px[rng.choice(len(px), max_count, replace=False)]
    return px


@dataclass
class ColorMatch:
    """A fitted reference->target color transform at full strength."""

    lut_full: np.ndarray

    def lut(self, amount: float = 1.0, tone: float = 1.0, color: float = 1.0) -> np.ndarray:
        """LUT with strengths applied: `tone` scales the lightness change,
        `color` scales the chroma/hue change, `amount` scales both."""
        if amount == tone == color == 1.0:
            return self.lut_full
        size = self.lut_full.shape[0]
        lab_in = rgb_to_lab(identity_lut(size))
        lab_out = rgb_to_lab(self.lut_full)
        k = np.array([amount * tone, amount * color, amount * color], dtype=np.float32)
        return lab_to_rgb(lab_in + (lab_out - lab_in) * k)


def fit_color_match(
    reference: np.ndarray,
    target: np.ndarray,
    method: str = "ot",
    lut_size: int = 33,
    analysis_size: int = 384,
    max_samples: int = 80_000,
    seed: int = 0,
) -> ColorMatch:
    """Fit a transform that gives `target` the color look of `reference`."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    rng = np.random.default_rng(seed)
    src_rgb = _samples(target, analysis_size, max_samples, rng)
    ref_rgb = _samples(reference, analysis_size, max_samples, rng)
    src, ref = rgb_to_nlab(src_rgb), rgb_to_nlab(ref_rgb)

    base = transfer.reinhard(src, ref) if method == "reinhard" else transfer.mkl(src, ref)
    lut = lut_from_function(lambda c: nlab_to_rgb(base(rgb_to_nlab(c))), lut_size)

    if method == "ot":
        moved = transfer.sliced_ot(src, ref, seed=seed)
        lut = splat_residual(lut, src_rgb, nlab_to_rgb(moved))
    return ColorMatch(lut)


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Soft skin-tone mask in [0, 1] from YCrCb ranges, feathered."""
    r, g, b = (rgb[..., i] * 255.0 for i in range(3))
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b

    def band(x, lo, hi, soft=6.0):
        return np.clip((x - lo) / soft, 0, 1) * np.clip((hi - x) / soft, 0, 1)

    m = band(cr, 135, 175) * band(cb, 80, 128)
    radius = max(1.0, max(rgb.shape[:2]) / 400.0)
    img = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def apply_color_match(
    target: np.ndarray,
    lut: np.ndarray,
    skin_protect: float = 0.0,
) -> np.ndarray:
    out = apply_lut(target, lut)
    if skin_protect > 0:
        m = skin_mask(target)[..., None] * float(skin_protect)
        out = out + (target - out) * m
    return np.clip(out, 0.0, 1.0)


def color_match(
    reference: np.ndarray,
    target: np.ndarray,
    method: str = "ot",
    amount: float = 1.0,
    tone: float = 1.0,
    color: float = 1.0,
    skin_protect: float = 0.0,
    lut_size: int = 33,
    seed: int = 0,
):
    """One-shot helper. Returns (result_image, lut)."""
    cm = fit_color_match(reference, target, method=method, lut_size=lut_size, seed=seed)
    lut = cm.lut(amount, tone, color)
    return apply_color_match(target, lut, skin_protect), lut
