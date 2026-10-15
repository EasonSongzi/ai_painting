"""3D LUT construction, application and .cube I/O.

A LUT is a float32 array of shape (N, N, N, 3) indexed as lut[r, g, b]
with inputs and outputs in [0, 1] sRGB.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter


def identity_lut(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def lut_from_function(fn: Callable[[np.ndarray], np.ndarray], size: int) -> np.ndarray:
    grid = identity_lut(size)
    return fn(grid.reshape(-1, 3)).reshape(grid.shape).astype(np.float32)


def _corner_weights(rgb: np.ndarray, size: int):
    """Yield (flat_index, weight) for the 8 trilinear corners of each color."""
    pos = np.clip(rgb, 0.0, 1.0) * (size - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), size - 2)
    f = pos - i0
    for dr in (0, 1):
        wr = f[:, 0] if dr else 1.0 - f[:, 0]
        for dg in (0, 1):
            wg = f[:, 1] if dg else 1.0 - f[:, 1]
            for db in (0, 1):
                wb = f[:, 2] if db else 1.0 - f[:, 2]
                idx = ((i0[:, 0] + dr) * size + (i0[:, 1] + dg)) * size + (i0[:, 2] + db)
                yield idx, wr * wg * wb


def splat_residual(
    lut: np.ndarray,
    src_rgb: np.ndarray,
    dst_rgb: np.ndarray,
    sigma: float = 1.2,
    confidence: float = 0.5,
) -> np.ndarray:
    """Add a smoothed, data-driven correction to `lut`.

    For every sample the residual (dst - lut(src)) is splatted into the grid,
    then numerator and weights are Gaussian-blurred and divided. Grid cells
    far from any sample keep the base LUT, so colors that do not appear in the
    sample image still get a sane (global) mapping.
    """
    size = lut.shape[0]
    resid = dst_rgb - apply_lut(src_rgb.reshape(-1, 1, 3), lut).reshape(-1, 3)
    num = np.zeros((size**3, 3), dtype=np.float64)
    den = np.zeros(size**3, dtype=np.float64)
    for idx, w in _corner_weights(src_rgb, size):
        np.add.at(den, idx, w)
        for c in range(3):
            np.add.at(num[:, c], idx, w * resid[:, c])
    num = num.reshape(size, size, size, 3)
    den = den.reshape(size, size, size)
    for c in range(3):
        num[..., c] = gaussian_filter(num[..., c], sigma, mode="nearest")
    den = gaussian_filter(den, sigma, mode="nearest")
    # Normalize density so `confidence` is independent of the sample count.
    den_n = den / (den.mean() + 1e-12)
    blend = den_n / (den_n + confidence)
    corr = num / (den[..., None] + 1e-12) * blend[..., None]
    return np.clip(lut + corr.astype(np.float32), 0.0, 1.0)


def apply_lut(rgb: np.ndarray, lut: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """Trilinear 3D-LUT lookup for an (H, W, 3) image in [0, 1]."""
    size = lut.shape[0]
    flat_lut = lut.reshape(-1, 3)
    src = rgb.reshape(-1, 3)
    out = np.empty_like(src, dtype=np.float32)
    for s in range(0, len(src), chunk):
        part = src[s : s + chunk]
        acc = np.zeros((len(part), 3), dtype=np.float32)
        for idx, w in _corner_weights(part, size):
            acc += flat_lut[idx] * w[:, None].astype(np.float32)
        out[s : s + chunk] = acc
    return out.reshape(rgb.shape)


def save_cube(path: str, lut: np.ndarray, title: str = "AI Color Match") -> None:
    size = lut.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        # .cube order: red varies fastest, then green, then blue.
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    v = lut[r, g, b]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def load_cube(path: str) -> np.ndarray:
    size, rows = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
            elif line[0].isdigit() or line[0] in "-.":
                rows.append([float(x) for x in line.split()[:3]])
    if size is None or len(rows) != size**3:
        raise ValueError(f"{path}: not a valid 3D .cube file")
    data = np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)  # [b, g, r]
    return np.ascontiguousarray(data.transpose(2, 1, 0, 3))
