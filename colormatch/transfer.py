"""Color-distribution transfer on N x 3 sample arrays (normalized Lab).

All functions take `src` (colors of the photo being edited) and `ref`
(colors of the reference photo) as float arrays of shape (N, 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sqrtm_psd(m: np.ndarray, inv: bool = False, eps: float = 1e-6) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, eps)
    d = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (v * d) @ v.T


@dataclass
class AffineMap:
    """x -> (x - mu_src) @ A + mu_ref, the "weights + bias" form of a global transfer."""

    A: np.ndarray
    mu_src: np.ndarray
    mu_ref: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu_src) @ self.A + self.mu_ref


def reinhard(src: np.ndarray, ref: np.ndarray) -> AffineMap:
    """Per-channel mean/std matching (Reinhard et al. 2001)."""
    mu_s, mu_r = src.mean(0), ref.mean(0)
    sd_s, sd_r = src.std(0) + 1e-6, ref.std(0) + 1e-6
    return AffineMap(np.diag(sd_r / sd_s).astype(np.float32), mu_s, mu_r)


def mkl(src: np.ndarray, ref: np.ndarray) -> AffineMap:
    """Monge-Kantorovich linear transfer (Pitie & Kokaram 2007).

    The optimal-transport linear map between two Gaussians: it matches the
    full covariance (so cross-channel correlations such as "shadows go teal,
    highlights go orange" carry over), not only per-channel spread.
    """
    mu_s, mu_r = src.mean(0), ref.mean(0)
    cs = np.cov(src - mu_s, rowvar=False) + np.eye(3) * 1e-5
    cr = np.cov(ref - mu_r, rowvar=False) + np.eye(3) * 1e-5
    cs_h = _sqrtm_psd(cs)
    cs_ih = _sqrtm_psd(cs, inv=True)
    t = cs_ih @ _sqrtm_psd(cs_h @ cr @ cs_h) @ cs_ih
    # Row-vector convention: y = x @ A, with A = T^T (T is symmetric anyway).
    return AffineMap(t.T.astype(np.float32), mu_s, mu_r)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


def _match_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Monotone 1-D transport of samples x onto the distribution of y."""
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(x))
    q = (ranks + 0.5) / len(x)
    ys = np.sort(y)
    yq = (np.arange(len(ys)) + 0.5) / len(ys)
    return np.interp(q, yq, ys)


def sliced_ot(
    src: np.ndarray,
    ref: np.ndarray,
    iterations: int = 24,
    step: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Iterative distribution transfer / sliced optimal transport (Pitie 2005).

    Repeatedly projects both point clouds onto a random 3-D basis and
    matches each 1-D marginal. Converges to a non-linear map whose output
    has the reference's full 3-D color distribution. Returns the moved
    `src` samples.
    """
    rng = np.random.default_rng(seed)
    x = src.astype(np.float64).copy()
    y = ref.astype(np.float64)
    for _ in range(iterations):
        rot = _random_rotation(rng)
        xp, yp = x @ rot, y @ rot
        delta = np.empty_like(xp)
        for c in range(3):
            delta[:, c] = _match_1d(xp[:, c], yp[:, c]) - xp[:, c]
        x += step * (delta @ rot.T)
    return x.astype(np.float32)
