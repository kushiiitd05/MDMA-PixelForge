"""Degradation and reconstruction operators for Assignment 2 pilots.

Lattice conventions (square images only)
-----------------------------------------
centered : LR sample i summarises HR block [s*i, s*i + s); its centre is s*i + (s-1)/2.
           Output coordinate u maps to input x = (u + 0.5)/s - 0.5.  Nearest neighbour
           under this mapping is exactly the PDF's s x s block copy, and it is the
           convention used by OpenCV / skimage / PIL resize.
corner   : LR sample i sits on HR pixel s*i (course-style G[::2, ::2]); x = u/s.

All arrays are float64 on the 0..255 scale.  Functions never modify their inputs.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage

Kernel = Callable[[np.ndarray], np.ndarray]


# ----------------------------------------------------------------------------- lattice
def lattice_offset(s: int, lattice: str) -> float:
    if lattice == "centered":
        return 0.5 / s - 0.5
    if lattice == "corner":
        return 0.0
    raise ValueError(f"unknown lattice {lattice!r}")


def input_coords(n_out: int, s: int, lattice: str) -> np.ndarray:
    return np.arange(n_out, dtype=np.float64) / s + lattice_offset(s, lattice)


def reflect_index(idx: np.ndarray, n: int) -> np.ndarray:
    """Half-sample symmetric boundary: ... b a | a b c ... """
    m = np.mod(idx, 2 * n)
    return np.where(m >= n, 2 * n - 1 - m, m)


def apply_1d(x: np.ndarray, idx: np.ndarray, w: np.ndarray, axis: int) -> np.ndarray:
    """Gather idx (n_out, taps) along axis and combine with weights w (n_out, taps)."""
    moved = np.moveaxis(x, axis, -1)
    out = np.einsum("...ot,ot->...o", moved[..., idx], w, optimize=True)
    return np.moveaxis(out, -1, axis)


def separable(x: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    return apply_1d(apply_1d(x, idx, w, 0), idx, w, 1)


# ----------------------------------------------------------------------------- downsampling
def gaussian_taps(sigma: float, centered: bool, truncate: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    radius = int(np.ceil(truncate * sigma))
    pos = (np.arange(-radius, radius) + 0.5) if centered else np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-0.5 * (pos / sigma) ** 2)
    return pos, w / w.sum()


def box_taps(factor: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.arange(factor) - (factor - 1) / 2
    return pos, np.full(factor, 1.0 / factor)


def filter_decimate(img: np.ndarray, factor: int, pos: np.ndarray, w: np.ndarray, lattice: str) -> np.ndarray:
    n_in = img.shape[0]
    n_out = n_in // factor
    shift = (factor - 1) / 2 if lattice == "centered" else 0.0
    centres = factor * np.arange(n_out) + shift
    idx = reflect_index(np.rint(centres[:, None] + pos[None, :]).astype(int), n_in)
    return separable(img, idx, np.broadcast_to(w, idx.shape))


def lanczos3_taps(factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Lanczos3 anti-alias kernel stretched by `factor`, half-sample centred taps (sensitivity protocol L3_rec)."""
    pos = np.arange(-3 * factor, 3 * factor) + 0.5
    w = lanczos(3)(pos / factor)
    return pos, w / w.sum()


def gaussian_sigma_effective(s: int, stage_sigma: float = 1.0) -> float:
    """sigma_eff^2 = sum_i sigma_i^2 (prod_{t<i} 2)^2 for log2(s) factor-2 stages."""
    stages = int(np.log2(s))
    return float(np.sqrt(sum((stage_sigma * 2 ** k) ** 2 for k in range(stages))))


DOWN_PROTOCOLS = ("P1_gauss_rec", "P1_gauss_direct", "area", "P0_course_rec", "decimate", "L3_rec")


def make_lr(img: np.ndarray, s: int, protocol: str = "P1_gauss_rec") -> tuple[np.ndarray, str]:
    """Return (low-resolution array, lattice).  P1_gauss_rec is the frozen assignment protocol."""
    stages = int(np.log2(s))
    if protocol == "P1_gauss_rec":
        pos, w = gaussian_taps(1.0, centered=True)
        out = img
        for _ in range(stages):
            out = filter_decimate(out, 2, pos, w, "centered")
        return out, "centered"
    if protocol == "P1_gauss_direct":
        pos, w = gaussian_taps(gaussian_sigma_effective(s), centered=True)
        return filter_decimate(img, s, pos, w, "centered"), "centered"
    if protocol == "area":
        pos, w = box_taps(s)
        return filter_decimate(img, s, pos, w, "centered"), "centered"
    if protocol == "P0_course_rec":
        pos, w = gaussian_taps(1.0, centered=False)
        out = img
        for _ in range(stages):
            out = filter_decimate(out, 2, pos, w, "corner")
        return out, "corner"
    if protocol == "L3_rec":  # sensitivity only; P1_gauss_rec stays the frozen protocol
        pos, w = lanczos3_taps(2)
        out = img
        for _ in range(stages):
            out = filter_decimate(out, 2, pos, w, "centered")
        return out, "centered"
    if protocol == "decimate":
        return img[::s, ::s].copy(), "corner"
    raise ValueError(f"unknown protocol {protocol!r}")


# ----------------------------------------------------------------------------- kernels
def triangle(t: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.abs(t), 0.0, None)


def dodgson_quadratic(t: np.ndarray) -> np.ndarray:
    """Interpolating local quadratic (Dodgson 1997, r = 0.5); 3 taps, not a B-spline."""
    a = np.abs(t)
    inner = 1.0 - 2.0 * a**2
    outer = a**2 - 2.5 * a + 1.5
    return np.where(a <= 0.5, inner, np.where(a <= 1.5, outer, 0.0))


def keys_cubic(a_param: float) -> Kernel:
    def k(t: np.ndarray) -> np.ndarray:
        x = np.abs(t)
        near = (a_param + 2) * x**3 - (a_param + 3) * x**2 + 1
        far = a_param * x**3 - 5 * a_param * x**2 + 8 * a_param * x - 4 * a_param
        return np.where(x <= 1, near, np.where(x < 2, far, 0.0))
    return k


def mitchell_netravali(b: float, c: float) -> Kernel:
    def k(t: np.ndarray) -> np.ndarray:
        x = np.abs(t)
        near = (12 - 9 * b - 6 * c) * x**3 + (-18 + 12 * b + 6 * c) * x**2 + (6 - 2 * b)
        far = (-b - 6 * c) * x**3 + (6 * b + 30 * c) * x**2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)
        return np.where(x < 1, near, np.where(x < 2, far, 0.0)) / 6.0
    return k


def lanczos(a: int) -> Kernel:
    def k(t: np.ndarray) -> np.ndarray:
        return np.where(np.abs(t) < a, np.sinc(t) * np.sinc(t / a), 0.0)
    return k


def kernel_upsample(lr: np.ndarray, s: int, kernel: Kernel, support: int, lattice: str,
                    normalize: bool = True) -> np.ndarray:
    n_out = lr.shape[0] * s
    x = input_coords(n_out, s, lattice)
    offs = np.arange(-support + 1, support + 1)
    taps = np.floor(x).astype(int)[:, None] + offs[None, :]
    w = kernel(x[:, None] - taps)
    if normalize:
        w = w / w.sum(axis=1, keepdims=True)
    return separable(lr, reflect_index(taps, lr.shape[0]), w)


def spline_upsample(lr: np.ndarray, s: int, order: int, lattice: str) -> np.ndarray:
    """Cardinal B-spline interpolation of degree `order` (IIR coefficient prefilter for order > 1)."""
    n_out = lr.shape[0] * s
    off = lattice_offset(s, lattice)
    return ndimage.affine_transform(lr, np.diag([1.0 / s, 1.0 / s]), offset=[off, off], output_shape=(n_out, n_out),
                                    order=order, mode="reflect", prefilter=True)


# name -> (family, degree label, callable(lr, s, lattice))
METHODS: dict[str, tuple[str, str, Callable[[np.ndarray, int, str], np.ndarray]]] = {
    "nearest": ("B-spline deg 0 / box", "0", lambda y, s, L: spline_upsample(y, s, 0, L)),
    "bilinear": ("B-spline deg 1 / triangle", "1", lambda y, s, L: spline_upsample(y, s, 1, L)),
    "quad_local": ("Dodgson interpolating quadratic", "2 (local)", lambda y, s, L: kernel_upsample(y, s, dodgson_quadratic, 2, L)),
    "spline2": ("Cardinal quadratic B-spline", "2 (spline)", lambda y, s, L: spline_upsample(y, s, 2, L)),
    "bicubic_a05": ("Keys cubic conv. a=-0.5", "3 (local)", lambda y, s, L: kernel_upsample(y, s, keys_cubic(-0.5), 2, L)),
    "bicubic_a075": ("Keys cubic conv. a=-0.75 (OpenCV)", "3 (local)", lambda y, s, L: kernel_upsample(y, s, keys_cubic(-0.75), 2, L)),
    "mitchell": ("Mitchell-Netravali B=C=1/3", "3 (approx.)", lambda y, s, L: kernel_upsample(y, s, mitchell_netravali(1 / 3, 1 / 3), 2, L)),
    "spline3": ("Cardinal cubic B-spline", "3 (spline)", lambda y, s, L: spline_upsample(y, s, 3, L)),
    "spline4": ("Cardinal quartic B-spline", "4 (spline)", lambda y, s, L: spline_upsample(y, s, 4, L)),
    "spline5": ("Cardinal quintic B-spline", "5 (spline)", lambda y, s, L: spline_upsample(y, s, 5, L)),
    "lanczos3": ("Windowed sinc, a=3", "n/a (6 taps)", lambda y, s, L: kernel_upsample(y, s, lanczos(3), 3, L)),
    "lanczos4": ("Windowed sinc, a=4", "n/a (8 taps)", lambda y, s, L: kernel_upsample(y, s, lanczos(4), 4, L)),
}


def upsample(lr: np.ndarray, s: int, method: str, lattice: str) -> np.ndarray:
    return METHODS[method][2](lr, s, lattice)
