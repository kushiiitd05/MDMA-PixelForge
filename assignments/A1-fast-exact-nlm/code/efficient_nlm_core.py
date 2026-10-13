"""Minimal, self-contained implementation of direct and accelerated NLM.

This file contains the assignment's primary contribution without any D5/fusion
dependencies. Images are float arrays in the range [0, 255].
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage


def shift_symmetric(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Sample q=p+(dy,dx) with half-sample symmetric boundary extension."""
    height, width = image.shape
    yy = np.arange(height) + dy
    xx = np.arange(width) + dx
    yy = np.where(yy < 0, -yy - 1, yy)
    yy = np.where(yy >= height, 2 * height - yy - 1, yy)
    xx = np.where(xx < 0, -xx - 1, xx)
    xx = np.where(xx >= width, 2 * width - xx - 1, xx)
    return image[np.ix_(yy, xx)]


def direct_patch_nlm(
    noisy: np.ndarray,
    sigma_n: float,
    h: float,
    patch_k: int = 7,
    search_s: int = 11,
) -> np.ndarray:
    """Direct O(N*S^2*K^2) reference implementation."""
    if patch_k % 2 != 1 or search_s % 2 != 1:
        raise ValueError("patch_k and search_s must be odd")
    x = np.asarray(noisy, dtype=np.float64)
    numerator = np.zeros_like(x)
    denominator = np.zeros_like(x)
    search_r, patch_r = search_s // 2, patch_k // 2
    inv_h2 = 1.0 / max(float(h) ** 2, 1e-12)
    noise_bias = 2.0 * float(sigma_n) ** 2

    for dy in range(-search_r, search_r + 1):
        for dx in range(-search_r, search_r + 1):
            candidate = shift_symmetric(x, dy, dx)
            squared = (x - candidate) ** 2
            patch_distance = np.zeros_like(x)
            for py in range(-patch_r, patch_r + 1):
                for px in range(-patch_r, patch_r + 1):
                    patch_distance += shift_symmetric(squared, py, px)
            patch_distance /= patch_k * patch_k
            weight = np.exp(-np.maximum(patch_distance - noise_bias, 0.0) * inv_h2)
            numerator += weight * candidate
            denominator += weight
    return numerator / np.maximum(denominator, 1e-12)


def fast_exact_nlm(
    noisy: np.ndarray,
    sigma_n: float,
    h: float,
    patch_k: int = 7,
    search_s: int = 11,
) -> np.ndarray:
    """Output-equivalent O(N*S^2) displacement/running-box NLM."""
    if patch_k % 2 != 1 or search_s % 2 != 1:
        raise ValueError("patch_k and search_s must be odd")
    x = np.asarray(noisy, dtype=np.float64)
    numerator = np.zeros_like(x)
    denominator = np.zeros_like(x)
    search_r = search_s // 2
    inv_h2 = 1.0 / max(float(h) ** 2, 1e-12)
    noise_bias = 2.0 * float(sigma_n) ** 2

    for dy in range(-search_r, search_r + 1):
        for dx in range(-search_r, search_r + 1):
            candidate = shift_symmetric(x, dy, dx)
            squared = (x - candidate) ** 2
            patch_distance = ndimage.uniform_filter(
                squared, size=patch_k, mode="reflect"
            )
            weight = np.exp(-np.maximum(patch_distance - noise_bias, 0.0) * inv_h2)
            numerator += weight * candidate
            denominator += weight
    return numerator / np.maximum(denominator, 1e-12)


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, 1000.0 * (time.perf_counter() - start)

