"""Ringing / overshoot proxy used in the assignment tables."""
from __future__ import annotations

import numpy as np

import a2_ops as ops


def local_range_overshoot(lr: np.ndarray, raw: np.ndarray, s: int) -> float:
    """Mean grey levels by which the output leaves the [min, max] of the 2x2 LR cell it lies in."""
    n = raw.shape[0]
    x = ops.input_coords(n, s, "centered")
    i0 = ops.reflect_index(np.floor(x).astype(int), lr.shape[0])
    i1 = ops.reflect_index(np.floor(x).astype(int) + 1, lr.shape[0])
    corners = np.stack([lr[np.ix_(a, b)] for a in (i0, i1) for b in (i0, i1)])
    lo, hi = corners.min(0), corners.max(0)
    return float(np.mean(np.clip(raw - hi, 0, None) + np.clip(lo - raw, 0, None)))
