"""M0: exact separable observation operator for the frozen protocols, its adjoint, right inverse and consistency maps.

D(X) = A X A^T with A (n/s x n) the exact 1-D matrix of the implemented reduction (filter taps, every x2 stage,
half-pixel lattice, symmetric borders), read off impulse responses of a2_ops.make_lr and verified against it.
K = A A^T = Q diag(l) Q^T (symmetric positive definite for these operators). With the eigenbasis,
  (D D^T + lam I)^-1 R = Q [ (Q^T R Q) / (l_i l_j + lam) ] Q^T,
so exact (lam = 0) and regularised consistency share one implementation and no 2-D matrix is ever formed.
  B(R)          = A^T K^-1 R K^-1 A                (right inverse, D B = I)
  P(Z)          = Z - B(D Z)                       (orthogonal projector onto ker D)
  C_y(z)        = z + B(y - D z) = B y + P z       (Euclidean consistency; cannot increase squared error, clean case)
  C_y,lam(z)    = z + D^T (D D^T + lam I)^-1 (y - D z)
  IBP_T(z)      = T steps of x <- x + U(y - D x)   (oblique; U = bicubic interpolation, the existing refinement)
Only scoring code receives the original image.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "pilot")]
import a2_ops as ops  # noqa: E402

PROTOCOLS = {"P1": "P1_gauss_rec", "L3": "L3_rec", "area": "area"}


@dataclass(frozen=True)
class Operator:
    protocol: str
    scale: int
    n: int
    A: np.ndarray          # (n/s, n)
    Q: np.ndarray          # eigenvectors of K = A A^T
    lam: np.ndarray        # eigenvalues of K
    setup_s: float

    @property
    def cond_2d(self) -> float:
        return float((self.lam.max() / self.lam.min()) ** 2)


def _impulse_matrix(n: int, s: int, protocol: str) -> np.ndarray:
    cols = []
    for i in range(n):
        imp = np.zeros((n, n))
        imp[:, i] = 1.0
        cols.append(ops.make_lr(imp, s, protocol)[0][0, :])
    return np.stack(cols, 1)


@lru_cache(maxsize=None)
def get_operator(protocol: str, scale: int, n: int = 512) -> Operator:
    t0 = time.perf_counter()
    a = _impulse_matrix(n, scale, PROTOCOLS[protocol])
    lam, q = np.linalg.eigh(a @ a.T)
    return Operator(protocol, scale, n, a, q, lam, time.perf_counter() - t0)


def apply_D(x: np.ndarray, op: Operator) -> np.ndarray:
    return op.A @ x @ op.A.T


def apply_D_adjoint(r: np.ndarray, op: Operator) -> np.ndarray:
    return op.A.T @ r @ op.A


def solve_gram(r: np.ndarray, op: Operator, lam: float = 0.0) -> np.ndarray:
    """(D D^T + lam I)^-1 applied to an LR image r."""
    c = op.Q.T @ r @ op.Q
    return op.Q @ (c / (np.outer(op.lam, op.lam) + lam)) @ op.Q.T


def apply_right_inverse(r: np.ndarray, op: Operator) -> np.ndarray:
    return apply_D_adjoint(solve_gram(r, op), op)


def null_project(z: np.ndarray, op: Operator) -> np.ndarray:
    return z - apply_right_inverse(apply_D(z, op), op)


def enforce_consistency(z: np.ndarray, y: np.ndarray, op: Operator, mode: str = "exact", lam: float = 0.0,
                        iters: int = 20) -> np.ndarray:
    if mode == "exact":
        return z + apply_right_inverse(y - apply_D(z, op), op)
    if mode == "regularized":
        return z + apply_D_adjoint(solve_gram(y - apply_D(z, op), op, lam), op)
    if mode == "ibp":
        x = z.copy()
        for _ in range(iters):
            x = x + ops.upsample(y - apply_D(x, op), op.scale, "bicubic_a05", "centered")
        return x
    raise ValueError(mode)


def decompose_reconstruction_error(x: np.ndarray, z: np.ndarray, op: Operator) -> dict[str, np.ndarray | float]:
    """e = z - x = (I-P)e + P e; the two parts are orthogonal. Returns images and squared norms (per pixel mean)."""
    e = z - x
    observable = apply_right_inverse(apply_D(e, op), op)
    null = e - observable
    return {"observable_img": observable, "null_img": null,
            "mse_total": float(np.mean(e**2)), "mse_observable": float(np.mean(observable**2)), "mse_null": float(np.mean(null**2))}
