"""Research extension (TA demo path): consistent null-space super-resolution for the frozen P1 downsampler.

    x_hat = F(y) + B (y - D F(y)) = B y + P F(y),   B = D^T (D D^T)^-1,  P = I - B D

D  : the exact linear P1 operator of a2_ops.make_lr (Gaussian sigma=1 per x2 stage, half-pixel lattice, symmetric
     borders), separable: D(X) = A X A^T. A is read off impulse responses of make_lr, K = A A^T = Q diag(l) Q^T.
F  : DirectLR, a plain LR-space CNN (width 96, depth 12, 3x3 convs + ReLU) whose head predicts s^2 channels that are
     pixel-shuffled onto a bicubic (torch, a=-0.75, half-pixel) upsampling of y. F is the complete raw prediction.
The projection is applied once, after the network; output is clipped to [0, 255] (clipping can break exact consistency
by a small amount, which is logged in the results). Weights are listed in extension/model_manifest.json.
Needs torch (runs on Apple MPS or CPU). Training code is not part of this demo path (see method_development/).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import a2_ops as ops

EXT = Path(__file__).resolve().parents[1] / "extension"


@lru_cache(maxsize=None)
def p1_operator(scale: int, n: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A, Q, lam) of the exact 1-D P1 reduction for an n-pixel axis."""
    cols = []
    for i in range(n):
        imp = np.zeros((n, n))
        imp[:, i] = 1.0
        cols.append(ops.make_lr(imp, scale, "P1_gauss_rec")[0][0, :])
    a = np.stack(cols, 1)
    lam, q = np.linalg.eigh(a @ a.T)
    return a, q, lam


class Projector:
    """Exact Euclidean consistency z -> z + B(y - Dz) on a torch device (no 2-D matrix is formed)."""

    def __init__(self, scale: int, device: torch.device, n: int = 512) -> None:
        a, q, lam = p1_operator(scale, n)
        self.A = torch.tensor(a, dtype=torch.float32, device=device)
        self.Q = torch.tensor(q, dtype=torch.float32, device=device)
        self.inv_ll = torch.tensor(1.0 / np.outer(lam, lam), dtype=torch.float32, device=device)

    def D(self, x: torch.Tensor) -> torch.Tensor:
        return self.A @ x @ self.A.T

    def B(self, r: torch.Tensor) -> torch.Tensor:
        c = self.Q.T @ r @ self.Q
        return self.A.T @ (self.Q @ (c * self.inv_ll) @ self.Q.T) @ self.A

    def __call__(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return z + self.B(y - self.D(z))


class DirectLR(nn.Module):
    def __init__(self, scale: int, width: int = 96, depth: int = 12) -> None:
        super().__init__()
        self.s = scale
        layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, scale * scale, 3, padding=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(y, scale_factor=self.s, mode="bicubic", align_corners=False)
        return up + F.pixel_shuffle(self.head(self.body(y)), self.s)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def manifest() -> dict:
    return json.loads((EXT / "model_manifest.json").read_text())


def load_model(scale: int, role: str = "promoted", dev: torch.device | None = None) -> tuple[DirectLR, dict]:
    entry = manifest()["models"][str(scale)][role]
    path = EXT / entry["weights"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"checksum mismatch for {path.name}: {digest[:12]} != {entry['sha256'][:12]}")
    net = DirectLR(scale, 96, 12)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(dev or device()).eval(), entry


@torch.no_grad()
def super_resolve(lr: np.ndarray, scale: int, net: DirectLR, project: bool = True, proj: Projector | None = None) -> np.ndarray:
    """lr: float array in [0, 255] of shape (512/s, 512/s). Returns the clipped 512x512 reconstruction in [0, 255]."""
    dev = next(net.parameters()).device
    y = torch.from_numpy((lr / 255.0).astype(np.float32))[None, None].to(dev)
    z = net(y)
    if project:
        z = (proj or Projector(scale, dev, lr.shape[0] * scale))(z, y)
    return np.clip(z[0, 0].float().cpu().numpy().astype(np.float64) * 255.0, 0.0, 255.0)


def lr_residual(x: np.ndarray, lr: np.ndarray, scale: int) -> float:
    """RMS of D(x) - y in grey levels (0 for an exactly consistent, unclipped reconstruction)."""
    a, _, _ = p1_operator(scale, x.shape[0])
    return float(np.sqrt(np.mean((a @ x @ a.T - lr) ** 2)))
