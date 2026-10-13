"""M2: trainable null-space predictors  x_hat = B y + P F_theta(y)  under the exact frozen P1 operator (torch).

TorchOperator mirrors operators.Operator for any square size n (crops use their own exact n-pixel operator, so every
training pair is exactly y = D_n x). Models (grayscale, [0,1]):
  DirectLR     : wide features only on the LR grid, one pixel-shuffle to full size, bicubic skip.
  Progressive  : three x2 stages; widest feature map at 1/2 output size; each stage predicts a signed residual on a
                 bicubic x2 of the current estimate (course-style G_k = E_k G_{k+1} + L_k with E_k = bicubic).
Consistency: 'train' puts the exact projection inside the training graph; 'infer' applies it only at test time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "method_development")]
import operators as O  # noqa: E402


class TorchOperator:
    def __init__(self, protocol: str, scale: int, n: int, device, dtype=torch.float32) -> None:
        op = O.get_operator(protocol, scale, n)
        self.s, self.n = scale, n
        self.A = torch.tensor(op.A, dtype=dtype, device=device)
        self.Q = torch.tensor(op.Q, dtype=dtype, device=device)
        self.inv_ll = torch.tensor(1.0 / np.outer(op.lam, op.lam), dtype=dtype, device=device)

    def D(self, x: torch.Tensor) -> torch.Tensor:
        return self.A @ x @ self.A.T

    def B(self, r: torch.Tensor) -> torch.Tensor:
        c = self.Q.T @ r @ self.Q
        return self.A.T @ (self.Q @ (c * self.inv_ll) @ self.Q.T) @ self.A

    def consistent(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return z + self.B(y - self.D(z))


def conv_block(c_in: int, c: int, n: int) -> nn.Sequential:
    layers = [nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(inplace=True)]
    for _ in range(n - 1):
        layers += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class DirectLR(nn.Module):
    def __init__(self, scale: int = 8, width: int = 64, depth: int = 8) -> None:
        super().__init__()
        self.s = scale
        self.body = conv_block(1, width, depth)
        self.head = nn.Conv2d(width, scale * scale, 3, padding=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(y, scale_factor=self.s, mode="bicubic", align_corners=False)
        return up + F.pixel_shuffle(self.head(self.body(y)), self.s)


class Stage(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.body = conv_block(1, width, depth)
        self.head = nn.Conv2d(width, 4, 3, padding=1)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return F.interpolate(g, scale_factor=2, mode="bicubic", align_corners=False) + F.pixel_shuffle(self.head(self.body(g)), 2)


class Progressive(nn.Module):
    """Widths chosen so parameters and FLOPs are comparable to DirectLR(64, 8): wide at 64^2, narrower at 128^2 / 256^2."""

    def __init__(self, widths=(64, 40, 24), depths=(6, 4, 3)) -> None:
        super().__init__()
        self.stages = nn.ModuleList(Stage(w, d) for w, d in zip(widths, depths))

    def forward(self, y: torch.Tensor, return_all: bool = False):
        outs, g = [], y
        for st in self.stages:
            g = st(g)
            outs.append(g)
        return outs if return_all else g


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.c2(F.relu(self.c1(x)))


class ResDirect(nn.Module):
    """Direct LR-space predictor with residual blocks (trainable depth).
    context: adds pooled x2/x4 LR context features (coarser grids -> cheap), fused before the head.
    hr_refine: light full-resolution head (16 ch, 2 convs) on [pixel-shuffled output, bicubic(y)]."""

    def __init__(self, scale: int = 8, width: int = 96, blocks: int = 8, context: bool = False, hr_refine: bool = False) -> None:
        super().__init__()
        self.s, self.context, self.hr_refine = scale, context, hr_refine
        self.inp = nn.Conv2d(1, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        if context:
            self.ctx = nn.ModuleList(nn.Sequential(nn.Conv2d(width, width // 2, 3, padding=1), nn.ReLU(inplace=True),
                                                   nn.Conv2d(width // 2, width // 2, 3, padding=1)) for _ in (2, 4))
            self.fuse = nn.Conv2d(width * 2, width, 1)
        self.head = nn.Conv2d(width, scale * scale, 3, padding=1)
        if hr_refine:
            self.refine = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(y, scale_factor=self.s, mode="bicubic", align_corners=False)
        f = self.blocks(self.inp(y))
        if self.context:
            h, w = f.shape[-2:]
            extra = [F.interpolate(m(F.avg_pool2d(f, k)), size=(h, w), mode="bilinear", align_corners=False) for m, k in zip(self.ctx, (2, 4))]
            f = self.fuse(torch.cat([f, *extra], 1))
        out = up + F.pixel_shuffle(self.head(f), self.s)
        return out + self.refine(torch.cat([out, up], 1)) if self.hr_refine else out
