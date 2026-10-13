"""SRD: Spatial Residual Decoder branch on the completed plain core (spec: DIP A2 Final SRD method, 15 Sep 2026).

x_hat = C_y[ z0(y) + rho(y, h(y)) ],  z0 = existing raw output, h = 96-ch LR feature map of the core.
Branch (8x): geometry j (5 ch, LR structure tensor) ; q/k 8-d (RMS-normalised h), v 16-d ;
 logits l_{p,d} = q_p.k_{p+d}/sqrt(8) + b_d - beta_I (y_{p+d}-y_p)^2/(trJ_p+eps) - beta_S d^T Jn_p d ; a = softmax_d
 moments m0 = sum a (v_{p+d}-v_p), m_r = sum a d_r (.), m_c = sum a d_c (.)
 phi = SiLU(W_m [m0,m_r,m_c]) * sigmoid(W_g [h, j] + b_g) ; a_p = W_o phi ; r = C a ; rho = pixel_shuffle(r)
mode="fixed" (control B): a_d = softmax(-|d|^2/2) fixed; no q/k/beta/b_d.
W_o starts at zero -> exact identity with the core before fitting. All gates act before the single global projection.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def core_forward_with_features(core, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DirectLR: h = body(y); z0 = bicubic(y) + pixel_shuffle(head(h)). Identical arithmetic to DirectLR.forward."""
    h = core.body(y)
    up = F.interpolate(y, scale_factor=core.s, mode="bicubic", align_corners=False)
    return h, up + F.pixel_shuffle(core.head(h), core.s)


def geometry(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns j (B,5,H,W), trJ (B,1,H,W), normalised tensor components (Jrr, Jrc, Jcc) each (B,1,H,W)."""
    yp = F.pad(y, (1, 1, 1, 1), mode="replicate")
    gr = 0.5 * (yp[..., 2:, 1:-1] - yp[..., :-2, 1:-1])
    gc = 0.5 * (yp[..., 1:-1, 2:] - yp[..., 1:-1, :-2])
    avg = lambda t: F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, 1)
    jrr, jrc, jcc = avg(gr * gr), avg(gr * gc), avg(gc * gc)
    tr = jrr + jcc
    n = tr + EPS
    coh = torch.sqrt((jrr - jcc) ** 2 + 4 * jrc ** 2) / n
    j = torch.cat([jrr / n, jrc / n, jcc / n, torch.log1p(torch.sqrt(tr) / 0.01), coh], 1)
    return j, tr, (jrr / n, jrc / n, jcc / n)


def neighbours(t: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B,C,9,H,W) values at the 3x3 offsets (replicate padding keeps constants constant)."""
    B, C, H, W = t.shape
    u = F.unfold(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3)  # (B, C*9, H*W) in row-major (dr, dc) order
    return u.view(B, C, 9, H, W)


class SRDBranch(nn.Module):
    def __init__(self, feat: int = 96, scale: int = 8, rank: int = 16, mode: str = "full", seed: int = 20260915) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.s, self.mode, self.rank = scale, mode, rank
        self.W_v = nn.Conv2d(feat, 16, 1, bias=False)
        if mode == "full":
            self.W_q = nn.Conv2d(feat, 8, 1, bias=False)
            self.W_k = nn.Conv2d(feat, 8, 1, bias=False)
            self.b_d = nn.Parameter(torch.tensor([-(dr * dr + dc * dc) / 2.0 for dr, dc in OFFSETS]))
            self.log_beta = nn.Parameter(torch.log(torch.expm1(torch.tensor([0.1, 0.1]))))  # softplus -> 0.1
        self.W_m = nn.Conv2d(48, 32, 1, bias=False)
        self.W_g = nn.Conv2d(feat + 5, 32, 1, bias=True)
        self.W_o = nn.Conv2d(32, rank, 1, bias=False)
        self.C = nn.Conv2d(rank, scale * scale, 1, bias=False)
        for m in (self.W_v, self.W_m, self.W_g, self.C) + ((self.W_q, self.W_k) if mode == "full" else ()):
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * math.sqrt(1.0 / m.weight.shape[1]))
        nn.init.zeros_(self.W_g.bias)
        nn.init.zeros_(self.W_o.weight)
        d = torch.tensor(OFFSETS, dtype=torch.float32)
        self.register_buffer("dr", d[:, 0].view(1, 1, 9, 1, 1))
        self.register_buffer("dc", d[:, 1].view(1, 1, 9, 1, 1))

    def attention(self, y: torch.Tensor, h: torch.Tensor, tr: torch.Tensor, jn) -> torch.Tensor:
        if self.mode == "fixed":
            w = torch.softmax(-(self.dr ** 2 + self.dc ** 2).view(9) / 2, 0)
            return w.view(1, 1, 9, 1, 1).expand(y.shape[0], 1, 9, *y.shape[-2:])
        hn = h / torch.sqrt(h.pow(2).mean(1, keepdim=True) + EPS)
        q, k = self.W_q(hn), neighbours(self.W_k(hn))
        logits = (q.unsqueeze(2) * k).sum(1, keepdim=True) / math.sqrt(8) + self.b_d.view(1, 1, 9, 1, 1)
        beta = F.softplus(self.log_beta)
        dy2 = (neighbours(y) - y.unsqueeze(2)) ** 2
        logits = logits - beta[0] * dy2 / (tr.unsqueeze(2) + EPS)
        jrr, jrc, jcc = (t.unsqueeze(2) for t in jn)
        quad = self.dr ** 2 * jrr + 2 * self.dr * self.dc * jrc + self.dc ** 2 * jcc
        return torch.softmax(logits - beta[1] * quad, 2)

    def latent(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        j, tr, jn = geometry(y)
        a = self.attention(y, h, tr, jn)
        v = self.W_v(h)
        d = neighbours(v) - v.unsqueeze(2)
        m0, mr, mc = (a * d).sum(2), (a * self.dr * d).sum(2), (a * self.dc * d).sum(2)
        return F.silu(self.W_m(torch.cat([m0, mr, mc], 1))) * torch.sigmoid(self.W_g(torch.cat([h, j], 1)))

    def forward(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.C(self.W_o(self.latent(y, h))), self.s)


class SRDNet(nn.Module):
    """core (DirectLR) + optional branch; forward returns the raw (unprojected) prediction z0 + rho."""

    def __init__(self, core: nn.Module, branch: SRDBranch | None) -> None:
        super().__init__()
        self.core, self.branch = core, branch

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h, z0 = core_forward_with_features(self.core, y)
        return z0 if self.branch is None else z0 + self.branch(y, h)


class RegressionStats:
    """Streaming sufficient statistics for the reduced-rank head: S = E[phi phi^T], R = E[e phi^T] (no centring)."""

    def __init__(self, d_phi: int, d_e: int) -> None:
        self.S, self.R, self.n = np.zeros((d_phi, d_phi)), np.zeros((d_e, d_phi)), 0

    def add(self, phi: np.ndarray, e: np.ndarray) -> None:
        self.S += phi.T @ phi
        self.R += e.T @ phi
        self.n += len(phi)

    def fit(self, rank: int, eta_rel: float = 1e-3) -> tuple[np.ndarray, np.ndarray, dict]:
        S, R = self.S / self.n, self.R / self.n
        eta = eta_rel * max(np.trace(S) / S.shape[0], 1e-12)
        lam, V = np.linalg.eigh(S + eta * np.eye(S.shape[0]))
        s_inv_half = V @ np.diag(lam ** -0.5) @ V.T
        U, sig, Vt = np.linalg.svd(R @ s_inv_half, full_matrices=False)
        C = U[:, :rank]
        W_o = np.diag(sig[:rank]) @ Vt[:rank] @ s_inv_half
        return C, W_o, {"eta": eta, "n": self.n, "singular_values": sig.tolist(), "cond_S": float(lam.max() / lam.min())}
