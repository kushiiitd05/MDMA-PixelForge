"""Integration checks for SRD on the real core checkpoint and the real P1 operator. Run: python method_development/test_srd.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "method_development"), str(ROOT / "pilot")]
import a2_ops as ops  # noqa: E402
import m2_models as M  # noqa: E402
import srd_model as S  # noqa: E402

CK = ROOT / "method_development" / "checkpoints" / "final_bal_s20260915.pt"


def main() -> None:
    torch.manual_seed(0)
    out = {}
    core = M.DirectLR(8, 96, 12)
    core.load_state_dict(torch.load(CK, map_location="cpu"))
    core.eval()
    for mode in ("full", "fixed"):
        br = S.SRDBranch(96, 8, 16, mode)
        out[f"branch_params_{mode}"] = sum(p.numel() for p in br.parameters())
    out["core_params"] = M.count_params(core)
    out["total_params_full"] = out["core_params"] + out["branch_params_full"]
    assert out["branch_params_full"] == 9419, out["branch_params_full"]
    img = torch.rand(1, 1, 256, 256, dtype=torch.float64) if False else None
    y = torch.rand(2, 1, 32, 32)
    net = S.SRDNet(core, S.SRDBranch(96, 8, 16, "full")).eval()
    with torch.no_grad():
        out["zero_head_identity_max_abs"] = float((net(y) - core(y)).abs().max())
    assert out["zero_head_identity_max_abs"] == 0.0
    br = S.SRDBranch(96, 8, 16, "full")
    torch.nn.init.normal_(br.W_o.weight)
    yc = torch.full((1, 1, 16, 16), 0.4)
    hc = torch.randn(1, 96, 1, 1).expand(1, 96, 16, 16).contiguous()
    with torch.no_grad():
        out["constant_features_correction_max_abs"] = float(br(yc, hc).abs().max())
    assert out["constant_features_correction_max_abs"] < 1e-6
    r = torch.randn(2, 64, 16, 16)
    out["phase_pack_roundtrip"] = float((F.pixel_unshuffle(F.pixel_shuffle(r, 8), 8) - r).abs().max())
    assert out["phase_pack_roundtrip"] == 0.0
    net = S.SRDNet(core, br).train()
    x = torch.rand(2, 1, 128, 128)
    loss = (net(F.avg_pool2d(x, 8)) - x).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    out["finite_gradients"] = bool(all(torch.isfinite(g).all() for g in grads))
    assert out["finite_gradients"]
    rng = np.random.default_rng(0)
    Wtrue = rng.normal(size=(64, 4)) @ rng.normal(size=(4, 32))
    st = S.RegressionStats(32, 64)
    phi = rng.normal(size=(50000, 32))
    st.add(phi, phi @ Wtrue.T + 0.1 * rng.normal(size=(50000, 64)))
    C, Wo, info = st.fit(4)
    out["reduced_rank_recovery_rel_err"] = float(np.linalg.norm(C @ Wo - Wtrue) / np.linalg.norm(Wtrue))
    assert out["reduced_rank_recovery_rel_err"] < 0.01
    op = M.TorchOperator("P1", 8, 512, torch.device("cpu"), torch.float64)
    x5 = torch.from_numpy(np.random.default_rng(1).random((1, 1, 512, 512)))
    y5 = op.D(x5)
    z = torch.from_numpy(np.random.default_rng(2).random((1, 1, 512, 512)))
    out["projection_residual_after_branch_sum"] = float((op.D(op.consistent(z, y5)) - y5).abs().max())
    assert out["projection_residual_after_branch_sum"] < 1e-9
    (ROOT / "method_development" / "results" / "srd_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
