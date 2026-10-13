"""M0 numerical checks on the REAL frozen operators (not a synthetic stand-in). Run: python method_development/test_operators.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "method_development"), str(ROOT / "pilot")]
import a2_ops as ops  # noqa: E402
import operators as O  # noqa: E402


def check(protocol: str, s: int, rng: np.random.Generator) -> dict:
    op = O.get_operator(protocol, s)
    x = rng.normal(size=(512, 512)) * 40 + 128
    r = rng.normal(size=(512 // s, 512 // s))
    y = O.apply_D(x, op)
    z = x + rng.normal(size=x.shape) * 12
    xc = O.enforce_consistency(z, y, op)
    pz = O.null_project(z, op)
    res = {
        "protocol": protocol, "scale": s, "setup_s": op.setup_s, "cond_2d": op.cond_2d,
        "D_vs_make_lr": float(np.abs(y - ops.make_lr(x, s, O.PROTOCOLS[protocol])[0]).max()),
        "adjoint_identity": float(abs(np.sum(y * r) - np.sum(x * O.apply_D_adjoint(r, op))) / abs(np.sum(y * r))),
        "DB_minus_I": float(np.abs(O.apply_D(O.apply_right_inverse(r, op), op) - r).max()),
        "DP": float(np.abs(O.apply_D(pz, op)).max()),
        "P_idempotent": float(np.abs(O.null_project(pz, op) - pz).max()),
        "consistency_residual": float(np.abs(O.apply_D(xc, op) - y).max()),
        "projection_error_identity": float(abs(np.sum((xc - x) ** 2) - (np.sum((z - x) ** 2) - np.sum(O.apply_right_inverse(O.apply_D(z - x, op), op) ** 2)))
                                           / np.sum((z - x) ** 2)),
    }
    d = O.decompose_reconstruction_error(x, z, op)
    res["decomposition_sum"] = float(abs(d["mse_total"] - d["mse_observable"] - d["mse_null"]) / d["mse_total"])
    return res


def main() -> None:
    rng = np.random.default_rng(20260915)
    rows = [check(p, s, rng) for p in ("P1", "L3") for s in (2, 4, 8)]
    limits = {"D_vs_make_lr": 1e-9, "adjoint_identity": 1e-10, "DB_minus_I": 1e-6, "DP": 1e-6, "P_idempotent": 1e-6,
              "consistency_residual": 1e-6, "projection_error_identity": 1e-8, "decomposition_sum": 1e-8}
    for r in rows:
        print({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()})
        for k, lim in limits.items():
            assert r[k] < lim, (r["protocol"], r["scale"], k, r[k])
    out = ROOT / "method_development" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "M0_operator_checks.json").write_text(json.dumps(rows, indent=1))
    print("M0: all operator identities passed on the real P1 and L3 operators")


if __name__ == "__main__":
    main()
