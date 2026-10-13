"""Freeze the four roles per scale from VALIDATION rows + matched latency only (policy fixed before the comparison).

For each scale s and candidate pipeline m: p_m, q_m = mean per-image val PSNR / SSIM; t_m = median full-decode latency.
  PSNR winner : max p (ties: SSIM, then latency)
  SSIM winner : max q (ties: PSNR, then latency)
  Fastest     : min t among valid full-output pipelines
  Trade-off   : eligible t <= 10 * t_bicubic,s (torch bicubic, same harness); drop methods dominated on (p, q, t);
                near-best set p >= p* - 0.10 dB and q >= q* - 0.002 (p*, q* = best eligible); promote its fastest member;
                if empty, minimise r = max((p*-p)/0.10, (q*-q)/0.002), ties by latency (reported as fallback).
These tolerances are a project preference, not a noise floor or significance test.
The same rules are also applied within our own family ("ours_best"). Output: results/final_eval/roles_frozen.json
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent / "results" / "final_eval"
BUDGET_X, TOL_P, TOL_Q = 10.0, 0.10, 0.002


def latency_key(s: int, mid: str) -> str:
    sfx = "+proj" if mid.endswith("+proj") else ""
    if mid.startswith("ours:"):
        return f"x{s}|ours:DirectLR w96 d12" + sfx
    if mid.startswith("finetuned:"):  # same SwinIR-M architecture and untiled code path as the pretrained weights
        return f"x{s}|pretrained:SwinIR-M" + sfx
    return f"x{s}|{mid}"


def candidates(val: pd.DataFrame, lat: dict, s: int) -> pd.DataFrame:
    v = val[val.scale == s]
    g = v.groupby("method_id").agg(p=("psnr_db", "mean"), q=("ssim", "mean"), n=("image_id", "nunique"), family=("family", "first")).reset_index()
    assert (g.n == 4).all(), g[g.n != 4]
    g["lat_key"] = [latency_key(s, m) for m in g.method_id]
    missing = [k for k in g.lat_key if k not in lat]
    assert not missing, f"no latency for {missing}"
    g["t"] = [lat[k]["median_of_medians_ms"] for k in g.lat_key]
    g["t_iqr"] = [lat[k]["max_iqr_ms"] for k in g.lat_key]
    g["t_ratio"] = g.t / lat[f"x{s}|classical:torch_bicubic"]["median_of_medians_ms"]
    return g


def pick(g: pd.DataFrame) -> dict:
    out = {}
    out["psnr"] = g.sort_values(["p", "q", "t"], ascending=[False, False, True]).iloc[0].method_id
    out["ssim"] = g.sort_values(["q", "p", "t"], ascending=[False, False, True]).iloc[0].method_id
    out["fastest"] = g.sort_values(["t", "p", "q"], ascending=[True, False, False]).iloc[0].method_id
    e = g[g.t_ratio <= BUDGET_X].copy()
    if e.empty:
        out["tradeoff"], out["tradeoff_rule"] = None, "no candidate within budget"
        return out
    dom = []
    for _, a in e.iterrows():
        dom.append(bool(((e.p >= a.p) & (e.q >= a.q) & (e.t <= a.t) & ((e.p > a.p) | (e.q > a.q) | (e.t < a.t))).any()))
    e["dominated"] = dom
    p_star, q_star = e.p.max(), e.q.max()
    nd = e[~e.dominated]
    near = nd[(nd.p >= p_star - TOL_P) & (nd.q >= q_star - TOL_Q)]
    out.update(p_star=float(p_star), q_star=float(q_star), eligible=e.method_id.tolist(), dominated=e[e.dominated].method_id.tolist(), near_best=near.method_id.tolist())
    if len(near):
        out["tradeoff"], out["tradeoff_rule"] = near.sort_values(["t", "p", "q"], ascending=[True, False, False]).iloc[0].method_id, "fastest member of near-best set"
    else:
        r = np.maximum((p_star - nd.p) / TOL_P, (q_star - nd.q) / TOL_Q)
        out["tradeoff"] = nd.assign(r=r).sort_values(["r", "t"]).iloc[0].method_id
        out["tradeoff_rule"] = "FALLBACK: minimum normalised quality deficit"
    return out


def main() -> None:
    rows = [json.loads(l) for l in (RES / "per_image.jsonl").read_text().splitlines()]
    val = pd.DataFrame([r for r in rows if r["stage"] == "val"])
    lat = json.loads((RES / "latency.json").read_text())
    frozen = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": __doc__,
              "val_rows_sha256": hashlib.sha256(json.dumps(sorted((r["scale"], r["method_id"], r["image_id"], round(r["psnr_db"], 6), round(r["ssim"], 6)) for r in rows if r["stage"] == "val")).encode()).hexdigest(),
              "latency_sha256": hashlib.sha256((RES / "latency.json").read_bytes()).hexdigest(), "scales": {}}
    for s in (2, 4, 8):
        g = candidates(val, lat, s)
        roles = pick(g)
        ours = pick(g[g.family == "ours"])
        frozen["scales"][str(s)] = {"bicubic_ref_ms": lat[f"x{s}|classical:torch_bicubic"]["median_of_medians_ms"], "roles": roles, "ours_best": ours,
                                    "candidates": g.sort_values("p", ascending=False).round(6).to_dict(orient="records")}
        print(f"x{s}:", {k: roles[k] for k in ("psnr", "ssim", "fastest", "tradeoff", "tradeoff_rule")}, "| ours:", {k: ours[k] for k in ("psnr", "ssim", "fastest", "tradeoff")})
    (RES / "roles_frozen.json").write_text(json.dumps(frozen, indent=1, default=str))


if __name__ == "__main__":
    main()
