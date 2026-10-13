"""Full decode latency of every role candidate at 2x/4x/8x, one pipeline per fresh Python process (idle machine).

Timed path = the scored path: host LR (float64 0-255 numpy, 512/s square) -> method -> projection if any -> clip ->
host numpy. Excludes file I/O, model construction/loading and operator construction (reported as setup_s).
MPS paths are synchronised by the device->host copy at the end of each call. Warm-up then measured repeats; median and IQR.
5 rotations with shuffled order for fast pipelines (raised from 3 after isolated processes showed a bimodal MPS state, IQR up to 4.8 ms; applied to every pipeline); 1 rotation for pretrained transformers (seconds per call).
Ours: timing depends on architecture, not weights; timed with the real checkpoint named in the record and applied to all
checkpoints of the same DirectLR(s, 96, 12) architecture.
Output: method_development/results/final_eval/latency.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "method_development" / "results" / "final_eval" / "latency.json"
INTERP = ("nearest", "bilinear", "quad_local", "spline2", "bicubic_a05", "spline3", "spline4", "spline5", "lanczos3")
INTERP_PROJ = ("nearest", "bicubic_a05", "spline3", "spline5")

CHILD = r'''
import sys, time, json, numpy as np, torch, torch.nn.functional as F
sys.path[:0] = [{md!r}, {pilot!r}, {dl!r}]
import a2_ops as ops, operators as O, m2_models as M
kind, arg, proj, s = {kind!r}, {arg!r}, {proj!r}, {s!r}
dev = torch.device("mps")
t0 = time.perf_counter()
op = O.get_operator("P1", s)
if kind == "ours":
    net = M.DirectLR(s, 96, 12); net.load_state_dict(torch.load(arg, map_location="cpu")); net = net.to(dev).eval()
if kind in ("ours", "torch_bicubic"):
    op_t = M.TorchOperator("P1", s, 512, dev)
if kind == "pretrained":
    import sota_kmp as kmp
y = np.random.default_rng(0).random((512 // s, 512 // s)) * 255.0
def run():
    if kind == "classical":
        z = ops.upsample(y, s, arg, "centered")
        x = O.enforce_consistency(z, y, op) if proj else z
    elif kind == "ibp":
        x = O.enforce_consistency(ops.upsample(y, s, "bicubic_a05", "centered"), y, op, "ibp")
    elif kind == "pretrained":
        z = np.clip(kmp.net_upscale(arg, y, s), 0, 255)
        x = O.enforce_consistency(z, y, op) if proj else z
    else:
        yt = torch.from_numpy((y / 255.0).astype(np.float32))[None, None].to(dev)
        z = net(yt) if kind == "ours" else F.interpolate(yt, scale_factor=s, mode="bicubic", align_corners=False)
        x = (op_t.consistent(z, yt) if proj else z)[0, 0].cpu().numpy() * 255.0
    return np.clip(x, 0, 255)
warm, reps = {warm!r}, {reps!r}
with torch.no_grad():
    run()
    setup = time.perf_counter() - t0
    for _ in range(warm): run()
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); run(); ts.append((time.perf_counter() - a) * 1e3)
q = np.percentile(ts, [25, 50, 75])
print(json.dumps({{"median_ms": float(q[1]), "iqr_ms": float(q[2] - q[0]), "reps": reps, "warm": warm, "setup_s_incl_first_call": setup}}))
'''


def pipelines(s: int, ours_ck: Path) -> dict[str, tuple]:
    p = {f"classical:{m}": ("classical", m, False) for m in INTERP}
    p |= {f"classical:{m}+proj": ("classical", m, True) for m in INTERP_PROJ}
    p |= {"classical:ibp20_bicubic_a05+proj": ("ibp", None, True), "classical:torch_bicubic": ("torch_bicubic", None, False),
          "classical:torch_bicubic+proj": ("torch_bicubic", None, True)}
    p |= {"ours:DirectLR w96 d12": ("ours", str(ours_ck), False), "ours:DirectLR w96 d12+proj": ("ours", str(ours_ck), True)}
    p |= {"pretrained:SwinIR-M": ("pretrained", "SwinIR-M", False), "pretrained:SwinIR-M+proj": ("pretrained", "SwinIR-M", True),
          "pretrained:HAT-L": ("pretrained", "HAT: HAT-L_SRxS_ImageNet-pretrain", False),
          "pretrained:HAT-L+proj": ("pretrained", "HAT: HAT-L_SRxS_ImageNet-pretrain", True)}
    return p


def main() -> None:
    import platform
    import torch
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["_protocol"] = {"device": "Apple M2 Pro 8GB, MPS (torch %s), numpy/scipy CPU default threads" % torch.__version__, "machine": platform.platform(),
                        "path": "host LR numpy -> method -> projection -> clip -> host numpy", "idle": "run after training finished; one process per measurement"}
    rng = np.random.default_rng(0)
    ck_dir = ROOT / "method_development" / "checkpoints"
    for s in (int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "8,4,2").split(",")):
        cks = sorted(ck_dir.glob(f"fs_x{s}_*_s7_psnr.pt"))
        ours_ck = cks[-1] if cks else ck_dir / "final_bal_s7.pt"
        pipes = pipelines(s, ours_ck)
        for rot in range(5):
            for key in rng.permutation(list(pipes)):
                kind, arg, proj = pipes[key]
                if kind == "pretrained" and rot > 0:
                    continue
                rec_key = f"x{s}|{key}"
                if len(res.get(rec_key, {}).get("runs", [])) > rot:
                    continue
                warm, reps = ((1, 5) if "HAT" in str(arg) else (2, 10)) if kind == "pretrained" else ((5, 30) if kind == "ibp" else (20, 100))
                code = CHILD.format(md=str(ROOT / "method_development"), pilot=str(ROOT / "pilot"), dl=str(ROOT / "dl"), kind=kind, arg=arg, proj=proj, s=s, warm=warm, reps=reps)
                r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
                if r.returncode != 0:
                    print("FAILED", rec_key, r.stderr[-800:], flush=True)
                    continue
                rec = res.setdefault(rec_key, {"scale": s, "pipeline": key, "kind": kind, "timed_with": arg, "projection": proj, "runs": []})
                rec["runs"].append(json.loads(r.stdout.strip().splitlines()[-1]))
                rec["median_of_medians_ms"] = float(np.median([x["median_ms"] for x in rec["runs"]]))
                rec["max_iqr_ms"] = float(max(x["iqr_ms"] for x in rec["runs"]))
                rec["median_iqr_ms"] = float(np.median([x["iqr_ms"] for x in rec["runs"]]))
                print(rot, rec_key, round(rec["runs"][-1]["median_ms"], 3), flush=True)
                OUT.write_text(json.dumps(res, indent=1))
                time.sleep(1)
        bic = res[f"x{s}|classical:torch_bicubic"]["median_of_medians_ms"]
        for k, v in res.items():
            if k.startswith(f"x{s}|"):
                v["ratio_vs_torch_bicubic"] = v["median_of_medians_ms"] / bic
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
