"""Final comparable evaluation at 2x/4x/8x (frozen P1). One canonical per-image table for every report.

Stages
  val       : every candidate pipeline on the 4 validation images (the only selection data).
  external  : after roles are frozen (select_roles.py), every candidate on assignment / fresh / div2k_eval20 / urban100c.
Families (--families): classical, ours, pretrained. Resumable: (stage, dataset, scale, method_id) blocks already in the
table are skipped.

Datasets
  val (4), fresh (12 crops of 3 sources), assignment (4)   : run_m2.splits()
  div2k_eval20 (20)                                        : DIV2K 0881-0900 centre 512^2 luma (run_m2e_data.div2k)
  urban100c (100)                                          : Urban100 genuine HR (SRF_4_HR copies, 1024-px), PIL "L" luma,
                                                             centre 512^2 crop (content-independent rule, fixed before
                                                             scoring). Custom benchmark, not leaderboard Urban100.
Metrics (metric_config_id M255_gauss1.5_full_clip): clip to [0,255]; PSNR R=255 on the full 512^2 frame; SSIM
  skimage Gaussian window sigma=1.5, data_range=255, population covariance, no border crop. Mean per-image values.
Projection: exact Euclidean consistency x = z + B(y - Dz). Classical/pretrained: numpy float64 (operators.py);
  ours: torch float32 on MPS (m2_models.TorchOperator), as in all earlier records. LR residual of the projected output is
  logged before and after clipping (0-255 units).
Output: method_development/results/final_eval/per_image.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "method_development"), str(ROOT / "pilot"), str(ROOT / "dl")]
import a2_eval as ev  # noqa: E402
import a2_ops as ops  # noqa: E402
import m2_models as M  # noqa: E402
import operators as O  # noqa: E402
import run_m2 as R  # noqa: E402
import run_m2e_data as E  # noqa: E402

OUT = R.RES / "final_eval"
TABLE = OUT / "per_image.jsonl"
URBAN = ROOT.parent / "A1" / "paper_research" / "data" / "_sources" / "urban100"
SCALES = (2, 4, 8)
METRIC_ID = "M255_gauss1.5_full_clip"
INTERP = ("nearest", "bilinear", "quad_local", "spline2", "bicubic_a05", "spline3", "spline4", "spline5", "lanczos3")
INTERP_PROJ = ("nearest", "bicubic_a05", "spline3", "spline5")
PRETRAINED = {"SwinIR-M": "SwinIR-M", "HAT-L": "HAT: HAT-L_SRxS_ImageNet-pretrain"}
OLD_CORE = ("final_bal_s7", "final_bal_s20260915")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def datasets(stage: str) -> dict[str, dict[str, np.ndarray]]:
    s = R.splits()
    if stage == "val":
        return {"val": s["val"]}
    _, dv = E.div2k()
    urban = {}
    for f in sorted(URBAN.glob("**/*SRF_4_HR.png")):
        g = np.asarray(Image.open(f).convert("L"), np.float64)
        h, w = g.shape
        urban[f.stem.replace("_SRF_4_HR", "")] = g[h // 2 - 256:h // 2 + 256, w // 2 - 256:w // 2 + 256]
    assert len(urban) == 100
    return {"assignment": s["assignment"], "fresh": s["fresh"], "div2k_eval20": {str(int(k)): v for k, v in dv.items()}, "urban100c": urban}


def source_of(dataset: str, image: str) -> str:
    return image.split("_")[0] if dataset == "fresh" else image


def our_checkpoints(scale: int) -> list[dict]:
    """All trained DirectLR w96 d12 checkpoints for this scale; identical weights are collapsed (first identity kept)."""
    cands = []
    for p in sorted(R.CK.glob(f"fs_x{scale}_*_s*_*.pt")):
        _, _, pool, seed, crit = p.stem.split("_")
        cands.append({"method_id": f"ours:{p.stem}", "checkpoint_id": p.name, "training_pool_id": pool, "seed": int(seed[1:]), "select_crit": f"best val {crit}", "path": p})
    if scale == 8:
        for c in OLD_CORE:
            p = R.CK / f"{c}.pt"
            cands.append({"method_id": f"ours:{c}", "checkpoint_id": p.name, "training_pool_id": "orig", "seed": int(c.split("_s")[1]), "select_crit": "best val psnr", "path": p})
    seen, out = {}, []
    for c in cands:
        h = sha(c["path"])
        if h in seen:
            seen[h].setdefault("identical_to", []).append(c["method_id"])
            continue
        c["sha256_16"] = h
        seen[h] = c
        out.append(c)
    return out


def lr_residuals(x: np.ndarray, y: np.ndarray, op) -> dict:
    return {"lr_resid_max_preclip": float(np.abs(O.apply_D(x, op) - y).max()),
            "lr_resid_rms_postclip": float(np.sqrt(np.mean((O.apply_D(np.clip(x, 0, 255), op) - y) ** 2)))}


def score_rows(img: np.ndarray, est: np.ndarray, y: np.ndarray, op, projected: bool) -> dict:
    sc = ev.score(img, est)
    row = {"psnr_db": sc["psnr"], "ssim": sc["ssim"], "overshoot_frac": sc["overshoot_frac"]}
    return {**row, **lr_residuals(est, y, op)} if projected else row


def run_classical(stage, name, imgs, scale, done, write):
    op = O.get_operator("P1", scale)
    blocks = [(m, False) for m in INTERP] + [(m, True) for m in INTERP_PROJ] + [("ibp20_bicubic_a05", True), ("torch_bicubic", False), ("torch_bicubic", True)]
    for m, proj in blocks:
        mid = f"classical:{m}{'+proj' if proj else ''}"
        if (stage, name, scale, mid) in done:
            continue
        rows = []
        for image, img in imgs.items():
            y = ops.make_lr(img, scale, "P1_gauss_rec")[0]
            if m == "ibp20_bicubic_a05":
                est = O.enforce_consistency(ops.upsample(y, scale, "bicubic_a05", "centered"), y, op, "ibp")
                impl, proj_impl = "numpy (a2_ops) + 20 IBP steps", "ibp (oblique, not exact)"
            elif m == "torch_bicubic":
                dev = R.DEV
                op_t = M.TorchOperator("P1", scale, 512, dev)
                yt = torch.from_numpy((y / 255.0).astype(np.float32))[None, None].to(dev)
                z = F.interpolate(yt, scale_factor=scale, mode="bicubic", align_corners=False)
                est = (op_t.consistent(z, yt) if proj else z)[0, 0].cpu().double().numpy() * 255.0
                impl, proj_impl = "torch MPS bicubic (a=-0.75, half-pixel)", "torch float32 MPS" if proj else "none"
            else:
                z = ops.upsample(y, scale, m, "centered")
                est = O.enforce_consistency(z, y, op) if proj else z
                impl, proj_impl = "numpy/scipy (a2_ops, centred lattice, reflect)", "numpy float64" if proj else "none"
            rows.append({"image_id": image, "impl": impl, "projection_impl": proj_impl, "projection": proj and m != "ibp20_bicubic_a05",
                         **score_rows(img, est, y, op, proj)})
        write(stage, name, scale, mid, rows, {"family": "classical", "checkpoint_id": None, "training_pool_id": None, "seed": None})


def run_ours(stage, name, imgs, scale, done, write, only: set | None):
    op = O.get_operator("P1", scale)
    op_t = M.TorchOperator("P1", scale, 512, R.DEV)
    for c in our_checkpoints(scale):
        if only is not None and c["method_id"] not in only:
            continue
        net = M.DirectLR(scale, 96, 12)
        net.load_state_dict(torch.load(c["path"], map_location="cpu"))
        net = net.to(R.DEV).eval()
        for proj in (False, True):
            mid = c["method_id"] + ("+proj" if proj else "")
            if (stage, name, scale, mid) in done:
                continue
            rows = []
            for image, img in imgs.items():
                y = ops.make_lr(img, scale, "P1_gauss_rec")[0]
                est = R.reconstruct(net, y, op_t, proj).astype(np.float64)
                rows.append({"image_id": image, "impl": "torch MPS float32", "projection_impl": "torch float32 MPS" if proj else "none", "projection": proj,
                             **score_rows(img, est, y, op, proj)})
            meta = {k: c.get(k) for k in ("checkpoint_id", "training_pool_id", "seed", "select_crit", "sha256_16", "identical_to")}
            write(stage, name, scale, mid, rows, {"family": "ours", "arch": f"DirectLR({scale}, 96, 12)", "params": M.count_params(net), **meta})


def run_pretrained(stage, name, imgs, scale, done, write):
    import sota_kmp as kmp
    from run_track_d1 import plan
    op = O.get_operator("P1", scale)
    for short, zoo in PRETRAINED.items():
        if all((stage, name, scale, f"pretrained:{short}{p}") in done for p in ("", "+proj")):
            continue
        raw, prj = [], []
        for image, img in imgs.items():
            y = ops.make_lr(img, scale, "P1_gauss_rec")[0]
            with torch.no_grad():
                z = np.clip(kmp.net_upscale(zoo, y, scale), 0, 255)
            base = {"image_id": image, "impl": f"sr_zoo {zoo}", "path": " > ".join(f"x{k}" for k in plan(kmp._registry()[zoo].native_scales, scale))}
            raw.append({**base, "projection_impl": "none", "projection": False, **score_rows(img, z, y, op, False)})
            x = O.enforce_consistency(z, y, op)
            prj.append({**base, "projection_impl": "numpy float64", "projection": True, **score_rows(img, x, y, op, True)})
            print(f"    {short} x{scale} {name}/{image} {raw[-1]['psnr_db']:.3f} -> {prj[-1]['psnr_db']:.3f}", flush=True)
        meta = {"family": "pretrained", "checkpoint_id": zoo, "training_pool_id": "upstream (bicubic degradation)", "seed": None}
        write(stage, name, scale, f"pretrained:{short}", raw, meta)
        write(stage, name, scale, f"pretrained:{short}+proj", prj, meta)


def run_finetuned(stage, name, imgs, scale, done, write):
    """SwinIR-M fine-tuned on P1 (run_swinir_p1.py): full-image forward, same grey adapter, raw and + exact projection."""
    import run_swinir_p1 as W
    op = O.get_operator("P1", scale)
    op_t = M.TorchOperator("P1", scale, 512, R.DEV)
    for p in sorted(R.CK.glob(f"swinirP1_x{scale}_*_s*_*.pt")):
        _, _, pool, seed, crit = p.stem.split("_")
        base = f"finetuned:{p.stem}"
        if all((stage, name, scale, base + sfx) in done for sfx in ("", "+proj")):
            continue
        net = W.load_swinir(scale)
        net.load_state_dict(torch.load(p, map_location="cpu"))
        wrap = W.GrayWrapper(net.to(R.DEV).eval())
        raw, prj = [], []
        for image, img in imgs.items():
            y = ops.make_lr(img, scale, "P1_gauss_rec")[0]
            z = R.reconstruct(wrap, y, op_t, False).astype(np.float64)
            x = R.reconstruct(wrap, y, op_t, True).astype(np.float64)
            raw.append({"image_id": image, "impl": "SwinIR-M P1 fine-tuned, CUDA float32, untiled", "projection_impl": "none", "projection": False, **score_rows(img, z, y, op, False)})
            prj.append({"image_id": image, "impl": "SwinIR-M P1 fine-tuned, CUDA float32, untiled", "projection_impl": "torch float32", "projection": True, **score_rows(img, x, y, op, True)})
        meta = {"family": "finetuned", "checkpoint_id": p.name, "training_pool_id": pool, "seed": int(seed[1:]), "select_crit": f"best val {crit}", "sha256_16": sha(p),
                "params": sum(q.numel() for q in net.parameters())}
        if (stage, name, scale, base) not in done:
            write(stage, name, scale, base, raw, meta)
        if (stage, name, scale, base + "+proj") not in done:
            write(stage, name, scale, base + "+proj", prj, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("val", "external"))
    ap.add_argument("--families", default="classical,ours,pretrained")
    ap.add_argument("--scales", default="2,4,8")
    ap.add_argument("--only-ours", default=None, help="comma-separated ours:* method ids (external stage: frozen finalists)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if TABLE.exists():
        for line in TABLE.read_text().splitlines():
            r = json.loads(line)
            done.add((r["stage"], r["dataset"], r["scale"], r["method_id"]))

    def write(stage, name, scale, mid, rows, meta):
        with TABLE.open("a") as f:
            for r in rows:
                f.write(json.dumps({"stage": stage, "dataset": name, "source_id": source_of(name, r["image_id"]), "scale": scale, "method_id": mid,
                                    **meta, "tta": False, "clip_policy": "final clip [0,255] (pretrained: also per stage, before projection)",
                                    "metric_config_id": METRIC_ID, **r}, default=str) + "\n")
        print(f"  {stage} {name} x{scale} {mid}: PSNR {np.mean([r['psnr_db'] for r in rows]):.3f} SSIM {np.mean([r['ssim'] for r in rows]):.4f}", flush=True)

    only = set(a.only_ours.split(",")) if a.only_ours else None
    fams = a.families.split(",")
    for scale in (int(v) for v in a.scales.split(",")):
        for name, imgs in datasets(a.stage).items():
            if "classical" in fams:
                run_classical(a.stage, name, imgs, scale, done, write)
            if "ours" in fams:
                run_ours(a.stage, name, imgs, scale, done, write, only)
            if "pretrained" in fams:
                run_pretrained(a.stage, name, imgs, scale, done, write)
            if "finetuned" in fams:
                run_finetuned(a.stage, name, imgs, scale, done, write)


if __name__ == "__main__":
    main()
