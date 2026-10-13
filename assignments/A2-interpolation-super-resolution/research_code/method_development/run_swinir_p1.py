"""Matched-degradation transformer baseline: pretrained SwinIR-M (DIV2K, bicubic) fine-tuned on the frozen P1 operator.

Purpose: separate "degradation mismatch" from "architecture" in the pretrained comparison. Same training images and
group-balanced sampler as the final DirectLR runs (run_final_scales.pool_and_weights), same exact P1 crop operator,
same validation-only selection with exact projection. Budget fixed before training (one seed, 20260915):
  SwinIR-M x{s} official weights, grey->RGB replicate / BT.601 luma adapter (as in the zoo evaluation),
  LR crop 32 (8x) / 48 (4x) / 64 (2x) -> HR 256 / 192 / 128, batch 16 (8x, 4x) / 8 (2x: memory and time on a shared GPU), L1 on the raw output (SwinIR training loss),
  Adam 1e-4 cosine -> 0, 20k updates, fp32. Val every 1000 updates; best-val-PSNR and best-val-SSIM states saved.
Usage: python method_development/run_swinir_p1.py --scale 8 --pool bsd
Output: checkpoints/swinirP1_x{s}_{pool}_s{seed}_{psnr|ssim}.pt, results/final_scales/swinirP1_x{s}_{pool}_s{seed}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "method_development"), str(ROOT / "pilot")]
import a2_eval as ev  # noqa: E402
import a2_ops as ops  # noqa: E402
import m2_models as M  # noqa: E402
import run_final_scales as FS  # noqa: E402
import run_m2 as R  # noqa: E402

LR_CROP = {8: 32, 4: 48, 2: 64}
ITERS, EVAL_EVERY, SEED = 20000, 1000, 20260915
BATCH_BY_SCALE = {8: 16, 4: 16, 2: 8}
BATCH = 16


def load_swinir(s: int) -> torch.nn.Module:
    import spandrel
    import spandrel_extra_arches
    spandrel_extra_arches.install(ignore_duplicates=True)
    return spandrel.ModelLoader(device=R.DEV).load_from_file(str(ROOT / "weights" / "swinir" / f"001_classicalSR_DIV2K_s48w8_SwinIR-M_x{s}.pth")).model


def forward_gray(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    xp = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="reflect")
    y = net(xp.repeat(1, 3, 1, 1))
    y = 0.299 * y[:, :1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    s = y.shape[-1] // xp.shape[-1]
    return y[..., : h * s, : w * s]


class GrayWrapper(torch.nn.Module):
    """Makes a SwinIR net look like DirectLR to run_m2.reconstruct: (B,1,h,w) [0,1] -> (B,1,sh,sw) raw luma."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return forward_gray(self.net, y)


def batch(pool, rng, weights, crop: int) -> torch.Tensor:
    xs = []
    for _ in range(BATCH):
        im = pool[int(rng.choice(len(pool), p=weights))]
        if min(im.shape) < crop:
            im = pool[0]
        i, j = int(rng.integers(im.shape[0] - crop + 1)), int(rng.integers(im.shape[1] - crop + 1))
        c = im[i:i + crop, j:j + crop]
        c = torch.rot90(c.float() / 255.0 if c.dtype == torch.uint8 else c, int(rng.integers(4)))
        xs.append(torch.flip(c, (1,)) if rng.integers(2) else c)
    return torch.stack(xs)[:, None].to(R.DEV)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, required=True, choices=(2, 4, 8))
    ap.add_argument("--pool", required=True, choices=("orig", "bsd"))
    ap.add_argument("--iters", type=int, default=ITERS)
    a = ap.parse_args()
    global BATCH
    s, crop = a.scale, LR_CROP[a.scale] * a.scale
    BATCH = BATCH_BY_SCALE[s]
    tag = f"swinirP1_x{s}_{a.pool}_s{SEED}"
    info_path = FS.OUT / f"{tag}.json"
    FS.OUT.mkdir(parents=True, exist_ok=True)
    sets = R.splits()
    pool, weights, keys = FS.pool_and_weights(sets, a.pool)
    ok = np.array([min(im.shape) >= crop for im in pool])
    weights = np.where(ok, weights, 0.0)
    weights /= weights.sum()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = load_swinir(s).train()
    wrap = GrayWrapper(net)
    op_crop, op_full = M.TorchOperator("P1", s, crop, R.DEV), M.TorchOperator("P1", s, 512, R.DEV)
    info = {"tag": tag, "scale": s, "pool": a.pool, "seed": SEED, "arch": "SwinIR-M (official DIV2K bicubic weights) fine-tuned", "params": sum(p.numel() for p in net.parameters()),
            "iters": a.iters, "batch": BATCH, "hr_crop": crop, "lr": 1e-4, "schedule": "cosine", "loss": "L1(raw luma)", "sampler": "group-balanced",
            "n_pool_images_used": int(ok.sum()), "status": "running", "curve": []}

    def val() -> tuple[float, float]:
        wrap.eval()
        sc = [ev.score(img, R.reconstruct(wrap, ops.make_lr(img, s, "P1_gauss_rec")[0], op_full, True)) for img in sets["val"].values()]
        wrap.train()
        return float(np.mean([c["psnr"] for c in sc])), float(np.mean([c["ssim"] for c in sc]))

    p0, q0 = val()
    info.update(pretrained_val_psnr_proj=p0, pretrained_val_ssim_proj=q0)
    print(tag, "pretrained val (+proj)", round(p0, 3), round(q0, 4), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    best_p, best_q, t0 = -1.0, -1.0, time.perf_counter()
    for it in range(1, a.iters + 1):
        x = batch(pool, rng, weights, crop)
        loss = (wrap(op_crop.D(x)) - x).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % EVAL_EVERY == 0:
            p, q = val()
            info["curve"].append([it, round(p, 4), round(q, 5)])
            if p > best_p:
                best_p, info["best_psnr_it"], info["best_psnr_val_ssim"] = p, it, q
                torch.save(net.state_dict(), R.CK / f"{tag}_psnr.pt")
            if q > best_q:
                best_q, info["best_ssim_it"], info["best_ssim_val_psnr"] = q, it, p
                torch.save(net.state_dict(), R.CK / f"{tag}_ssim.pt")
            info.update(best_val_psnr=best_p, best_val_ssim=best_q, train_s=time.perf_counter() - t0, last_it=it)
            info_path.write_text(json.dumps(info, indent=1))
            print(f"  [{tag}] it {it} loss {loss.item():.5f} val {p:.3f} / {q:.4f} {time.perf_counter() - t0:.0f}s", flush=True)
    info["status"] = "done"
    info_path.write_text(json.dumps(info, indent=1))
    print(tag, "best val PSNR", round(best_p, 3), "at", info["best_psnr_it"], "| best val SSIM", round(best_q, 4), "at", info["best_ssim_it"], flush=True)


if __name__ == "__main__":
    main()
