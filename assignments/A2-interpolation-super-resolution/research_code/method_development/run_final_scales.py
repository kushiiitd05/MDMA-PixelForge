"""Final bounded campaign: plain DirectLR w96 d12 at scale s in {8, 4, 2}, 64k updates, 2 seeds, frozen recipe.

Recipe (unchanged from the selected 8x core, run_final_candidate.py): HR crop 128, y = exact P1 operator of the 128-px
crop (symmetric borders at the crop edge; established convention), batch 16, Adam 2e-4 cosine, plain MSE on the raw
output, group-balanced source sampling (each still = 1 unit, each Vid4 sequence = 1 unit), dihedral augmentation.
Pools:
  orig : misc train + Vid4 + DIV2K 0801-0880                      (the pool of the selected 8x core)
  bsd  : orig + 68 BSD68 grayscale originals (481x321), one unit each
Selection data only: the 4 validation images, scored with exact projection every 500 updates (PSNR and SSIM).
Saves the best-val-PSNR and best-val-SSIM checkpoints (distinct identities). No held-out split is scored here.
Usage: python method_development/run_final_scales.py --scale 8 --pool bsd --seeds 7,20260915
Output: checkpoints/fs_x{s}_{pool}_s{seed}_{psnr|ssim}.pt, results/final_scales/x{s}_{pool}_s{seed}.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "method_development"), str(ROOT / "pilot")]
import a2_eval as ev  # noqa: E402
import a2_ops as ops  # noqa: E402
import m2_models as M  # noqa: E402
import run_m2 as R  # noqa: E402
import run_m2e_data as E  # noqa: E402

BSD = ROOT.parent / "A1" / "paper_research" / "data" / "_sources" / "bsd68"
OUT = R.RES / "final_scales"
ITERS, EVAL_EVERY = 64000, 500


def bsd68() -> dict[str, torch.Tensor]:
    files = sorted(BSD.glob("**/*.png"))
    assert len(files) == 68, len(files)
    return {f"bsd68/{f.stem}": torch.from_numpy(np.asarray(Image.open(f).convert("L")).copy()) for f in files}


def pool_and_weights(sets: dict, pool_id: str) -> tuple[list[torch.Tensor], np.ndarray, list[str]]:
    keys = [k for k, im in sets["train"].items() if min(im.shape) >= R.CROP]
    pool = [torch.from_numpy((sets["train"][k] / 255.0).astype(np.float32)) for k in keys]
    div, _ = E.div2k()
    pool += div
    keys += [f"div2k/{801 + i:04d}" for i in range(len(div))]
    if pool_id == "bsd":
        extra = bsd68()
        pool += list(extra.values())
        keys += list(extra)
    groups = [("vid4/" + k.split("/")[1]) if k.startswith("vid4/") else k for k in keys]
    size = {g: groups.count(g) for g in set(groups)}
    w = np.array([1.0 / size[g] for g in groups])
    return pool, w / w.sum(), keys


def val_scores(net, sets: dict, op_full: M.TorchOperator, s: int) -> tuple[float, float]:
    net.eval()
    sc = [ev.score(img, R.reconstruct(net, ops.make_lr(img, s, "P1_gauss_rec")[0], op_full, True)) for img in sets["val"].values()]
    net.train()
    return float(np.mean([c["psnr"] for c in sc])), float(np.mean([c["ssim"] for c in sc]))


def train(s: int, pool_id: str, seed: int, sets: dict) -> dict:
    tag = f"fs_x{s}_{pool_id}_s{seed}"
    info_path = OUT / f"x{s}_{pool_id}_s{seed}.json"
    if info_path.exists() and json.loads(info_path.read_text()).get("status") == "done":
        print(tag, "already done; skipped", flush=True)
        return json.loads(info_path.read_text())
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = M.DirectLR(s, 96, 12).to(R.DEV)
    op_crop, op_full = M.TorchOperator("P1", s, R.CROP, R.DEV), M.TorchOperator("P1", s, 512, R.DEV)
    pool, weights, keys = pool_and_weights(sets, pool_id)
    info = {"tag": tag, "scale": s, "pool": pool_id, "seed": seed, "arch": f"DirectLR({s}, 96, 12)", "params": M.count_params(net),
            "iters": ITERS, "batch": R.BATCH, "hr_crop": R.CROP, "lr": 2e-4, "schedule": "cosine", "loss": "MSE(raw)",
            "sampler": "group-balanced", "n_pool_images": len(pool), "n_units": len(set(("vid4/" + k.split("/")[1]) if k.startswith("vid4/") else k for k in keys)),
            "pool_keys": keys, "status": "running", "curve": []}
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    best_p, best_q, t0 = -1.0, -1.0, time.perf_counter()
    for it in range(1, ITERS + 1):
        x = R.batch(pool, rng, weights)
        loss = torch.mean((net(op_crop.D(x)) - x) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % EVAL_EVERY == 0:
            p, q = val_scores(net, sets, op_full, s)
            info["curve"].append([it, round(p, 4), round(q, 5)])
            if p > best_p:
                best_p, info["best_psnr_it"], info["best_psnr_val_ssim"] = p, it, q
                torch.save(net.state_dict(), R.CK / f"{tag}_psnr.pt")
            if q > best_q:
                best_q, info["best_ssim_it"], info["best_ssim_val_psnr"] = q, it, p
                torch.save(net.state_dict(), R.CK / f"{tag}_ssim.pt")
            print(f"  [{tag}] it {it} loss {loss.item():.5f} val {p:.3f} / {q:.4f} {time.perf_counter() - t0:.0f}s", flush=True)
            info.update(best_val_psnr=best_p, best_val_ssim=best_q, train_s=time.perf_counter() - t0, last_it=it)
            info_path.write_text(json.dumps(info, indent=1))
    info["status"] = "done"
    info_path.write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, required=True, choices=(2, 4, 8))
    ap.add_argument("--pool", required=True, choices=("orig", "bsd"))
    ap.add_argument("--seeds", default="7,20260915")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sets = R.splits()
    for seed in (int(v) for v in a.seeds.split(",")):
        i = train(a.scale, a.pool, seed, sets)
        print(i["tag"], "best val PSNR", round(i["best_val_psnr"], 3), "at", i["best_psnr_it"], "| best val SSIM", round(i["best_val_ssim"], 4), "at", i["best_ssim_it"], flush=True)


if __name__ == "__main__":
    main()
