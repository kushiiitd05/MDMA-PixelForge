"""SRD bounded comparison (spec section 8): A = core continuation, B = core + fixed-stencil branch, C = core + full SRD.

All three start from the same completed core checkpoint and share: LR generated once per FULL training image with the real
P1 operator (no reflected crop boundaries), LR input crops of 32^2 (8-px real halo) with MSE on the centre 16^2 LR
(128^2 HR), source-unit-uniform sampling (each still image = 1 unit, each Vid4 sequence = 1 unit, frame uniform inside),
identical minibatch sequence (seeded), Adam lr 5e-5 cosine -> 0, batch 16, 16k updates, raw MSE (lambda = 1).
B/C first get a reduced-rank fit of (C, W_o) on <=200k training block samples with the core and branch frozen.
Selection: full-image projected val PSNR every 1000 updates. Held-out splits are scored only for the val-selected state.
Usage: python method_development/run_srd.py --core final_bal_s20260915 --runs A,B,C
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "method_development"), str(ROOT / "pilot"), str(ROOT / "dl")]
import a2_eval as ev  # noqa: E402
import a2_ops as ops  # noqa: E402
import m2_models as M  # noqa: E402
import run_m2 as R  # noqa: E402
import srd_model as S  # noqa: E402
from run_sota_finetune import degrade  # noqa: E402  (verified == make_lr to 1e-9 on square images)

HALO, CORE_LR, BATCH, S8 = 8, 16, 16, 8
IN_LR = CORE_LR + 2 * HALO
DIV = ROOT / "data_ext" / "DIV2K_valid_HR"


def build_units() -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """List of source units; each unit is a list of (HR float32 [0,1], LR float32 [0,1]) full-image pairs."""
    sets = R.splits()
    units, video = [], {}
    for k, im in sets["train"].items():
        hr = im[: im.shape[0] // S8 * S8, : im.shape[1] // S8 * S8]
        if min(hr.shape) // S8 < IN_LR:
            continue
        pair = ((hr / 255.0).astype(np.float32), (degrade(hr, S8) / 255.0).astype(np.float32))
        if k.startswith("vid4/"):
            video.setdefault(k.split("/")[1], []).append(pair)
        else:
            units.append([pair])
    for f in sorted(DIV.glob("*.png")):
        if int(f.stem) <= 880:
            g = np.asarray(Image.open(f).convert("L"), np.float64)
            g = g[: g.shape[0] // S8 * S8, : g.shape[1] // S8 * S8]
            units.append([((g / 255.0).astype(np.float32), (degrade(g, S8) / 255.0).astype(np.float32))])
    units += list(video.values())
    return units


def sample(units, rng) -> tuple[torch.Tensor, torch.Tensor]:
    ys, xs = [], []
    for _ in range(BATCH):
        unit = units[int(rng.integers(len(units)))]
        hr, lr = unit[int(rng.integers(len(unit)))]
        i, j = int(rng.integers(lr.shape[0] - IN_LR + 1)), int(rng.integers(lr.shape[1] - IN_LR + 1))
        a = lr[i:i + IN_LR, j:j + IN_LR]
        b = hr[(i + HALO) * S8:(i + HALO + CORE_LR) * S8, (j + HALO) * S8:(j + HALO + CORE_LR) * S8]
        rot, flip = int(rng.integers(4)), bool(rng.integers(2))
        a, b = np.rot90(a, rot), np.rot90(b, rot)
        if flip:
            a, b = a[:, ::-1], b[:, ::-1]
        ys.append(np.ascontiguousarray(a))
        xs.append(np.ascontiguousarray(b))
    return torch.from_numpy(np.stack(ys))[:, None].to(R.DEV), torch.from_numpy(np.stack(xs))[:, None].to(R.DEV)


def centre(t: torch.Tensor) -> torch.Tensor:
    o = HALO * S8
    return t[..., o:o + CORE_LR * S8, o:o + CORE_LR * S8]


@torch.no_grad()
def fit_head(net: S.SRDNet, units, seed: int, max_samples: int = 200_000) -> dict:
    rng = np.random.default_rng(seed + 1)
    st = S.RegressionStats(32, S8 * S8)
    t0 = time.perf_counter()
    while st.n < max_samples:
        y, x = sample(units, rng)
        h, z0 = S.core_forward_with_features(net.core, y)
        phi = net.branch.latent(y, h)[..., HALO:HALO + CORE_LR, HALO:HALO + CORE_LR]
        e = F.pixel_unshuffle(x - centre(z0), S8)
        st.add(phi.permute(0, 2, 3, 1).reshape(-1, 32).cpu().double().numpy(), e.permute(0, 2, 3, 1).reshape(-1, S8 * S8).cpu().double().numpy())
    C, W_o, info = st.fit(net.branch.rank)
    net.branch.C.weight.copy_(torch.from_numpy(C).float().view_as(net.branch.C.weight))
    net.branch.W_o.weight.copy_(torch.from_numpy(W_o).float().view_as(net.branch.W_o.weight))
    info["fit_s"] = time.perf_counter() - t0
    info["singular_values"] = info["singular_values"][:20]
    return info


def val_psnr(net, sets, op) -> float:
    net.eval()
    v = float(np.mean([ev.psnr(img, R.reconstruct(net, ops.make_lr(img, S8, "P1_gauss_rec")[0], op, True)) for img in sets["val"].values()]))
    net.train()
    return v


def run(label: str, core_ck: str, units, sets, iters: int, seed: int) -> list[dict]:
    core = M.DirectLR(8, 96, 12)
    core.load_state_dict(torch.load(R.CK / f"{core_ck}.pt", map_location="cpu"))
    branch = None if label == "A" else S.SRDBranch(96, 8, 16, "fixed" if label == "B" else "full", seed=seed)
    net = S.SRDNet(core, branch).to(R.DEV)
    op = M.TorchOperator("P1", 8, 512, R.DEV)
    info = {"label": label, "core": core_ck, "params": sum(p.numel() for p in net.parameters())}
    info["val_start"] = val_psnr(net, sets, op)
    if branch is not None:
        info["fit"] = fit_head(net, units, seed)
        info["val_after_fit"] = val_psnr(net, sets, op)
    print(label, {k: v for k, v in info.items() if k != "fit"}, flush=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    ck = R.CK / f"srd_{label}_{core_ck}.pt"
    best, t0 = info.get("val_after_fit", info["val_start"]), time.perf_counter()
    torch.save(net.state_dict(), ck)
    info["curve"] = []
    for it in range(1, iters + 1):
        y, x = sample(units, rng)
        loss = (centre(net(y)) - x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0:
            v = val_psnr(net, sets, op)
            info["curve"].append((it, round(v, 4)))
            print(f"  [{label}] it {it} loss {loss.item():.5f} val {v:.3f} {time.perf_counter() - t0:.0f}s", flush=True)
            if v > best:
                best = v
                torch.save(net.state_dict(), ck)
    info.update(train_s=time.perf_counter() - t0, best_val=best, updates=iters, examples=iters * BATCH)
    net.load_state_dict(torch.load(ck, map_location=R.DEV))
    net.eval()
    rows = R.evaluate(net, sets, op, f"SRD-{label} from {core_ck}")
    from run_m2e_data import div2k
    _, dv = div2k()
    for name, img in dv.items():
        yv = ops.make_lr(img, 8, "P1_gauss_rec")[0]
        rows.append({"variant": f"SRD-{label} from {core_ck}", "split": "div2k_eval20", "image": str(int(name)), "consistency_at_output": True, "tta": False,
                     **ev.score(img, R.reconstruct(net, yv, op, True))})
    for r in rows:
        r.update({k: v for k, v in info.items() if k not in ("curve", "fit")}, label=label)
    (R.RES / f"srd_{label}_{core_ck}_info.json").write_text(json.dumps(info, indent=1, default=float))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="final_bal_s20260915")
    ap.add_argument("--runs", default="A,B,C")
    ap.add_argument("--iters", type=int, default=16000)
    args = ap.parse_args()
    sets = R.splits()
    units = build_units()
    n_video = sum(len(u) > 1 for u in units)
    print(f"units {len(units)} (still {len(units) - n_video}, video sequences {n_video}); video sampling mass {n_video / len(units):.4f}", flush=True)
    log = R.RES / "SRD.jsonl"
    for label in args.runs.split(","):
        rows = run(label, args.core, units, sets, args.iters, 20260915)
        with log.open("a") as f:
            f.writelines(json.dumps(r, default=float) + "\n" for r in rows)
        import pandas as pd
        d = pd.DataFrame(rows)
        print(label, d[d.consistency_at_output].groupby("split").psnr.mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
