"""M2: direct vs progressive null-space predictors at 8x under exact P1 consistency (controlled training).

Splits (image level, fixed before training):
  train : misc dev images except VAL, misc 256^2 natural images, Vid4 GT frames (luma)
  val   : 4 held-out dev images (checkpoint selection)
  fresh : 12 non-overlapping 512^2 crops of the three misc 1024^2 images (never trained on; generalisation check)
  assignment : the 4 frozen assignment test images (historical comparison; not used for any selection)
Budget (same for every variant): 4000 iterations, batch 16, HR crop 128, Adam 2e-4 cosine, final-image MSE, seed fixed.
Variants: {DirectLR, Progressive} x {consistency in training, consistency at inference only}.
Output: method_development/results/M2.jsonl, checkpoints method_development/checkpoints/.
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

RES = ROOT / "method_development" / "results"
CK = ROOT / "method_development" / "checkpoints"
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
VAL = ("4.2.01.tiff", "7.1.05.tiff", "boat.512.tiff", "5.2.09.tiff")
FRESH_SRC = ("5.3.01.tiff", "5.3.02.tiff", "7.2.01.tiff")
S, CROP, BATCH, SEED = 8, 128, 16, 20260915


def splits() -> dict[str, dict[str, np.ndarray]]:
    chosen = json.loads((ROOT / "pilot_outputs" / "image_selection.json").read_text())["chosen"]
    out = {"assignment": {n: ev.load_gray(ev.DATASET / n) for n in chosen}, "val": {n: ev.load_gray(ev.DATASET / n) for n in VAL}, "fresh": {}, "train": {}}
    for n in FRESH_SRC:
        big = ev.load_gray(ev.DATASET / n)
        for i in range(2):
            for j in range(2):
                out["fresh"][f"{n[:-5]}_{i}{j}"] = big[i * 512:(i + 1) * 512, j * 512:(j + 1) * 512]
    for p in sorted(ev.DATASET.glob("*.tiff")):
        if p.name in chosen or p.name in VAL or p.name in FRESH_SRC or p.name in ev.SYNTHETIC:
            continue
        out["train"][p.name] = ev.load_gray(p)
    for f in sorted((ROOT / "data_ext" / "Vid4" / "GT").glob("*/*.png")):
        rgb = np.asarray(Image.open(f).convert("RGB"), np.float64)
        out["train"][f"vid4/{f.parent.name}/{f.name}"] = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return out


def batch(pool: list[torch.Tensor], rng: np.random.Generator, weights: np.ndarray | None = None) -> torch.Tensor:
    xs = []
    for _ in range(BATCH):
        im = pool[int(rng.choice(len(pool), p=weights)) if weights is not None else int(rng.integers(len(pool)))]
        i, j = int(rng.integers(im.shape[0] - CROP + 1)), int(rng.integers(im.shape[1] - CROP + 1))
        c = im[i:i + CROP, j:j + CROP]
        c = torch.rot90(c.float() / 255.0 if c.dtype == torch.uint8 else c, int(rng.integers(4)))
        xs.append(torch.flip(c, (1,)) if rng.integers(2) else c)
    return torch.stack(xs)[:, None].to(DEV)


@torch.no_grad()
def reconstruct(net, y: np.ndarray, op_full: M.TorchOperator, consistent: bool, tta: bool = False) -> np.ndarray:
    yt = torch.from_numpy((y / 255.0).astype(np.float32))[None, None].to(DEV)
    if tta:  # x8 dihedral self-ensemble; D is dihedral-equivariant (separable symmetric), so y transforms with x
        zs = []
        for r in range(4):
            for fl in (False, True):
                a = torch.rot90(yt, r, (2, 3))
                a = torch.flip(a, (3,)) if fl else a
                o = net(a)
                o = torch.flip(o, (3,)) if fl else o
                zs.append(torch.rot90(o, -r, (2, 3)))
        z = torch.stack(zs).mean(0)
    else:
        z = net(yt)
    x = op_full.consistent(z, yt) if consistent else z
    return x[0, 0].float().cpu().numpy() * 255.0


def evaluate(net, sets: dict, op_full: M.TorchOperator, tag: str, tta: bool = False) -> list[dict]:
    rows = []
    for split in ("val", "fresh", "assignment"):
        for name, img in sets[split].items():
            y = ops.make_lr(img, S, "P1_gauss_rec")[0]
            nn_psnr = ev.psnr(img, np.kron(y, np.ones((S, S))))
            for cons in (False, True):
                sc = ev.score(img, reconstruct(net, y, op_full, cons))
                rows.append({"variant": tag, "split": split, "image": name, "consistency_at_output": cons, "tta": False, **sc, "d_psnr_vs_nn": sc["psnr"] - nn_psnr})
            if tta:
                sc = ev.score(img, reconstruct(net, y, op_full, True, tta=True))
                rows.append({"variant": tag, "split": split, "image": name, "consistency_at_output": True, "tta": True, **sc, "d_psnr_vs_nn": sc["psnr"] - nn_psnr})
    return rows


def latency_ms(net, op_full: M.TorchOperator, runs: int = 30) -> dict:
    y = torch.rand(1, 1, 64, 64, device=DEV)
    out = {}
    for label, fn in (("net", lambda: net(y)), ("net+projection", lambda: op_full.consistent(net(y), y)),
                      ("torch bicubic", lambda: torch.nn.functional.interpolate(y, scale_factor=8, mode="bicubic"))):
        with torch.no_grad():
            for _ in range(5):
                fn()
            ts = []
            for _ in range(runs):
                t0 = time.perf_counter(); fn()
                if DEV.type == "mps":
                    torch.mps.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
        out[label] = float(np.median(ts))
    return out


def train_variant(arch: str, cons_train: bool, sets: dict, iters: int, width: int = 64, depth: int = 8, name: str | None = None, tta: bool = False, model_fn=None, ck_tag: str | None = None, extra_pool_u8: list | None = None, loss_lambda: float | None = None, group_balanced: bool = False, seed: int = SEED) -> list[dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (model_fn() if model_fn else M.DirectLR(S, width, depth) if arch == "direct" else M.Progressive()).to(DEV)
    op_crop, op_full = M.TorchOperator("P1", S, CROP, DEV), M.TorchOperator("P1", S, 512, DEV)
    pool = [torch.from_numpy((im[: im.shape[0] // S * S, : im.shape[1] // S * S] / 255.0).astype(np.float32))
            for im in sets["train"].values() if min(im.shape) >= CROP]
    pool += extra_pool_u8 or []
    weights = None
    if group_balanced:  # every Vid4 sequence counts as one source image; every other image is its own group
        keys = [k for k, im in sets["train"].items() if min(im.shape) >= CROP] + [f"extra/{i}" for i in range(len(extra_pool_u8 or []))]
        groups = [("vid4/" + k.split("/")[1]) if k.startswith("vid4/") else k for k in keys]
        size = {g: groups.count(g) for g in set(groups)}
        weights = np.array([1.0 / size[g] for g in groups]); weights /= weights.sum()
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    tag = name or f"{arch} | consistency {'in training' if cons_train else 'at inference only'}"
    ck_name = ck_tag or (f"m2_{arch}_{'ctrain' if cons_train else 'cinfer'}" + (f"_w{width}d{depth}i{iters}" if name else ""))
    best, t0 = -1.0, time.perf_counter()
    for it in range(1, iters + 1):
        x = batch(pool, rng, weights)
        y = op_crop.D(x)
        z = net(y)
        if loss_lambda is not None:  # L = ||P(z-x)||^2 + lambda ||(I-P)(z-x)||^2  (lambda = 1 equals plain MSE)
            e = z - x
            obs = op_crop.B(op_crop.D(e))
            loss = torch.mean((e - obs) ** 2) + loss_lambda * torch.mean(obs ** 2)
        else:
            pred = op_crop.consistent(z, y) if cons_train else z
            loss = torch.mean((pred - x) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == iters:
            net.eval()
            v = float(np.mean([ev.psnr(img, reconstruct(net, ops.make_lr(img, S, "P1_gauss_rec")[0], op_full, True)) for img in sets["val"].values()]))
            net.train()
            print(f"  [{tag}] it {it} loss {loss.item():.5f} val(consistent) {v:.3f} {time.perf_counter() - t0:.0f}s", flush=True)
            if v > best:
                best = v
                torch.save(net.state_dict(), CK / f"{ck_name}.pt")
    train_s = time.perf_counter() - t0
    net.load_state_dict(torch.load(CK / f"{ck_name}.pt", map_location=DEV))
    net.eval()
    rows = evaluate(net, sets, op_full, tag, tta)
    lat = latency_ms(net, op_full)
    for r in rows:
        r.update(width=width, depth=depth, params=M.count_params(net), train_s=train_s, iters=iters, examples_seen=iters * BATCH, best_val=best, **{f"ms_{k}": v for k, v in lat.items()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True)
    sets = splits()
    print({k: len(v) for k, v in sets.items()}, "device", DEV, flush=True)
    chk = M.TorchOperator("P1", S, CROP, torch.device("cpu"), torch.float64)
    xc = torch.rand(2, 1, CROP, CROP, dtype=torch.float64)
    ref = ops.make_lr(xc[0, 0].numpy(), S, "P1_gauss_rec")[0]
    assert float((chk.D(xc)[0, 0] - torch.from_numpy(ref)).abs().max()) < 1e-10
    z = torch.rand_like(xc)
    assert float((chk.D(chk.consistent(z, chk.D(xc))) - chk.D(xc)).abs().max()) < 1e-10
    log = RES / ("M2_smoke.jsonl" if args.smoke else "M2.jsonl")
    log.unlink(missing_ok=True)
    for arch in ("direct", "progressive"):
        for cons_train in (True, False):
            rows = train_variant(arch, cons_train, sets, 20 if args.smoke else args.iters)
            with log.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            import pandas as pd
            d = pd.DataFrame(rows)
            print(d.pivot_table(index="split", columns="consistency_at_output", values=["psnr", "ssim"], aggfunc="mean").round(3),
                  "params", rows[0]["params"], "ms", {k: round(v, 2) for k, v in rows[0].items() if k.startswith("ms_")}, flush=True)
            if args.smoke:
                return


if __name__ == "__main__":
    main()
