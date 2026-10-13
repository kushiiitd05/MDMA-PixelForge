"""Pre-declared pool rule (before any 4x/2x training): prefer the BSD68-expanded pool iff its two-seed mean validation PSNR
exceeds the original pool's and its mean validation SSIM is not more than 0.002 below. Both sides use the best-val-PSNR
checkpoint of each seed (the original pool's 8x runs are FINAL.jsonl; same recipe, same seeds).
Prints the chosen pool id on the last line and writes results/final_scales/pool_decision.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent / "results"


def main() -> None:
    f = pd.read_json(RES / "FINAL.jsonl", lines=True)
    f = f[(f.split == "val") & f.consistency_at_output & ~f.tta]
    orig = {int(s): (float(g.psnr.mean()), float(g.ssim.mean())) for s, g in f.groupby("seed")}
    new = {}
    for s in orig:
        i = json.loads((RES / "final_scales" / f"x8_bsd_s{s}.json").read_text())
        assert i["status"] == "done", s
        new[s] = (i["best_val_psnr"], i["best_psnr_val_ssim"])
    mp = lambda d, k: sum(v[k] for v in d.values()) / len(d)
    out = {"orig_per_seed": orig, "bsd_per_seed": new, "orig_mean_psnr": mp(orig, 0), "orig_mean_ssim": mp(orig, 1),
           "bsd_mean_psnr": mp(new, 0), "bsd_mean_ssim": mp(new, 1)}
    out["delta_psnr"] = out["bsd_mean_psnr"] - out["orig_mean_psnr"]
    out["delta_ssim"] = out["bsd_mean_ssim"] - out["orig_mean_ssim"]
    out["chosen_pool"] = "bsd" if out["delta_psnr"] > 0 and out["delta_ssim"] >= -0.002 else "orig"
    (RES / "final_scales" / "pool_decision.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(out["chosen_pool"])


if __name__ == "__main__":
    main()
