"""M2e: is the direct model data-limited?  Same architecture/budget, training pool with vs without DIV2K.

DIV2K_valid_HR (100 images, public, 2K): image-level split fixed here before training:
  0801-0880 -> training (luma, uint8, cropped to multiples of 8)
  0881-0900 -> 'div2k_eval20' external evaluation: centre 512x512 luma crop (never trained on; never used for selection)
Upstream caveat: pretrained SwinIR/HAT may have used DIV2K validation images for model selection, so their scores on
div2k_eval20 are not a clean hold-out; our models never saw 0881-0900.
Runs (val selection): plain w96 d12 @16k (+DIV2K) vs the existing @16k without DIV2K; plain w96 d12 @64k (+DIV2K);
res w96 b8 @64k (+DIV2K).  Output: method_development/results/M2e.jsonl
"""
from __future__ import annotations

import json
import sys
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

DIV = ROOT / "data_ext" / "DIV2K_valid_HR"


def div2k() -> tuple[list[torch.Tensor], dict[str, np.ndarray]]:
    train, evald = [], {}
    for f in sorted(DIV.glob("*.png")):
        k = int(f.stem)
        g = np.asarray(Image.open(f).convert("L"))
        if k <= 880:
            train.append(torch.from_numpy(np.ascontiguousarray(g[: g.shape[0] // 8 * 8, : g.shape[1] // 8 * 8])))
        else:
            h, w = g.shape
            evald[f.stem] = g[h // 2 - 256:h // 2 + 256, w // 2 - 256:w // 2 + 256].astype(np.float64)
    return train, evald


RUNS = [
    ("plain w96 d12 it16k +DIV2K", 16000, lambda: M.DirectLR(8, 96, 12), "m2e_plain_w96d12_16k_div2k"),
    ("plain w96 d12 it64k +DIV2K", 64000, lambda: M.DirectLR(8, 96, 12), "m2e_plain_w96d12_64k_div2k"),
    ("res w96 b8 it64k +DIV2K", 64000, lambda: M.ResDirect(8, 96, 8), "m2e_res_w96b8_64k_div2k"),
]


def main() -> None:
    sets = R.splits()
    extra, sets["div2k_eval20"] = div2k()
    print("DIV2K train", len(extra), "eval", len(sets["div2k_eval20"]), flush=True)
    orig_eval = R.evaluate

    def evaluate_with_div2k(net, s, op_full, tag, tta=False):
        rows = orig_eval(net, s, op_full, tag, tta)
        for name, img in s["div2k_eval20"].items():
            y = ops.make_lr(img, R.S, "P1_gauss_rec")[0]
            nn_psnr = ev.psnr(img, np.kron(y, np.ones((R.S, R.S))))
            sc = ev.score(img, R.reconstruct(net, y, op_full, True))
            rows.append({"variant": tag, "split": "div2k_eval20", "image": name, "consistency_at_output": True, "tta": False, **sc, "d_psnr_vs_nn": sc["psnr"] - nn_psnr})
        return rows

    R.evaluate = evaluate_with_div2k
    log = R.RES / "M2e.jsonl"
    log.unlink(missing_ok=True)
    for name, iters, fn, ck in RUNS:
        rows = R.train_variant("direct", False, sets, iters, name=name, tta=True, model_fn=fn, ck_tag=ck, extra_pool_u8=extra)
        with log.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        import pandas as pd
        df = pd.DataFrame(rows)
        print(name, df[df.consistency_at_output & ~df.tta].groupby("split").psnr.mean().round(3).to_dict(), "tta",
              df[df.tta].groupby("split").psnr.mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
