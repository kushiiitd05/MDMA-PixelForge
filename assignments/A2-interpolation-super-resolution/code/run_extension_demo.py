"""TA demo for the research extension: load the manifest weights (SHA-256 checked), super-resolve the four assignment
images at 2x/4x/8x, score them and compare with the stored final-evaluation rows.

Usage (from submission/):  python code/run_extension_demo.py
Output: results/extension_demo.csv; exit code 1 if any score differs from the stored table by more than 0.01 dB.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import a2_eval as ev  # noqa: E402
import a2_ops as ops  # noqa: E402
import sr_extension as X  # noqa: E402

IMAGES = ("4.2.03.tiff", "7.1.02.tiff", "7.1.03.tiff", "4.2.06.tiff")


def main() -> int:
    stored = pd.read_csv(ROOT / "extension" / "results" / "per_image.csv")
    stored = stored[(stored.stage == "external") & (stored.dataset == "assignment")]
    dev = X.device()
    rows = []
    for s in (2, 4, 8):
        roles = X.manifest()["models"][str(s)]
        for role in roles:
            net, entry = X.load_model(s, role, dev)
            proj = X.Projector(s, dev)
            for name in IMAGES:
                img = ev.load_gray(ROOT / "data" / name)
                lr = ops.make_lr(img, s, "P1_gauss_rec")[0]
                X.super_resolve(lr, s, net, entry["projection"], proj)  # warm-up
                t0 = time.perf_counter()
                out = X.super_resolve(lr, s, net, entry["projection"], proj)
                ms = (time.perf_counter() - t0) * 1e3
                sc = ev.score(img, out)
                ref = stored[(stored.scale == s) & (stored.method_id == entry["method_id"]) & (stored.image_id == name)]
                rows.append({"scale": s, "role": role, "method_id": entry["method_id"], "image": name, "psnr": sc["psnr"], "ssim": sc["ssim"], "ms_single_run": ms,
                             "stored_psnr": float(ref.psnr_db.iloc[0]) if len(ref) else np.nan, "lr_residual_rms": X.lr_residual(out, lr, s)})
    df = pd.DataFrame(rows)
    df["abs_diff_db"] = (df.psnr - df.stored_psnr).abs()
    df.to_csv(ROOT / "results" / "extension_demo.csv", index=False)
    print(df.groupby(["scale", "role", "method_id"])[["psnr", "ssim", "ms_single_run", "abs_diff_db", "lr_residual_rms"]].mean().round(4).to_string())
    bad = df[~(df.abs_diff_db <= 0.01)]
    if len(bad):
        print("MISMATCH vs stored results:\n", bad.to_string())
        return 1
    print("all scores reproduce the stored final-evaluation table (|diff| <= 0.01 dB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
