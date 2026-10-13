"""Collect the completed downsampler x interpolation sensitivity records (4 assignment images) into one table.

Sources (no new experiments): pilot_outputs/B1_prefilters.csv (anti-alias strength, clean), B2_down_path.csv (area,
decimation, recursive vs direct, course lattice), B8_lanczos_down.jsonl (Lanczos-3 direct / per-stage, IBP with matched D),
B9_signal.jsonl sections T2 (DCT/FFT band crop + zero-pad) and T4 (LS co-designed LR for a fixed upscaler).
Output: submission/results/downsampler_joint.csv (mean PSNR/SSIM per downsampler, upsampler, scale; n images = 4).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PO = ROOT / "pilot_outputs"
UP = {"spline3": "cubic B-spline", "cubic spline": "cubic B-spline", "spline5": "quintic B-spline", "bicubic_a05": "Keys bicubic", "lanczos3": "Lanczos-3",
      "nearest": "nearest", "NN block copy (reference)": "nearest", "bilinear": "bilinear", "IBP20 (matched D)": "IBP-20 (matched D)",
      "DCT zero-pad": "DCT zero-pad", "FFT zero-pad": "FFT zero-pad"}
DOWN = {"gauss0.5": "Gaussian σ=0.5", "gauss0.7": "Gaussian σ=0.7", "gauss1.0": "Gaussian σ=1 (P1, frozen)", "gauss1.4": "Gaussian σ=1.4", "box": "box / area",
        "P1_gauss_rec": "Gaussian σ=1 (P1, frozen)", "P1_gauss_direct": "Gaussian σ_eff direct", "area": "box / area", "decimate": "no smoothing (decimation)",
        "P0_course_rec": "Gaussian σ=1, corner lattice", "P1 gauss sigma=1 per-stage (frozen)": "Gaussian σ=1 (P1, frozen)", "lanczos3 direct": "Lanczos-3 direct",
        "lanczos3 per-stage": "Lanczos-3 per ×2 stage", "DCT-II band crop": "DCT band crop (ideal low-pass)", "FFT band crop (periodic)": "FFT band crop (periodic)",
        "LS-optimal for this upscaler (8-bit)": "co-designed for the upscaler (8-bit LR)", "LS-optimal, unquantised": "co-designed for the upscaler (float LR)",
        "Lanczos3 direct": "Lanczos-3 direct", "P1 Gaussian (frozen)": "Gaussian σ=1 (P1, frozen)"}


def main() -> None:
    parts = []
    b1 = pd.read_csv(PO / "B1_prefilters.csv")
    b1 = b1[(b1.condition == "clean") & (b1.prefilter == "none")].assign(down=lambda d: d.anti_alias, up="cubic spline", source="B1")
    parts.append(b1[["image", "scale", "down", "up", "psnr", "ssim", "source"]])
    b2 = pd.read_csv(PO / "B2_down_path.csv").rename(columns={"method": "up"}).assign(source="B2")
    parts.append(b2[["image", "scale", "down", "up", "psnr", "ssim", "source"]])
    b8 = pd.read_json(PO / "B8_lanczos_down.jsonl", lines=True).rename(columns={"degradation": "down", "method": "up"}).assign(source="B8")
    parts.append(b8[["image", "scale", "down", "up", "psnr", "ssim", "source"]])
    b9 = pd.read_json(PO / "B9_signal.jsonl", lines=True)
    b9 = b9[b9.section.str.startswith(("T2", "T4"))].rename(columns={"degradation": "down", "method": "up"}).assign(source=lambda d: "B9 " + d.section.str[:2])
    parts.append(b9[["image", "scale", "down", "up", "psnr", "ssim", "source"]])
    df = pd.concat(parts, ignore_index=True)
    df["down"], df["up"] = df.down.map(lambda v: DOWN.get(v, v)), df.up.map(lambda v: UP.get(v, v))
    df = df[df.up.isin(set(UP.values()))]
    per_img = df.groupby(["down", "up", "scale", "image"]).agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"), dup_spread=("psnr", lambda v: v.max() - v.min()),
                                                              source=("source", lambda v: "+".join(sorted(set(v))))).reset_index()
    assert (per_img.dup_spread < 0.02).all(), per_img[per_img.dup_spread >= 0.02]  # same pipeline recorded by two tracks must agree
    agg = per_img.groupby(["down", "up", "scale"]).agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"), n=("image", "nunique"), source=("source", "first")).reset_index()
    assert (agg.n == 4).all()
    out = ROOT / "submission" / "results" / "downsampler_joint.csv"
    agg.to_csv(out, index=False)
    print(agg.pivot_table(index=["down", "up"], columns="scale", values="psnr").round(2).sort_values(8, ascending=False).head(25).to_string())
    print("wrote", out)
    b9_all = pd.read_json(PO / "B9_signal.jsonl", lines=True)
    t1 = b9_all[b9_all.section.str.startswith("T1")][["scale", "filter", "gain_0.5fc", "gain_0.9fc", "gain_fc", "alias_energy_frac", "passband_energy_kept"]].drop_duplicates()
    t1.to_csv(ROOT / "submission" / "results" / "downsampler_transfer.csv", index=False)
    print("wrote downsampler_transfer.csv", len(t1))


if __name__ == "__main__":
    main()
