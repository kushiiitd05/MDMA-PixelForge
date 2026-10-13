"""Assignment 2 — Interpolation-Based Super-Resolution: compute every required result and figure.

Usage (from submission/):  python code/run_assignment.py
Outputs: results/*.csv, results/lr/*.png, figures/*.png
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import platform
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import a2_eval as ev  # noqa: E402
import a2_ops as ops  # noqa: E402

ev.DATASET = ROOT / "data"
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
SCALES = (2, 4, 8)
IMAGES = {  # selected by a pre-registered content rule (see README), before any SR score was computed
    "4.2.03.tiff": "Mandrill — dense fur/hair texture",
    "7.1.02.tiff": "Aerial jet on sand — large smooth areas, sparse strong edges",
    "7.1.03.tiff": "Tank — mid-frequency structure on grainy background",
    "4.2.06.tiff": "Sailboat on lake — mixed: foliage texture, water, sky, thin mast",
}
METHODS = ("nearest", "bilinear", "quad_local", "spline2", "bicubic_a05", "spline3", "spline4", "spline5", "lanczos3")
LABELS = {
    "nearest": "Nearest (0)", "bilinear": "Bilinear (1)", "quad_local": "Quadratic local (2)",
    "spline2": "Quadratic B-spline (2)", "bicubic_a05": "Bicubic Keys a=-0.5 (3)", "spline3": "Cubic B-spline (3)",
    "spline4": "Quartic B-spline (4)", "spline5": "Quintic B-spline (5)", "lanczos3": "Lanczos-3 (sinc ref.)",
}
cv2.setNumThreads(1)


def load() -> dict[str, np.ndarray]:
    return {n: ev.load_gray(ev.DATASET / n) for n in IMAGES}


def make_lrs(images):
    (RESULTS / "lr").mkdir(parents=True, exist_ok=True)
    lrs = {}
    for name, img in images.items():
        for s in SCALES:
            lr, _ = ops.make_lr(img, s, "P1_gauss_rec")
            lrs[(name, s)] = lr
            cv2.imwrite(str(RESULTS / "lr" / f"{Path(name).stem}_{512 // s}.png"), np.clip(np.rint(lr), 0, 255).astype(np.uint8))
    return lrs


def local_overshoot(lr, raw, s):
    import a2_neighborhood_min as nbm
    return nbm.local_range_overshoot(lr, raw, s)


def quality(images, lrs) -> pd.DataFrame:
    import a2_neighborhood_min as nbm
    rows = []
    for name, img in images.items():
        masks = ev.region_masks(img)
        for s in SCALES:
            for m in METHODS:
                raw = ops.upsample(lrs[(name, s)], s, m, "centered")
                rows.append({"image": name, "scale": s, "input": f"{512 // s}x{512 // s}", "method": m, "label": LABELS[m],
                             **ev.score(img, raw), **ev.region_psnr(img, raw, masks),
                             "local_overshoot": nbm.local_range_overshoot(lrs[(name, s)], raw, s)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "quality.csv", index=False)
    return df


def runtime(lrs) -> pd.DataFrame:
    name = next(iter(IMAGES))
    rows = []
    for s in SCALES:
        lr = lrs[(name, s)]
        lr32 = lr.astype(np.float32)
        for m in METHODS:
            track = "SciPy B-spline (orders 0-5, same code path)" if m in ("nearest", "bilinear") or m.startswith("spline") \
                else "NumPy separable kernel (reference)"
            rows.append({"scale": s, "track": track, "method": m, "label": LABELS[m],
                         **ev.time_call(lambda: ops.upsample(lr, s, m, "centered"), repeats=30, warmup=3)})
        for label, flag in (("Nearest (0)", cv2.INTER_NEAREST), ("Bilinear (1)", cv2.INTER_LINEAR),
                            ("Bicubic Keys a=-0.75 (3)", cv2.INTER_CUBIC), ("Lanczos-4 (sinc ref.)", cv2.INTER_LANCZOS4)):
            rows.append({"scale": s, "track": "OpenCV optimised", "method": f"cv2_{flag}", "label": label,
                         **ev.time_call(lambda: cv2.resize(lr32, (512, 512), interpolation=flag), repeats=100, warmup=5)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "runtime.csv", index=False)
    return df


def crop_boxes(img: np.ndarray) -> dict[str, tuple[int, int]]:
    """Fixed 96x96 crops chosen from the HR image for display only (edge-richest and texture-richest)."""
    masks = ev.region_masks(img)
    out = {}
    for key in ("edge", "texture"):
        dens = masks[key][:480, :480].reshape(20, 24, 20, 24).mean(axis=(1, 3))
        dens[[0, -1], :] = dens[:, [0, -1]] = -1
        r, c = np.unravel_index(np.argmax(dens), dens.shape)
        out[key] = (int(r) * 24, int(c) * 24)
    return out


def fig_originals_and_lr(images, lrs):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.6))
    for r, (name, img) in enumerate(images.items()):
        axes[r, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title(f"{name}  512x512", fontsize=8)
        for c, s in enumerate(SCALES, start=1):
            axes[r, c].imshow(lrs[(name, s)], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[r, c].set_title(f"{512 // s}x{512 // s} (shown enlarged)", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_originals_and_lr.png", dpi=150)
    plt.close(fig)


def fig_sr_grids(images, lrs):
    show = ("nearest", "bilinear", "spline2", "bicubic_a05", "spline3", "spline5", "lanczos3")
    for name, img in images.items():
        boxes = crop_boxes(img)
        fig, axes = plt.subplots(6, len(show) + 1, figsize=(1.75 * (len(show) + 1), 1.95 * 6))
        row = 0
        for key, (y, x) in boxes.items():
            for s in SCALES:
                axes[row, 0].imshow(img[y:y + 96, x:x + 96], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                axes[row, 0].set_ylabel(f"{key}\n{s}x", fontsize=8)
                for c, m in enumerate(show, start=1):
                    out = np.clip(ops.upsample(lrs[(name, s)], s, m, "centered"), 0, 255)
                    axes[row, c].imshow(out[y:y + 96, x:x + 96], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                    if row == 0:
                        axes[row, c].set_title(LABELS[m], fontsize=6.5)
                row += 1
        axes[0, 0].set_title("original crop", fontsize=6.5)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(FIGURES / f"fig2_sr_crops_{Path(name).stem}.png", dpi=150)
        plt.close(fig)


def fig_full_examples(images, lrs):
    name = "4.2.06.tiff"
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    for r, s in enumerate(SCALES):
        for c, m in enumerate(("nearest", "bilinear", "bicubic_a05", "spline5")):
            axes[r, c].imshow(np.clip(ops.upsample(lrs[(name, s)], s, m, "centered"), 0, 255), cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_title(f"{512 // s}->512 ({s}x)  {LABELS[m]}", fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_full_sr_sailboat.png", dpi=120)
    plt.close(fig)


def fig_quality_vs_scale(df):
    agg = df.groupby(["scale", "method"])[["psnr", "ssim"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in METHODS:
        sub = agg[agg.method == m]
        style = "--" if m in ("quad_local", "lanczos3") else "-"
        axes[0].plot(sub.scale, sub.psnr, style, marker="o", label=LABELS[m])
        axes[1].plot(sub.scale, sub.ssim, style, marker="o", label=LABELS[m])
    for ax, lab in zip(axes, ("mean PSNR (dB)", "mean SSIM")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(SCALES, ["2x (256)", "4x (128)", "8x (64)"])
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_quality_vs_scale.png", dpi=150)
    plt.close(fig)


def fig_runtime_vs_quality(df, rt):
    q = df.groupby(["scale", "method"]).psnr.mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    for ax, s in zip(axes, SCALES):
        sub = rt[(rt.scale == s) & (rt.method.isin(METHODS))]
        for _, r in sub.iterrows():
            ax.scatter(r.ms_median, q[(s, r.method)], s=25)
            ax.annotate(r.label.split(" (")[0], (r.ms_median, q[(s, r.method)]), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_title(f"{s}x", fontsize=9)
        ax.set_xlabel("median runtime (ms, log)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean PSNR (dB)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_runtime_vs_psnr.png", dpi=150)
    plt.close(fig)


def fig_artifacts(images, lrs):
    name = "7.1.02.tiff"
    img = images[name]
    s = 4
    methods = ("nearest", "bilinear", "bicubic_a05", "spline5", "lanczos3")
    y, x = crop_boxes(img)["edge"]
    fig, axes = plt.subplots(2, len(methods), figsize=(2.3 * len(methods), 4.8))
    for c, m in enumerate(methods):
        raw = ops.upsample(lrs[(name, s)], s, m, "centered")
        err = (raw - img)[y:y + 96, x:x + 96]
        axes[0, c].imshow(np.clip(raw, 0, 255)[y:y + 96, x:x + 96], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0, c].set_title(LABELS[m], fontsize=7)
        axes[1, c].imshow(err, cmap="RdBu_r", vmin=-60, vmax=60, interpolation="nearest")
    axes[1, 0].set_ylabel("signed error\n(-60..60)", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_error_maps.png", dpi=150)
    plt.close(fig)

    step = np.zeros((32, 32))
    step[:, 16:] = 200.0
    step += 20.0
    fig, ax = plt.subplots(figsize=(7, 3))
    xs = np.arange(32 * 8)
    for m in ("nearest", "bilinear", "bicubic_a05", "spline3", "spline5", "lanczos3"):
        ax.plot(xs, ops.upsample(step, 8, m, "centered")[16], label=LABELS[m], lw=1.2)
    ax.set_xlim(96, 160)
    ax.axhline(20, color="k", lw=0.5, ls=":")
    ax.axhline(220, color="k", lw=0.5, ls=":")
    ax.set_title("8x enlargement of an ideal step edge (row profile): staircase, blur, overshoot/ringing", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_step_edge_profile.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    images = load()
    lrs = make_lrs(images)
    df = quality(images, lrs)
    rt = runtime(lrs)
    fig_originals_and_lr(images, lrs)
    fig_sr_grids(images, lrs)
    fig_full_examples(images, lrs)
    fig_quality_vs_scale(df)
    fig_runtime_vs_quality(df, rt)
    fig_artifacts(images, lrs)
    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
           "scipy": scipy.__version__, "scikit-image": skimage.__version__, "opencv": cv2.__version__,
           "threads": 1, "conda_env": "eda311",
           "hashes": {n: ev.sha256(ev.DATASET / n) for n in IMAGES}}
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2))
    pd.set_option("display.width", 200)
    print(df.pivot_table(index="label", columns="scale", values=["psnr", "ssim"], aggfunc="mean").round(3))
    print(rt.pivot_table(index=["track", "label"], columns="scale", values="ms_median").round(3))


if __name__ == "__main__":
    main()
