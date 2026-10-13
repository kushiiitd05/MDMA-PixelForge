"""Figure: every test image reconstructed back to 512² from each downsampled version.

Columns: original, then 256²→512² (2×), 128²→512² (4×), 64²→512² (8×), each reconstructed by the report's promoted
extension pipeline for that factor (model manifest) and by the best interpolation (quintic B-spline) for comparison.
Usage (from submission/):  python code/make_recon_figure.py  ->  figures/fig8_reconstructions.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import a2_eval as ev  # noqa: E402
import a2_ops as ops  # noqa: E402
import sr_extension as X  # noqa: E402

IMAGES = {"4.2.03.tiff": "Mandrill", "7.1.02.tiff": "Jet", "7.1.03.tiff": "Tank", "4.2.06.tiff": "Sailboat"}


def main() -> None:
    dev = X.device()
    nets = {}
    for s in (2, 4, 8):
        roles = X.manifest()["models"][str(s)]
        role = "promoted" if "promoted" in roles else "psnr"
        nets[s] = (*X.load_model(s, role, dev), X.Projector(s, dev))
    fig, axes = plt.subplots(4, 7, figsize=(15.4, 9.2))
    for r, (name, short) in enumerate(IMAGES.items()):
        img = ev.load_gray(ROOT / "data" / name)
        axes[r, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(short, fontsize=10)
        if r == 0:
            axes[r, 0].set_title("original 512²", fontsize=9)
        for c, s in enumerate((2, 4, 8)):
            lr = ops.make_lr(img, s, "P1_gauss_rec")[0]
            net, entry, proj = nets[s]
            outs = [("quintic B-spline", np.clip(ops.upsample(lr, s, "spline5", "centered"), 0, 255)),
                    ("OURS", X.super_resolve(lr, s, net, entry["projection"], proj))]
            for k, (lab, out) in enumerate(outs):
                ax = axes[r, 1 + 2 * c + k]
                sc = ev.score(img, out)
                ax.imshow(out, cmap="gray", vmin=0, vmax=255)
                ax.set_xlabel(f"{sc['psnr']:.2f} dB / {sc['ssim']:.3f}", fontsize=7.5)
                if r == 0:
                    ax.set_title(f"{512 // s}² → 512² ({s}×)\n{lab}", fontsize=8.5)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Reconstructions back to 512² from each downsampled version — best interpolation vs the promoted extension pipeline", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / "figures" / "fig8_reconstructions.png"
    fig.savefig(out, dpi=125)
    print("wrote", out)


if __name__ == "__main__":
    main()
