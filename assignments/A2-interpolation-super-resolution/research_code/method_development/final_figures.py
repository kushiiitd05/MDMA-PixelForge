"""Visual comparison figures for the extension (4 assignment images, 2x/4x/8x). Display only; no scoring decisions.

Pipelines: Keys bicubic a=-0.5 (interpolation), quintic B-spline (interpolation), quintic + exact projection,
HAT-L + exact projection (pretrained), OURS (PSNR-role checkpoint of our family, + exact projection).
Crops: run_assignment.crop_boxes (edge-richest 96x96 window of the HR image, chosen before any SR output).
Output: submission/extension/figures/ext_crops_x{s}.png, ext_errors_x8.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

MD = Path(__file__).resolve().parent
ROOT = MD.parent
sys.path[:0] = [str(MD), str(ROOT / "pilot"), str(ROOT / "dl"), str(ROOT / "submission" / "code")]
import a2_eval as ev  # noqa: E402
import a2_ops as ops  # noqa: E402
import m2_models as M  # noqa: E402
import operators as O  # noqa: E402
import run_m2 as R  # noqa: E402

OUT = ROOT / "submission" / "extension" / "figures"
SHORT = {"4.2.03.tiff": "Mandrill", "7.1.02.tiff": "Jet/sand", "7.1.03.tiff": "Tank", "4.2.06.tiff": "Sailboat"}


def predictions(img: np.ndarray, s: int, ours_ck: str, ft_ck: str | None) -> dict[str, np.ndarray]:
    import sota_kmp as kmp
    op = O.get_operator("P1", s)
    op_t = M.TorchOperator("P1", s, 512, R.DEV)
    y = ops.make_lr(img, s, "P1_gauss_rec")[0]
    q5 = ops.upsample(y, s, "spline5", "centered")
    net = M.DirectLR(s, 96, 12)
    net.load_state_dict(torch.load(R.CK / ours_ck, map_location="cpu"))
    net = net.to(R.DEV).eval()
    with torch.no_grad():
        hat = np.clip(kmp.net_upscale("HAT: HAT-L_SRxS_ImageNet-pretrain", y, s), 0, 255)
    out = {"Keys bicubic (3)": ops.upsample(y, s, "bicubic_a05", "centered"), "Quintic B-spline (5)": q5,
           "Quintic + exact proj.": O.enforce_consistency(q5, y, op), "HAT-L + exact proj.": O.enforce_consistency(hat, y, op)}
    if ft_ck:
        import run_swinir_p1 as W
        sw = W.load_swinir(s)
        sw.load_state_dict(torch.load(R.CK / ft_ck, map_location="cpu"))
        out["SwinIR-M P1-ft + exact proj."] = R.reconstruct(W.GrayWrapper(sw.to(R.DEV).eval()), y, op_t, True).astype(np.float64)
    out["OURS + exact proj."] = R.reconstruct(net, y, op_t, True).astype(np.float64)
    return out


def main() -> None:
    import run_assignment as ra
    OUT.mkdir(parents=True, exist_ok=True)
    roles = json.loads((MD / "results" / "final_eval" / "roles_frozen.json").read_text())
    ims = {n: ev.load_gray(ev.DATASET / n) for n in SHORT}
    cache = {}
    for s in (2, 4, 8):
        mid = roles["scales"][str(s)]["ours_best"]["psnr"]
        ck = mid.replace("ours:", "").replace("+proj", "") + ".pt"
        fts = [c for c in roles["scales"][str(s)]["candidates"] if c["family"] == "finetuned" and c["method_id"].endswith("+proj")]
        ft = sorted(fts, key=lambda c: -c["p"])[0]["method_id"].replace("finetuned:", "").replace("+proj", "") + ".pt" if fts else None
        ncol = 6 + (ft is not None)
        fig, axes = plt.subplots(4, ncol, figsize=(2 * ncol, 8.6))
        for r, (name, img) in enumerate(ims.items()):
            preds = predictions(img, s, ck, ft)
            cache[(name, s)] = preds
            y0, x0 = ra.crop_boxes(img)["edge"]
            axes[r, 0].imshow(img[y0:y0 + 96, x0:x0 + 96], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[r, 0].set_ylabel(SHORT[name], fontsize=9)
            if r == 0:
                axes[r, 0].set_title("original", fontsize=8)
            for c, (lab, p) in enumerate(preds.items(), start=1):
                sc = ev.score(img, p)
                axes[r, c].imshow(np.clip(p, 0, 255)[y0:y0 + 96, x0:x0 + 96], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                axes[r, c].set_xlabel(f"{sc['psnr']:.2f} dB / {sc['ssim']:.3f}", fontsize=7)
                if r == 0:
                    axes[r, c].set_title(lab, fontsize=8)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{s}x: {512 // s}² → 512², 96×96 edge crops (full-image PSNR / SSIM under each crop); OURS = {mid.replace('ours:', '')}", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"ext_crops_x{s}.png", dpi=150)
        plt.close(fig)
    name = "4.2.06.tiff"
    img = ims[name]
    keys = [k for k in ("Quintic B-spline (5)", "Quintic + exact proj.", "HAT-L + exact proj.", "SwinIR-M P1-ft + exact proj.", "OURS + exact proj.") if k in cache[(name, 8)]]
    fig, axes = plt.subplots(2, len(keys), figsize=(3 * len(keys), 6.4))
    for c, k in enumerate(keys):
        p = np.clip(cache[(name, 8)][k], 0, 255)
        axes[0, c].imshow(p, cmap="gray", vmin=0, vmax=255)
        axes[0, c].set_title(f"{k}\n{ev.score(img, p)['psnr']:.2f} dB", fontsize=8)
        axes[1, c].imshow(np.abs(p - img), cmap="magma", vmin=0, vmax=60)
        axes[1, c].set_title("|error| (0–60 grey levels)", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "ext_errors_x8.png", dpi=130)
    plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
