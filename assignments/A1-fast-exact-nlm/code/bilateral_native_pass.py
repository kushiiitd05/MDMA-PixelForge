"""Native-resolution visual pass for the four selected images.

This pass uses the original selected image sizes instead of resizing to 256x256.
It is still a controlled denoising experiment: synthetic Gaussian noise is added
to the native clean image, then outputs are scored against that native clean
image and saved as visual comparison sheets.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from bilateral_dataset_pipeline import read_image
from bilateral_foundations import dataframe_markdown, gaussian_from_scratch, metrics
from bilateral_recovery import NOISE_LEVELS, apply_method, joint_bilateral
from bilateral_v1 import VALIDATION_SEED
from bilateral_v2 import masked_ltp_sbf, v2_noise_aware_selective, gradient_map


METHOD_ORDER = [
    "noisy",
    "G",
    "BF_exact",
    "BF_skimage",
    "V2_G_BF_selective",
    "JBF_gaussian_guide",
    "V2_JBF_LTP_edge",
]


def add_noise(clean: np.ndarray, sigma_n: int, image_index: int) -> np.ndarray:
    rng = np.random.default_rng(VALIDATION_SEED + image_index * 1000 + sigma_n)
    return np.clip(clean + rng.normal(0, sigma_n, clean.shape), 0, 255).astype(np.float32)


def v2_jbf_ltp_edge(noisy: np.ndarray, sigma_n: int, payload: dict,
                    ltp_best: dict) -> tuple[np.ndarray, dict]:
    jp = payload["best"]["JBF_gaussian_guide"][str(sigma_n)]
    ep = payload["best"]["BF_exact"][str(sigma_n)]
    lp = ltp_best[str(sigma_n)]
    guide = ndimage.gaussian_filter(noisy.astype(float), sigma=jp["guide_sigma"], mode="reflect")
    jbf, jt = joint_bilateral(noisy, guide, jp["sigma_s"], jp["sigma_r"], jp["k"])
    grad = gradient_map(noisy, lp["guide_sigma"])
    edge = grad >= lp["grad_threshold"]
    sbf, sd = masked_ltp_sbf(noisy, edge, ep["sigma_s"], ep["sigma_r"], ep["k"],
                             lp["ltp_ratio"] * sigma_n, lp["sigma_t"])
    out = np.where(edge, sbf, jbf).astype(np.float32)
    diag = dict(runtime_ms=jt + sd["runtime_ms"], P_G=0.0, P_BF=float((~edge).mean()),
                P_SBF=float(edge.mean()), K=ep["k"], sigma_s=ep["sigma_s"],
                sigma_r=ep["sigma_r"], guide_sigma=jp["guide_sigma"],
                ltp_threshold=lp["ltp_ratio"] * sigma_n, sigma_t=lp["sigma_t"],
                range_eval_count=noisy.size * jp["k"] * jp["k"] + sd["range_eval_count"],
                structural_eval_count=sd["structural_eval_count"],
                route_note="JBF except edge pixels use LTP-SBF")
    return out, diag


def run_one_image(clean: np.ndarray, sigma_n: int, image_index: int,
                  payload: dict, ltp_best: dict) -> list[tuple[str, np.ndarray, dict]]:
    noisy = add_noise(clean, sigma_n, image_index)
    exact_p = payload["best"]["BF_exact"][str(sigma_n)]
    outputs: list[tuple[str, np.ndarray, dict]] = [
        ("noisy", noisy, dict(runtime_ms=0.0, P_G=0.0, P_BF=0.0, P_SBF=0.0,
                              K=np.nan, sigma_s=np.nan, sigma_r=np.nan)),
    ]

    g, gd = gaussian_from_scratch(noisy, exact_p["sigma_s"], exact_p["k"])
    gd = dict(gd)
    gd.update(P_G=1.0, P_BF=0.0, P_SBF=0.0, sigma_r=np.nan, guide_sigma=np.nan)
    outputs.append(("G", g, gd))

    for method in ("BF_exact", "BF_skimage"):
        out, runtime = apply_method(method, noisy, payload["best"][method][str(sigma_n)])
        p = payload["best"][method][str(sigma_n)]
        outputs.append((method, out, dict(runtime_ms=runtime, P_G=0.0, P_BF=1.0, P_SBF=0.0,
                                          K=p["k"], sigma_s=p["sigma_s"], sigma_r=p["sigma_r"],
                                          guide_sigma=np.nan)))

    selective, sdiag, _ = v2_noise_aware_selective(noisy, sigma_n, payload)
    outputs.append(("V2_G_BF_selective", selective, sdiag))

    jbf, jt = apply_method("JBF_gaussian_guide", noisy, payload["best"]["JBF_gaussian_guide"][str(sigma_n)])
    jp = payload["best"]["JBF_gaussian_guide"][str(sigma_n)]
    outputs.append(("JBF_gaussian_guide", jbf,
                    dict(runtime_ms=jt, P_G=0.0, P_BF=1.0, P_SBF=0.0, K=jp["k"],
                         sigma_s=jp["sigma_s"], sigma_r=jp["sigma_r"],
                         guide_sigma=jp["guide_sigma"])))

    ltp, ldiag = v2_jbf_ltp_edge(noisy, sigma_n, payload, ltp_best)
    outputs.append(("V2_JBF_LTP_edge", ltp, ldiag))
    return outputs


def save_sheet(clean: np.ndarray, rows: list[tuple[str, np.ndarray, dict]],
               path: Path, title: str) -> None:
    panels = [("clean", clean, {})] + rows
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    for ax, (name, image, diag) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        runtime = diag.get("runtime_ms")
        suffix = "" if runtime is None else f"\n{runtime:.1f} ms"
        ax.set_title(f"{name}{suffix}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(out_dir: Path, selected: pd.DataFrame, results: pd.DataFrame) -> None:
    agg = results.groupby(["noise_sigma", "method"]).agg(
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        edge=("edge_metric", "mean"),
        texture=("texture_metric", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        P_G=("P_G", "mean"),
        P_BF=("P_BF", "mean"),
        P_SBF=("P_SBF", "mean"),
    ).reset_index().sort_values(["noise_sigma", "SSIM"], ascending=[True, False])
    best = []
    for sigma_n, part in agg[~agg.method.eq("noisy")].groupby("noise_sigma"):
        peak = part.SSIM.max()
        tied = part[np.isclose(part.SSIM, peak, rtol=0.0, atol=1e-9)]
        best.append(tied.sort_values("runtime_ms").iloc[0])
    best_df = pd.DataFrame(best)
    agg.to_csv(out_dir / "native_aggregate.csv", index=False)
    best_df.to_csv(out_dir / "native_best_by_noise.csv", index=False)

    text = (
        "# Native-resolution selected-image pass\n\n"
        "## Scope\n\n"
        "This pass runs the four selected images at their original native sizes. It is still a controlled denoising "
        "test because synthetic Gaussian noise is added and the native clean image is used as reference. It is not "
        "yet an all-image dataset pass.\n\n"
        "## Selected images\n\n"
        + dataframe_markdown(selected[["image_id", "role", "filename", "width", "height"]], 5)
        + "\n\n## Native aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Best denoising method by noise level\n\n"
        + dataframe_markdown(best_df, 5)
        + "\n\n## Reading the outputs\n\n"
        "`native_visual_sheets/` contains one side-by-side figure per image and noise level. The sheets show the clean "
        "native image, noisy input, Gaussian, exact BF, scikit BF, V2 selective G/BF, JBF, and V2 JBF+LTP.\n\n"
        "## Caution\n\n"
        "These native results can differ from the earlier `256x256` benchmark because image size, interpolation, "
        "and native texture frequency changed. The strongest final claim should use this pass plus the earlier "
        "controlled benchmark together.\n"
    )
    (out_dir / "Native_Selected_Image_Pass.md").write_text(text)


def run_all(output_root: str | Path = "outputs") -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "10_native_selected_pass"
    sheet_dir = out_dir / "native_visual_sheets"
    image_dir = out_dir / "native_outputs"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(root / "02_image_selection" / "final_four.csv")
    payload = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((root / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())

    rows = []
    for image_index, r in selected.reset_index(drop=True).iterrows():
        _, clean, _ = read_image(Path("misc") / r.relative_path)
        clean = clean.astype(np.float32)
        for sigma_n in NOISE_LEVELS:
            outputs = run_one_image(clean, sigma_n, image_index, payload, ltp_best)
            save_sheet(clean, outputs, sheet_dir / f"{r.image_id}_noise_{sigma_n}_native.png",
                       f"{r.image_id} / {r.filename} native {int(r.width)}x{int(r.height)} sigma_n={sigma_n}")
            method_dir = image_dir / r.image_id / f"noise_{sigma_n}"
            method_dir.mkdir(parents=True, exist_ok=True)
            for method, out, diag in outputs:
                plt.imsave(method_dir / f"{method}.png", out, cmap="gray", vmin=0, vmax=255)
                rows.append(dict(image_id=r.image_id, role=r.role, filename=r.filename,
                                 native_width=int(r.width), native_height=int(r.height),
                                 noise_sigma=sigma_n, method=method,
                                 runtime_ms=diag.get("runtime_ms", np.nan),
                                 K=diag.get("K", np.nan), sigma_s=diag.get("sigma_s", np.nan),
                                 sigma_r=diag.get("sigma_r", np.nan),
                                 guide_sigma=diag.get("guide_sigma", np.nan),
                                 ltp_threshold=diag.get("ltp_threshold", np.nan),
                                 sigma_t=diag.get("sigma_t", np.nan),
                                 P_G=diag.get("P_G", np.nan), P_BF=diag.get("P_BF", np.nan),
                                 P_SBF=diag.get("P_SBF", np.nan), **metrics(clean, out)))
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "native_results.csv", index=False)
    write_report(out_dir, selected, results)
    return results


if __name__ == "__main__":
    df = run_all()
    print(pd.read_csv("outputs/10_native_selected_pass/native_best_by_noise.csv").to_string(index=False))
