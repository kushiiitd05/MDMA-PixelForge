"""PCA-NLM compact patch representation baseline.

This stage follows the updated MD after transparent NLM: fit an image-specific
PCA patch descriptor and compare reduced descriptor dimensions against exact
NLM/PBF/BF-stage baselines on the four selected images.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA

from bilateral_foundations import dataframe_markdown, metrics
from bilateral_nlm_bridge import (PATCH_K, SEARCH_RADIUS, SEARCH_S, _shift_reflect,
                                  skimage_nlm)
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank
from bilateral_v2 import v2_noise_aware_selective


DIMS = (4, 6, 8, 12, 16)


def extract_reflect_patches(image: np.ndarray, k: int = PATCH_K) -> np.ndarray:
    radius = k // 2
    x = image.astype(np.float32)
    padded = np.pad(x, radius, mode="symmetric")
    patches = sliding_window_view(padded, (k, k)).reshape(x.size, k * k)
    return patches.astype(np.float32)


def fit_pca_descriptors(image: np.ndarray, max_dim: int = max(DIMS)) -> tuple[np.ndarray, PCA]:
    patches = extract_reflect_patches(image)
    model = PCA(n_components=max_dim, svd_solver="randomized", random_state=20260827)
    desc = model.fit_transform(patches)
    return desc.astype(np.float32).reshape(image.shape[0], image.shape[1], max_dim), model


def pca_nlm(noisy: np.ndarray, dim: int, h: float, desc: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    x = noisy.astype(np.float32)
    z = fit_pca_descriptors(x, max(DIMS))[0] if desc is None else desc
    z = z[..., :dim]
    num = np.zeros_like(x, dtype=np.float64)
    den = np.zeros_like(x, dtype=np.float64)
    candidates = 0
    start = time.perf_counter()
    inv_h2 = 1.0 / max(h * h, 1e-12)
    for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            dz = z - np.stack([_shift_reflect(z[..., c], dy, dx) for c in range(dim)], axis=-1)
            dist = np.sum(dz * dz, axis=-1) / dim
            weight = np.exp(-dist * inv_h2)
            num += weight * _shift_reflect(x, dy, dx)
            den += weight
            candidates += 1
    out = (num / np.maximum(den, 1e-12)).astype(np.float32)
    diag = dict(runtime_ms=(time.perf_counter() - start) * 1000.0, dim=dim,
                search_s=SEARCH_S, search_candidates=candidates,
                descriptor_comparison_count=x.size * candidates * dim)
    return out, diag


def tune_pca(data: dict, bank: dict, h_grid: dict) -> dict:
    best = {}
    desc_cache = {}
    for sigma_n in NOISE_LEVELS:
        best[str(sigma_n)] = {}
        for image_id in data:
            desc_cache[(image_id, sigma_n)] = fit_pca_descriptors(bank[(image_id, sigma_n)])[0]
        for dim in DIMS:
            best_score = -np.inf
            best_h = None
            for h in h_grid[sigma_n]:
                vals = []
                for image_id, item in data.items():
                    out, _ = pca_nlm(bank[(image_id, sigma_n)], dim, h, desc_cache[(image_id, sigma_n)])
                    vals.append(metrics(item["clean"], out)["SSIM"])
                score = float(np.mean(vals))
                if score > best_score:
                    best_score = score
                    best_h = h
            best[str(sigma_n)][str(dim)] = dict(h=best_h, tune_SSIM=best_score)
    return best


def run_controlled(output_root: str | Path = "outputs", side: int = 128) -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "12_pca_nlm"
    sheet_dir = out_dir / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    data = load_selected(root / "02_image_selection" / "final_four.csv", Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    nlm_h = json.loads((root / "11_nlm_bridge" / "nlm_bridge_best_h.json").read_text())
    h_grid = {
        10: (4.0, 6.0, 10.0, 16.0, 24.0, 32.0),
        25: (8.0, 10.0, 16.0, 24.0, 40.0, 60.0),
        50: (14.0, 20.0, 32.0, 50.0, 80.0, 120.0),
    }
    best = tune_pca(data, tune_bank, h_grid)
    (out_dir / "pca_nlm_best_parameters.json").write_text(json.dumps(best, indent=2))

    rows = []
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            variants = [("noisy", noisy, dict(runtime_ms=0.0))]
            bf, bf_time = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            variants.append(("BF_exact", bf, dict(runtime_ms=bf_time)))
            jbf, jbf_time = apply_method("JBF_gaussian_guide", noisy, recovery["best"]["JBF_gaussian_guide"][str(sigma_n)])
            variants.append(("JBF_gaussian_guide", jbf, dict(runtime_ms=jbf_time)))
            v2, v2diag, _ = v2_noise_aware_selective(noisy, sigma_n, recovery)
            variants.append(("V2_G_BF_selective", v2, v2diag))
            sk, skdiag = skimage_nlm(noisy, sigma_n, nlm_h["skimage"][str(sigma_n)]["h_factor"])
            variants.append(("NLM_skimage_fast", sk, skdiag))
            desc = fit_pca_descriptors(noisy)[0]
            for dim in DIMS:
                out, diag = pca_nlm(noisy, dim, best[str(sigma_n)][str(dim)]["h"], desc)
                variants.append((f"PCA_NLM_d{dim}", out, diag))
            for method, out, diag in variants:
                dim = diag.get("dim", np.nan)
                rows.append(dict(image_id=image_id, role=item["role"], noise_sigma=sigma_n,
                                 method=method, side=side, dim=dim,
                                 h=best.get(str(sigma_n), {}).get(str(int(dim)) if not np.isnan(dim) else "", {}).get("h", np.nan),
                                 runtime_ms=diag.get("runtime_ms", np.nan),
                                 descriptor_comparison_count=diag.get("descriptor_comparison_count", np.nan),
                                 **metrics(clean, out)))
            if sigma_n == 25:
                show = [("clean", clean, {})] + [v for v in variants if v[0] in
                        ("BF_exact", "NLM_skimage_fast", "PCA_NLM_d4", "PCA_NLM_d8", "PCA_NLM_d16")]
                fig, axes = plt.subplots(2, 3, figsize=(12, 7))
                axes = axes.ravel()
                for ax, (name, image, diag) in zip(axes, show):
                    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"{name}\n{diag.get('runtime_ms', 0):.1f} ms", fontsize=8)
                    ax.axis("off")
                fig.suptitle(f"{image_id}: PCA-NLM bridge, side={side}, sigma_n=25")
                fig.tight_layout()
                fig.savefig(sheet_dir / f"{image_id}_pca_nlm.png", dpi=160)
                plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pca_nlm_results.csv", index=False)
    write_report(out_dir, df, best, side)
    return df


def write_report(out_dir: Path, df: pd.DataFrame, best: dict, side: int) -> None:
    agg = df.groupby(["noise_sigma", "method"]).agg(
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        edge=("edge_metric", "mean"),
        texture=("texture_metric", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        descriptor_comparison_count=("descriptor_comparison_count", "mean"),
    ).reset_index().sort_values(["noise_sigma", "SSIM"], ascending=[True, False])
    agg.to_csv(out_dir / "pca_nlm_aggregate.csv", index=False)
    best_rows = []
    for sigma_n, part in agg[~agg.method.eq("noisy")].groupby("noise_sigma"):
        best_rows.append(part.loc[part.SSIM.idxmax()])
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(out_dir / "pca_nlm_best_by_noise.csv", index=False)
    params = []
    for noise, dims in best.items():
        for dim, vals in dims.items():
            params.append(dict(noise_sigma=int(noise), dim=int(dim), **vals))
    text = (
        "# PCA-NLM compact descriptor baseline\n\n"
        "## Scope\n\n"
        f"This runs image-specific PCA-NLM on the four selected images at side `{side}`. PCA is fit separately "
        "on each noisy image's K=7 patch cloud. Dimensions `4, 6, 8, 12, 16` are tuned on the tuning seed and "
        "evaluated on held-out validation seed. Reconstruction is standard R0 center-sample NLM.\n\n"
        "## Tuned PCA-NLM parameters\n\n"
        + dataframe_markdown(pd.DataFrame(params), 5)
        + "\n\n## Aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Best method by noise level\n\n"
        + dataframe_markdown(best_df, 5)
        + "\n\n## Reading\n\n"
        "With the widened h-grid, PCA-NLM is a strong quality baseline on this controlled 128px benchmark. "
        "The current implementation is brute-force over PCA descriptors, so its quality gain comes with high runtime "
        "and many descriptor comparisons. The next retrieval-stage question is whether ANN/PatchMatch/MSB pruning can "
        "recover much of this quality at lower total cost.\n"
    )
    (out_dir / "PCA_NLM_Baseline.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/12_pca_nlm/pca_nlm_best_by_noise.csv").to_string(index=False))
