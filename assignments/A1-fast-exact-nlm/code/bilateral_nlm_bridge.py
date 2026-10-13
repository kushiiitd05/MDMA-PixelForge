"""Transparent NLM and patch-based BF bridge experiments.

Implements the next source-of-truth step after the BF-stage work:
K=7, S=21 exact patch weighting, library NLM comparison, and reconstruction
rules R0--R3 using the same exact NLM weights. R4 is documented as pending
because it changes the output aggregation topology.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.restoration import denoise_nl_means, estimate_sigma

from bilateral_dataset_pipeline import read_image
from bilateral_foundations import dataframe_markdown, metrics
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_native_pass import v2_jbf_ltp_edge
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank
from bilateral_v2 import v2_noise_aware_selective


PATCH_K = 7
SEARCH_S = 21
SEARCH_RADIUS = SEARCH_S // 2
NLM_METHODS = ("NLM_R0_center", "NLM_R1_patch_mean", "NLM_R2_patch_median",
               "NLM_R3_shrink_center", "NLM_R4_patch_consensus", "PBF_R0")


def _shift_reflect(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift image with reflect boundary, returning values at q=p+(dy,dx)."""
    h, w = image.shape
    y = np.arange(h) + dy
    x = np.arange(w) + dx
    y = np.where(y < 0, -y - 1, y)
    y = np.where(y >= h, 2 * h - y - 1, y)
    x = np.where(x < 0, -x - 1, x)
    x = np.where(x >= w, 2 * w - x - 1, x)
    return image[np.ix_(y, x)]


def patch_stats(image: np.ndarray, k: int = PATCH_K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = image.astype(np.float32)
    mean = ndimage.uniform_filter(x, size=k, mode="reflect")
    mean2 = ndimage.uniform_filter(x * x, size=k, mode="reflect")
    var = np.maximum(mean2 - mean * mean, 0.0)
    median = ndimage.median_filter(x, size=k, mode="reflect")
    return mean.astype(np.float32), var.astype(np.float32), median.astype(np.float32)


def exact_nlm_reconstruction(noisy: np.ndarray, h: float, sigma_n: float,
                             patch_k: int = PATCH_K, search_s: int = SEARCH_S,
                             mode: str = "R0") -> tuple[np.ndarray, dict]:
    """Fast exact NLM by displacement shifts plus box aggregation of patch SSD."""
    if mode not in {"R0", "R1", "R2", "R3", "PBF_R0"}:
        raise ValueError("mode must be R0, R1, R2, R3, or PBF_R0")
    x = noisy.astype(np.float32)
    radius = search_s // 2
    patch_mean, patch_var, patch_median = patch_stats(x, patch_k)
    signal_var = np.maximum(patch_var - sigma_n * sigma_n, 0.0)
    alpha = signal_var / (signal_var + sigma_n * sigma_n + 1e-6)
    shrink_center = patch_mean + alpha * (x - patch_mean)

    num = np.zeros_like(x, dtype=np.float64)
    den = np.zeros_like(x, dtype=np.float64)
    dist_sum = np.zeros_like(x, dtype=np.float64)
    dist2_sum = np.zeros_like(x, dtype=np.float64)
    weight_entropy = np.zeros_like(x, dtype=np.float64)
    top_weight = np.zeros_like(x, dtype=np.float64)
    candidates = 0
    patch_comparisons = 0
    start = time.perf_counter()
    inv_h2 = 1.0 / max(h * h, 1e-12)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_reflect(x, dy, dx)
            diff2 = (x - shifted) ** 2
            patch_dist = ndimage.uniform_filter(diff2, size=patch_k, mode="reflect")
            if mode == "PBF_R0":
                spatial = np.exp(-(dx * dx + dy * dy) / (2.0 * (radius / 2.0) ** 2))
                weight = spatial * np.exp(-patch_dist * inv_h2)
            else:
                weight = np.exp(-patch_dist * inv_h2)
            if mode == "R0" or mode == "PBF_R0":
                contrib = shifted
            elif mode == "R1":
                contrib = _shift_reflect(patch_mean, dy, dx)
            elif mode == "R2":
                contrib = _shift_reflect(patch_median, dy, dx)
            else:
                contrib = _shift_reflect(shrink_center, dy, dx)
            num += weight * contrib
            den += weight
            dist_sum += weight * patch_dist
            dist2_sum += weight * patch_dist * patch_dist
            weight_entropy -= weight * np.log(np.maximum(weight, 1e-12))
            top_weight = np.maximum(top_weight, weight)
            candidates += 1
            patch_comparisons += x.size * patch_k * patch_k

    out = (num / np.maximum(den, 1e-12)).astype(np.float32)
    mean_dist = dist_sum / np.maximum(den, 1e-12)
    var_dist = dist2_sum / np.maximum(den, 1e-12) - mean_dist * mean_dist
    prob_top = top_weight / np.maximum(den, 1e-12)
    normalized_entropy = (weight_entropy / np.maximum(den, 1e-12) + np.log(np.maximum(den, 1e-12))) / np.log(candidates)
    diag = dict(runtime_ms=(time.perf_counter() - start) * 1000.0,
                patch_k=patch_k, search_s=search_s, search_candidates=candidates,
                patch_comparison_count=patch_comparisons,
                mean_weighted_patch_distance=float(mean_dist.mean()),
                mean_patch_distance_variance=float(np.maximum(var_dist, 0).mean()),
                mean_top_weight_fraction=float(prob_top.mean()),
                mean_weight_entropy=float(np.clip(normalized_entropy, 0, 1).mean()))
    return out, diag


def exact_nlm_r4_patch_consensus(noisy: np.ndarray, h: float,
                                 patch_k: int = PATCH_K, search_s: int = SEARCH_S) -> tuple[np.ndarray, dict]:
    """R4 whole-patch reconstruction with overlap consensus.

    For each reference center p, this reconstructs a whole KxK patch from the
    same exact NLM weights, then averages all overlapping estimates that vote
    for each output pixel.
    """
    x = noisy.astype(np.float32)
    radius = search_s // 2
    patch_radius = patch_k // 2
    weights = []
    den = np.zeros_like(x, dtype=np.float64)
    candidates = 0
    patch_comparisons = 0
    start = time.perf_counter()
    inv_h2 = 1.0 / max(h * h, 1e-12)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_reflect(x, dy, dx)
            diff2 = (x - shifted) ** 2
            patch_dist = ndimage.uniform_filter(diff2, size=patch_k, mode="reflect")
            weight = np.exp(-patch_dist * inv_h2).astype(np.float32)
            weights.append((dy, dx, weight))
            den += weight
            candidates += 1
            patch_comparisons += x.size * patch_k * patch_k

    consensus_num = np.zeros_like(x, dtype=np.float64)
    consensus_den = np.zeros_like(x, dtype=np.float64)
    for uy in range(-patch_radius, patch_radius + 1):
        for ux in range(-patch_radius, patch_radius + 1):
            patch_num = np.zeros_like(x, dtype=np.float64)
            for dy, dx, weight in weights:
                patch_num += weight * _shift_reflect(x, dy + uy, dx + ux)
            patch_estimate = patch_num / np.maximum(den, 1e-12)
            consensus_num += _shift_reflect(patch_estimate, -uy, -ux)
            consensus_den += 1.0

    out = (consensus_num / np.maximum(consensus_den, 1e-12)).astype(np.float32)
    diag = dict(runtime_ms=(time.perf_counter() - start) * 1000.0,
                patch_k=patch_k, search_s=search_s, search_candidates=candidates,
                patch_comparison_count=patch_comparisons,
                mean_weight_entropy=np.nan, mean_top_weight_fraction=np.nan,
                note="R4 whole-patch reconstruction with uniform overlap consensus")
    return out, diag


def skimage_nlm(noisy: np.ndarray, sigma_n: float, h_factor: float = 0.8) -> tuple[np.ndarray, dict]:
    start = time.perf_counter()
    sigma = sigma_n / 255.0
    out = denoise_nl_means(noisy.astype(np.float32) / 255.0, patch_size=PATCH_K,
                           patch_distance=SEARCH_RADIUS, h=h_factor * sigma,
                           fast_mode=True, sigma=sigma, preserve_range=False,
                           channel_axis=None)
    diag = dict(runtime_ms=(time.perf_counter() - start) * 1000.0,
                patch_k=PATCH_K, search_s=SEARCH_S + 2,
                note="scikit patch_distance=10 gives radius 10; implementation includes library boundary conventions")
    return (out * 255.0).astype(np.float32), diag


def run_controlled(output_root: str | Path = "outputs", side: int = 128) -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "11_nlm_bridge"
    sheet_dir = out_dir / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    selection_csv = root / "02_image_selection" / "final_four.csv"
    data = load_selected(selection_csv, Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((root / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())

    h_grid = {10: (6.0, 8.0, 10.0, 12.0), 25: (15.0, 20.0, 25.0, 30.0), 50: (30.0, 40.0, 50.0, 60.0)}
    sk_h_factor_grid = (0.6, 0.8, 1.0, 1.2)
    rows = []
    best_h = {}
    best_sk_h_factor = {}
    for sigma_n in NOISE_LEVELS:
        best_h[str(sigma_n)] = {}
        for mode in ("R0", "R1", "R2", "R3", "R4", "PBF_R0"):
            best_score = -np.inf
            best_value = None
            for h in h_grid[sigma_n]:
                vals = []
                for image_id, item in data.items():
                    if mode == "R4":
                        out, _ = exact_nlm_r4_patch_consensus(tune_bank[(image_id, sigma_n)], h)
                    else:
                        out, _ = exact_nlm_reconstruction(tune_bank[(image_id, sigma_n)], h, sigma_n, mode=mode)
                    vals.append(metrics(item["clean"], out)["SSIM"])
                score = float(np.mean(vals))
                if score > best_score:
                    best_score = score
                    best_value = h
            best_h[str(sigma_n)][mode] = dict(h=best_value, tune_SSIM=best_score)
        best_sk_score = -np.inf
        best_sk_factor = None
        for factor in sk_h_factor_grid:
            vals = []
            for image_id, item in data.items():
                out, _ = skimage_nlm(tune_bank[(image_id, sigma_n)], sigma_n, factor)
                vals.append(metrics(item["clean"], out)["SSIM"])
            score = float(np.mean(vals))
            if score > best_sk_score:
                best_sk_score = score
                best_sk_factor = factor
        best_sk_h_factor[str(sigma_n)] = dict(h_factor=best_sk_factor, tune_SSIM=best_sk_score)

    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            variants = [("noisy", noisy, dict(runtime_ms=0.0))]
            bf, bf_time = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            variants.append(("BF_exact", bf, dict(runtime_ms=bf_time)))
            jbf, jbf_time = apply_method("JBF_gaussian_guide", noisy, recovery["best"]["JBF_gaussian_guide"][str(sigma_n)])
            variants.append(("JBF_gaussian_guide", jbf, dict(runtime_ms=jbf_time)))
            v2_sel, v2_sel_diag, _ = v2_noise_aware_selective(noisy, sigma_n, recovery)
            variants.append(("V2_G_BF_selective", v2_sel, v2_sel_diag))
            v2_ltp, v2_ltp_diag = v2_jbf_ltp_edge(noisy, sigma_n, recovery, ltp_best)
            variants.append(("V2_JBF_LTP_edge", v2_ltp, v2_ltp_diag))
            sk, sk_diag = skimage_nlm(noisy, sigma_n, best_sk_h_factor[str(sigma_n)]["h_factor"])
            variants.append(("NLM_skimage_fast", sk, sk_diag))
            for mode in ("R0", "R1", "R2", "R3", "R4", "PBF_R0"):
                if mode == "R4":
                    out, diag = exact_nlm_r4_patch_consensus(noisy, best_h[str(sigma_n)][mode]["h"])
                else:
                    out, diag = exact_nlm_reconstruction(noisy, best_h[str(sigma_n)][mode]["h"], sigma_n, mode=mode)
                variants.append((f"NLM_{mode}" if mode != "PBF_R0" else "PBF_R0", out, diag))
            for method, out, diag in variants:
                rows.append(dict(image_id=image_id, role=item["role"], noise_sigma=sigma_n,
                                 method=method, side=side, tune_seed=TUNE_SEED, validation_seed=VALIDATION_SEED,
                                 h=best_h.get(str(sigma_n), {}).get(method.replace("NLM_", ""), {}).get("h", np.nan),
                                 sk_h_factor=best_sk_h_factor.get(str(sigma_n), {}).get("h_factor", np.nan) if method == "NLM_skimage_fast" else np.nan,
                                 runtime_ms=diag.get("runtime_ms", np.nan),
                                 patch_k=diag.get("patch_k", np.nan),
                                 search_s=diag.get("search_s", np.nan),
                                 search_candidates=diag.get("search_candidates", np.nan),
                                 patch_comparison_count=diag.get("patch_comparison_count", np.nan),
                                 mean_weight_entropy=diag.get("mean_weight_entropy", np.nan),
                                 mean_top_weight_fraction=diag.get("mean_top_weight_fraction", np.nan),
                                 **metrics(clean, out)))
            if sigma_n == 25:
                fig, axes = plt.subplots(2, 5, figsize=(17, 7))
                axes = axes.ravel()
                panels = [("clean", clean, {})] + variants
                for ax, (name, image, diag) in zip(axes, panels):
                    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"{name}\n{diag.get('runtime_ms', 0):.1f} ms", fontsize=7)
                    ax.axis("off")
                for ax in axes[len(panels):]:
                    ax.axis("off")
                fig.suptitle(f"{image_id}: transparent NLM/PBF bridge, side={side}, sigma_n=25")
                fig.tight_layout()
                fig.savefig(sheet_dir / f"{image_id}_nlm_bridge.png", dpi=160)
                plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "nlm_bridge_results.csv", index=False)
    (out_dir / "nlm_bridge_best_h.json").write_text(json.dumps({"exact": best_h, "skimage": best_sk_h_factor}, indent=2))
    write_report(out_dir, df, best_h, best_sk_h_factor, side)
    return df


def write_report(out_dir: Path, df: pd.DataFrame, best_h: dict, best_sk_h_factor: dict, side: int) -> None:
    agg = df.groupby(["noise_sigma", "method"]).agg(
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        edge=("edge_metric", "mean"),
        texture=("texture_metric", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        mean_weight_entropy=("mean_weight_entropy", "mean"),
        mean_top_weight_fraction=("mean_top_weight_fraction", "mean"),
    ).reset_index().sort_values(["noise_sigma", "SSIM"], ascending=[True, False])
    agg.to_csv(out_dir / "nlm_bridge_aggregate.csv", index=False)
    best_rows = []
    for sigma_n, part in agg[~agg.method.eq("noisy")].groupby("noise_sigma"):
        best_rows.append(part.loc[part.SSIM.idxmax()])
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(out_dir / "nlm_bridge_best_by_noise.csv", index=False)
    h_table = []
    for noise, modes in best_h.items():
        for mode, vals in modes.items():
            h_table.append(dict(noise_sigma=int(noise), mode=mode, **vals))
    sk_table = [dict(noise_sigma=int(noise), **vals) for noise, vals in best_sk_h_factor.items()]
    text = (
        "# Transparent NLM / PBF bridge\n\n"
        "## Scope\n\n"
        f"This implements the first NLM bridge on the four selected images at side `{side}`. It uses exact "
        "`K=7, S=21` patch weights by displacement-shift patch SSD, compares against scikit's fast NLM, and tests "
        "R0--R4 reconstruction variants plus a patch-bilateral `PBF_R0` bridge. Hyperparameters are tuned on "
        f"seed `{TUNE_SEED}` and evaluated on held-out seed `{VALIDATION_SEED}`.\n\n"
        "## Tuned exact NLM h values\n\n"
        + dataframe_markdown(pd.DataFrame(h_table), 5)
        + "\n\n## Tuned scikit NLM h factors\n\n"
        + dataframe_markdown(pd.DataFrame(sk_table), 5)
        + "\n\n## Aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Best method by noise level\n\n"
        + dataframe_markdown(best_df, 5)
        + "\n\n## Go / no-go reading\n\n"
        "If an NLM/PBF row beats the BF/JBF rows at a noise level, patch information is useful there. If not, "
        "the BF-family remains the stronger baseline and later learned retrieval should be justified by cost or "
        "by cases where exact NLM already helps.\n"
    )
    (out_dir / "Transparent_NLM_PBF_Bridge.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/11_nlm_bridge/nlm_bridge_best_by_noise.csv").to_string(index=False))
