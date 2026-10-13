"""Optuna search over the local-radius candidate pool (stage 17) plus the
embedding itself — jointly tuning spatial locality, a weight-percentile
clip, retrieval/rerank dimensionality, patch kernel size `K`, KD-tree bag
size, and the range bandwidth `h`, on all four selected images at once.

New knob not in `bilateral_local_retrieval.py`: `weight_percentile_clip`.
After computing the bag's exact-descriptor weights, zero out the bottom
`clip` percent of each pixel's own bag by weight before renormalizing and
averaging — a continuous, per-pixel-adaptive generalization of "keep only
the top-K most similar candidates" (as opposed to `bag_size`, which is a
fixed count decided before any weight is even computed).

`K` (patch kernel size) is also searched — bigger patches are more
distinctive (less likely to falsely match) but blur under high noise and
cost more per PCA/descriptor comparison; this asks Optuna to find the
noise-dependent sweet spot instead of assuming the project's fixed `K=7`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from bilateral_foundations import dataframe_markdown, metrics
from bilateral_pca_nlm import extract_reflect_patches
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank

K_OPTIONS = (5, 7, 9)
DIM_OPTIONS = (4, 6, 8, 12, 16)
RETRIEVAL_DIM_OPTIONS = (2, 4, 6)
BAG_SIZE_OPTIONS = (16, 32, 64, 128)
H_RANGE = {10: (3.0, 40.0), 25: (6.0, 80.0), 50: (10.0, 150.0)}
N_TRIALS = 30


def _intra_patch_gaussian_weights(k: int, sigma_patch: float) -> np.ndarray:
    """Per-position weight for a KxK patch, center=1.0, fading toward the
    edges with sigma_patch (BF's own spatial kernel, reused one level down
    inside the patch comparison instead of at the candidate-selection level).
    `sigma_patch<=0` returns uniform (flat box) weights — today's default."""
    radius = k // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    if sigma_patch <= 0:
        return np.ones((k, k), dtype=np.float32)
    return np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma_patch ** 2)).astype(np.float32)


def fit_descriptor(image: np.ndarray, k: int, max_dim: int, sigma_patch: float = 0.0) -> np.ndarray:
    patches = extract_reflect_patches(image, k)
    if sigma_patch > 0:
        # Pre-scale each patch's raw K^2 values by sqrt(Gaussian weight):
        # (sqrt(w)*a - sqrt(w)*b)^2 = w*(a-b)^2, so a plain Euclidean/PCA
        # distance on the rescaled vector equals the Gaussian-weighted SSD
        # for free — no change needed anywhere downstream (PCA, KD-tree,
        # rerank all stay untouched).
        w = _intra_patch_gaussian_weights(k, sigma_patch).ravel()
        patches = patches * np.sqrt(w)[None, :]
    n_comp = min(max_dim, patches.shape[1])
    model = PCA(n_components=n_comp, svd_solver="randomized", random_state=20260827)
    desc = model.fit_transform(patches)
    if n_comp < max_dim:
        desc = np.pad(desc, ((0, 0), (0, max_dim - n_comp)))
    return desc.reshape(image.shape[0], image.shape[1], max_dim).astype(np.float32)


def local_pool_with_clip(noisy: np.ndarray, desc: np.ndarray, retrieval_dim: int, rerank_dim: int,
                         spatial_weight: float, bag_size: int, h: float,
                         percentile_clip: float) -> tuple[np.ndarray, dict]:
    x = noisy.astype(np.float32)
    height, width = x.shape
    flat_x = x.ravel()
    z_ret = desc[..., :retrieval_dim].reshape(-1, retrieval_dim)
    z_rer = desc[..., :rerank_dim].reshape(-1, rerank_dim)

    ys, xs = np.indices((height, width))
    coords = np.stack([ys.ravel(), xs.ravel()], axis=-1).astype(np.float32) * spatial_weight
    z_query = np.concatenate([z_ret, coords], axis=-1)

    start = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=bag_size, algorithm="kd_tree", metric="euclidean")
    nn.fit(z_query)
    _, indices = nn.kneighbors(z_query)
    kd_ms = (time.perf_counter() - start) * 1000.0

    self_idx = np.arange(height * width)[:, None]
    diff = z_rer[self_idx] - z_rer[indices]
    dist2 = np.sum(diff * diff, axis=-1) / rerank_dim
    weight = np.exp(-dist2 / max(h * h, 1e-12))

    if percentile_clip > 0:
        thresh = np.percentile(weight, percentile_clip, axis=1, keepdims=True)
        weight = np.where(weight >= thresh, weight, 0.0)
        row_sum = weight.sum(axis=1)
        zero_rows = row_sum <= 1e-12
        if np.any(zero_rows):
            weight[zero_rows, 0] = 1.0

    values = flat_x[indices]
    out = (weight * values).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-12)
    diag = dict(runtime_ms=kd_ms, bag_size=bag_size, spatial_weight=spatial_weight,
               percentile_clip=percentile_clip, retrieval_dim=retrieval_dim, rerank_dim=rerank_dim)
    return out.reshape(x.shape).astype(np.float32), diag


def run_optuna(output_root: str | Path = "outputs", side: int = 128) -> dict:
    root = Path(output_root)
    out_dir = root / "18_local_retrieval_optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_selected(root / "02_image_selection" / "final_four.csv", Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())

    results = {}
    per_image_rows = []
    for sigma_n in NOISE_LEVELS:
        h_lo, h_hi = H_RANGE[sigma_n]

        def objective(trial, sigma_n=sigma_n, h_lo=h_lo, h_hi=h_hi):
            k = trial.suggest_categorical("K", K_OPTIONS)
            rerank_dim = trial.suggest_categorical("rerank_dim", DIM_OPTIONS)
            retrieval_dim = min(trial.suggest_categorical("retrieval_dim", RETRIEVAL_DIM_OPTIONS), rerank_dim)
            bag_size = trial.suggest_categorical("bag_size", BAG_SIZE_OPTIONS)
            spatial_weight = trial.suggest_float("spatial_weight", 0.3, 60.0, log=True)
            percentile_clip = trial.suggest_float("percentile_clip", 0.0, 90.0)
            h = trial.suggest_float("h", h_lo, h_hi, log=True)
            # sigma_patch<=0 (a third of the range) disables intra-patch Gaussian
            # weighting entirely, reproducing today's uniform-box patch distance.
            sigma_patch = trial.suggest_float("sigma_patch", -1.0, k / 2.0)
            sigma_patch = max(sigma_patch, 0.0)

            vals = []
            for image_id, item in data.items():
                noisy = tune_bank[(image_id, sigma_n)]
                desc = fit_descriptor(noisy, k, rerank_dim, sigma_patch)
                out, _ = local_pool_with_clip(noisy, desc, retrieval_dim, rerank_dim,
                                              spatial_weight, bag_size, h, percentile_clip)
                vals.append(metrics(item["clean"], out)["SSIM"])
            return float(np.mean(vals))

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=sigma_n))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best = study.best_params
        best_value = study.best_value

        best_sigma_patch = max(best["sigma_patch"], 0.0)
        eval_vals = {}
        for image_id, item in data.items():
            noisy = eval_bank[(image_id, sigma_n)]
            desc = fit_descriptor(noisy, best["K"], best["rerank_dim"], best_sigma_patch)
            retrieval_dim = min(best["retrieval_dim"], best["rerank_dim"])
            out, diag = local_pool_with_clip(noisy, desc, retrieval_dim, best["rerank_dim"],
                                             best["spatial_weight"], best["bag_size"], best["h"],
                                             best["percentile_clip"])
            score = metrics(item["clean"], out)["SSIM"]
            eval_vals[image_id] = score
            per_image_rows.append(dict(noise_sigma=sigma_n, image_id=image_id, role=item["role"],
                                       held_out_SSIM=score, **{**best, "sigma_patch_used": best_sigma_patch}))

        exact_vals = []
        for image_id, item in data.items():
            noisy = eval_bank[(image_id, sigma_n)]
            bf, _ = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            exact_vals.append(metrics(item["clean"], bf)["SSIM"])

        best_reported = {**best, "sigma_patch_used": best_sigma_patch}
        results[str(sigma_n)] = dict(best_params=best_reported, tune_SSIM=best_value,
                                     eval_SSIM_mean=float(np.mean(list(eval_vals.values()))),
                                     eval_SSIM_by_image=eval_vals,
                                     bf_exact_eval_SSIM_mean=float(np.mean(exact_vals)),
                                     n_trials=N_TRIALS)

    (out_dir / "local_optuna_best_parameters.json").write_text(json.dumps(results, indent=2))
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(out_dir / "local_optuna_per_image.csv", index=False)
    write_report(out_dir, results, per_image_df)
    return results


def write_report(out_dir: Path, results: dict, per_image_df: pd.DataFrame) -> None:
    summary_rows = []
    for noise, r in results.items():
        summary_rows.append(dict(noise_sigma=int(noise), **r["best_params"],
                                 tune_SSIM=r["tune_SSIM"], eval_SSIM_mean=r["eval_SSIM_mean"],
                                 bf_exact_eval_SSIM_mean=r["bf_exact_eval_SSIM_mean"]))
    summary_df = pd.DataFrame(summary_rows)
    text = (
        "# Optuna: local-radius pool + embedding search\n\n"
        "## Scope\n\n"
        f"Joint search ({N_TRIALS} trials/noise level, TPE) over `K` (patch kernel size), `rerank_dim` "
        "(embedding/PCA dimension), `retrieval_dim` (cheap KD-tree query dim), `bag_size`, `spatial_weight` "
        "(local-radius bias strength, the CANDIDATE-SELECTION spatial term), `percentile_clip` (drop the bottom "
        "N% of each pixel's own bag by weight before averaging — a continuous top-K generalization), `h`, and "
        "`sigma_patch` (a second, INDEPENDENT spatial term — Gaussian-weights each of the K^2 pixel positions "
        "*inside* the patch-distance computation itself by distance from the patch's own center, `<=0` disables "
        "it and falls back to today's uniform box SSD). Objective is mean SSIM across all four selected images "
        "jointly on the tuning seed; reported `eval_SSIM_mean`/`eval_SSIM_by_image` are on the held-out "
        "validation seed.\n\n"
        "## Best config per noise level\n\n"
        + dataframe_markdown(summary_df, 5)
        + "\n\n## Per-image held-out SSIM at the winning config\n\n"
        + dataframe_markdown(per_image_df[["noise_sigma", "image_id", "role", "held_out_SSIM"]], 5)
        + "\n\n## Reading\n\n"
        "Compare `eval_SSIM_mean` here against stage 17's hand-picked `spatial_weight` grid and against "
        "`PCA_NLM_exact`/`BF_exact` in `outputs/17_local_retrieval/local_retrieval_aggregate.csv` and "
        "`outputs/12_pca_nlm/pca_nlm_aggregate.csv`. If `percentile_clip` lands near 0 at the optimum, the "
        "continuous clip isn't earning its keep over plain `bag_size` truncation; if `K` or `rerank_dim` drift "
        "noise-dependently, that's the noise-adaptive sweet spot the fixed `K=7`/dim grid search never tested.\n"
    )
    (out_dir / "Local_Retrieval_Optuna.md").write_text(text)


if __name__ == "__main__":
    results = run_optuna()
    for noise, r in results.items():
        print(noise, r["best_params"], "tune_SSIM=", round(r["tune_SSIM"], 4),
              "eval_SSIM_mean=", round(r["eval_SSIM_mean"], 4),
              "bf_exact=", round(r["bf_exact_eval_SSIM_mean"], 4))
