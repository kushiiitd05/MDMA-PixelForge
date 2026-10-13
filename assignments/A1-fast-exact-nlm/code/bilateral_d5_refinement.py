"""D5 internal refinement: noise-corrected distances and adaptive h(p).

FusionV4 defended KD-tree D5 as the practical branch. This stage keeps the
same D5 candidate pool and tests whether its weighting can improve:

  - subtract a tunable noise floor from descriptor distances;
  - use a simple per-pixel bandwidth h(p) based on cheap local cues.

The candidate bag is cached per image/noise so every variant uses the same
neighbors. That isolates weighting math from retrieval membership.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from bilateral_foundations import dataframe_markdown, metrics
from bilateral_fusion_v1 import _robust_norm, expert_maps
from bilateral_local_retrieval_optuna import fit_descriptor, local_pool_with_clip
from bilateral_recovery import NOISE_LEVELS
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank


VARIANTS = [
    dict(name="D5_base_recomputed", h_mult=1.0, noise_alpha=0.0, h_mode="none", h_strength=0.0),
    dict(name="global_h_0.75", h_mult=0.75, noise_alpha=0.0, h_mode="none", h_strength=0.0),
    dict(name="global_h_1.25", h_mult=1.25, noise_alpha=0.0, h_mode="none", h_strength=0.0),
    dict(name="global_h_1.50", h_mult=1.50, noise_alpha=0.0, h_mode="none", h_strength=0.0),
    dict(name="noise_floor_0.02", h_mult=1.0, noise_alpha=0.02, h_mode="none", h_strength=0.0),
    dict(name="noise_floor_0.05", h_mult=1.0, noise_alpha=0.05, h_mode="none", h_strength=0.0),
    dict(name="noise_floor_0.10", h_mult=1.0, noise_alpha=0.10, h_mode="none", h_strength=0.0),
    dict(name="noise_floor_0.20", h_mult=1.0, noise_alpha=0.20, h_mode="none", h_strength=0.0),
    dict(name="h_signal_high_0.25", h_mult=1.0, noise_alpha=0.0, h_mode="signal_high", h_strength=0.25),
    dict(name="h_signal_high_0.50", h_mult=1.0, noise_alpha=0.0, h_mode="signal_high", h_strength=0.50),
    dict(name="h_signal_low_0.25", h_mult=1.0, noise_alpha=0.0, h_mode="signal_low", h_strength=0.25),
    dict(name="h_signal_low_0.50", h_mult=1.0, noise_alpha=0.0, h_mode="signal_low", h_strength=0.50),
    dict(name="h_residual_high_0.25", h_mult=1.0, noise_alpha=0.0, h_mode="residual_high", h_strength=0.25),
    dict(name="h_residual_high_0.50", h_mult=1.0, noise_alpha=0.0, h_mode="residual_high", h_strength=0.50),
    dict(name="h_neff_low_0.25", h_mult=1.0, noise_alpha=0.0, h_mode="neff_low", h_strength=0.25),
    dict(name="h_neff_low_0.50", h_mult=1.0, noise_alpha=0.0, h_mode="neff_low", h_strength=0.50),
]


def prepare_bag(noisy: np.ndarray, desc: np.ndarray, d5: dict, mask: np.ndarray | None = None) -> dict:
    x = noisy.astype(np.float32)
    height, width = x.shape
    rerank_dim = int(d5["rerank_dim"])
    retrieval_dim = min(int(d5["retrieval_dim"]), rerank_dim)
    bag_size = int(d5["bag_size"])
    z_ret = desc[..., :retrieval_dim].reshape(-1, retrieval_dim)
    z_rer = desc[..., :rerank_dim].reshape(-1, rerank_dim)
    yy, xx = np.indices((height, width))
    coords = np.stack([yy.ravel(), xx.ravel()], axis=-1).astype(np.float32) * float(d5["spatial_weight"])
    z_query = np.concatenate([z_ret, coords], axis=-1)
    flat_mask = np.ones(x.size, dtype=bool) if mask is None else mask.ravel().astype(bool)
    query_idx = np.where(flat_mask)[0]

    start = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=bag_size, algorithm="kd_tree", metric="euclidean")
    nn.fit(z_query)
    _, indices = nn.kneighbors(z_query[query_idx])
    bag_ms = (time.perf_counter() - start) * 1000.0
    start_dist = time.perf_counter()
    diff = z_rer[query_idx][:, None, :] - z_rer[indices]
    dist2 = np.sum(diff * diff, axis=-1) / float(rerank_dim)
    dist_ms = (time.perf_counter() - start_dist) * 1000.0
    return dict(
        query_idx=query_idx,
        indices=indices,
        z_rer=z_rer,
        dist2=dist2.astype(np.float32),
        flat_x=x.ravel(),
        shape=x.shape,
        bag_runtime_ms=bag_ms + dist_ms,
        neighbor_runtime_ms=bag_ms,
        dist_runtime_ms=dist_ms,
        retrieval_dim=retrieval_dim,
        rerank_dim=rerank_dim,
        bag_size=bag_size,
    )


def h_map_for_variant(noisy: np.ndarray, experts: dict, base_h: float, variant: dict) -> np.ndarray:
    h0 = float(base_h) * float(variant["h_mult"])
    mode = variant["h_mode"]
    strength = float(variant["h_strength"])
    if mode == "none" or strength <= 0:
        return np.full(noisy.shape, h0, dtype=np.float32)
    if mode == "signal_high":
        cue = _robust_norm(experts["signal"])
    elif mode == "signal_low":
        cue = 1.0 - _robust_norm(experts["signal"])
    elif mode == "residual_high":
        cue = _robust_norm(np.abs(noisy.astype(np.float32) - experts["bf"]))
    elif mode == "neff_low":
        cue = _robust_norm(-experts["neff"])
    else:
        raise ValueError(f"unknown h_mode: {mode}")
    return (h0 * (1.0 + strength * cue)).astype(np.float32)


def pool_from_bag(noisy: np.ndarray, bag: dict, h_map: np.ndarray, percentile_clip: float,
                  sigma_n: int, noise_alpha: float) -> tuple[np.ndarray, dict]:
    query_idx = bag["query_idx"]
    indices = bag["indices"]
    z_rer = bag["z_rer"]
    rerank_dim = bag["rerank_dim"]
    flat_x = bag["flat_x"]
    out = flat_x.copy()
    if query_idx.size == 0:
        return out.reshape(bag["shape"]).astype(np.float32), dict(runtime_ms=0.0)

    start = time.perf_counter()
    dist2 = bag["dist2"]
    if noise_alpha > 0:
        # Descriptor-space proxy for the expected squared difference due to
        # independent Gaussian noise in two patches. Tuned alpha keeps this
        # conservative because PCA/sigma_patch make the exact bias nontrivial.
        dist2 = np.maximum(dist2 - float(noise_alpha) * 2.0 * float(sigma_n) ** 2, 0.0)
    hq = h_map.ravel()[query_idx][:, None]
    weight = np.exp(-dist2 / np.maximum(hq * hq, 1e-12))
    if percentile_clip > 0:
        thresh = np.percentile(weight, float(percentile_clip), axis=1, keepdims=True)
        weight = np.where(weight >= thresh, weight, 0.0)
        zero_rows = weight.sum(axis=1) <= 1e-12
        if np.any(zero_rows):
            weight[zero_rows, 0] = 1.0
    values = flat_x[indices]
    out[query_idx] = (weight * values).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-12)
    weight_ms = (time.perf_counter() - start) * 1000.0
    diag = dict(
        runtime_ms=bag["bag_runtime_ms"] + weight_ms,
        bag_runtime_ms=bag["bag_runtime_ms"],
        weight_runtime_ms=weight_ms,
        mean_h=float(h_map.ravel()[query_idx].mean()),
        min_h=float(h_map.ravel()[query_idx].min()),
        max_h=float(h_map.ravel()[query_idx].max()),
        descriptor_comparison_count=int(query_idx.size) * bag["bag_size"] * (bag["retrieval_dim"] + rerank_dim),
    )
    return out.reshape(bag["shape"]).astype(np.float32), diag


def run_variant(noisy: np.ndarray, sigma_n: int, recovery: dict, ltp_best: dict,
                d5_params: dict, variant: dict) -> tuple[np.ndarray, dict]:
    d5 = d5_params[str(sigma_n)]["best_params"]
    desc = fit_descriptor(noisy, int(d5["K"]), int(d5["rerank_dim"]), float(d5.get("sigma_patch_used", 0.0)))
    bag = prepare_bag(noisy, desc, d5)
    experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
    hmap = h_map_for_variant(noisy, experts, float(d5["h"]), variant)
    out, diag = pool_from_bag(noisy, bag, hmap, float(d5["percentile_clip"]),
                              sigma_n, float(variant["noise_alpha"]))
    diag.update({k: variant[k] for k in ("h_mult", "noise_alpha", "h_mode", "h_strength")})
    return out, diag


def tune_variants(data: dict, bank: dict, recovery: dict, ltp_best: dict, d5_params: dict) -> dict:
    cache = {}
    for image_id in data:
        for sigma_n in NOISE_LEVELS:
            noisy = bank[(image_id, sigma_n)]
            d5 = d5_params[str(sigma_n)]["best_params"]
            desc = fit_descriptor(noisy, int(d5["K"]), int(d5["rerank_dim"]), float(d5.get("sigma_patch_used", 0.0)))
            bag = prepare_bag(noisy, desc, d5)
            experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
            cache[(image_id, sigma_n)] = dict(noisy=noisy, d5=d5, bag=bag, experts=experts)

    best = {}
    for sigma_n in NOISE_LEVELS:
        best_score = -np.inf
        best_variant = None
        for variant in VARIANTS:
            vals = []
            for image_id, item in data.items():
                c = cache[(image_id, sigma_n)]
                hmap = h_map_for_variant(c["noisy"], c["experts"], float(c["d5"]["h"]), variant)
                out, _ = pool_from_bag(c["noisy"], c["bag"], hmap, float(c["d5"]["percentile_clip"]),
                                       sigma_n, float(variant["noise_alpha"]))
                vals.append(metrics(item["clean"], out)["SSIM"])
            score = float(np.mean(vals))
            if score > best_score:
                best_score = score
                best_variant = dict(variant, tune_SSIM=score)
        best[str(sigma_n)] = best_variant
    return best


def run_controlled(output_root: str | Path = "outputs", side: int = 128) -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "26_d5_refinement"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_selected(root / "02_image_selection" / "final_four.csv", Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((root / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())
    d5_params = json.loads((root / "18_local_retrieval_optuna" / "local_optuna_best_parameters.json").read_text())

    tuned = tune_variants(data, tune_bank, recovery, ltp_best, d5_params)
    (out_dir / "d5_refinement_best_parameters.json").write_text(json.dumps(tuned, indent=2))

    rows = []
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            d5 = d5_params[str(sigma_n)]["best_params"]
            desc = fit_descriptor(noisy, int(d5["K"]), int(d5["rerank_dim"]), float(d5.get("sigma_patch_used", 0.0)))
            bag = prepare_bag(noisy, desc, d5)
            experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
            variants_to_eval = [v for v in VARIANTS if v["name"] == "D5_base_recomputed"]
            selected = tuned[str(sigma_n)]
            if selected["name"] != "D5_base_recomputed":
                variants_to_eval.append({k: selected[k] for k in ("name", "h_mult", "noise_alpha", "h_mode", "h_strength")})
            for variant in variants_to_eval:
                hmap = h_map_for_variant(noisy, experts, float(d5["h"]), variant)
                out, diag = pool_from_bag(noisy, bag, hmap, float(d5["percentile_clip"]),
                                          sigma_n, float(variant["noise_alpha"]))
                rows.append(dict(
                    image_id=image_id,
                    role=item["role"],
                    noise_sigma=sigma_n,
                    method=variant["name"],
                    side=side,
                    **{k: variant[k] for k in ("h_mult", "noise_alpha", "h_mode", "h_strength")},
                    **diag,
                    **metrics(clean, out),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "d5_refinement_results.csv", index=False)
    write_report(out_dir, df, tuned)
    return df


def write_report(out_dir: Path, df: pd.DataFrame, tuned: dict) -> None:
    agg = df.groupby(["noise_sigma", "method"]).agg(
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        edge=("edge_metric", "mean"),
        texture=("texture_metric", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        mean_h=("mean_h", "mean"),
        descriptor_comparison_count=("descriptor_comparison_count", "mean"),
    ).reset_index().sort_values(["noise_sigma", "SSIM"], ascending=[True, False])
    agg.to_csv(out_dir / "d5_refinement_aggregate.csv", index=False)
    tune_df = pd.DataFrame([dict(noise_sigma=int(k), **v) for k, v in tuned.items()])
    text = (
        "# D5 refinement: noise floor and adaptive bandwidth\n\n"
        "## Scope\n\n"
        "Keeps the same D5 KD-tree candidate bag and tests only the weighting math: global `h` multipliers, "
        "descriptor-distance noise-floor subtraction, and simple adaptive `h(p)` maps from signal/residual/`N_eff`. "
        "The bag is cached per image/noise so variants are candidate-matched.\n\n"
        "## Tuned variant by noise\n\n"
        + dataframe_markdown(tune_df, 5)
        + "\n\n## Held-out aggregate\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Reading\n\n"
        "If a refined variant beats `D5_base_recomputed` on held-out SSIM, carry that weighting into fusion. "
        "If tuning picks a variant that does not generalize, treat it as overfit and keep the old D5 branch.\n"
    )
    (out_dir / "D5_Refinement.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/26_d5_refinement/d5_refinement_aggregate.csv").to_string(index=False))
