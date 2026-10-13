"""FusionV1: multi-cue BF/G/D5 routing.

This implements the design in:

    outputs/reports/G_BF_NLM_Gating_Theory_and_FusionV1_Design.md

Unlike F0, D5 is allowed to compete across *all* pixels under a budget. The
old homogeneity gate no longer permanently blocks NLM. D5 is still queried only
for selected pixels, so P_NLM is real compute scope.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bilateral_foundations import (BOUNDARY, bilateral_from_scratch,
                                   dataframe_markdown, gaussian_from_scratch,
                                   metrics)
from bilateral_graded_funnel import graded_funnel, local_pool_masked
from bilateral_local_retrieval_optuna import fit_descriptor
from bilateral_recovery import NOISE_LEVELS, apply_method, noise_corrected_signal_std
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank
from bilateral_v2 import gradient_map


BUDGETS = (0.10, 0.25, 0.50, 0.75)
CUE_COMBOS = {
    "residual": ("residual",),
    "neff_low": ("neff_low",),
    "signal": ("signal",),
    "gradient": ("gradient",),
    "residual_neff": ("residual", "neff_low"),
    "residual_neff_signal": ("residual", "neff_low", "signal"),
    "residual_neff_signal_gradient": ("residual", "neff_low", "signal", "gradient"),
}


def _robust_norm(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Percentile robust [0,1] normalization."""
    vals = x[mask] if mask is not None else x.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(vals, [5, 95])
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _top_budget_mask(score: np.ndarray, budget: float) -> np.ndarray:
    n = score.size
    n_take = min(max(int(np.ceil(float(budget) * n)), 1), n)
    flat = score.ravel()
    order = np.argpartition(-flat, n_take - 1)[:n_take]
    mask = np.zeros(n, dtype=bool)
    mask[order] = True
    return mask.reshape(score.shape)


def expert_maps(noisy: np.ndarray, sigma_n: int, recovery: dict, ltp_best: dict) -> dict:
    p = recovery["best"]["BF_exact"][str(sigma_n)]
    g_threshold = recovery["noise_aware_gate"][str(sigma_n)]["signal_std_threshold"]
    guide_sigma = ltp_best[str(sigma_n)]["guide_sigma"]

    start = time.perf_counter()
    g, gdiag = gaussian_from_scratch(noisy, p["sigma_s"], p["k"], BOUNDARY)
    bf, bdiag, neff = bilateral_from_scratch(
        noisy, p["sigma_s"], p["sigma_r"], p["k"], BOUNDARY, return_neff=True
    )
    signal = noise_corrected_signal_std(noisy, sigma_n, p["k"])
    grad = gradient_map(noisy, guide_sigma)
    elapsed = (time.perf_counter() - start) * 1000.0

    return dict(
        gaussian=g,
        bf=bf,
        neff=neff,
        signal=signal,
        gradient=grad,
        g_threshold=g_threshold,
        cheap_runtime_ms=elapsed,
        gaussian_runtime_ms=gdiag["runtime_ms"],
        bf_runtime_ms=bdiag["runtime_ms"],
        bf_range_eval_count=bdiag["range_eval_count"],
    )


def ambiguity_score(noisy: np.ndarray, experts: dict, sigma_n: int, combo: str) -> np.ndarray:
    bf = experts["bf"]
    neff = experts["neff"]
    signal = experts["signal"]
    grad = experts["gradient"]
    cue_maps = {
        "residual": _robust_norm(np.abs(noisy.astype(np.float32) - bf) / max(float(sigma_n), 1e-6)),
        "neff_low": _robust_norm(-neff),
        "signal": _robust_norm(signal / max(float(sigma_n), 1e-6)),
        "gradient": _robust_norm(grad),
    }
    parts = CUE_COMBOS[combo]
    score = np.zeros_like(noisy, dtype=np.float32)
    for name in parts:
        score += cue_maps[name]
    return score / float(len(parts))


def fusion_v1(noisy: np.ndarray, sigma_n: int, recovery: dict, ltp_best: dict,
              d5_params: dict, combo: str, budget: float) -> tuple[np.ndarray, dict]:
    experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
    score = ambiguity_score(noisy, experts, sigma_n, combo)
    nlm_mask = _top_budget_mask(score, budget)

    # Gaussian remains available for low-signal non-D5 pixels, but it no
    # longer gets first refusal before D5.
    gmask = (experts["signal"] <= experts["g_threshold"]) & ~nlm_mask
    bfmask = ~(nlm_mask | gmask)

    d5 = d5_params[str(sigma_n)]["best_params"]
    rerank_dim = int(d5["rerank_dim"])
    retrieval_dim = min(int(d5["retrieval_dim"]), rerank_dim)
    sigma_patch = float(d5.get("sigma_patch_used", 0.0))
    desc = fit_descriptor(noisy, int(d5["K"]), rerank_dim, sigma_patch)
    nlm, ndiag = local_pool_masked(
        noisy,
        desc,
        retrieval_dim,
        rerank_dim,
        float(d5["spatial_weight"]),
        int(d5["bag_size"]),
        float(d5["h"]),
        float(d5["percentile_clip"]),
        nlm_mask,
    )

    out = np.where(nlm_mask, nlm, np.where(gmask, experts["gaussian"], experts["bf"])).astype(np.float32)
    diag = dict(
        P_G=float(gmask.mean()),
        P_BF=float(bfmask.mean()),
        P_NLM=float(nlm_mask.mean()),
        combo=combo,
        budget=budget,
        runtime_ms=experts["cheap_runtime_ms"] + ndiag["runtime_ms"],
        range_eval_count=experts["bf_range_eval_count"],
        nlm_escalated_pixels=ndiag["escalated_pixels"],
        nlm_descriptor_comparison_count=ndiag["descriptor_comparison_count"],
    )
    return out, diag


def tune_fusion_v1(data: dict, bank: dict, recovery: dict, ltp_best: dict, d5_params: dict) -> dict:
    # Cache expensive per-image/per-noise ingredients once. During tuning we
    # only need to compare masks, so a precomputed full-D5 image is enough.
    # The final selected config is still evaluated with true masked D5 queries
    # in `run_controlled`.
    cache = {}
    for image_id, item in data.items():
        for sigma_n in NOISE_LEVELS:
            noisy = bank[(image_id, sigma_n)]
            experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
            d5 = d5_params[str(sigma_n)]["best_params"]
            rerank_dim = int(d5["rerank_dim"])
            retrieval_dim = min(int(d5["retrieval_dim"]), rerank_dim)
            desc = fit_descriptor(noisy, int(d5["K"]), rerank_dim, float(d5.get("sigma_patch_used", 0.0)))
            d5_full, _ = local_pool_masked(
                noisy,
                desc,
                retrieval_dim,
                rerank_dim,
                float(d5["spatial_weight"]),
                int(d5["bag_size"]),
                float(d5["h"]),
                float(d5["percentile_clip"]),
                np.ones_like(noisy, dtype=bool),
            )
            cache[(image_id, sigma_n)] = dict(noisy=noisy, clean=item["clean"], experts=experts, d5_full=d5_full)

    best = {}
    for sigma_n in NOISE_LEVELS:
        best_score = -np.inf
        best_cfg = None
        for combo in CUE_COMBOS:
            for budget in BUDGETS:
                vals = []
                for image_id, item in data.items():
                    c = cache[(image_id, sigma_n)]
                    score = ambiguity_score(c["noisy"], c["experts"], sigma_n, combo)
                    nlm_mask = _top_budget_mask(score, budget)
                    gmask = (c["experts"]["signal"] <= c["experts"]["g_threshold"]) & ~nlm_mask
                    out = np.where(
                        nlm_mask,
                        c["d5_full"],
                        np.where(gmask, c["experts"]["gaussian"], c["experts"]["bf"]),
                    ).astype(np.float32)
                    vals.append(metrics(c["clean"], out)["SSIM"])
                score = float(np.mean(vals))
                if score > best_score:
                    best_score = score
                    best_cfg = dict(combo=combo, budget=budget, tune_SSIM=best_score)
        best[str(sigma_n)] = best_cfg
    return best


def run_controlled(output_root: str | Path = "outputs", side: int = 128) -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "22_fusion_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_selected(root / "02_image_selection" / "final_four.csv", Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((root / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())
    d5_params = json.loads((root / "18_local_retrieval_optuna" / "local_optuna_best_parameters.json").read_text())
    f0_tuned = json.loads((root / "20_graded_funnel" / "graded_funnel_best_parameters.json").read_text())

    tuned = tune_fusion_v1(data, tune_bank, recovery, ltp_best, d5_params)
    (out_dir / "fusion_v1_best_parameters.json").write_text(json.dumps(tuned, indent=2))

    rows = []
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            variants = [("noisy", noisy, dict(runtime_ms=0.0))]
            bf, bf_time = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            variants.append(("BF_exact", bf, dict(runtime_ms=bf_time)))

            f0_out, f0_diag = graded_funnel(
                noisy,
                sigma_n,
                recovery,
                ltp_best,
                d5_params,
                f0_tuned[str(sigma_n)]["grad_threshold"],
            )
            variants.append(("F0_Sobel_graded_funnel", f0_out, f0_diag))

            cfg = tuned[str(sigma_n)]
            f1_out, f1_diag = fusion_v1(
                noisy, sigma_n, recovery, ltp_best, d5_params, cfg["combo"], cfg["budget"]
            )
            variants.append(("FusionV1_multicue_budget", f1_out, f1_diag))

            d5 = d5_params[str(sigma_n)]["best_params"]
            rerank_dim = int(d5["rerank_dim"])
            retrieval_dim = min(int(d5["retrieval_dim"]), rerank_dim)
            desc = fit_descriptor(noisy, int(d5["K"]), rerank_dim, float(d5.get("sigma_patch_used", 0.0)))
            full, full_diag = local_pool_masked(
                noisy,
                desc,
                retrieval_dim,
                rerank_dim,
                float(d5["spatial_weight"]),
                int(d5["bag_size"]),
                float(d5["h"]),
                float(d5["percentile_clip"]),
                np.ones_like(noisy, dtype=bool),
            )
            variants.append(("D5_full_image", full, full_diag))

            for method, out, diag in variants:
                rows.append(dict(
                    image_id=image_id,
                    role=item["role"],
                    noise_sigma=sigma_n,
                    method=method,
                    side=side,
                    combo=diag.get("combo", ""),
                    budget=diag.get("budget", np.nan),
                    P_G=diag.get("P_G", np.nan),
                    P_BF=diag.get("P_BF", np.nan),
                    P_NLM=diag.get("P_NLM", np.nan),
                    runtime_ms=diag.get("runtime_ms", np.nan),
                    range_eval_count=diag.get("range_eval_count", np.nan),
                    nlm_escalated_pixels=diag.get("nlm_escalated_pixels", np.nan),
                    nlm_descriptor_comparison_count=diag.get("nlm_descriptor_comparison_count", np.nan),
                    **metrics(clean, out),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fusion_v1_results.csv", index=False)
    write_report(out_dir, df, tuned)
    return df


def write_report(out_dir: Path, df: pd.DataFrame, tuned: dict) -> None:
    agg = df.groupby(["noise_sigma", "method"]).agg(
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        edge=("edge_metric", "mean"),
        texture=("texture_metric", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        P_G=("P_G", "mean"),
        P_BF=("P_BF", "mean"),
        P_NLM=("P_NLM", "mean"),
        nlm_escalated_pixels=("nlm_escalated_pixels", "mean"),
        nlm_descriptor_comparison_count=("nlm_descriptor_comparison_count", "mean"),
    ).reset_index().sort_values(["noise_sigma", "SSIM"], ascending=[True, False])
    agg.to_csv(out_dir / "fusion_v1_aggregate.csv", index=False)
    tune_df = pd.DataFrame([dict(noise_sigma=int(k), **v) for k, v in tuned.items()])

    text = (
        "# FusionV1: multi-cue all-pixel D5 budget gate\n\n"
        "## Scope\n\n"
        "FusionV1 replaces F0's Sobel-only, non-flat-only escalation with a budgeted ambiguity score. D5 is allowed "
        "to compete across all pixels first; Gaussian remains available for low-signal pixels only after the D5 "
        "budget is assigned. The D5 branch is queried only for selected pixels, so `P_NLM` is real compute scope.\n\n"
        "## Tuned gate by noise level\n\n"
        + dataframe_markdown(tune_df, 5)
        + "\n\n## Aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Reading\n\n"
        "A win over `F0_Sobel_graded_funnel` means the problem really was the gate. A result close to "
        "`D5_full_image` with `P_NLM < 1` would be the desired MD-style graded funnel. If FusionV1 still needs a "
        "large D5 budget, the next move is a risk-style selector or a learned cheap classifier against the oracle "
        "diagnostic mask from stage 21.\n"
    )
    (out_dir / "FusionV1.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/22_fusion_v1/fusion_v1_aggregate.csv").to_string(index=False))
