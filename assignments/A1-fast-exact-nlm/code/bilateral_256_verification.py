"""256px verification of the selected BF/D5/fusion candidates.

This is not a full ablation rerun. It verifies the candidates that survived:

  - BF_exact;
  - D5 original;
  - D5 refined;
  - F0 Sobel funnel;
  - FusionV5 refined D5 with quality and efficient masks.

Selectors are trained/tuned on the 256px tune seed and evaluated on the 256px
held-out seed, but the D5 hyperparameters themselves are the already-selected
128px D5 parameters. This tests whether the selected route generalizes upward
in resolution before final assessment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilateral_d5_refinement import h_map_for_variant, pool_from_bag, prepare_bag
from bilateral_foundations import dataframe_markdown, metrics
from bilateral_fusion_v1 import _top_budget_mask, expert_maps
from bilateral_fusion_v2_cost_aware import d5_full, feature_stack, train_gate
from bilateral_fusion_v3_pareto import RidgeGate, score_from_name, train_ridge_gate
from bilateral_fusion_v5_refined_d5 import refined_d5_masked, route_from_mask
from bilateral_graded_funnel import graded_funnel
from bilateral_local_retrieval_optuna import fit_descriptor
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank


SELECTED_MASKS = {
    10: {
        "quality": ("logistic::C=0.1", 0.75),
        "efficient": ("ridge_benefit::alpha=0.1", 0.10),
    },
    25: {
        "quality": ("logistic::C=1.0", 0.75),
        "efficient": ("cheap::residual_neff", 0.25),
    },
    50: {
        "quality": ("cheap::signal", 0.75),
        "efficient": ("cheap::residual", 0.35),
    },
}


def base_and_d5_for_training(noisy: np.ndarray, clean: np.ndarray, sigma_n: int,
                             recovery: dict, ltp_best: dict, d5_params: dict) -> dict:
    experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
    base = np.where(
        experts["signal"] <= experts["g_threshold"],
        experts["gaussian"],
        experts["bf"],
    ).astype(np.float32)
    d5, _ = d5_full(noisy, d5_params, sigma_n)
    features = feature_stack(noisy, sigma_n, experts)
    label = (np.abs(d5 - clean) < np.abs(base - clean)).astype(np.int32)
    benefit = (np.abs(base - clean) - np.abs(d5 - clean)).astype(np.float32)
    return dict(noisy=noisy, clean=clean, experts=experts, base=base, d5=d5,
                features=features, label=label, benefit=benefit)


def prepare_training_cache(data: dict, bank: dict, recovery: dict, ltp_best: dict, d5_params: dict) -> dict:
    cache = {}
    for image_id, item in data.items():
        for sigma_n in NOISE_LEVELS:
            cache[(image_id, sigma_n)] = base_and_d5_for_training(
                bank[(image_id, sigma_n)], item["clean"], sigma_n, recovery, ltp_best, d5_params
            )
    return cache


def train_required_models(data: dict, cache: dict) -> dict:
    models = {}
    for sigma_n in NOISE_LEVELS:
        models[str(sigma_n)] = {}
        needed = {selector for selector, _budget in SELECTED_MASKS[sigma_n].values()}
        features = [cache[(image_id, sigma_n)]["features"] for image_id in data]
        labels = [cache[(image_id, sigma_n)]["label"] for image_id in data]
        benefits = [cache[(image_id, sigma_n)]["benefit"] for image_id in data]
        for selector in needed:
            if selector.startswith("cheap::"):
                models[str(sigma_n)][selector] = None
            elif selector.startswith("logistic::C="):
                c_value = float(selector.split("=", 1)[1])
                models[str(sigma_n)][selector] = train_gate(features, labels, c_value)
            elif selector.startswith("ridge_benefit::alpha="):
                alpha = float(selector.split("=", 1)[1])
                models[str(sigma_n)][selector] = train_ridge_gate(features, benefits, alpha)
            else:
                raise ValueError(f"unknown selector: {selector}")
    return models


def fusion_masked_refined(noisy: np.ndarray, sigma_n: int, recovery: dict, ltp_best: dict,
                          d5_params: dict, refine_params: dict,
                          selector: str, budget: float, model) -> tuple[np.ndarray, dict]:
    experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
    score = score_from_name(selector, model, noisy, sigma_n, experts)
    mask = _top_budget_mask(score, budget)
    branch, bdiag = refined_d5_masked(noisy, sigma_n, recovery, ltp_best, d5_params, refine_params, mask)
    out, rdiag = route_from_mask(noisy, experts, branch, mask)
    return out, {**bdiag, **rdiag, "selector": selector, "budget": budget}


def run_controlled(output_root: str | Path = "outputs", side: int = 256) -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "28_256_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_selected(root / "02_image_selection" / "final_four.csv", Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((root / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())
    d5_params = json.loads((root / "18_local_retrieval_optuna" / "local_optuna_best_parameters.json").read_text())
    refine_params = json.loads((root / "26_d5_refinement" / "d5_refinement_best_parameters.json").read_text())
    f0_tuned = json.loads((root / "20_graded_funnel" / "graded_funnel_best_parameters.json").read_text())

    train_cache = prepare_training_cache(data, tune_bank, recovery, ltp_best, d5_params)
    models = train_required_models(data, train_cache)

    rows = []
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            variants = [("noisy", "", np.nan, noisy, dict(runtime_ms=0.0))]

            bf, bf_time = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            variants.append(("BF_exact", "", np.nan, bf, dict(runtime_ms=bf_time)))

            f0, f0diag = graded_funnel(
                noisy, sigma_n, recovery, ltp_best, d5_params,
                f0_tuned[str(sigma_n)]["grad_threshold"],
            )
            variants.append(("F0_Sobel_graded_funnel", "", np.nan, f0, f0diag))

            d5_orig, d5diag = d5_full(noisy, d5_params, sigma_n)
            variants.append(("D5_full_original", "", np.nan, d5_orig, d5diag))

            d5_ref, refdiag = refined_d5_masked(
                noisy, sigma_n, recovery, ltp_best, d5_params, refine_params, None
            )
            variants.append(("D5_full_refined", "", np.nan, d5_ref, refdiag))

            for mode_name, (selector, budget) in SELECTED_MASKS[sigma_n].items():
                model = models[str(sigma_n)][selector]
                out, diag = fusion_masked_refined(
                    noisy, sigma_n, recovery, ltp_best, d5_params, refine_params,
                    selector, budget, model
                )
                variants.append((f"FusionV5_{mode_name}_refined", selector, budget, out, diag))

            for method, selector, budget, out, diag in variants:
                rows.append(dict(
                    image_id=image_id,
                    role=item["role"],
                    noise_sigma=sigma_n,
                    method=method,
                    selector=selector,
                    budget=budget,
                    side=side,
                    P_G=diag.get("P_G", np.nan),
                    P_BF=diag.get("P_BF", np.nan),
                    P_NLM=diag.get("P_NLM", np.nan),
                    runtime_ms=diag.get("runtime_ms", np.nan),
                    descriptor_comparison_count=diag.get("descriptor_comparison_count", np.nan),
                    **metrics(clean, out),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "verification_256_results.csv", index=False)
    write_report(out_dir, df)
    return df


def write_report(out_dir: Path, df: pd.DataFrame) -> None:
    agg = df.groupby(["noise_sigma", "method", "selector", "budget"], dropna=False).agg(
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        edge=("edge_metric", "mean"),
        texture=("texture_metric", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        P_G=("P_G", "mean"),
        P_BF=("P_BF", "mean"),
        P_NLM=("P_NLM", "mean"),
        descriptor_comparison_count=("descriptor_comparison_count", "mean"),
    ).reset_index().sort_values(["noise_sigma", "SSIM"], ascending=[True, False])
    agg.to_csv(out_dir / "verification_256_aggregate.csv", index=False)
    text = (
        "# 256px verification of selected candidates\n\n"
        "## Scope\n\n"
        "Selected-method verification at side `256`, not a full ablation rerun. D5 hyperparameters come from the "
        "128px selected route; fusion selectors are trained on the 256px tuning seed and evaluated on the 256px "
        "held-out seed.\n\n"
        "## Aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Reading\n\n"
        "If D5/fusion rankings hold at 256, the 128px development line generalizes. If winners flip, final "
        "assessment needs size-specific caveats or a 256px retuning pass.\n"
    )
    (out_dir / "Verification_256.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/28_256_verification/verification_256_aggregate.csv").to_string(index=False))
