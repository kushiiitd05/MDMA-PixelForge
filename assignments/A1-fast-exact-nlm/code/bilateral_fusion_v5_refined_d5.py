"""FusionV5: V3 masks with refined D5 weighting.

Earlier stages:
  - V3 gave the selector Pareto curve.
  - V4 defended KD-tree D5 as the practical selected-pixel branch.
  - D5 refinement found small held-out wins for D5 weighting at noise 10/25.

FusionV5 asks whether those refined D5 weights help the actual fusion routes.
It keeps the same V3 masks/selectors and swaps only the D5 weighting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilateral_d5_refinement import h_map_for_variant, pool_from_bag, prepare_bag
from bilateral_foundations import dataframe_markdown, metrics
from bilateral_fusion_v1 import _top_budget_mask, expert_maps, fusion_v1
from bilateral_fusion_v2_cost_aware import d5_full
from bilateral_fusion_v3_pareto import prepare_tune_cache, score_from_name, tune_pareto
from bilateral_graded_funnel import graded_funnel
from bilateral_local_retrieval_optuna import fit_descriptor
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank


# Same representative tradeoff masks used in FusionV4.
MASK_LAMBDAS = {
    10: ("0.0", "0.06"),
    25: ("0.0", "0.16"),
    50: ("0.0", "0.1"),
}


def route_from_mask(noisy: np.ndarray, experts: dict, nlm_image: np.ndarray,
                    mask: np.ndarray) -> tuple[np.ndarray, dict]:
    gmask = (experts["signal"] <= experts["g_threshold"]) & ~mask
    bfmask = ~(mask | gmask)
    out = np.where(mask, nlm_image, np.where(gmask, experts["gaussian"], experts["bf"])).astype(np.float32)
    return out, dict(P_G=float(gmask.mean()), P_BF=float(bfmask.mean()), P_NLM=float(mask.mean()))


def refined_d5_masked(noisy: np.ndarray, sigma_n: int, recovery: dict, ltp_best: dict,
                      d5_params: dict, refine_params: dict,
                      mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    d5 = d5_params[str(sigma_n)]["best_params"]
    variant = refine_params[str(sigma_n)]
    variant = {k: variant[k] for k in ("name", "h_mult", "noise_alpha", "h_mode", "h_strength")}
    desc = fit_descriptor(noisy, int(d5["K"]), int(d5["rerank_dim"]), float(d5.get("sigma_patch_used", 0.0)))
    bag = prepare_bag(noisy, desc, d5, mask)
    experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
    hmap = h_map_for_variant(noisy, experts, float(d5["h"]), variant)
    out, diag = pool_from_bag(noisy, bag, hmap, float(d5["percentile_clip"]),
                              sigma_n, float(variant["noise_alpha"]))
    diag.update({f"refine_{k}": v for k, v in variant.items()})
    return out, diag


def run_controlled(output_root: str | Path = "outputs", side: int = 128) -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "27_fusion_v5_refined_d5"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_selected(root / "02_image_selection" / "final_four.csv", Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((root / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())
    d5_params = json.loads((root / "18_local_retrieval_optuna" / "local_optuna_best_parameters.json").read_text())
    refine_params = json.loads((root / "26_d5_refinement" / "d5_refinement_best_parameters.json").read_text())
    f0_tuned = json.loads((root / "20_graded_funnel" / "graded_funnel_best_parameters.json").read_text())
    v1_tuned = json.loads((root / "22_fusion_v1" / "fusion_v1_best_parameters.json").read_text())

    cache = prepare_tune_cache(data, tune_bank, recovery, ltp_best, d5_params)
    tuned, models, _ = tune_pareto(data, cache)
    (out_dir / "fusion_v5_masks_used.json").write_text(json.dumps(tuned, indent=2))

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

            v1cfg = v1_tuned[str(sigma_n)]
            v1, v1diag = fusion_v1(noisy, sigma_n, recovery, ltp_best, d5_params,
                                   v1cfg["combo"], v1cfg["budget"])
            variants.append(("FusionV1_original_D5", "", np.nan, v1, v1diag))

            full_old, full_old_diag = d5_full(noisy, d5_params, sigma_n)
            variants.append(("D5_full_original", "", np.nan, full_old, full_old_diag))

            full_ref, full_ref_diag = refined_d5_masked(
                noisy, sigma_n, recovery, ltp_best, d5_params, refine_params, None
            )
            variants.append(("D5_full_refined", "", np.nan, full_ref, full_ref_diag))

            experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
            for lam in MASK_LAMBDAS[sigma_n]:
                cfg = tuned[str(sigma_n)][lam]
                selector = cfg["selector"]
                model = models[str(sigma_n)][selector]
                score = score_from_name(selector, model, noisy, sigma_n, experts)
                mask = _top_budget_mask(score, cfg["budget"])

                branch_ref, bdiag = refined_d5_masked(
                    noisy, sigma_n, recovery, ltp_best, d5_params, refine_params, mask
                )
                routed_ref, rdiag = route_from_mask(noisy, experts, branch_ref, mask)
                merged_diag = {**bdiag, **rdiag, "budget": cfg["budget"], "selector": selector}
                variants.append((f"FusionV5_refined_lambda_{lam}", selector, cfg["budget"],
                                 routed_ref, merged_diag))

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
                    refine_name=diag.get("refine_name", ""),
                    refine_noise_alpha=diag.get("refine_noise_alpha", np.nan),
                    refine_h_mode=diag.get("refine_h_mode", ""),
                    refine_h_strength=diag.get("refine_h_strength", np.nan),
                    **metrics(clean, out),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fusion_v5_results.csv", index=False)
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
    agg.to_csv(out_dir / "fusion_v5_aggregate.csv", index=False)

    focus = agg[agg["method"].str.contains("D5_full|FusionV5|FusionV1|F0|BF_exact", na=False)]
    text = (
        "# FusionV5: refined D5 weights inside fusion\n\n"
        "## Scope\n\n"
        "Uses the same V3 representative masks/selectors, keeps KD-tree D5 as the branch, and swaps in the "
        "D5 weighting refinements from stage 26. This asks whether the small full-D5 gains transfer to fusion.\n\n"
        "## Focus leaderboard\n\n"
        + dataframe_markdown(focus, 5)
        + "\n\n## Full aggregate\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Reading\n\n"
        "A useful result is not just a higher full-D5 score; the refined branch should improve the same fusion "
        "mask at comparable `P_NLM`. If gains are tiny, keep the refinement only as a final quality tweak and "
        "move to 256/native verification.\n"
    )
    (out_dir / "FusionV5_Refined_D5.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/27_fusion_v5_refined_d5/fusion_v5_aggregate.csv").to_string(index=False))
