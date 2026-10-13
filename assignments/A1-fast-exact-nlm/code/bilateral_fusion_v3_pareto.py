"""FusionV3 selector Pareto study.

FusionV2 constrained D5 to <= 50%, but its selector family was only one
logistic classifier. This stage compares several cheap selector score families:

  - hand-built ambiguity scores from FusionV1,
  - oracle-trained logistic classifiers from FusionV2,
  - oracle-benefit ridge regressors.

It then chooses configs by an explicit utility:

    utility = SSIM - lambda_cost * P_NLM

The selected configs are validated with true masked D5 queries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from bilateral_foundations import dataframe_markdown, metrics
from bilateral_fusion_v1 import BUDGETS as V1_BUDGETS
from bilateral_fusion_v1 import CUE_COMBOS, ambiguity_score, _top_budget_mask, expert_maps, fusion_v1
from bilateral_fusion_v2_cost_aware import (C_GRID, FEATURE_NAMES, GateModel,
                                            d5_full, d5_masked, feature_stack,
                                            prepare_tune_cache, train_gate)
from bilateral_graded_funnel import graded_funnel
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank


BUDGET_GRID = tuple(sorted(set((0.10, 0.25, 0.35, 0.50) + tuple(V1_BUDGETS))))
LAMBDA_GRID = (0.00, 0.03, 0.06, 0.10, 0.16)
RIDGE_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass
class RidgeGate:
    coef: np.ndarray
    intercept: float

    def score(self, features: np.ndarray) -> np.ndarray:
        flat = features.reshape(-1, features.shape[-1])
        return (flat @ self.coef + self.intercept).reshape(features.shape[:2])


def train_ridge_gate(features: list[np.ndarray], targets: list[np.ndarray], alpha: float) -> RidgeGate:
    x = np.concatenate([f.reshape(-1, f.shape[-1]) for f in features], axis=0)
    y = np.concatenate([t.ravel() for t in targets], axis=0)
    model = Ridge(alpha=float(alpha), random_state=20260827)
    model.fit(x, y)
    return RidgeGate(model.coef_.astype(np.float32), float(model.intercept_))


def build_score_models(data: dict, cache: dict, sigma_n: int) -> dict:
    features = [cache[(image_id, sigma_n)]["features"] for image_id in data]
    labels = [cache[(image_id, sigma_n)]["label"] for image_id in data]
    targets = []
    for image_id in data:
        c = cache[(image_id, sigma_n)]
        benefit = np.abs(c["base"] - c["clean"]) - np.abs(c["d5"] - c["clean"])
        targets.append(benefit.astype(np.float32))

    models = {}
    for combo in CUE_COMBOS:
        models[f"cheap::{combo}"] = None
    for c_value in C_GRID:
        models[f"logistic::C={c_value}"] = train_gate(features, labels, c_value)
    for alpha in RIDGE_ALPHA_GRID:
        models[f"ridge_benefit::alpha={alpha}"] = train_ridge_gate(features, targets, alpha)
    return models


def score_from_name(name: str, model, noisy: np.ndarray, sigma_n: int, experts: dict) -> np.ndarray:
    if name.startswith("cheap::"):
        combo = name.split("::", 1)[1]
        return ambiguity_score(noisy, experts, sigma_n, combo)
    features = feature_stack(noisy, sigma_n, experts)
    return model.score(features)


def route_simulated(noisy: np.ndarray, experts: dict, d5_image: np.ndarray,
                    score: np.ndarray, budget: float) -> tuple[np.ndarray, dict]:
    mask = _top_budget_mask(score, budget)
    gmask = (experts["signal"] <= experts["g_threshold"]) & ~mask
    bfmask = ~(mask | gmask)
    out = np.where(mask, d5_image, np.where(gmask, experts["gaussian"], experts["bf"])).astype(np.float32)
    return out, dict(P_G=float(gmask.mean()), P_BF=float(bfmask.mean()), P_NLM=float(mask.mean()))


def tune_pareto(data: dict, cache: dict) -> tuple[dict, dict, pd.DataFrame]:
    tuned = {}
    all_models = {}
    tune_rows = []
    for sigma_n in NOISE_LEVELS:
        models = build_score_models(data, cache, sigma_n)
        all_models[str(sigma_n)] = models
        for selector, model in models.items():
            for budget in BUDGET_GRID:
                vals = []
                ps = []
                for image_id in data:
                    c = cache[(image_id, sigma_n)]
                    score = score_from_name(selector, model, c["noisy"], sigma_n, c["experts"])
                    out, diag = route_simulated(c["noisy"], c["experts"], c["d5"], score, budget)
                    vals.append(metrics(c["clean"], out)["SSIM"])
                    ps.append(diag["P_NLM"])
                mean_ssim = float(np.mean(vals))
                mean_p = float(np.mean(ps))
                for lam in LAMBDA_GRID:
                    tune_rows.append(dict(
                        noise_sigma=sigma_n,
                        lambda_cost=lam,
                        selector=selector,
                        budget=budget,
                        tune_SSIM=mean_ssim,
                        tune_P_NLM=mean_p,
                        utility=mean_ssim - lam * mean_p,
                    ))

        rows_sigma = [r for r in tune_rows if r["noise_sigma"] == sigma_n]
        df_sigma = pd.DataFrame(rows_sigma)
        best_by_lambda = {}
        for lam in LAMBDA_GRID:
            sub = df_sigma[df_sigma["lambda_cost"] == lam].sort_values("utility", ascending=False).iloc[0]
            best_by_lambda[str(lam)] = {
                "selector": sub["selector"],
                "budget": float(sub["budget"]),
                "lambda_cost": float(lam),
                "tune_SSIM": float(sub["tune_SSIM"]),
                "tune_P_NLM": float(sub["tune_P_NLM"]),
                "utility": float(sub["utility"]),
            }
        tuned[str(sigma_n)] = best_by_lambda
    return tuned, all_models, pd.DataFrame(tune_rows)


def eval_selected(noisy: np.ndarray, sigma_n: int, recovery: dict, ltp_best: dict,
                  d5_params: dict, selector: str, model, budget: float) -> tuple[np.ndarray, dict]:
    experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
    score = score_from_name(selector, model, noisy, sigma_n, experts)
    mask = _top_budget_mask(score, budget)
    nlm, ndiag = d5_masked(noisy, d5_params, sigma_n, mask)
    out, route_diag = route_simulated(noisy, experts, nlm, score, budget)
    diag = dict(
        **route_diag,
        runtime_ms=experts["cheap_runtime_ms"] + ndiag["runtime_ms"],
        range_eval_count=experts["bf_range_eval_count"],
        nlm_escalated_pixels=ndiag["escalated_pixels"],
        nlm_descriptor_comparison_count=ndiag["descriptor_comparison_count"],
    )
    return out, diag


def run_controlled(output_root: str | Path = "outputs", side: int = 128) -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "24_fusion_v3_pareto"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_selected(root / "02_image_selection" / "final_four.csv", Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((root / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())
    d5_params = json.loads((root / "18_local_retrieval_optuna" / "local_optuna_best_parameters.json").read_text())
    f0_tuned = json.loads((root / "20_graded_funnel" / "graded_funnel_best_parameters.json").read_text())
    v1_tuned = json.loads((root / "22_fusion_v1" / "fusion_v1_best_parameters.json").read_text())

    cache = prepare_tune_cache(data, tune_bank, recovery, ltp_best, d5_params)
    tuned, models, tune_df = tune_pareto(data, cache)
    tune_df.to_csv(out_dir / "fusion_v3_tune_pareto.csv", index=False)
    (out_dir / "fusion_v3_selected_parameters.json").write_text(json.dumps(tuned, indent=2))

    rows = []
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            variants = [("noisy", "", np.nan, noisy, dict(runtime_ms=0.0))]

            bf, bf_time = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            variants.append(("BF_exact", "", np.nan, bf, dict(runtime_ms=bf_time)))

            f0, f0diag = graded_funnel(noisy, sigma_n, recovery, ltp_best, d5_params,
                                       f0_tuned[str(sigma_n)]["grad_threshold"])
            variants.append(("F0_Sobel_graded_funnel", "", np.nan, f0, f0diag))

            v1cfg = v1_tuned[str(sigma_n)]
            v1, v1diag = fusion_v1(noisy, sigma_n, recovery, ltp_best, d5_params,
                                   v1cfg["combo"], v1cfg["budget"])
            variants.append(("FusionV1_quality_gate", "", np.nan, v1, v1diag))

            full, fulldiag = d5_full(noisy, d5_params, sigma_n)
            variants.append(("D5_full_image", "", np.nan, full, fulldiag))

            for lam_str, cfg in tuned[str(sigma_n)].items():
                selector = cfg["selector"]
                model = models[str(sigma_n)][selector]
                v3, v3diag = eval_selected(noisy, sigma_n, recovery, ltp_best, d5_params,
                                           selector, model, cfg["budget"])
                v3diag.update(dict(selector=selector, budget=cfg["budget"]))
                variants.append((f"FusionV3_lambda_{lam_str}", selector, cfg["budget"], v3, v3diag))

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
                    range_eval_count=diag.get("range_eval_count", np.nan),
                    nlm_escalated_pixels=diag.get("nlm_escalated_pixels", np.nan),
                    nlm_descriptor_comparison_count=diag.get("nlm_descriptor_comparison_count", np.nan),
                    **metrics(clean, out),
                ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fusion_v3_results.csv", index=False)
    write_report(out_dir, df, tune_df, tuned)
    return df


def write_report(out_dir: Path, df: pd.DataFrame, tune_df: pd.DataFrame, tuned: dict) -> None:
    agg = df.groupby(["noise_sigma", "method", "selector", "budget"], dropna=False).agg(
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
    agg.to_csv(out_dir / "fusion_v3_aggregate.csv", index=False)

    selected_rows = []
    for noise, by_lam in tuned.items():
        for lam, cfg in by_lam.items():
            row = dict(cfg)
            row["lambda_cost_key"] = float(lam)
            selected_rows.append(dict(noise_sigma=int(noise), **row))
    selected_df = pd.DataFrame(selected_rows)

    v3_rows = agg[agg["method"].str.startswith("FusionV3")].copy()
    text = (
        "# FusionV3 selector Pareto study\n\n"
        "## Scope\n\n"
        "This stage compares cheap hand scores, oracle-trained logistic gates, and oracle-benefit ridge regressors. "
        "Selection is made by explicit utility: `SSIM - lambda_cost * P_NLM`. Validation uses true masked D5 "
        "queries for the selected configs.\n\n"
        "## Selected configs from tuning\n\n"
        + dataframe_markdown(selected_df, 5)
        + "\n\n## Validation aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## FusionV3 rows only\n\n"
        + dataframe_markdown(v3_rows, 5)
        + "\n\n## Reading\n\n"
        "If increasing `lambda_cost` moves the selected budget below 0.50 while holding most of the SSIM, we have "
        "a real cost-aware route. If the selected config stays at high budget, the fixed D5 branch is still doing "
        "work that cheap features cannot confidently predict away. In that case, the next accuracy-per-cost lever "
        "is not another scalar gate but changing the selected-pixel NLM branch itself: exact local selected-pixel "
        "search, adaptive `h`, or adaptive `K`.\n"
    )
    (out_dir / "FusionV3_Pareto.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/24_fusion_v3_pareto/fusion_v3_aggregate.csv").to_string(index=False))
