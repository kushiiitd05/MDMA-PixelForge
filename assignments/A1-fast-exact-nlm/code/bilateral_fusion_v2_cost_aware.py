"""FusionV2: cost-aware learned cheap gate for G/BF/D5.

FusionV1 proved the old F0 gate was the bottleneck, but it selected a heavy
75% D5 budget. FusionV2 learns a cheap per-pixel D5-benefit score from the
oracle diagnostic on the tuning seed, then constrains the selected budget to
<= 50% of pixels.

Important: clean images are used only to train/tune the gate on the synthetic
tuning seed. Validation uses only cheap features from the noisy image plus
true masked D5 queries for selected pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from bilateral_foundations import dataframe_markdown, metrics
from bilateral_graded_funnel import graded_funnel, local_pool_masked
from bilateral_fusion_v1 import _robust_norm, _top_budget_mask, expert_maps, fusion_v1
from bilateral_local_retrieval_optuna import fit_descriptor
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank


FEATURE_NAMES = (
    "bf_residual_norm",
    "neff_low_norm",
    "neff_high_norm",
    "signal_norm",
    "gradient_norm",
    "gaussian_safe_flag",
)
C_GRID = (0.05, 0.1, 0.3, 1.0)
BUDGET_GRID = (0.10, 0.25, 0.35, 0.50)
ORACLE_MARGIN = 0.0


@dataclass
class GateModel:
    coef: np.ndarray
    intercept: float

    def score(self, features: np.ndarray) -> np.ndarray:
        flat = features.reshape(-1, features.shape[-1])
        logits = flat @ self.coef + self.intercept
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))).reshape(features.shape[:2])


def feature_stack(noisy: np.ndarray, sigma_n: int, experts: dict) -> np.ndarray:
    sigma = max(float(sigma_n), 1e-6)
    residual = np.abs(noisy.astype(np.float32) - experts["bf"]) / sigma
    signal = experts["signal"] / sigma
    gsafe = (experts["signal"] <= experts["g_threshold"]).astype(np.float32)
    maps = [
        _robust_norm(residual),
        _robust_norm(-experts["neff"]),
        _robust_norm(experts["neff"]),
        _robust_norm(signal),
        _robust_norm(experts["gradient"]),
        gsafe,
    ]
    return np.stack(maps, axis=-1).astype(np.float32)


def d5_full(noisy: np.ndarray, d5_params: dict, sigma_n: int) -> tuple[np.ndarray, dict]:
    d5 = d5_params[str(sigma_n)]["best_params"]
    rerank_dim = int(d5["rerank_dim"])
    retrieval_dim = min(int(d5["retrieval_dim"]), rerank_dim)
    desc = fit_descriptor(noisy, int(d5["K"]), rerank_dim, float(d5.get("sigma_patch_used", 0.0)))
    return local_pool_masked(
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


def d5_masked(noisy: np.ndarray, d5_params: dict, sigma_n: int, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    d5 = d5_params[str(sigma_n)]["best_params"]
    rerank_dim = int(d5["rerank_dim"])
    retrieval_dim = min(int(d5["retrieval_dim"]), rerank_dim)
    desc = fit_descriptor(noisy, int(d5["K"]), rerank_dim, float(d5.get("sigma_patch_used", 0.0)))
    return local_pool_masked(
        noisy,
        desc,
        retrieval_dim,
        rerank_dim,
        float(d5["spatial_weight"]),
        int(d5["bag_size"]),
        float(d5["h"]),
        float(d5["percentile_clip"]),
        mask,
    )


def route_from_score(noisy: np.ndarray, experts: dict, d5_image: np.ndarray,
                     score: np.ndarray, budget: float) -> tuple[np.ndarray, dict]:
    nlm_mask = _top_budget_mask(score, budget)
    gmask = (experts["signal"] <= experts["g_threshold"]) & ~nlm_mask
    bfmask = ~(nlm_mask | gmask)
    out = np.where(nlm_mask, d5_image, np.where(gmask, experts["gaussian"], experts["bf"])).astype(np.float32)
    return out, dict(P_G=float(gmask.mean()), P_BF=float(bfmask.mean()), P_NLM=float(nlm_mask.mean()))


def prepare_tune_cache(data: dict, bank: dict, recovery: dict, ltp_best: dict, d5_params: dict) -> dict:
    cache = {}
    for image_id, item in data.items():
        for sigma_n in NOISE_LEVELS:
            noisy = bank[(image_id, sigma_n)]
            experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
            base = np.where(
                experts["signal"] <= experts["g_threshold"],
                experts["gaussian"],
                experts["bf"],
            ).astype(np.float32)
            d5, _ = d5_full(noisy, d5_params, sigma_n)
            feat = feature_stack(noisy, sigma_n, experts)
            label = (np.abs(d5 - item["clean"]) + ORACLE_MARGIN < np.abs(base - item["clean"]))
            cache[(image_id, sigma_n)] = dict(
                noisy=noisy,
                clean=item["clean"],
                experts=experts,
                base=base,
                d5=d5,
                features=feat,
                label=label.astype(np.int32),
            )
    return cache


def train_gate(features: list[np.ndarray], labels: list[np.ndarray], c_value: float) -> GateModel:
    x = np.concatenate([f.reshape(-1, f.shape[-1]) for f in features], axis=0)
    y = np.concatenate([lab.ravel() for lab in labels], axis=0)
    # Guard rare degenerate cases.
    if np.unique(y).size < 2:
        prior = float(y.mean())
        prior = np.clip(prior, 1e-3, 1 - 1e-3)
        return GateModel(np.zeros(x.shape[1], dtype=np.float32), float(np.log(prior / (1 - prior))))
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        random_state=20260827,
    )
    model.fit(x, y)
    return GateModel(model.coef_[0].astype(np.float32), float(model.intercept_[0]))


def tune_cost_aware(data: dict, cache: dict) -> tuple[dict, dict]:
    tuned = {}
    models = {}
    for sigma_n in NOISE_LEVELS:
        features = [cache[(image_id, sigma_n)]["features"] for image_id in data]
        labels = [cache[(image_id, sigma_n)]["label"] for image_id in data]
        best_score = -np.inf
        best = None
        best_model = None
        for c_value in C_GRID:
            model = train_gate(features, labels, c_value)
            for budget in BUDGET_GRID:
                vals = []
                ps = []
                for image_id in data:
                    c = cache[(image_id, sigma_n)]
                    score = model.score(c["features"])
                    out, diag = route_from_score(c["noisy"], c["experts"], c["d5"], score, budget)
                    vals.append(metrics(c["clean"], out)["SSIM"])
                    ps.append(diag["P_NLM"])
                mean_ssim = float(np.mean(vals))
                mean_p = float(np.mean(ps))
                if mean_ssim > best_score:
                    best_score = mean_ssim
                    best = dict(C=c_value, budget=budget, tune_SSIM=mean_ssim, tune_P_NLM=mean_p)
                    best_model = model
        tuned[str(sigma_n)] = best
        models[str(sigma_n)] = best_model
    return tuned, models


def fusion_v2(noisy: np.ndarray, sigma_n: int, recovery: dict, ltp_best: dict,
              d5_params: dict, model: GateModel, budget: float) -> tuple[np.ndarray, dict]:
    experts = expert_maps(noisy, sigma_n, recovery, ltp_best)
    features = feature_stack(noisy, sigma_n, experts)
    score = model.score(features)
    nlm_mask = _top_budget_mask(score, budget)
    nlm, ndiag = d5_masked(noisy, d5_params, sigma_n, nlm_mask)
    out, route_diag = route_from_score(noisy, experts, nlm, score, budget)
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
    out_dir = root / "23_fusion_v2_cost_aware"
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
    tuned, models = tune_cost_aware(data, cache)

    serializable = {}
    for noise, cfg in tuned.items():
        model = models[noise]
        serializable[noise] = {
            **cfg,
            "feature_names": list(FEATURE_NAMES),
            "coef": model.coef.tolist(),
            "intercept": model.intercept,
            "oracle_margin": ORACLE_MARGIN,
        }
    (out_dir / "fusion_v2_best_parameters.json").write_text(json.dumps(serializable, indent=2))

    rows = []
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            variants = [("noisy", noisy, dict(runtime_ms=0.0))]

            bf, bf_time = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            variants.append(("BF_exact", bf, dict(runtime_ms=bf_time)))

            f0, f0diag = graded_funnel(
                noisy, sigma_n, recovery, ltp_best, d5_params,
                f0_tuned[str(sigma_n)]["grad_threshold"],
            )
            variants.append(("F0_Sobel_graded_funnel", f0, f0diag))

            v1cfg = v1_tuned[str(sigma_n)]
            v1, v1diag = fusion_v1(
                noisy, sigma_n, recovery, ltp_best, d5_params,
                v1cfg["combo"], v1cfg["budget"],
            )
            variants.append(("FusionV1_quality_gate", v1, v1diag))

            v2cfg = tuned[str(sigma_n)]
            v2, v2diag = fusion_v2(
                noisy, sigma_n, recovery, ltp_best, d5_params,
                models[str(sigma_n)], v2cfg["budget"],
            )
            v2diag.update(dict(C=v2cfg["C"], budget=v2cfg["budget"]))
            variants.append(("FusionV2_cost_aware", v2, v2diag))

            full, fulldiag = d5_full(noisy, d5_params, sigma_n)
            variants.append(("D5_full_image", full, fulldiag))

            for method, out, diag in variants:
                rows.append(dict(
                    image_id=image_id,
                    role=item["role"],
                    noise_sigma=sigma_n,
                    method=method,
                    side=side,
                    C=diag.get("C", np.nan),
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
    df.to_csv(out_dir / "fusion_v2_results.csv", index=False)
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
    agg.to_csv(out_dir / "fusion_v2_aggregate.csv", index=False)
    tune_df = pd.DataFrame([dict(noise_sigma=int(k), **v) for k, v in tuned.items()])

    text = (
        "# FusionV2: cost-aware learned gate\n\n"
        "## Scope\n\n"
        "FusionV2 trains a cheap logistic gate on the tuning seed to predict where D5 beats the G/BF base. "
        "The selected D5 budget is constrained to `<=0.50`, unlike FusionV1's quality-only gate which selected "
        "`0.75` everywhere. Validation uses true masked D5 queries, not precomputed full-D5 routing.\n\n"
        "Features: "
        + ", ".join(f"`{x}`" for x in FEATURE_NAMES)
        + ".\n\n"
        "## Tuned gate\n\n"
        + dataframe_markdown(tune_df, 5)
        + "\n\n## Aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Reading\n\n"
        "If `FusionV2_cost_aware` beats F0 and approaches FusionV1 while using `P_NLM<=0.50`, the learned "
        "cost-aware gate is a better fusion candidate. If it falls far below FusionV1, the next refinement is "
        "not more D5 math but a better risk/proxy target for the selector.\n"
    )
    (out_dir / "FusionV2_Cost_Aware.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/23_fusion_v2_cost_aware/fusion_v2_aggregate.csv").to_string(index=False))
