"""Native-resolution verification of the final candidates.

Every prior verification (128px controlled ablations, 256px selected-method
check) forced a uniform square resize across all four images. This runs the
FINAL candidates only (`BF_exact`, `F0`, `D5_full_original/refined`,
`FusionV5_*_refined`) at each image's own true native size instead --
`img_007`/`img_003` at 256x256, `img_039`/`img_025` at 512x512, no resize
either way. Everything downstream (D5 retrieval, the fusion gate, metrics)
operates per-image on that image's own array shape already, so mixing native
sizes in one pass needs no other code changes -- only the data loader differs
from `bilateral_256_verification.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bilateral_dataset_pipeline import read_image
from bilateral_foundations import dataframe_markdown, metrics
from bilateral_fusion_v2_cost_aware import d5_full
from bilateral_fusion_v5_refined_d5 import refined_d5_masked
from bilateral_graded_funnel import graded_funnel
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED
from bilateral_256_verification import (SELECTED_MASKS, fusion_masked_refined,
                                        prepare_training_cache, train_required_models)


def load_selected_native(selection_csv: Path, dataset_dir: Path) -> dict[str, dict]:
    selected = pd.read_csv(selection_csv)
    data = {}
    for _, r in selected.iterrows():
        _, native, _ = read_image(dataset_dir / r.relative_path)
        data[r.image_id] = dict(clean=native.astype(np.float32), role=r.role, filename=r.filename,
                                relative_path=r.relative_path,
                                native_resolution=f"{int(r.height)}x{int(r.width)}")
    return data


def noisy_bank_native(data: dict, noise_levels: tuple, seed: int) -> dict:
    bank = {}
    for idx, (image_id, item) in enumerate(data.items()):
        for sigma_n in noise_levels:
            rng = np.random.default_rng(seed + idx * 1000 + sigma_n)
            bank[(image_id, sigma_n)] = np.clip(
                item["clean"] + rng.normal(0, sigma_n, item["clean"].shape), 0, 255).astype(np.float32)
    return bank


def run_controlled(output_root: str | Path = "outputs") -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "31_native_final_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_selected_native(root / "02_image_selection" / "final_four.csv", Path("misc"))
    tune_bank = noisy_bank_native(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank_native(data, NOISE_LEVELS, VALIDATION_SEED)
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
        native_res = item["native_resolution"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            variants = [("noisy", "", np.nan, noisy, dict(runtime_ms=0.0))]

            bf, bf_time = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            variants.append(("BF_exact", "", np.nan, bf, dict(runtime_ms=bf_time)))

            f0, f0diag = graded_funnel(noisy, sigma_n, recovery, ltp_best, d5_params,
                                       f0_tuned[str(sigma_n)]["grad_threshold"])
            variants.append(("F0_Sobel_graded_funnel", "", np.nan, f0, f0diag))

            d5_orig, d5diag = d5_full(noisy, d5_params, sigma_n)
            variants.append(("D5_full_original", "", np.nan, d5_orig, d5diag))

            d5_ref, refdiag = refined_d5_masked(noisy, sigma_n, recovery, ltp_best, d5_params, refine_params, None)
            variants.append(("D5_full_refined", "", np.nan, d5_ref, refdiag))

            for mode_name, (selector, budget) in SELECTED_MASKS[sigma_n].items():
                model = models[str(sigma_n)][selector]
                out, diag = fusion_masked_refined(noisy, sigma_n, recovery, ltp_best, d5_params,
                                                  refine_params, selector, budget, model)
                variants.append((f"FusionV5_{mode_name}_refined", selector, budget, out, diag))

            for method, selector, budget, out, diag in variants:
                rows.append(dict(image_id=image_id, role=item["role"], native_resolution=native_res,
                                 noise_sigma=sigma_n, method=method, selector=selector, budget=budget,
                                 P_G=diag.get("P_G", np.nan), P_BF=diag.get("P_BF", np.nan),
                                 P_NLM=diag.get("P_NLM", np.nan),
                                 runtime_ms=diag.get("runtime_ms", np.nan),
                                 descriptor_comparison_count=diag.get("descriptor_comparison_count", np.nan),
                                 **metrics(clean, out)))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "native_final_verification_results.csv", index=False)
    write_report(out_dir, df, data)
    return df


def write_report(out_dir: Path, df: pd.DataFrame, data: dict) -> None:
    size_rows = [dict(image_id=k, role=v["role"], native_resolution=v["native_resolution"]) for k, v in data.items()]
    agg = df.groupby(["noise_sigma", "method", "selector", "budget"], dropna=False).agg(
        PSNR=("PSNR", "mean"), SSIM=("SSIM", "mean"), edge=("edge_metric", "mean"),
        texture=("texture_metric", "mean"), runtime_ms=("runtime_ms", "mean"),
        P_NLM=("P_NLM", "mean"),
    ).reset_index().sort_values(["noise_sigma", "SSIM"], ascending=[True, False])
    agg.to_csv(out_dir / "native_final_verification_aggregate.csv", index=False)
    best_rows = []
    for sigma_n, part in agg[~agg.method.eq("noisy")].groupby("noise_sigma"):
        best_rows.append(part.loc[part.SSIM.idxmax()])
    best_df = pd.DataFrame(best_rows)
    text = (
        "# Native-resolution verification of the final candidates\n\n"
        "## Scope\n\n"
        "Every prior benchmark (128px controlled ablations, 256px selected-method check) forced a uniform "
        "square resize across all four images. This runs the final candidates (`BF_exact`, `F0`, "
        "`D5_full_original/refined`, `FusionV5_quality/efficient_refined`) at each image's own TRUE native size "
        "instead -- no resize either direction. D5/gate hyperparameters are the already-selected 128px-tuned "
        "values (same precedent as the 256px verification pass); only the fusion gate is retrained fresh, on "
        "native-resolution features.\n\n"
        "## Native sizes\n\n"
        + dataframe_markdown(pd.DataFrame(size_rows), 5)
        + "\n\n## Aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Best method by noise level\n\n"
        + dataframe_markdown(best_df, 5)
        + "\n\n## Reading\n\n"
        "If the ranking (D5 best quality, FusionV5 efficient best tradeoff, both clearly beating BF_exact) "
        "holds at native size too, the whole project's conclusions generalize beyond the 128px/256px controlled "
        "benchmark. If the ranking flips at native size, that's a real external-validity gap the final writeup "
        "needs to state plainly, not smooth over.\n"
    )
    (out_dir / "Native_Final_Verification.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/31_native_final_verification/native_final_verification_aggregate.csv").to_string(index=False))
