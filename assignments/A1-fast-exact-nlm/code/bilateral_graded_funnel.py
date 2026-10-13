"""BF-to-NLM graded funnel (MD Step 10) -- the actual routing fusion the
MD's abstract describes, distinct from D4/D5's "borrow BF's spatial-Gaussian
formula inside NLM's own math."

Three-way per-pixel route, in increasing cost order:

    Gaussian  -- noise-corrected local variation says the neighborhood is
                 flat; reuses V2's own tuned signal_std threshold unchanged.
    BF        -- the default workhorse for ordinary, non-ambiguous pixels.
    NLM (D5)  -- escalate only where a Sobel-gradient-on-a-smoothed-guide
                 signal says the pixel sits in a structurally rich/edge
                 region -- exactly the same gradient_map + threshold V2's own
                 LTP-edge ablation already validated as a real structural cue,
                 reused here to gate the patch route instead of the LTP route.

The NLM branch is queried ONLY for escalated pixels (KD-tree query restricted
to the escalated subset), so the escalation fraction directly controls real
compute, not just which output gets selected after the fact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from bilateral_foundations import (BOUNDARY, dataframe_markdown,
                                   gaussian_from_scratch, metrics)
from bilateral_local_retrieval_optuna import fit_descriptor
from bilateral_recovery import NOISE_LEVELS, apply_method, noise_corrected_signal_std
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED, load_selected, noisy_bank
from bilateral_v2 import gradient_map, masked_bilateral

GRAD_THRESHOLD_GRID = (4.0, 8.0, 12.0, 20.0, 30.0, 45.0, 60.0)


def local_pool_masked(noisy: np.ndarray, desc: np.ndarray, retrieval_dim: int, rerank_dim: int,
                      spatial_weight: float, bag_size: int, h: float, percentile_clip: float,
                      query_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Same math as `local_pool_with_clip`, but the KD-tree is only QUERIED
    for pixels where `query_mask` is true -- the tree is still built over the
    whole image (candidates can come from anywhere), but escalation fraction
    directly controls query cost, giving real compute savings, not just
    output selection after the fact."""
    x = noisy.astype(np.float32)
    height, width = x.shape
    flat_x = x.ravel()
    flat_mask = query_mask.ravel()
    query_idx = np.where(flat_mask)[0]
    if query_idx.size == 0:
        return x.copy(), dict(runtime_ms=0.0, escalated_pixels=0, candidate_count=0,
                              descriptor_comparison_count=0)

    z_ret = desc[..., :retrieval_dim].reshape(-1, retrieval_dim)
    z_rer = desc[..., :rerank_dim].reshape(-1, rerank_dim)
    ys, xs = np.indices((height, width))
    coords = np.stack([ys.ravel(), xs.ravel()], axis=-1).astype(np.float32) * spatial_weight
    z_query_all = np.concatenate([z_ret, coords], axis=-1)

    start = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=bag_size, algorithm="kd_tree", metric="euclidean")
    nn.fit(z_query_all)
    _, indices = nn.kneighbors(z_query_all[query_idx])
    kd_ms = (time.perf_counter() - start) * 1000.0

    start_rerank = time.perf_counter()
    diff = z_rer[query_idx][:, None, :] - z_rer[indices]
    dist2 = np.sum(diff * diff, axis=-1) / rerank_dim
    weight = np.exp(-dist2 / max(h * h, 1e-12))
    if percentile_clip > 0:
        thresh = np.percentile(weight, percentile_clip, axis=1, keepdims=True)
        weight = np.where(weight >= thresh, weight, 0.0)
        zero_rows = weight.sum(axis=1) <= 1e-12
        if np.any(zero_rows):
            weight[zero_rows, 0] = 1.0
    values = flat_x[indices]
    sub_out = (weight * values).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-12)
    rerank_ms = (time.perf_counter() - start_rerank) * 1000.0

    out = flat_x.copy()
    out[query_idx] = sub_out
    diag = dict(runtime_ms=kd_ms + rerank_ms, escalated_pixels=int(query_idx.size),
               candidate_count=int(query_idx.size) * bag_size,
               descriptor_comparison_count=int(query_idx.size) * bag_size * (retrieval_dim + rerank_dim))
    return out.reshape(x.shape).astype(np.float32), diag


def graded_funnel(noisy: np.ndarray, sigma_n: int, recovery: dict, ltp_best: dict,
                  d5_params: dict, grad_threshold: float) -> tuple[np.ndarray, dict]:
    p = recovery["best"]["BF_exact"][str(sigma_n)]
    g_threshold = recovery["noise_aware_gate"][str(sigma_n)]["signal_std_threshold"]
    guide_sigma = ltp_best[str(sigma_n)]["guide_sigma"]
    d5 = d5_params[str(sigma_n)]["best_params"]
    k, rerank_dim = d5["K"], d5["rerank_dim"]
    retrieval_dim = min(d5["retrieval_dim"], rerank_dim)
    sigma_patch = d5.get("sigma_patch_used", 0.0)

    g, gdiag = gaussian_from_scratch(noisy, p["sigma_s"], p["k"], BOUNDARY)
    signal = noise_corrected_signal_std(noisy, sigma_n, p["k"])
    gmask = signal <= g_threshold
    rest = ~gmask

    grad = gradient_map(noisy, guide_sigma)
    escalate = rest & (grad >= grad_threshold)
    bf_active = rest & ~escalate

    bf, bdiag = masked_bilateral(noisy, bf_active, p["sigma_s"], p["sigma_r"], p["k"], BOUNDARY)

    desc = fit_descriptor(noisy, k, rerank_dim, sigma_patch)
    nlm, ndiag = local_pool_masked(noisy, desc, retrieval_dim, rerank_dim, d5["spatial_weight"],
                                   d5["bag_size"], d5["h"], d5["percentile_clip"], escalate)

    out = np.where(gmask, g, np.where(escalate, nlm, bf)).astype(np.float32)
    diag = dict(P_G=float(gmask.mean()), P_BF=float(bf_active.mean()), P_NLM=float(escalate.mean()),
               grad_threshold=grad_threshold,
               runtime_ms=gdiag["runtime_ms"] + bdiag["runtime_ms"] + ndiag["runtime_ms"],
               range_eval_count=bdiag["range_eval_count"],
               nlm_escalated_pixels=ndiag["escalated_pixels"],
               nlm_descriptor_comparison_count=ndiag["descriptor_comparison_count"])
    return out, diag


def tune_grad_threshold(data: dict, bank: dict, recovery: dict, ltp_best: dict, d5_params: dict) -> dict:
    best = {}
    for sigma_n in NOISE_LEVELS:
        best_score, best_t = -np.inf, None
        for t in GRAD_THRESHOLD_GRID:
            vals = []
            for image_id, item in data.items():
                out, _ = graded_funnel(bank[(image_id, sigma_n)], sigma_n, recovery, ltp_best, d5_params, t)
                vals.append(metrics(item["clean"], out)["SSIM"])
            score = float(np.mean(vals))
            if score > best_score:
                best_score, best_t = score, t
        best[str(sigma_n)] = dict(grad_threshold=best_t, tune_SSIM=best_score)
    return best


def run_controlled(output_root: str | Path = "outputs", side: int = 128) -> pd.DataFrame:
    root = Path(output_root)
    out_dir = root / "20_graded_funnel"
    sheet_dir = out_dir / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    data = load_selected(root / "02_image_selection" / "final_four.csv", Path("misc"), side)
    tune_bank = noisy_bank(data, NOISE_LEVELS, TUNE_SEED)
    eval_bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    recovery = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((root / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())
    d5_params = json.loads((root / "18_local_retrieval_optuna" / "local_optuna_best_parameters.json").read_text())

    tuned = tune_grad_threshold(data, tune_bank, recovery, ltp_best, d5_params)
    (out_dir / "graded_funnel_best_parameters.json").write_text(json.dumps(tuned, indent=2))

    rows = []
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            t = tuned[str(sigma_n)]["grad_threshold"]
            variants = [("noisy", noisy, dict(runtime_ms=0.0))]
            bf, bf_time = apply_method("BF_exact", noisy, recovery["best"]["BF_exact"][str(sigma_n)])
            variants.append(("BF_exact", bf, dict(runtime_ms=bf_time)))
            funnel_out, funnel_diag = graded_funnel(noisy, sigma_n, recovery, ltp_best, d5_params, t)
            variants.append(("Graded_Funnel_BF_NLM", funnel_out, funnel_diag))
            all_escalate = np.ones_like(noisy, dtype=bool)
            desc = fit_descriptor(noisy, d5_params[str(sigma_n)]["best_params"]["K"],
                                  d5_params[str(sigma_n)]["best_params"]["rerank_dim"],
                                  d5_params[str(sigma_n)]["best_params"].get("sigma_patch_used", 0.0))
            d5 = d5_params[str(sigma_n)]["best_params"]
            retrieval_dim = min(d5["retrieval_dim"], d5["rerank_dim"])
            nlm_full, nlm_full_diag = local_pool_masked(noisy, desc, retrieval_dim, d5["rerank_dim"],
                                                        d5["spatial_weight"], d5["bag_size"], d5["h"],
                                                        d5["percentile_clip"], all_escalate)
            variants.append(("NLM_D5_full_image", nlm_full, nlm_full_diag))

            for method, out, diag in variants:
                rows.append(dict(image_id=image_id, role=item["role"], noise_sigma=sigma_n,
                                 method=method, side=side,
                                 P_G=diag.get("P_G", np.nan), P_BF=diag.get("P_BF", np.nan),
                                 P_NLM=diag.get("P_NLM", np.nan),
                                 runtime_ms=diag.get("runtime_ms", np.nan),
                                 range_eval_count=diag.get("range_eval_count", np.nan),
                                 nlm_escalated_pixels=diag.get("nlm_escalated_pixels", np.nan),
                                 nlm_descriptor_comparison_count=diag.get("nlm_descriptor_comparison_count", np.nan),
                                 **metrics(clean, out)))
            if sigma_n == 25:
                fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
                panels = [("clean", clean, {})] + variants
                for ax, (name, image, diag) in zip(axes, panels):
                    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"{name}\n{diag.get('runtime_ms', 0):.1f} ms", fontsize=7)
                    ax.axis("off")
                fig.suptitle(f"{image_id}: BF-to-NLM graded funnel, side={side}, sigma_n=25")
                fig.tight_layout()
                fig.savefig(sheet_dir / f"{image_id}_graded_funnel.png", dpi=160)
                plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "graded_funnel_results.csv", index=False)
    write_report(out_dir, df, tuned)
    return df


def write_report(out_dir: Path, df: pd.DataFrame, tuned: dict) -> None:
    agg = df.groupby(["noise_sigma", "method"]).agg(
        PSNR=("PSNR", "mean"), SSIM=("SSIM", "mean"),
        edge=("edge_metric", "mean"), texture=("texture_metric", "mean"),
        runtime_ms=("runtime_ms", "mean"), P_G=("P_G", "mean"), P_BF=("P_BF", "mean"),
        P_NLM=("P_NLM", "mean"), nlm_escalated_pixels=("nlm_escalated_pixels", "mean"),
        nlm_descriptor_comparison_count=("nlm_descriptor_comparison_count", "mean"),
    ).reset_index().sort_values(["noise_sigma", "SSIM"], ascending=[True, False])
    agg.to_csv(out_dir / "graded_funnel_aggregate.csv", index=False)
    tune_df = pd.DataFrame([dict(noise_sigma=int(k), **v) for k, v in tuned.items()])
    text = (
        "# BF-to-NLM graded funnel (MD Step 10)\n\n"
        "## Scope\n\n"
        "Three-way per-pixel route: Gaussian (V2's own noise-corrected homogeneity gate, unchanged), BF "
        "(default), NLM/D5 (escalate only where a Sobel-gradient structural signal -- the same cue V2's LTP-edge "
        "ablation already validated -- crosses a tuned threshold). The NLM branch's KD-tree is queried ONLY for "
        "escalated pixels, so `P_NLM` directly controls real compute (`nlm_descriptor_comparison_count`), not "
        "just which output gets picked after the fact. `NLM_D5_full_image` (escalate=100%) is included as a "
        "reference for what running D5 everywhere would cost/score, to make the funnel's actual savings legible.\n\n"
        "## Tuned escalation threshold\n\n"
        + dataframe_markdown(tune_df, 5)
        + "\n\n## Aggregate leaderboard\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Reading\n\n"
        "This is the MD's actual \"graded funnel\" (Step 10) -- distinct from D4/D5, which only reused BF's "
        "spatial-Gaussian formula INSIDE the NLM branch and never touched BF/Gaussian routing at all. A strong "
        "result here is `Graded_Funnel_BF_NLM` landing close to `NLM_D5_full_image`'s SSIM while escalating only "
        "a minority of pixels (`P_NLM` well under 1.0) -- that demonstrates the MD's central hypothesis "
        "(escalate expensive reasoning only where cheap reasoning is locally ambiguous) rather than just running "
        "the expensive method everywhere and calling it a router.\n"
    )
    (out_dir / "Graded_Funnel.md").write_text(text)


if __name__ == "__main__":
    result = run_controlled()
    print(pd.read_csv("outputs/20_graded_funnel/graded_funnel_aggregate.csv").to_string(index=False))
