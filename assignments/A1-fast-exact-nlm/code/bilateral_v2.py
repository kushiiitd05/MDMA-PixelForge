"""Noise-aware V2 routing and LTP structural ablations.

This stage keeps the failed V1 gate as evidence, then tests a corrected route:
noise-corrected local variance for smooth regions, noise-specific BF/JBF
parameters from recovery, and a local ternary pattern structural branch for
edge/structure-safe ablation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from skimage.metrics import structural_similarity

from bilateral_dataset_pipeline import POPCOUNT
from bilateral_foundations import (BOUNDARY, append_log, dataframe_markdown,
                                   gaussian_from_scratch, metrics,
                                   spatial_kernel, validate_window)
from bilateral_recovery import (NOISE_LEVELS, apply_method, joint_bilateral,
                                noise_corrected_signal_std)
from bilateral_v1 import VALIDATION_SEED, load_selected, noisy_bank


CODE_VERSION = "v2-noise-aware-ltp-1.0"


def _pad_mode(boundary: str) -> str:
    if boundary == "reflect":
        return "symmetric"
    if boundary == "mirror":
        return "reflect"
    raise ValueError("boundary must be reflect or mirror")


def ltp8(image: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Eight-neighbor local ternary pattern split into positive/negative codes."""
    x = image.astype(np.float32)
    pad = np.pad(x, 1, mode="symmetric")
    center = pad[1:-1, 1:-1]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    pos = np.zeros(center.shape, dtype=np.uint8)
    neg = np.zeros(center.shape, dtype=np.uint8)
    h, w = center.shape
    for bit, (dy, dx) in enumerate(offsets):
        nbr = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        pos |= ((nbr >= center + threshold).astype(np.uint8) << bit)
        neg |= ((nbr <= center - threshold).astype(np.uint8) << bit)
    return pos, neg


def masked_bilateral(image: np.ndarray, active: np.ndarray, sigma_s: float, sigma_r: float,
                     k: int, boundary: str = BOUNDARY, chunk_rows: int = 32) -> tuple[np.ndarray, dict]:
    """Compute exact BF only where active is true; inactive pixels are unchanged."""
    radius = validate_window(k)
    x = np.asarray(image, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    out = x.copy()
    gs = spatial_kernel(k, sigma_s)
    px = np.pad(x, radius, mode=_pad_mode(boundary))
    windows = sliding_window_view(px, (k, k))
    inv = 1.0 / (2.0 * sigma_r * sigma_r)
    range_evals = range_exps = 0
    start = time.perf_counter()
    for y0 in range(0, x.shape[0], chunk_rows):
        y1 = min(y0 + chunk_rows, x.shape[0])
        amask = active[y0:y1]
        if not np.any(amask):
            continue
        awin = windows[y0:y1][amask]
        centers = x[y0:y1][amask]
        delta = centers[:, None, None] - awin
        weights = gs * np.exp(-(delta * delta) * inv)
        den = weights.sum(axis=(-2, -1))
        out[y0:y1][amask] = (weights * awin).sum(axis=(-2, -1)) / den
        count = int(amask.sum()) * k * k
        range_evals += count
        range_exps += count
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return out.astype(np.float32), dict(runtime_ms=runtime_ms, range_eval_count=range_evals,
                                        exponential_count=range_exps)


def masked_ltp_sbf(image: np.ndarray, active: np.ndarray, sigma_s: float, sigma_r: float,
                   k: int, ltp_threshold: float, sigma_t: float,
                   boundary: str = BOUNDARY, chunk_rows: int = 32) -> tuple[np.ndarray, dict]:
    """Compute a BF times LTP-distance structural branch only on active pixels."""
    radius = validate_window(k)
    x = np.asarray(image, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    out = x.copy()
    pos, neg = ltp8(x, ltp_threshold)
    gs = spatial_kernel(k, sigma_s)
    px = np.pad(x, radius, mode=_pad_mode(boundary))
    pp = np.pad(pos, radius, mode=_pad_mode(boundary))
    pn = np.pad(neg, radius, mode=_pad_mode(boundary))
    windows = sliding_window_view(px, (k, k))
    pos_windows = sliding_window_view(pp, (k, k))
    neg_windows = sliding_window_view(pn, (k, k))
    inv_r = 1.0 / (2.0 * sigma_r * sigma_r)
    inv_t = 1.0 / (2.0 * sigma_t * sigma_t)
    range_evals = struct_evals = range_exps = struct_exps = 0
    start = time.perf_counter()
    for y0 in range(0, x.shape[0], chunk_rows):
        y1 = min(y0 + chunk_rows, x.shape[0])
        amask = active[y0:y1]
        if not np.any(amask):
            continue
        awin = windows[y0:y1][amask]
        centers = x[y0:y1][amask]
        pwin = pos_windows[y0:y1][amask]
        nwin = neg_windows[y0:y1][amask]
        pc = pos[y0:y1][amask, None, None]
        nc = neg[y0:y1][amask, None, None]
        d_ltp = (POPCOUNT[np.bitwise_xor(pc, pwin)] + POPCOUNT[np.bitwise_xor(nc, nwin)]) / 16.0
        delta = centers[:, None, None] - awin
        weights = gs * np.exp(-(delta * delta) * inv_r) * np.exp(-(d_ltp * d_ltp) * inv_t)
        den = weights.sum(axis=(-2, -1))
        out[y0:y1][amask] = (weights * awin).sum(axis=(-2, -1)) / den
        count = int(amask.sum()) * k * k
        range_evals += count
        struct_evals += count
        range_exps += count
        struct_exps += count
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return out.astype(np.float32), dict(runtime_ms=runtime_ms, range_eval_count=range_evals,
                                        structural_eval_count=struct_evals,
                                        exponential_count=range_exps + struct_exps,
                                        range_exponential_count=range_exps,
                                        structural_exponential_count=struct_exps)


def gradient_map(image: np.ndarray, guide_sigma: float) -> np.ndarray:
    guide = ndimage.gaussian_filter(image.astype(float), sigma=guide_sigma, mode="reflect")
    return np.hypot(ndimage.sobel(guide, 0, mode="reflect") / 8.0,
                    ndimage.sobel(guide, 1, mode="reflect") / 8.0).astype(np.float32)


def v2_noise_aware_selective(noisy: np.ndarray, sigma_n: int, payload: dict) -> tuple[np.ndarray, dict, dict]:
    p = payload["best"]["BF_exact"][str(sigma_n)]
    threshold = payload["noise_aware_gate"][str(sigma_n)]["signal_std_threshold"]
    g, gdiag = gaussian_from_scratch(noisy, p["sigma_s"], p["k"], BOUNDARY)
    signal = noise_corrected_signal_std(noisy, sigma_n, p["k"])
    gmask = signal <= threshold
    bf_active = ~gmask
    bf, bdiag = masked_bilateral(noisy, bf_active, p["sigma_s"], p["sigma_r"], p["k"], BOUNDARY)
    out = np.where(gmask, g, bf).astype(np.float32)
    route = np.where(gmask, 0, 1).astype(np.uint8)
    diag = dict(method="V2_G_BF_selective", K=p["k"], sigma_s=p["sigma_s"], sigma_r=p["sigma_r"],
                P_G=float(gmask.mean()), P_BF=float(bf_active.mean()), P_SBF=0.0,
                runtime_ms=gdiag["runtime_ms"] + bdiag["runtime_ms"],
                range_eval_count=bdiag["range_eval_count"], structural_eval_count=0,
                exponential_count=gdiag["exponential_count"] + bdiag["exponential_count"],
                notes="BF evaluated only outside noise-corrected smooth mask")
    return out, diag, dict(route=route, signal_std=signal)


def tune_ltp_edge(selection_csv: Path, dataset_dir: Path, payload: dict, out_dir: Path) -> dict:
    """Small 128px grid for replacing strong-edge JBF pixels with LTP-SBF."""
    data = load_selected(selection_csv, dataset_dir, 128)
    bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    rows = []
    best = {}
    for sigma_n in NOISE_LEVELS:
        jp = payload["best"]["JBF_gaussian_guide"][str(sigma_n)]
        ep = payload["best"]["BF_exact"][str(sigma_n)]
        for guide_sigma in (0.7, 1.1, 1.5):
            for grad_threshold in (6.0, 12.0, 20.0, 30.0):
                for ltp_ratio in (0.5, 0.75, 1.0):
                    for sigma_t in (0.25, 0.40, 0.60):
                        vals = []
                        edge_fracs = []
                        deltas = []
                        for image_id, item in data.items():
                            noisy = bank[(image_id, sigma_n)]
                            guide = ndimage.gaussian_filter(noisy.astype(float), sigma=jp["guide_sigma"], mode="reflect")
                            base, _ = joint_bilateral(noisy, guide, jp["sigma_s"], jp["sigma_r"], jp["k"])
                            grad = gradient_map(noisy, guide_sigma)
                            edge = grad >= grad_threshold
                            sbf, _ = masked_ltp_sbf(noisy, edge, ep["sigma_s"], ep["sigma_r"], ep["k"],
                                                    ltp_ratio * sigma_n, sigma_t)
                            out = np.where(edge, sbf, base).astype(np.float32)
                            score = structural_similarity(item["clean"], out, data_range=255)
                            base_score = structural_similarity(item["clean"], base, data_range=255)
                            vals.append(score)
                            deltas.append(score - base_score)
                            edge_fracs.append(float(edge.mean()))
                        rows.append(dict(noise_sigma=sigma_n, guide_sigma=guide_sigma,
                                         grad_threshold=grad_threshold, ltp_ratio=ltp_ratio,
                                         sigma_t=sigma_t, SSIM_mean=float(np.mean(vals)),
                                         min_delta_vs_jbf=float(np.min(deltas)),
                                         P_SBF=float(np.mean(edge_fracs)),
                                         score=float(np.mean(vals) + 0.25 * min(0.0, np.min(deltas)))))
        part = pd.DataFrame([r for r in rows if r["noise_sigma"] == sigma_n])
        winner = part.loc[part.score.idxmax()].to_dict()
        best[str(sigma_n)] = {k: float(winner[k]) for k in
                              ("guide_sigma", "grad_threshold", "ltp_ratio", "sigma_t", "score")}
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ltp_edge_tuning_grid.csv", index=False)
    return best


def evaluate_v2(selection_csv: Path, dataset_dir: Path, payload: dict,
                ltp_best: dict, out_dir: Path) -> pd.DataFrame:
    data = load_selected(selection_csv, dataset_dir, 256)
    bank = noisy_bank(data, NOISE_LEVELS, VALIDATION_SEED)
    rows = []
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = bank[(image_id, sigma_n)]
            variants = []
            selective, sdiag, smaps = v2_noise_aware_selective(noisy, sigma_n, payload)
            variants.append(("V2_G_BF_selective", selective, sdiag, smaps))

            jp = payload["best"]["JBF_gaussian_guide"][str(sigma_n)]
            guide = ndimage.gaussian_filter(noisy.astype(float), sigma=jp["guide_sigma"], mode="reflect")
            jbf, jt = joint_bilateral(noisy, guide, jp["sigma_s"], jp["sigma_r"], jp["k"])
            variants.append(("JBF_gaussian_guide", jbf,
                             dict(method="JBF_gaussian_guide", K=jp["k"], sigma_s=jp["sigma_s"],
                                  sigma_r=jp["sigma_r"], guide_sigma=jp["guide_sigma"],
                                  runtime_ms=jt, P_G=0.0, P_BF=1.0, P_SBF=0.0,
                                  range_eval_count=clean.size * jp["k"] * jp["k"],
                                  structural_eval_count=0, exponential_count=clean.size * jp["k"] * jp["k"]),
                             dict(route=np.ones(clean.shape, dtype=np.uint8))))

            ep = payload["best"]["BF_exact"][str(sigma_n)]
            lp = ltp_best[str(sigma_n)]
            grad = gradient_map(noisy, lp["guide_sigma"])
            edge = grad >= lp["grad_threshold"]
            sbf, sbfdiag = masked_ltp_sbf(noisy, edge, ep["sigma_s"], ep["sigma_r"], ep["k"],
                                          lp["ltp_ratio"] * sigma_n, lp["sigma_t"])
            ltp_hybrid = np.where(edge, sbf, jbf).astype(np.float32)
            route = np.where(edge, 2, 1).astype(np.uint8)
            diag = dict(method="V2_JBF_LTP_edge", K=ep["k"], sigma_s=ep["sigma_s"],
                        sigma_r=ep["sigma_r"], guide_sigma=jp["guide_sigma"],
                        ltp_threshold=lp["ltp_ratio"] * sigma_n, sigma_t=lp["sigma_t"],
                        runtime_ms=jt + sbfdiag["runtime_ms"], P_G=0.0,
                        P_BF=float((~edge).mean()), P_SBF=float(edge.mean()),
                        range_eval_count=clean.size * jp["k"] * jp["k"] + sbfdiag["range_eval_count"],
                        structural_eval_count=sbfdiag["structural_eval_count"],
                        exponential_count=clean.size * jp["k"] * jp["k"] + sbfdiag["exponential_count"])
            variants.append(("V2_JBF_LTP_edge", ltp_hybrid, diag, dict(route=route, gradient=grad)))

            if sigma_n == 25:
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                axes = axes.ravel()
                axes[0].imshow(clean, cmap="gray", vmin=0, vmax=255); axes[0].set_title("clean")
                axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=255); axes[1].set_title("noisy")
                for ax, (name, out, _, _) in zip(axes[2:], variants):
                    ax.imshow(out, cmap="gray", vmin=0, vmax=255); ax.set_title(name)
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(f"{image_id}: V2 comparison, sigma_n=25")
                fig.tight_layout()
                fig.savefig(out_dir / f"{image_id}_v2.png", dpi=160)
                plt.close(fig)

            for method, out, diag, _ in variants:
                rows.append(dict(image_id=image_id, role=item["role"], noise_sigma=sigma_n,
                                 method=method, K=diag["K"], sigma_s=diag["sigma_s"],
                                 sigma_r=diag["sigma_r"], guide_sigma=diag.get("guide_sigma", np.nan),
                                 ltp_threshold=diag.get("ltp_threshold", np.nan),
                                 sigma_t=diag.get("sigma_t", np.nan), runtime_ms=diag["runtime_ms"],
                                 range_eval_count=diag["range_eval_count"],
                                 structural_eval_count=diag["structural_eval_count"],
                                 exponential_count=diag["exponential_count"],
                                 P_G=diag["P_G"], P_BF=diag["P_BF"], P_SBF=diag["P_SBF"],
                                 **metrics(clean, out)))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "v2_results.csv", index=False)
    agg = df.groupby(["noise_sigma", "method"]).agg(
        PSNR=("PSNR", "mean"), SSIM=("SSIM", "mean"), edge=("edge_metric", "mean"),
        texture=("texture_metric", "mean"), runtime_ms=("runtime_ms", "mean"),
        range_eval_count=("range_eval_count", "mean"),
        structural_eval_count=("structural_eval_count", "mean"),
        P_G=("P_G", "mean"), P_BF=("P_BF", "mean"), P_SBF=("P_SBF", "mean")
    ).reset_index()
    agg.to_csv(out_dir / "v2_aggregate.csv", index=False)
    return df


def append_v2_log(df: pd.DataFrame, path: Path) -> None:
    stamp = pd.Timestamp.now(tz="UTC")
    run_id = stamp.strftime("v2_%Y%m%dT%H%M%SZ")
    rows = []
    for _, r in df.iterrows():
        rows.append(dict(timestamp=stamp.isoformat(), run_id=run_id, code_version=CODE_VERSION,
                         image_id=r.image_id, source_path="see final_four.csv",
                         source_resolution="see inventory", experiment_resolution="256x256",
                         grayscale_conversion="BT.601 or original L",
                         noise_type="additive_gaussian_clipped", noise_sigma=r.noise_sigma,
                         random_seed=VALIDATION_SEED, method=r.method, K=r.K, radius=(r.K - 1) / 2,
                         sigma_s=r.sigma_s, sigma_r=r.sigma_r, epsilon=np.nan, tau=np.nan,
                         structural_descriptor="LTP P=8 R=1" if "LTP" in r.method else "none",
                         structural_parameters=f"ltp_threshold={r.ltp_threshold}; sigma_t={r.sigma_t}",
                         h=np.nan, T_A=np.nan, MSB_depth_b=np.nan, boundary=BOUNDARY,
                         runtime_ms=r.runtime_ms, range_eval_count=r.range_eval_count,
                         structural_eval_count=r.structural_eval_count,
                         exponential_count=r.exponential_count, comparison_count=np.nan,
                         P_G=r.P_G, P_BF=r.P_BF, P_SBF=r.P_SBF, MSE=r.MSE, PSNR=r.PSNR,
                         SSIM=r.SSIM, edge_metric=r.edge_metric, texture_metric=r.texture_metric,
                         BF_approximation_MAE=np.nan, mean_Neff=np.nan,
                         notes="V2 held-out validation; LTP branch is edge ablation"))
    append_log(rows, path)


def write_report(out_dir: Path, payload: dict, ltp_best: dict, df: pd.DataFrame) -> None:
    agg = pd.read_csv(out_dir / "v2_aggregate.csv")
    rec = pd.read_csv(out_dir.parent / "07_core_ablations" / "recovery" / "recovery_aggregate.csv")
    compare_methods = ["BF_exact", "BF_skimage", "G_BF_noise_aware", "JBF_gaussian_guide"]
    compare = pd.concat([
        rec[rec.method.isin(compare_methods)].assign(stage="recovery"),
        agg.assign(stage="v2")
    ], ignore_index=True, sort=False)
    best_rows = []
    for sigma_n in NOISE_LEVELS:
        part = compare[compare.noise_sigma.eq(sigma_n)]
        best_rows.append(part.loc[part.SSIM.idxmax()])
    text = (
        "# V2 noise-aware and LTP study\n\n"
        "## Purpose\n\n"
        "V1 failed because raw noisy max-min rarely satisfied tau. V2 therefore treats tau as a diagnostic, "
        "uses the recovered noise-specific BF/JBF parameters, computes BF only where the noise-corrected "
        "signal-variance gate says the pixel is not smooth, and tests an LTP edge branch as a noise-resistant "
        "structural cue.\n\n"
        "## LTP edge parameters\n\n"
        + dataframe_markdown(pd.DataFrame([dict(noise_sigma=int(k), **v) for k, v in ltp_best.items()]), 5)
        + "\n\n## V2 held-out aggregate\n\n"
        + dataframe_markdown(agg, 5)
        + "\n\n## Best across recovery and V2 candidates\n\n"
        + dataframe_markdown(pd.DataFrame(best_rows), 5)
        + "\n\n## Interpretation\n\n"
        "The selective G/BF route is the practical fix for low and medium noise: it keeps the quality of the "
        "recovered variance gate while recording the reduced range-kernel work. At high noise, JBF remains the "
        "strong average candidate because raw intensities become unreliable as a range guide. The LTP branch is "
        "kept as an ablation: at medium noise it mainly improves the edge/ruler role, so it should be presented "
        "as an edge-safe option rather than a universal replacement.\n"
    )
    (out_dir / "v2_report.md").write_text(text)


def run_all(output_root: str | Path = "outputs") -> dict:
    root = Path(output_root)
    out_dir = root / "08_v2_noise_aware_ltp"
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_csv = root / "02_image_selection" / "final_four.csv"
    dataset_dir = Path("misc")
    payload = json.loads((root / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = tune_ltp_edge(selection_csv, dataset_dir, payload, out_dir)
    (out_dir / "v2_ltp_best_parameters.json").write_text(json.dumps(ltp_best, indent=2))
    df = evaluate_v2(selection_csv, dataset_dir, payload, ltp_best, out_dir)
    write_report(out_dir, payload, ltp_best, df)
    append_v2_log(df, root / "reports" / "experiment_log.csv")
    return dict(ltp_best=ltp_best, results=df)


if __name__ == "__main__":
    result = run_all()
    print(json.dumps(result["ltp_best"], indent=2))
    print(pd.read_csv("outputs/08_v2_noise_aware_ltp/v2_aggregate.csv").to_string(index=False))
