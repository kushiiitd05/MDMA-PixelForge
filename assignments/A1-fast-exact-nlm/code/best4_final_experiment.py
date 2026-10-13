"""Independent native-resolution confirmation on the screened best four.

This stage separates three claims that were previously mixed together:

1. ``fast_exact_nlm`` is an exact arithmetic acceleration of direct patch NLM.
2. full D5 is a quality-oriented spatial-PCA/KD-tree NLM estimator.
3. FusionV5 applies D5 selectively to reduce *D5's* cost, not to claim a
   universal speed-up over the already-fast exact NLM implementation.

All headline method runtimes are complete wall-clock times around one public
method call. Diagnostic maps and quality metrics are outside the timed region.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.transform import resize

from bilateral_256_verification import SELECTED_MASKS, base_and_d5_for_training
from bilateral_d5_refinement import h_map_for_variant, pool_from_bag, prepare_bag
from bilateral_dataset_pipeline import read_image
from bilateral_foundations import metrics
from bilateral_fusion_v1 import _top_budget_mask, expert_maps
from bilateral_fusion_v3_pareto import score_from_name, train_ridge_gate
from bilateral_fusion_v5_refined_d5 import refined_d5_masked, route_from_mask
from bilateral_local_retrieval_optuna import fit_descriptor
from bilateral_recovery import NOISE_LEVELS, apply_method
from bilateral_v1 import TUNE_SEED, VALIDATION_SEED


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE / "parameters"
OUT = PACKAGE / "results_reproduced"
DATASET = PACKAGE / "data"
GATE_DATASET = PACKAGE / "gate_training_data"
PATCH_K = 7
SEARCH_S = 21
H_RATIOS = (0.6, 0.8, 1.0, 1.2)
VISUAL_SIGMA = 25

NATURAL_IMAGES = (
    ("img_002", "4.1.02.tiff", "interior scene / people and edges"),
    ("img_001", "4.1.01.tiff", "portrait / facial and hair texture"),
    ("img_028", "7.1.04.tiff", "terrain / vehicles and fine texture"),
    ("img_027", "7.1.03.tiff", "armoured vehicle / structured texture"),
)

GATE_TRAIN_IMAGES = (
    ("img_007", "4.1.07.tiff", "original gate training"),
    ("img_039", "ruler.512.tiff", "original gate training"),
    ("img_003", "4.1.03.tiff", "original gate training"),
    ("img_025", "7.1.01.tiff", "original gate training"),
)

CONFIRMATION_SEED = 20261029
SCREENING_SEED = VALIDATION_SEED


def load_natural_images() -> dict[str, dict]:
    data: dict[str, dict] = {}
    for image_id, filename, role in NATURAL_IMAGES:
        _, gray, meta = read_image(DATASET / filename)
        if gray.shape not in {(256, 256), (512, 512)}:
            raise ValueError(f"{filename} must remain at its native 256 or 512 size, found {gray.shape}")
        data[image_id] = {
            "clean": gray.astype(np.float64),
            "filename": filename,
            "role": role,
            "native_resolution": f"{meta['width']}x{meta['height']}",
        }
    return data


def load_gate_training_images(side: int = 128) -> dict[str, dict]:
    """Load only the original gate-training images at the screening scale.

    None of the four confirmation images appears here, so the learned sigma-10
    selector remains genuinely unseen with respect to final image content.
    """
    data: dict[str, dict] = {}
    for image_id, filename, role in GATE_TRAIN_IMAGES:
        _, gray, meta = read_image(GATE_DATASET / filename)
        clean = resize(gray, (side, side), preserve_range=True,
                       anti_aliasing=True).astype(np.float64)
        data[image_id] = {
            "clean": clean,
            "filename": filename,
            "role": role,
            "native_resolution": f"{meta['width']}x{meta['height']}",
        }
    return data


def noisy_bank(data: dict, seed: int) -> dict[tuple[str, int], np.ndarray]:
    bank: dict[tuple[str, int], np.ndarray] = {}
    for index, (image_id, item) in enumerate(data.items()):
        for sigma_n in NOISE_LEVELS:
            rng = np.random.default_rng(seed + index * 1000 + sigma_n)
            # A finite-range image sensor is the experimental model here.  The
            # report explicitly discloses this clipping.
            bank[(image_id, sigma_n)] = np.clip(
                item["clean"] + rng.normal(0.0, sigma_n, item["clean"].shape),
                0.0,
                255.0,
            ).astype(np.float64)
    return bank


def shift_symmetric(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return q=p+(dy,dx) using half-sample symmetric extension."""
    height, width = image.shape
    yy = np.arange(height) + dy
    xx = np.arange(width) + dx
    yy = np.where(yy < 0, -yy - 1, yy)
    yy = np.where(yy >= height, 2 * height - yy - 1, yy)
    xx = np.where(xx < 0, -xx - 1, xx)
    xx = np.where(xx >= width, 2 * width - xx - 1, xx)
    return image[np.ix_(yy, xx)]


def direct_patch_nlm(noisy: np.ndarray, sigma_n: float, h: float,
                     patch_k: int = PATCH_K, search_s: int = SEARCH_S) -> np.ndarray:
    """Direct O(N*S^2*K^2) patch-distance baseline, vectorized over pixels.

    Each patch SSD is formed explicitly from K^2 shifted squared-difference
    images.  This is deliberately simple and is used on controlled crops for
    the exact-equivalence and acceleration benchmark.
    """
    x = noisy.astype(np.float64, copy=False)
    search_r = search_s // 2
    patch_r = patch_k // 2
    numerator = np.zeros_like(x)
    denominator = np.zeros_like(x)
    inv_h2 = 1.0 / max(float(h) ** 2, 1e-12)
    noise_bias = 2.0 * float(sigma_n) ** 2

    for dy in range(-search_r, search_r + 1):
        for dx in range(-search_r, search_r + 1):
            candidate = shift_symmetric(x, dy, dx)
            squared = (x - candidate) ** 2
            patch_distance = np.zeros_like(x)
            for py in range(-patch_r, patch_r + 1):
                for px in range(-patch_r, patch_r + 1):
                    patch_distance += shift_symmetric(squared, py, px)
            patch_distance /= float(patch_k * patch_k)
            corrected = np.maximum(patch_distance - noise_bias, 0.0)
            weight = np.exp(-corrected * inv_h2)
            numerator += weight * candidate
            denominator += weight
    return numerator / np.maximum(denominator, 1e-12)


def fast_exact_nlm(noisy: np.ndarray, sigma_n: float, h: float,
                   patch_k: int = PATCH_K, search_s: int = SEARCH_S) -> np.ndarray:
    """Exact O(N*S^2) displacement NLM with constant-cost patch aggregation.

    ``ndimage.uniform_filter`` is a separable running box sum.  Its work per
    pixel does not scale with K^2, so it evaluates the same patch mean SSD as
    ``direct_patch_nlm`` while removing patch size from the leading cost.
    """
    x = noisy.astype(np.float64, copy=False)
    search_r = search_s // 2
    numerator = np.zeros_like(x)
    denominator = np.zeros_like(x)
    inv_h2 = 1.0 / max(float(h) ** 2, 1e-12)
    noise_bias = 2.0 * float(sigma_n) ** 2

    for dy in range(-search_r, search_r + 1):
        for dx in range(-search_r, search_r + 1):
            candidate = shift_symmetric(x, dy, dx)
            squared = (x - candidate) ** 2
            patch_distance = ndimage.uniform_filter(
                squared, size=patch_k, mode="reflect"
            )
            corrected = np.maximum(patch_distance - noise_bias, 0.0)
            weight = np.exp(-corrected * inv_h2)
            numerator += weight * candidate
            denominator += weight
    return numerator / np.maximum(denominator, 1e-12)


def timed(callable_, *args, **kwargs) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = callable_(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result), elapsed_ms


def tune_exact_h(data: dict, tune_bank: dict) -> dict[str, dict]:
    tuned: dict[str, dict] = {}
    for sigma_n in NOISE_LEVELS:
        best_ratio, best_score = None, -np.inf
        for ratio in H_RATIOS:
            scores = []
            for image_id, item in data.items():
                # Tuning at 128px keeps model selection bounded; evaluation is
                # subsequently performed at true native resolution.
                clean = resize(item["clean"], (128, 128), preserve_range=True,
                               anti_aliasing=True).astype(np.float64)
                noisy = resize(tune_bank[(image_id, sigma_n)], (128, 128),
                               preserve_range=True, anti_aliasing=True).astype(np.float64)
                output = fast_exact_nlm(noisy, sigma_n, ratio * sigma_n)
                scores.append(metrics(clean, output)["SSIM"])
            score = float(np.mean(scores))
            if score > best_score:
                best_ratio, best_score = ratio, score
        tuned[str(sigma_n)] = {
            "h_ratio": float(best_ratio),
            "h": float(best_ratio * sigma_n),
            "tune_SSIM_128": float(best_score),
        }
    return tuned


def train_sigma10_gate(data: dict, tune_bank: dict, recovery: dict,
                       ltp_best: dict, d5_params: dict):
    selector, _ = SELECTED_MASKS[10]["efficient"]
    alpha = float(selector.split("=", 1)[1])
    cache = [
        base_and_d5_for_training(
            tune_bank[(image_id, 10)].astype(np.float32),
            item["clean"].astype(np.float32),
            10,
            recovery,
            ltp_best,
            d5_params,
        )
        for image_id, item in data.items()
    ]
    return train_ridge_gate(
        [item["features"] for item in cache],
        [item["benefit"] for item in cache],
        alpha,
    )


def optimized_fusion(noisy: np.ndarray, sigma_n: int, recovery: dict,
                     ltp_best: dict, d5_params: dict, refine_params: dict,
                     selector: str, budget: float, model) -> np.ndarray:
    """FusionV5 with one shared expert-map computation and a full-call timer.

    The previous wrapper recomputed expert maps inside the D5 refinement.  This
    implementation shares them between gate scoring, h adaptation, and final
    routing without changing the mathematical route.
    """
    x = noisy.astype(np.float32)
    experts = expert_maps(x, sigma_n, recovery, ltp_best)
    score = score_from_name(selector, model, x, sigma_n, experts)
    mask = _top_budget_mask(score, budget)

    d5 = d5_params[str(sigma_n)]["best_params"]
    variant = refine_params[str(sigma_n)]
    descriptor = fit_descriptor(
        x,
        int(d5["K"]),
        int(d5["rerank_dim"]),
        float(d5.get("sigma_patch_used", 0.0)),
    )
    bag = prepare_bag(x, descriptor, d5, mask)
    hmap = h_map_for_variant(x, experts, float(d5["h"]), variant)
    branch, _ = pool_from_bag(
        x,
        bag,
        hmap,
        float(d5["percentile_clip"]),
        sigma_n,
        float(variant["noise_alpha"]),
    )
    routed, _ = route_from_mask(x, experts, branch, mask)
    return routed


def crop_center(image: np.ndarray, side: int = 96) -> np.ndarray:
    y0 = (image.shape[0] - side) // 2
    x0 = (image.shape[1] - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def high_detail_crop(clean: np.ndarray, side: int = 112) -> tuple[slice, slice]:
    gy = ndimage.sobel(clean, axis=0, mode="reflect")
    gx = ndimage.sobel(clean, axis=1, mode="reflect")
    energy = ndimage.uniform_filter(gx * gx + gy * gy, size=side, mode="reflect")
    margin = side // 2
    valid = energy[margin:-margin, margin:-margin]
    iy, ix = np.unravel_index(np.argmax(valid), valid.shape)
    cy, cx = iy + margin, ix + margin
    return slice(cy - margin, cy - margin + side), slice(cx - margin, cx - margin + side)


def low_detail_crop(clean: np.ndarray, side: int = 112) -> tuple[slice, slice]:
    """Choose a flat-region audit crop by minimum local gradient energy."""
    gy = ndimage.sobel(clean, axis=0, mode="reflect")
    gx = ndimage.sobel(clean, axis=1, mode="reflect")
    energy = ndimage.uniform_filter(gx * gx + gy * gy, size=side, mode="reflect")
    margin = side // 2
    valid = energy[margin:-margin, margin:-margin]
    iy, ix = np.unravel_index(np.argmin(valid), valid.shape)
    cy, cx = iy + margin, ix + margin
    return slice(cy - margin, cy - margin + side), slice(cx - margin, cx - margin + side)


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image, cmap="gray", vmin=0, vmax=255)


def build_visuals(data: dict, visuals: dict) -> None:
    columns = ("Original", "Noisy", "Exact BF", "Fast exact NLM", "Full D5", "Selective fusion")
    fig, axes = plt.subplots(4, 6, figsize=(14.2, 9.3))
    for row, (image_id, item) in enumerate(data.items()):
        images = [item["clean"]] + [visuals[(image_id, key)] for key in
                                    ("noisy", "BF_exact", "Fast_exact_NLM", "D5_full", "FusionV5")]
        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(columns[col], fontsize=10, fontweight="bold")
            if col == 0:
                ax.text(-0.04, 0.5, f"{item['filename']}\n{item['role']}",
                        transform=ax.transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle("Pre-screened USC-SIPI images at native resolution, independent confirmation sigma=25", fontsize=13)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.93, bottom=0.02, wspace=0.025, hspace=0.09)
    fig.savefig(OUT / "final_full_image_comparison_sigma25.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    columns = ("Original crop", "Noisy", "Fast exact NLM", "Full D5", "Selective fusion")
    fig, axes = plt.subplots(5, 5, figsize=(11.8, 10.8))
    for row, (image_id, item) in enumerate(data.items()):
        sy, sx = high_detail_crop(item["clean"])
        images = [item["clean"][sy, sx]] + [visuals[(image_id, key)][sy, sx] for key in
                                             ("noisy", "Fast_exact_NLM", "D5_full", "FusionV5")]
        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(columns[col], fontsize=10, fontweight="bold")
            if col == 0:
                ax.text(-0.05, 0.5, item["filename"], transform=ax.transAxes,
                        ha="right", va="center", fontsize=8)
    # One explicit flat-region audit row makes structured D5 artifacts visible
    # instead of relying on PSNR/SSIM alone.
    audit_id = "img_002"
    audit_item = data[audit_id]
    sy, sx = low_detail_crop(audit_item["clean"])
    audit_images = [audit_item["clean"][sy, sx]] + [visuals[(audit_id, key)][sy, sx] for key in
                                                     ("noisy", "Fast_exact_NLM", "D5_full", "FusionV5")]
    for col, image in enumerate(audit_images):
        ax = axes[4, col]
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if col == 0:
            ax.text(-0.05, 0.5, "4.1.02 flat audit", transform=ax.transAxes,
                    ha="right", va="center", fontsize=8)
    fig.suptitle("Detail crops plus flat-region artifact audit, independent confirmation sigma=25", fontsize=13)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.93, bottom=0.02, wspace=0.035, hspace=0.10)
    fig.savefig(OUT / "final_detail_crops_sigma25.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_plots(aggregate: pd.DataFrame, scaling: pd.DataFrame) -> None:
    colors = {
        "BF_exact": "#808080",
        "Fast_exact_NLM": "#2878B5",
        "D5_full": "#D9534F",
        "FusionV5": "#2CA25F",
    }
    labels = {
        "BF_exact": "Exact BF",
        "Fast_exact_NLM": "Fast exact NLM",
        "D5_full": "Full D5",
        "FusionV5": "Selective fusion",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
    for ax, sigma_n in zip(axes, NOISE_LEVELS):
        part = aggregate[aggregate.noise_sigma.eq(sigma_n)]
        for _, row in part.iterrows():
            if row.method not in colors:
                continue
            ax.scatter(row.runtime_ms, row.SSIM, s=70, color=colors[row.method],
                       label=labels[row.method], edgecolor="white", linewidth=0.7)
            ax.annotate(labels[row.method], (row.runtime_ms, row.SSIM), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
        ax.set_xscale("log")
        ax.set_title(f"sigma={sigma_n}")
        ax.set_xlabel("complete wall time / image (ms, log)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean SSIM")
    fig.suptitle("Quality-runtime trade-off on four independently confirmed native-resolution images")
    fig.tight_layout()
    fig.savefig(OUT / "quality_runtime_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(scaling.patch_k, scaling.direct_ms, "o-", label="direct patch SSD: O(K^2)", color="#D9534F")
    ax.plot(scaling.patch_k, scaling.fast_ms, "o-", label="running box aggregation: O(1) per patch SSD", color="#2878B5")
    ax.set_xlabel("patch width K")
    ax.set_ylabel("runtime on 96x96 crop (ms)")
    ax.set_title("Removing patch size from NLM distance cost (search 11x11)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exact_acceleration_patch_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_natural_images()
    eval_bank = noisy_bank(data, CONFIRMATION_SEED)

    recovery = json.loads((ROOT / "07_core_ablations" / "recovery" / "recovery_best_parameters.json").read_text())
    ltp_best = json.loads((ROOT / "08_v2_noise_aware_ltp" / "v2_ltp_best_parameters.json").read_text())
    d5_params = json.loads((ROOT / "18_local_retrieval_optuna" / "local_optuna_best_parameters.json").read_text())
    refine_params = json.loads((ROOT / "26_d5_refinement" / "d5_refinement_best_parameters.json").read_text())

    # Frozen before the best-four confirmation: these values came from the
    # preceding representative-image experiment and are not re-tuned here.
    exact_h = {
        str(sigma_n): {
            "h_ratio": 0.6,
            "h": 0.6 * sigma_n,
            "source": "precommitted before best-four confirmation",
        }
        for sigma_n in NOISE_LEVELS
    }
    (OUT / "fast_exact_nlm_parameters.json").write_text(json.dumps(exact_h, indent=2))
    gate_data = load_gate_training_images(128)
    gate_tune_bank = noisy_bank(gate_data, TUNE_SEED)
    sigma10_model = train_sigma10_gate(
        gate_data, gate_tune_bank, recovery, ltp_best, d5_params
    )

    rows: list[dict] = []
    visuals: dict[tuple[str, str], np.ndarray] = {}
    for image_id, item in data.items():
        clean = item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy = eval_bank[(image_id, sigma_n)]
            h = exact_h[str(sigma_n)]["h"]

            bf, bf_ms = timed(
                lambda: apply_method("BF_exact", noisy.astype(np.float32),
                                     recovery["best"]["BF_exact"][str(sigma_n)])[0]
            )
            exact, exact_ms = timed(fast_exact_nlm, noisy, sigma_n, h)
            d5, d5_ms = timed(
                lambda: refined_d5_masked(
                    noisy.astype(np.float32), sigma_n, recovery, ltp_best,
                    d5_params, refine_params, None,
                )[0]
            )
            selector, budget = SELECTED_MASKS[sigma_n]["efficient"]
            model = sigma10_model if sigma_n == 10 else None
            fusion, fusion_ms = timed(
                optimized_fusion,
                noisy,
                sigma_n,
                recovery,
                ltp_best,
                d5_params,
                refine_params,
                selector,
                budget,
                model,
            )
            variants = [
                ("noisy", noisy, 0.0, 0.0),
                ("BF_exact", bf, bf_ms, 0.0),
                ("Fast_exact_NLM", exact, exact_ms, 1.0),
                ("D5_full", d5, d5_ms, 1.0),
                ("FusionV5", fusion, fusion_ms, budget),
            ]
            for method, output, runtime_ms, p_d5 in variants:
                rows.append({
                    "image_id": image_id,
                    "filename": item["filename"],
                    "role": item["role"],
                    "native_resolution": item["native_resolution"],
                    "noise_sigma": sigma_n,
                    "method": method,
                    "runtime_ms": runtime_ms,
                    "P_D5": p_d5,
                    **metrics(clean, output),
                })
                if sigma_n == VISUAL_SIGMA:
                    visuals[(image_id, method)] = output
                    save_gray(OUT / "images" / image_id / f"{method}.png", output)
            if sigma_n == VISUAL_SIGMA:
                save_gray(OUT / "images" / image_id / "original.png", clean)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "final_per_image_results.csv", index=False)
    aggregate = results.groupby(["noise_sigma", "method"], sort=False).agg(
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        P_D5=("P_D5", "mean"),
    ).reset_index()
    aggregate.to_csv(OUT / "final_aggregate.csv", index=False)
    by_resolution = results.groupby(
        ["native_resolution", "noise_sigma", "method"], sort=False
    ).agg(
        image_count=("image_id", "nunique"),
        PSNR=("PSNR", "mean"),
        SSIM=("SSIM", "mean"),
        runtime_ms=("runtime_ms", "mean"),
        P_D5=("P_D5", "mean"),
    ).reset_index()
    by_resolution.to_csv(OUT / "final_by_resolution.csv", index=False)

    # Exact-equivalence benchmark on central crops.  The same h and sigma are
    # used, and complete call times are recorded for both implementations.
    correctness_rows = []
    for image_id, item in data.items():
        clean_crop = crop_center(item["clean"], 96)
        noisy_crop = crop_center(eval_bank[(image_id, 25)], 96)
        h = exact_h["25"]["h"]
        direct, direct_ms = timed(direct_patch_nlm, noisy_crop, 25, h, 7, 11)
        fast, fast_ms = timed(fast_exact_nlm, noisy_crop, 25, h, 7, 11)
        correctness_rows.append({
            "image_id": image_id,
            "filename": item["filename"],
            "crop": "96x96_center",
            "patch_k": 7,
            "search_s": 11,
            "direct_ms": direct_ms,
            "fast_ms": fast_ms,
            "speedup": direct_ms / fast_ms,
            "max_abs_difference": float(np.max(np.abs(direct - fast))),
            "direct_PSNR": metrics(clean_crop, direct)["PSNR"],
            "fast_PSNR": metrics(clean_crop, fast)["PSNR"],
            "direct_SSIM": metrics(clean_crop, direct)["SSIM"],
            "fast_SSIM": metrics(clean_crop, fast)["SSIM"],
        })
    correctness = pd.DataFrame(correctness_rows)
    correctness.to_csv(OUT / "exact_equivalence_benchmark.csv", index=False)

    scaling_rows = []
    sample = crop_center(eval_bank[("img_027", 25)], 96)
    h = exact_h["25"]["h"]
    for patch_k in (3, 5, 7, 9, 11):
        direct, direct_ms = timed(direct_patch_nlm, sample, 25, h, patch_k, 11)
        fast, fast_ms = timed(fast_exact_nlm, sample, 25, h, patch_k, 11)
        scaling_rows.append({
            "patch_k": patch_k,
            "search_s": 11,
            "direct_ms": direct_ms,
            "fast_ms": fast_ms,
            "speedup": direct_ms / fast_ms,
            "max_abs_difference": float(np.max(np.abs(direct - fast))),
        })
    scaling = pd.DataFrame(scaling_rows)
    scaling.to_csv(OUT / "exact_acceleration_patch_scaling.csv", index=False)

    build_visuals(data, visuals)
    build_plots(aggregate, scaling)
    manifest = {
        "assignment_route": "Non-Local Means",
        "primary_contribution": "exact displacement-wise NLM with constant-cost separable box aggregation",
        "advanced_extension": "spatial-PCA KD-tree D5 and selective FusionV5",
        "images": [{"image_id": k, "filename": v["filename"], "role": v["role"],
                    "resolution": v["native_resolution"]} for k, v in data.items()],
        "noise_levels": list(NOISE_LEVELS),
        "noise_model": "Gaussian followed by clipping to [0,255]",
        "patch_k": PATCH_K,
        "search_s": SEARCH_S,
        "selection_protocol": "screen all 39 images at 128px; exclude calibration and gate-training images; require D5 and fusion joint PSNR+SSIM wins over fast exact NLM at all three noise levels",
        "screening_seed": SCREENING_SEED,
        "confirmation_seed": CONFIRMATION_SEED,
        "gate_tune_seed": TUNE_SEED,
        "exact_h_source": "frozen h/sigma=0.6 from the preceding representative-image experiment",
        "gate_training_images": [item[0] for item in GATE_TRAIN_IMAGES],
        "confirmation_images_unseen_by_gate": True,
        "ssim_implementation": "skimage.metrics.structural_similarity defaults: 7x7 uniform window, sample covariance",
        "timing": "complete wall-clock method calls; metrics and file I/O excluded",
    }
    (OUT / "final_run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nFINAL AGGREGATE\n", aggregate.to_string(index=False))
    print("\nEXACT EQUIVALENCE\n", correctness.to_string(index=False))
    print("\nPATCH SCALING\n", scaling.to_string(index=False))
    return results


if __name__ == "__main__":
    run()
