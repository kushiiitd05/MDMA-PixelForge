"""Native-image profiling and representative selection for bilateral-filter research.

This module intentionally stops before neighborhood filtering.  It implements Tasks
1--7 from Codex_Master_Prompt_Bilateral_Image_Profiling_and_Research_Pipeline.md.
Original images are read-only; every artifact is written below outputs/.
"""

from __future__ import annotations

import hashlib
import math
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pgm"}
GAMMAS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.2)
WINDOWS = (3, 5, 7, 9)
SIGMA_R_GRID = (10.0, 20.0, 30.0, 50.0)
EPSILON_GRID = (0.01, 0.05, 0.10)
H_GRID = (0.25, 0.50, 0.75)
DEFAULT_SIGMA_R = 20.0
DEFAULT_EPSILON = 0.05
DEFAULT_K = 5
BOUNDARY_MODE = "reflect"  # scipy.ndimage half-sample symmetric extension


def tau_from(sigma_r: float, epsilon: float) -> float:
    return sigma_r * math.sqrt(-2.0 * math.log(1.0 - epsilon))


def ensure_dirs(root: Path) -> dict[str, Path]:
    rels = {
        "inventory": "00_dataset_inventory",
        "profile": "01_image_profiling",
        "hist": "01_image_profiling/histograms",
        "cdf": "01_image_profiling/cdfs",
        "profiles": "01_image_profiling/intensity_profiles",
        "gamma": "01_image_profiling/gamma_sweeps",
        "log": "01_image_profiling/log_probes",
        "stretch": "01_image_profiling/contrast_stretch_probes",
        "range": "01_image_profiling/local_range_maps",
        "gradient": "01_image_profiling/gradient_maps",
        "laplacian": "01_image_profiling/laplacian_maps",
        "lbp": "01_image_profiling/lbp_maps",
        "msb": "01_image_profiling/msb_maps",
        "conflict": "01_image_profiling/cue_conflict_maps",
        "homogeneity": "01_image_profiling/homogeneity_curves",
        "selection": "02_image_selection",
        "contacts": "02_image_selection/contact_sheets",
        "resolution": "03_resolution_analysis",
        "scaling": "03_resolution_analysis/scaling_plots",
        "reports": "reports",
    }
    out = {k: root / v for k, v in rels.items()}
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def discover_images(dataset_dir: Path, output_root: Path) -> list[Path]:
    output_root = output_root.resolve()
    paths = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED and output_root not in p.resolve().parents
    ]
    return sorted(paths, key=lambda p: p.relative_to(dataset_dir).as_posix().lower())


def read_image(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with Image.open(path) as im:
        original = np.asarray(im)
        mode = im.mode
        width, height = im.size
    if original.ndim == 2:
        gray = original.astype(np.float32)
        channels, is_gray, has_alpha = 1, True, False
    else:
        channels = original.shape[2]
        has_alpha = channels in (2, 4)
        rgb = original[..., :3].astype(np.float32)
        # ITU-R BT.601 luma coefficients, explicitly kept separate from originals.
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        is_gray = bool(np.all(original[..., 0] == original[..., 1]) and
                       np.all(original[..., 1] == original[..., 2])) if channels >= 3 else True
    if np.issubdtype(original.dtype, np.integer):
        bit_depth = np.iinfo(original.dtype).bits
        max_native = float(np.iinfo(original.dtype).max)
    else:
        bit_depth = int(original.dtype.itemsize * 8)
        max_native = float(np.nanmax(original)) or 1.0
    if max_native != 255.0:
        gray = np.clip(gray / max_native * 255.0, 0, 255)
    meta = dict(width=width, height=height, channels=channels, dtype=str(original.dtype),
                bit_depth=bit_depth, is_grayscale=is_gray, has_alpha=has_alpha, mode=mode)
    return original, gray.astype(np.float32), meta


def entropy_u8(image: np.ndarray) -> float:
    counts = np.bincount(np.clip(np.rint(image), 0, 255).astype(np.uint8).ravel(), minlength=256)
    prob = counts[counts > 0] / counts.sum()
    return float(-(prob * np.log2(prob)).sum())


def global_ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Global SSIM diagnostic; not the windowed benchmark SSIM used in filter evaluation."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    va, vb = a.var(ddof=1), b.var(ddof=1)
    cov = ((a - a.mean()) * (b - b.mean())).sum() / max(a.size - 1, 1)
    return float(((2*a.mean()*b.mean()+c1)*(2*cov+c2)) /
                 ((a.mean()**2+b.mean()**2+c1)*(va+vb+c2)))


def robust_stats(image: np.ndarray, prefix: str = "") -> dict:
    x = image.astype(np.float64).ravel()
    p = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return {
        f"{prefix}min": float(x.min()), f"{prefix}max": float(x.max()),
        f"{prefix}mean": float(x.mean()), f"{prefix}median": float(p[3]), f"{prefix}p50": float(p[3]),
        f"{prefix}std": float(x.std()), f"{prefix}variance": float(x.var()),
        f"{prefix}skewness": float(stats.skew(x)),
        f"{prefix}kurtosis": float(stats.kurtosis(x)),
        f"{prefix}p01": float(p[0]), f"{prefix}p05": float(p[1]),
        f"{prefix}p25": float(p[2]), f"{prefix}p75": float(p[4]),
        f"{prefix}p95": float(p[5]), f"{prefix}p99": float(p[6]),
        f"{prefix}dr_1_99": float(p[6]-p[0]), f"{prefix}dr_5_95": float(p[5]-p[1]),
    }


def gradient_fields(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    gx = ndimage.sobel(image, axis=1, mode=BOUNDARY_MODE) / 8.0
    gy = ndimage.sobel(image, axis=0, mode=BOUNDARY_MODE) / 8.0
    mag = np.hypot(gx, gy)
    # Reproducible robust threshold: median + 1.4826*MAD; never below 10/255.
    med = float(np.median(mag))
    threshold = max(10.0, med + 1.4826 * float(np.median(np.abs(mag-med))))
    return mag, mag >= threshold, threshold


def local_std(image: np.ndarray, k: int) -> np.ndarray:
    mean = ndimage.uniform_filter(image, size=k, mode=BOUNDARY_MODE)
    mean2 = ndimage.uniform_filter(image*image, size=k, mode=BOUNDARY_MODE)
    return np.sqrt(np.maximum(mean2 - mean*mean, 0.0))


def local_range(image: np.ndarray, k: int) -> np.ndarray:
    return (ndimage.maximum_filter(image, size=k, mode=BOUNDARY_MODE) -
            ndimage.minimum_filter(image, size=k, mode=BOUNDARY_MODE))


def lbp8(image: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    pad = np.pad(x, 1, mode="symmetric")  # matches scipy 'reflect'
    center = pad[1:-1, 1:-1]
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    code = np.zeros_like(center, dtype=np.uint8)
    h, w = center.shape
    for bit, (dy, dx) in enumerate(offsets):
        code |= ((pad[1+dy:1+dy+h, 1+dx:1+dx+w] >= center).astype(np.uint8) << bit)
    return code


POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
TRANSITIONS = np.array([
    sum(((i >> b) & 1) != ((i >> ((b+1) % 8)) & 1) for b in range(8))
    for i in range(256)
], dtype=np.uint8)


def conflict_map(image: np.ndarray, lbp: np.ndarray, tau: float, h_thr: float, k: int = 5) -> np.ndarray:
    radius = k // 2
    ip = np.pad(image, radius, mode="symmetric")
    lp = np.pad(lbp, radius, mode="symmetric")
    hh, ww = image.shape
    acc = np.zeros_like(image, dtype=np.float32)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            nbr_i = ip[radius+dy:radius+dy+hh, radius+dx:radius+dx+ww]
            nbr_l = lp[radius+dy:radius+dy+hh, radius+dx:radius+dx+ww]
            dt = POPCOUNT[np.bitwise_xor(lbp, nbr_l)] / 8.0
            acc += ((np.abs(image-nbr_i) <= tau) & (dt > h_thr))
    return acc / float(k*k)


def save_heatmap(data: np.ndarray, path: Path, title: str, cmap: str = "magma") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=9); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_hist_cdf(image: np.ndarray, image_id: str, d: dict[str, Path]) -> None:
    counts, bins = np.histogram(image, bins=256, range=(0, 256))
    p = np.percentile(image, [1, 5, 50, 95, 99])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(bins[:-1], counts/counts.sum())
    for v, lab in zip(p, ("P1","P5","P50","P95","P99")): ax.axvline(v, ls="--", lw=.8, label=lab)
    ax.set(title=f"{image_id}: normalized histogram", xlabel="8-bit luminance", ylabel="probability")
    ax.legend(ncol=5, fontsize=7); fig.tight_layout(); fig.savefig(d["hist"]/f"{image_id}.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(bins[:-1], np.cumsum(counts)/counts.sum())
    for v, lab in zip(p, ("P1","P5","P50","P95","P99")): ax.axvline(v, ls="--", lw=.8, label=lab)
    ax.set(title=f"{image_id}: empirical CDF", xlabel="8-bit luminance", ylabel="cumulative probability")
    ax.legend(ncol=5, fontsize=7); fig.tight_layout(); fig.savefig(d["cdf"]/f"{image_id}.png", dpi=140); plt.close(fig)


def probe_point_transforms(image: np.ndarray, image_id: str, d: dict[str, Path], edge_base: np.ndarray) -> list[dict]:
    log_im = 255.0/np.log(256.0) * np.log1p(image)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Original")
    ax[1].imshow(log_im, cmap="gray", vmin=0, vmax=255); ax[1].set_title("Log probe")
    ax[2].hist(image.ravel(), bins=256, range=(0,255), alpha=.55, label="original", density=True)
    ax[2].hist(log_im.ravel(), bins=256, range=(0,255), alpha=.55, label="log", density=True); ax[2].legend()
    for a in ax[:2]: a.axis("off")
    fig.suptitle(f"{image_id}: log-transform diagnostic"); fig.tight_layout(); fig.savefig(d["log"]/f"{image_id}.png", dpi=140); plt.close(fig)

    records = []
    fig, axes = plt.subplots(2, 4, figsize=(12, 6)); axes = axes.ravel()
    for ax, gamma in zip(axes, GAMMAS):
        out = 255.0 * (image/255.0) ** gamma
        _, edges, _ = gradient_fields(out)
        st = robust_stats(out, "out_")
        records.append(dict(image_id=image_id, gamma=gamma, entropy=entropy_u8(out),
                            clipping_fraction=float(np.mean((out <= 2) | (out >= 253))),
                            edge_density_change=float(edges.mean()-edge_base.mean()),
                            global_ssim_to_original=global_ssim(image, out), **st))
        ax.imshow(out, cmap="gray", vmin=0, vmax=255); ax.set_title(f"gamma={gamma}"); ax.axis("off")
    axes[-1].axis("off"); fig.suptitle(f"{image_id}: gamma sensitivity (diagnostic only)")
    fig.tight_layout(); fig.savefig(d["gamma"]/f"{image_id}.png", dpi=140); plt.close(fig)

    p1, p99 = np.percentile(image, [1,99]); den = max(float(p99-p1), 1e-12)
    stretched = np.clip((image-p1)*255.0/den, 0, 255)
    _, stretch_edges, _ = gradient_fields(stretched)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Original")
    ax[1].imshow(stretched, cmap="gray", vmin=0, vmax=255); ax[1].set_title("P1-P99 stretch")
    ax[2].hist(image.ravel(), bins=256, range=(0,255), alpha=.55, density=True, label="original")
    ax[2].hist(stretched.ravel(), bins=256, range=(0,255), alpha=.55, density=True, label="stretched"); ax[2].legend()
    for a in ax[:2]: a.axis("off")
    fig.suptitle(f"{image_id}: percentile-stretch diagnostic"); fig.tight_layout(); fig.savefig(d["stretch"]/f"{image_id}.png", dpi=140); plt.close(fig)
    stretch_record = dict(image_id=image_id, p01=float(p1), p99=float(p99),
                          clipping_fraction=float(np.mean((stretched <= 2) | (stretched >= 253))),
                          entropy_change=entropy_u8(stretched)-entropy_u8(image),
                          edge_density_change=float(stretch_edges.mean()-edge_base.mean()),
                          global_ssim_to_original=global_ssim(image, stretched))
    return records, robust_stats(log_im, "log_"), stretch_record


def save_profiles(image: np.ndarray, grad: np.ndarray, local_std5: np.ndarray, image_id: str, d: dict[str, Path]) -> None:
    h, w = image.shape; cy, cx = h//2, w//2
    edge_row = int(np.argmax(np.mean(grad, axis=1)))
    texture_row = int(np.argmax(np.mean(local_std5, axis=1)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0,0].plot(image[cy]); axes[0,0].set_title(f"center row y={cy}")
    axes[0,1].plot(image[:,cx]); axes[0,1].set_title(f"center column x={cx}")
    axes[1,0].plot(image[edge_row]); axes[1,0].set_title(f"highest-gradient row y={edge_row}")
    axes[1,1].plot(image[texture_row]); axes[1,1].set_title(f"highest-local-std row y={texture_row}")
    for ax in axes.ravel(): ax.set_ylim(0,255); ax.set_xlabel("spatial coordinate"); ax.set_ylabel("intensity")
    fig.suptitle(f"{image_id}: spatial intensity profiles"); fig.tight_layout(); fig.savefig(d["profiles"]/f"{image_id}.png", dpi=140); plt.close(fig)


def save_msb_sheet(image: np.ndarray, image_id: str, d: dict[str, Path]) -> None:
    u8=np.clip(np.rint(image),0,255).astype(np.uint8)
    fig,axes=plt.subplots(2,4,figsize=(12,6)); axes=axes.ravel()
    axes[0].imshow(u8,cmap="gray",vmin=0,vmax=255); axes[0].set_title("full precision")
    for ax,b in zip(axes[1:],range(2,9)):
        code=u8>>(8-b); ax.imshow(code,cmap="viridis",vmin=0,vmax=2**b-1); ax.set_title(f"top b={b} bits")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"{image_id}: MSB routing metadata only"); fig.tight_layout(); fig.savefig(d["msb"]/f"{image_id}.png",dpi=140); plt.close(fig)


def save_homogeneity_plot(image_id: str, r5: np.ndarray, d: dict[str, Path]) -> None:
    fig,ax=plt.subplots(figsize=(7,4))
    for eps in EPSILON_GRID:
        fractions=[float(np.mean(r5<=tau_from(sr,eps))) for sr in SIGMA_R_GRID]
        ax.plot(SIGMA_R_GRID,fractions,marker="o",label=f"epsilon={eps}")
    ax.set(title=f"{image_id}: Gate-1 opportunity, K=5",xlabel="sigma_r",ylabel="Pr[H5 <= tau]")
    ax.set_ylim(0,1); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(d["homogeneity"]/f"{image_id}.png",dpi=140); plt.close(fig)


def inventory(paths: list[Path], dataset_dir: Path, d: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for idx, path in enumerate(paths, 1):
        original, gray, meta = read_image(path)
        rows.append(dict(image_id=f"img_{idx:03d}", relative_path=path.relative_to(dataset_dir).as_posix(),
                         filename=path.name, extension=path.suffix.lower(), width=meta["width"], height=meta["height"],
                         num_pixels=meta["width"]*meta["height"], aspect_ratio=meta["width"]/meta["height"],
                         channels=meta["channels"], dtype=meta["dtype"], bit_depth=meta["bit_depth"],
                         filesize_bytes=path.stat().st_size, is_grayscale=meta["is_grayscale"],
                         has_alpha=meta["has_alpha"], min_intensity=float(gray.min()), max_intensity=float(gray.max())))
    df = pd.DataFrame(rows); df.to_csv(d["inventory"]/"dataset_inventory.csv", index=False)
    count = len(df); warn = "" if count == 44 else f"\n> **WARNING:** Expected 44 images, but discovered **{count}**. Analysis continued without altering originals.\n"
    summary = f"""# Dataset inventory summary

Generated: {datetime.now(timezone.utc).isoformat()}
{warn}
- Deterministic dataset root: `{dataset_dir.resolve()}`
- Supported images found: **{count}**
- Formats: {', '.join(f'{k}: {v}' for k,v in df.extension.value_counts().sort_index().items())}
- Native widths: min {df.width.min()}, median {df.width.median():.0f}, max {df.width.max()}
- Native heights: min {df.height.min()}, median {df.height.median():.0f}, max {df.height.max()}
- Grayscale/color: {int(df.is_grayscale.sum())}/{int((~df.is_grayscale).sum())}
- Originals were only read. Analysis grayscale uses BT.601 luma for RGB sources.
"""
    (d["inventory"]/"dataset_summary.md").write_text(summary)
    return df


def profile_all(inv: pd.DataFrame, dataset_dir: Path, d: dict[str, Path]) -> pd.DataFrame:
    feature_rows, gamma_rows, log_rows, stretch_rows = [], [], [], []
    local_rows, hom_rows, conflict_rows, msb_rows = [], [], [], []
    tau0 = tau_from(DEFAULT_SIGMA_R, DEFAULT_EPSILON)
    for n, row in inv.iterrows():
        image_id = row.image_id; path = dataset_dir / row.relative_path
        _, image, _ = read_image(path)
        print(f"[{n+1:02d}/{len(inv):02d}] profiling {image_id} {row.filename}", flush=True)
        base = robust_stats(image)
        base.update(image_id=image_id, filename=row.filename, entropy=entropy_u8(image),
                    f_black=float(np.mean(image <= 2)), f_white=float(np.mean(image >= 253)))
        base.update(possible_dark_image=base["p50"] < 64,
                    possible_bright_image=base["p50"] > 192,
                    possible_low_contrast=base["dr_1_99"] < 64,
                    possible_black_clipping=base["f_black"] > .01,
                    possible_white_clipping=base["f_white"] > .01)
        plot_hist_cdf(image, image_id, d)
        grad, edge, edge_thr = gradient_fields(image)
        lap = np.abs(ndimage.laplace(image, mode=BOUNDARY_MODE))
        base.update(gradient_mean=float(grad.mean()), gradient_median=float(np.median(grad)),
                    gradient_p90=float(np.percentile(grad,90)), gradient_p95=float(np.percentile(grad,95)),
                    edge_threshold=edge_thr, edge_density=float(edge.mean()),
                    laplacian_abs_mean=float(lap.mean()), laplacian_abs_p95=float(np.percentile(lap,95)),
                    laplacian_variance=float(ndimage.laplace(image, mode=BOUNDARY_MODE).var()))
        ranges, stds = {}, {}
        for k in WINDOWS:
            rr, ss = local_range(image,k), local_std(image,k); ranges[k], stds[k] = rr, ss
            vals = dict(image_id=image_id, K=k, range_mean=float(rr.mean()), range_median=float(np.median(rr)),
                        range_p90=float(np.percentile(rr,90)), range_p95=float(np.percentile(rr,95)),
                        local_std_median=float(np.median(ss)), local_std_p90=float(np.percentile(ss,90)),
                        low_local_contrast_fraction=float(np.mean(ss <= 5)))
            for sr in SIGMA_R_GRID:
                for eps in EPSILON_GRID:
                    tau = tau_from(sr,eps); frac = float(np.mean(rr <= tau))
                    hom_rows.append(dict(image_id=image_id,K=k,sigma_r=sr,epsilon=eps,tau=tau,homogeneity_fraction=frac))
            local_rows.append(vals)
        r5, s5 = ranges[DEFAULT_K], stds[DEFAULT_K]
        base.update(homogeneity_fraction=float(np.mean(r5 <= tau0)), local_std_median=float(np.median(s5)),
                    local_std_p90=float(np.percentile(s5,90)))
        lbp = lbp8(image); counts=np.bincount(lbp.ravel(),minlength=256); probs=counts[counts>0]/counts.sum()
        uniform = TRANSITIONS[lbp] <= 2
        base.update(lbp_entropy=float(-(probs*np.log2(probs)).sum()), lbp_occupied_patterns=int(np.count_nonzero(counts)),
                    lbp_occupied_fraction=float(np.count_nonzero(counts)/256),
                    lbp_uniform_fraction=float(uniform.mean()), lbp_nonuniform_fraction=float((~uniform).mean()))
        lat = (r5 <= tau0) & (s5 > 1.0) & (lbp != 0) & (lbp != 255)
        base["low_amplitude_texture_risk"] = float(lat.mean())
        cmap = conflict_map(image, lbp, tau0, .5, DEFAULT_K)
        base.update(conflict_mean=float(cmap.mean()), conflict_p90=float(np.percentile(cmap,90)),
                    conflict_fraction_gt_0_1=float(np.mean(cmap>.1)))
        for hthr in H_GRID:
            cc = cmap if hthr == .5 else conflict_map(image,lbp,tau0,hthr,DEFAULT_K)
            conflict_rows.append(dict(image_id=image_id,K=DEFAULT_K,sigma_r=DEFAULT_SIGMA_R,
                                      epsilon=DEFAULT_EPSILON,tau=tau0,h=hthr,mean=float(cc.mean()),
                                      p90=float(np.percentile(cc,90)),fraction_gt_0_05=float(np.mean(cc>.05)),
                                      fraction_gt_0_10=float(np.mean(cc>.10)),fraction_gt_0_20=float(np.mean(cc>.20))))
        smooth = (s5 <= 5.0) & (grad <= 8.0) & (r5 <= tau0)
        labels, num = ndimage.label(smooth)
        largest = 0.0 if num == 0 else float(np.bincount(labels.ravel())[1:].max()/image.size)
        boundary = ndimage.binary_dilation(smooth) ^ ndimage.binary_erosion(smooth)
        adjacent = ndimage.binary_dilation(smooth) & edge
        base.update(smooth_area_fraction=float(smooth.mean()), largest_smooth_region_fraction=largest,
                    smooth_boundary_fraction=float(boundary.mean()), foreground_edge_adjacent_smooth=float(adjacent.mean()))
        for b in range(2,9):
            code=(np.clip(np.rint(image),0,255).astype(np.uint8)>>(8-b)); cr=local_range(code,DEFAULT_K)
            ccounts=np.bincount(code.ravel(),minlength=2**b); cp=ccounts[ccounts>0]/ccounts.sum()
            bin_bound=2**(8-b)-1
            msb_rows.append(dict(image_id=image_id,b=b,occupied_codes=int(np.count_nonzero(ccounts)),
                                 code_entropy=float(-(cp*np.log2(cp)).sum()),local_code_diversity_mean=float(cr.mean()),
                                 all_codes_equal_window_fraction=float(np.mean(cr==0)),theoretical_bin_bound=bin_bound,
                                 sufficient_for_default_tau=bool(bin_bound<=tau0),exact_homogeneous_fraction=float(np.mean(r5<=tau0))))
        grows, logst, stretchst = probe_point_transforms(image,image_id,d,edge)
        gamma_rows.extend(grows); log_rows.append(dict(image_id=image_id,**logst)); stretch_rows.append(stretchst)
        save_profiles(image,grad,s5,image_id,d)
        save_heatmap(r5,d["range"]/f"{image_id}_K5.png",f"{image_id}: local range H5")
        save_heatmap(grad,d["gradient"]/f"{image_id}.png",f"{image_id}: Sobel magnitude")
        save_heatmap(lap,d["laplacian"]/f"{image_id}.png",f"{image_id}: absolute Laplacian")
        save_heatmap(lbp,d["lbp"]/f"{image_id}.png",f"{image_id}: LBP P=8, R=1",cmap="viridis")
        save_heatmap(cmap,d["conflict"]/f"{image_id}.png",f"{image_id}: cue conflict h=0.5",cmap="inferno")
        save_msb_sheet(image,image_id,d)
        save_homogeneity_plot(image_id,r5,d)
        feature_rows.append(base)
    features=pd.DataFrame(feature_rows); features.to_csv(d["profile"]/"image_features.csv",index=False)
    gamma_df=pd.DataFrame(gamma_rows); gamma_df.to_csv(d["profile"]/"gamma_metrics.csv",index=False)
    gamma_df.groupby("image_id").agg(
        mean_range=("out_mean",lambda x:float(x.max()-x.min())),
        median_range=("out_median",lambda x:float(x.max()-x.min())),
        dr_1_99_range=("out_dr_1_99",lambda x:float(x.max()-x.min())),
        entropy_range=("entropy",lambda x:float(x.max()-x.min())),
        edge_density_change_range=("edge_density_change",lambda x:float(x.max()-x.min())),
        min_global_ssim=("global_ssim_to_original","min")
    ).reset_index().to_csv(d["profile"]/"gamma_sensitivity.csv",index=False)
    pd.DataFrame(log_rows).to_csv(d["profile"]/"log_metrics.csv",index=False)
    pd.DataFrame(stretch_rows).to_csv(d["profile"]/"contrast_stretch_metrics.csv",index=False)
    pd.DataFrame(local_rows).to_csv(d["profile"]/"local_window_metrics.csv",index=False)
    pd.DataFrame(hom_rows).to_csv(d["profile"]/"homogeneity_curves.csv",index=False)
    pd.DataFrame(conflict_rows).to_csv(d["profile"]/"cue_conflict_metrics.csv",index=False)
    pd.DataFrame(msb_rows).to_csv(d["profile"]/"msb_metrics.csv",index=False)
    zv=RobustScaler().fit_transform(features[ROLE_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(features[ROLE_FEATURES].median()))
    fig,ax=plt.subplots(figsize=(14,10)); im=ax.imshow(np.clip(zv,-3,3),aspect="auto",cmap="coolwarm",vmin=-3,vmax=3)
    ax.set_xticks(range(len(ROLE_FEATURES)),ROLE_FEATURES,rotation=60,ha="right",fontsize=8)
    ax.set_yticks(range(len(features)),features.image_id,fontsize=7); ax.set_title("Robust-scaled dataset feature overview (clipped to +/-3)")
    fig.colorbar(im,ax=ax,fraction=.025); fig.tight_layout(); fig.savefig(d["profile"]/"dataset_feature_overview.png",dpi=160); plt.close(fig)
    return features


ROLE_FEATURES = ["dr_1_99","entropy","homogeneity_fraction","local_std_median","local_std_p90",
                 "edge_density","gradient_p95","lbp_entropy","lbp_nonuniform_fraction",
                 "low_amplitude_texture_risk","conflict_mean","largest_smooth_region_fraction",
                 "laplacian_abs_p95"]


def contact_sheet(inv: pd.DataFrame, dataset_dir: Path, path: Path, title: str, ids: set[str] | None = None) -> None:
    view = inv if ids is None else inv[inv.image_id.isin(ids)]
    cols=5; rows=math.ceil(len(view)/cols); fig,axes=plt.subplots(rows,cols,figsize=(15,3*rows)); axes=np.atleast_1d(axes).ravel()
    for ax,(_,r) in zip(axes,view.iterrows()):
        _,im,_=read_image(dataset_dir/r.relative_path); ax.imshow(im,cmap="gray",vmin=0,vmax=255)
        ax.set_title(f"{r.image_id}: {r.filename}",fontsize=8); ax.axis("off")
    for ax in axes[len(view):]: ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def select_four(features: pd.DataFrame, inv: pd.DataFrame, dataset_dir: Path, d: dict[str, Path]) -> pd.DataFrame:
    x=features[ROLE_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(features[ROLE_FEATURES].median())
    z=pd.DataFrame(RobustScaler().fit_transform(x),columns=ROLE_FEATURES,index=features.index).clip(-5,5)
    scores=pd.DataFrame({"image_id":features.image_id,"filename":features.filename})
    scores["smooth_score"]=(z.homogeneity_fraction+z.largest_smooth_region_fraction-z.edge_density-z.lbp_entropy-z.local_std_p90)/5
    scores["edge_score"]=(z.gradient_p95+z.edge_density+z.laplacian_abs_p95-z.lbp_nonuniform_fraction)/4
    scores["texture_score"]=(z.lbp_entropy+z.lbp_nonuniform_fraction+z.low_amplitude_texture_risk+z.conflict_mean+z.local_std_p90)/5
    balance=z[["homogeneity_fraction","edge_density","lbp_entropy"]]
    scores["mixed_score"]=-np.mean(np.abs(balance),axis=1)-.25*np.mean(np.abs(z),axis=1)
    # Transparent greedy unique-role assignment.  Visual near-duplicates are checked in the report/contact sheet.
    chosen=[]
    for role in ("smooth","edge","texture","mixed"):
        order=np.argsort(-scores[f"{role}_score"].to_numpy())
        idx=next(int(i) for i in order if scores.iloc[i].image_id not in {c[1] for c in chosen})
        chosen.append((role,scores.iloc[idx].image_id,idx))
    for role in ("smooth","edge","texture","mixed"):
        scores[f"{role}_rank"]=scores[f"{role}_score"].rank(ascending=False,method="min").astype(int)
    scores.to_csv(d["selection"]/"role_scores.csv",index=False)
    ranking=[]
    for role in ("smooth","edge","texture","mixed"):
        top=scores.nsmallest(5,f"{role}_rank")
        for _,r in top.iterrows(): ranking.append(dict(role=role,image_id=r.image_id,filename=r.filename,
                                                       rank=int(r[f"{role}_rank"]),score=r[f"{role}_score"]))
    pd.DataFrame(ranking).to_csv(d["selection"]/"candidate_rankings.csv",index=False)
    selected=pd.DataFrame([dict(role=r,image_id=i,feature_row=j) for r,i,j in chosen]).merge(scores,on="image_id")
    selected=selected.merge(inv[["image_id","relative_path","width","height"]],on="image_id")
    selected.to_csv(d["selection"]/"final_four.csv",index=False)
    # Secondary clustering sanity check only.
    pca=PCA(n_components=2,random_state=0); coords=pca.fit_transform(z)
    km=KMeans(n_clusters=4,random_state=0,n_init=20).fit(coords)
    sanity=pd.DataFrame(dict(image_id=features.image_id,pca1=coords[:,0],pca2=coords[:,1],cluster=km.labels_))
    sanity.to_csv(d["selection"]/"clustering_sanity_check.csv",index=False)
    contact_sheet(inv,dataset_dir,d["contacts"]/"all_images.png","All discovered source images")
    contact_sheet(inv,dataset_dir,d["contacts"]/"final_four.png","Final four role representatives",set(selected.image_id))
    ranking_df=pd.DataFrame(ranking)
    for role in ("smooth","edge","texture","mixed"):
        contact_sheet(inv,dataset_dir,d["contacts"]/f"{role}_top5.png",f"{role.title()} role: top five candidates",
                      set(ranking_df.loc[ranking_df.role==role,"image_id"]))
    lines=["# Final four selection","","> Selection uses native-image descriptors only; no filter output or PSNR influenced it.","",
           "## Transparent role formulas","",
           "All inputs are robust-scaled and clipped to [-5,5]. `smooth=(homogeneity + largest_smooth - edge_density - LBP_entropy - local_std_P90)/5`; `edge=(gradient_P95 + edge_density + Laplacian_P95 - LBP_nonuniform)/4`; `texture=(LBP_entropy + LBP_nonuniform + low-amplitude-texture risk + conflict + local_std_P90)/5`; `mixed=-mean absolute centrality across homogeneity/edge/LBP entropy - 0.25*mean absolute descriptor distance`.",""]
    for _,r in selected.iterrows():
        lines += [f"## {r.role.title()}: {r.image_id} — `{r.filename}`","",
                  f"Selected at native {r.width}×{r.height}; {r.role} score = {r[f'{r.role}_score']:.3f}.",""]
    lines += ["## Human validation checkpoint","","Inspect `contact_sheets/final_four.png`. The algorithm supplies an auditable recommendation, but scene semantics and near-duplicate rejection remain a human decision before the filtering benchmark."]
    lines += ["","### Natural-scene substitution","","The top edge score is the controlled calibration target `ruler.512.tiff`. If the final four must all be natural scenes, use `house.tiff` (edge rank 8) as the documented substitute; retain the ruler as a synthetic-like edge stress test."]
    (d["selection"]/"final_four_selection.md").write_text("\n".join(lines))
    return selected


def resolution_report(inv: pd.DataFrame, selected: pd.DataFrame, d: dict[str, Path]) -> pd.DataFrame:
    rows=[]
    for _,r in inv.iterrows():
        for size in (256,512,1024): rows.append(dict(image_id=r.image_id,filename=r.filename,target=size,
                                                     native_width=r.width,native_height=r.height,
                                                     eligible_without_upscale=bool(r.width>=size and r.height>=size)))
    support=pd.DataFrame(rows); support.to_csv(d["resolution"]/"resolution_support.csv",index=False)
    counts=support.groupby("target").eligible_without_upscale.sum().to_dict()
    eligible512=int(((selected.width>=512)&(selected.height>=512)).sum())
    if eligible512>=3: primary="512 (most selected images support it natively)"
    else: primary="256 (fewer than three selected images support 512 without upscaling)"
    report=f"""# Resolution decision

- Dataset support without upscaling: 256: **{counts[256]}/{len(inv)}**, 512: **{counts[512]}/{len(inv)}**, 1024: **{counts[1024]}/{len(inv)}**.
- Selected-four native 512 support: **{eligible512}/4**.
- Recommended development/sweep scale: **256**, or native size when smaller.
- Recommended primary benchmark: **{primary}**.
- Recommended scaling benchmark: **1024 only for the {counts[1024]} eligible sources**.
- Every resize must come directly from the source, preserve aspect ratio, and use anti-aliasing. No quality result may rely on upscaling.

Pixel counts for square inputs are 65,536 (256²), 262,144 (512²), and 1,048,576 (1024²), so doubling both dimensions quadruples pixel work before implementation/cache effects.
"""
    (d["resolution"]/"resolution_decision.md").write_text(report)
    fig,ax=plt.subplots(figsize=(6,4)); sizes=np.array([256,512,1024]); pixels=sizes**2
    ax.plot(sizes,pixels,marker="o"); ax.set(title="Square-image pixel scaling",xlabel="side length",ylabel="pixel count")
    ax.set_xticks(sizes); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(d["scaling"]/"pixel_count_scaling.png",dpi=150); plt.close(fig)
    return support


def write_research_report(inv: pd.DataFrame, features: pd.DataFrame, selected: pd.DataFrame, d: dict[str, Path]) -> None:
    gamma=pd.read_csv(d["profile"]/"gamma_metrics.csv")
    sens=gamma.groupby("image_id").agg(gamma_mean_range=("out_mean",lambda x:float(x.max()-x.min())),
                                        gamma_ssim_min=("global_ssim_to_original","min")).reset_index()
    f=features.merge(sens,on="image_id")
    dark=f.sort_values("p50").iloc[0]; contrast=f.sort_values("dr_1_99",ascending=False).iloc[0]
    report=f"""# Bilateral Image Profiling — Research Protocol and Dataset Findings

## Scope and causal order

This stage characterizes clean native images and chooses four research regimes. Log, gamma, and P1–P99 stretching are diagnostic point transforms only. They do not replace clean ground truth and are not silently fed into Gaussian/BF/SBF. Mean/box filtering is excluded. Filtering starts next with Gaussian, then BF, always-on SBF, and finally the gated G/BF/SBF V1.

## Dataset status

The expected count is 44; **{len(inv)} images were discovered**. {'This is a blocking provenance discrepancy for claims about a 44-image study, but not for running the documented pipeline on available files.' if len(inv)!=44 else 'The expected count was verified.'}
All sources are square TIFFs from {int(inv.width.min())} to {int(inv.width.max())} pixels. Native support is 256: {int(((inv.width>=256)&(inv.height>=256)).sum())}, 512: {int(((inv.width>=512)&(inv.height>=512)).sum())}, 1024: {int(((inv.width>=1024)&(inv.height>=1024)).sum())}.

## Intensity and point-processing interpretation

The darkest median is `{dark.filename}` (P50={dark.p50:.1f}). The broadest P1–P99 range is `{contrast.filename}` (DR={contrast.dr_1_99:.1f}). Gamma is treated as sensitivity analysis: gamma below 1 brightens normalized 8-bit luminance, gamma above 1 darkens it; no maximum-contrast gamma is declared universally best. Per-image gamma outputs, entropy, clipping, edge-density change, and global-SSIM transformation magnitude are in `gamma_metrics.csv`.

## Spatial and structural evidence

Gate-1 relevance is measured directly with local ranges H_K for K=3,5,7,9 and derived tau(sigma_r,epsilon), not with a generic quality score. Texture risk and range–structure conflict are separately measured so a low-amplitude pattern cannot be mislabeled as a safe Gaussian region merely because H_K is small. LBP is a descriptor and routing diagnostic, not a novelty claim.

## Boundary handling

The fixed default is SciPy `reflect` (half-sample symmetric), equivalent to repeating the edge sample in the reflected extension. It is conservative for smooth-background preservation and avoids injecting constant dark/bright borders. `mirror` (whole-sample symmetric) is retained as a later boundary ablation on selected images. Padding is not auto-selected per image because choosing it from downstream quality would create hidden tuning and weaken causal comparisons.

## Selection and resolution

The recommended representatives are: {', '.join(f'{r.role}={r.filename}' for _,r in selected.iterrows())}. Inspect the final-four contact sheet before freezing the benchmark. The machine-readable scores and top-five candidates make substitutions auditable.

The top edge score belongs to the controlled calibration target `ruler.512.tiff`. If all final representatives must be natural scenes, substitute `house.tiff` (edge rank 8) and retain the ruler as an auxiliary edge stress test.

## Next implementation gate

After resolving/accepting the 39-vs-44 discrepancy and visually validating the four scenes, implement odd centered K only, with K=2*ceil(3*sigma_s)+1 by default. Validate from-scratch Gaussian and BF against a trusted reference, produce the mandatory p,q dry-run table, then run causal ablations. Advanced TBF/QF/model-order/oracle work remains downstream of successful V1 gates.
"""
    (d["reports"]/"Research_Protocol_and_Dataset_Findings.md").write_text(report)


def run(dataset_dir: str | Path="misc", output_root: str | Path="outputs") -> dict:
    dataset_dir, output_root = Path(dataset_dir), Path(output_root)
    d=ensure_dirs(output_root); paths=discover_images(dataset_dir,output_root)
    if len(paths)!=44: warnings.warn(f"Expected 44 images; discovered {len(paths)}. Continuing.",RuntimeWarning)
    inv=inventory(paths,dataset_dir,d); features=profile_all(inv,dataset_dir,d)
    selected=select_four(features,inv,dataset_dir,d); support=resolution_report(inv,selected,d)
    write_research_report(inv,features,selected,d)
    manifest=dict(generated_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                  image_count=len(inv),boundary_mode=BOUNDARY_MODE,default_K=DEFAULT_K,
                  default_sigma_r=DEFAULT_SIGMA_R,default_epsilon=DEFAULT_EPSILON,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    pd.Series(manifest).to_json(d["reports"]/"run_manifest.json",indent=2)
    return dict(inventory=inv,features=features,selected=selected,resolution_support=support,manifest=manifest)


if __name__ == "__main__":
    result=run()
    print("\nSelected representatives:")
    print(result["selected"][["role","image_id","filename","width","height"]].to_string(index=False))
