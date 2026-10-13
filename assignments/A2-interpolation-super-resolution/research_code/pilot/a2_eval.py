"""Data loading, content-based image selection, metrics, and timing."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity

DATASET = Path(__file__).resolve().parents[2] / "A1" / "misc"
SYNTHETIC = {"gray21.512.tiff", "ruler.512.tiff"}


def load_gray(path: Path) -> np.ndarray:
    """ITU-R BT.601 luma on encoded values (PIL 'L'), returned as float64 0..255."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def candidate_files() -> list[Path]:
    out = []
    for p in sorted(DATASET.glob("*.tiff")):
        with Image.open(p) as im:
            if im.size == (512, 512) and p.name not in SYNTHETIC:
                out.append(p)
    return out


def content_features(img: np.ndarray) -> dict[str, float]:
    hf = img - ndimage.gaussian_filter(img, 2.0, mode="reflect")
    blocks = img.reshape(64, 8, 64, 8).std(axis=(1, 3))
    smooth = ndimage.gaussian_filter(img, 1.0, mode="reflect")
    grad = np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(512))
    radius = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    return {
        "log_hf_energy": float(np.log10(hf.var() + 1e-9)),
        "smooth_block_frac": float((blocks < 3.0).mean()),
        "edge_density": float((grad > 100.0).mean()),
        "energy_above_nyq64": float(spec[radius > 1 / 16].sum() / spec.sum()),
    }


def farthest_point_selection(features: dict[str, dict[str, float]], k: int = 4) -> list[str]:
    """Deterministic rule: start at max high-frequency energy, add images maximising the
    minimum z-scored feature distance to those already chosen."""
    names = sorted(features)
    keys = sorted(next(iter(features.values())))
    mat = np.array([[features[n][c] for c in keys] for n in names])
    z = (mat - mat.mean(0)) / (mat.std(0) + 1e-12)
    chosen = [int(np.argmax(mat[:, keys.index("log_hf_energy")]))]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(z[:, None, :] - z[None, chosen, :], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return [names[i] for i in chosen]


# ----------------------------------------------------------------------------- metrics
def score(ref: np.ndarray, raw: np.ndarray) -> dict[str, float]:
    """Clip to [0,255] after logging overshoot; PSNR with R=255; SSIM (Wang 2004 Gaussian window)."""
    est = np.clip(raw, 0.0, 255.0)
    mse = float(np.mean((ref - est) ** 2))
    return {
        "psnr": float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse),
        "ssim": float(structural_similarity(ref, est, data_range=255.0, gaussian_weights=True,
                                            sigma=1.5, use_sample_covariance=False)),
        "mse": mse,
        "overshoot_frac": float(np.mean((raw > 255.0) | (raw < 0.0))),
        "overshoot_peak": float(max(raw.max() - 255.0, -raw.min(), 0.0)),
    }


def psnr(ref: np.ndarray, est: np.ndarray) -> float:
    mse = float(np.mean((ref - np.clip(est, 0, 255)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def region_masks(ref: np.ndarray) -> dict[str, np.ndarray]:
    """Offline scoring masks from the HR target. Never passed to a reconstruction method."""
    smooth = ndimage.gaussian_filter(ref, 1.0, mode="reflect")
    grad = np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))
    local_std = np.sqrt(np.clip(ndimage.uniform_filter(ref**2, 7) - ndimage.uniform_filter(ref, 7) ** 2, 0, None))
    edge = grad >= np.quantile(grad, 0.90)
    flat = (local_std < 4.0) & ~edge
    texture = (local_std >= 10.0) & ~edge
    return {"edge": edge, "smooth": flat, "texture": texture}


def region_psnr(ref: np.ndarray, raw: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    err = (ref - np.clip(raw, 0, 255)) ** 2
    return {f"psnr_{k}": float(10 * np.log10(255.0**2 / max(err[m].mean(), 1e-12))) for k, m in masks.items()}


# ----------------------------------------------------------------------------- timing
def time_call(fn, repeats: int = 20, warmup: int = 3) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    arr = np.array(samples) * 1e3
    return {"ms_median": float(np.median(arr)), "ms_iqr": float(np.subtract(*np.percentile(arr, [75, 25])))}
