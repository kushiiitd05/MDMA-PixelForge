# Fast Exact Non-Local Means

**Reordering the computation, then studying spatial-PCA D5 and selective fusion as quality–runtime extensions**

This is the reproducible Digital Image Processing Assignment 1 submission:

> Implement a more efficient solution to basic Bilateral Filtering or Non-Local Means, and report original/processed images, PSNR, SSIM, and runtimes for any four USC-SIPI images.

## Submitted solution

The selected route is **Non-Local Means (NLM)**. The primary contribution is output-equivalent acceleration:

- `direct_patch_nlm`: explicit patch SSD, **O(N S² K²)**;
- `fast_exact_nlm`: displacement-wise SSD plus separable running-box aggregation, **O(N S²)**;
- controlled mean speed-up at `K=7`, `S=11`: **17.26×**;
- largest direct-versus-fast pixel difference: **2.3e-13**.

The package also reports two distinct research extensions:

- **Full D5**: spatially weighted PCA patches, coordinate-aware KD-tree retrieval, exact reranking, and centre-pixel reconstruction.
- **FusionV5**: sends fixed budgets of difficult pixels to D5, reducing full-D5 cost while retaining much of its selected-set quality.

On the independent native-resolution confirmation, D5 and FusionV5 each beat fast exact NLM in both PSNR and SSIM at `σ=10,25,50`. These four images were selected using a disclosed 128-pixel screen, so this is a confirmed selected-set result—not a universal dominance claim. Fast exact NLM remains the primary efficiency answer.

## Where to start

1. Open `report/Efficient_NLM_Report.pdf` — the final three-page report.
2. Review `code/best4_final_experiment.py` — the authoritative native-image experiment.
3. Open `Efficient_NLM_Demo.ipynb` — a labelled exact-acceleration walkthrough.
4. See `results/` — every CSV and figure used in the report.

## Four confirmation images

All 39 locally available USC-SIPI images were screened at 128 px. Gate/calibration images were excluded; the fixed selection rule required both D5 and fusion to beat fast exact NLM in PSNR and SSIM at all three noise levels. The chosen images were then rerun at native size with independent seed `20261029`.

| ID | File | Native size | Structural role |
|---|---|---:|---|
| `img_002` | `4.1.02.tiff` | 256×256 | interior scene, people, and edges |
| `img_001` | `4.1.01.tiff` | 256×256 | portrait, facial and hair texture |
| `img_028` | `7.1.04.tiff` | 512×512 | terrain, vehicles, and fine texture |
| `img_027` | `7.1.03.tiff` | 512×512 | armoured vehicle and structured texture |

The separate `gate_training_data/` directory contains only the four original gate-training images. Confirmation images are unseen by the gate. Images are bundled for reproducibility; attribution remains with the USC-SIPI Image Database.

## Result files

| File | Contents |
|---|---|
| `results/final_per_image_results.csv` | PSNR, SSIM, runtime, and diagnostics for each image/noise/method |
| `results/final_aggregate.csv` | four-image means used in report Table 1 |
| `results/final_by_resolution.csv` | quality/runtime split into 256×256 and 512×512 groups |
| `results/exact_equivalence_benchmark.csv` | direct-versus-fast equality and controlled speed-up |
| `results/exact_acceleration_patch_scaling.csv` | measured removal of the K² runtime factor |
| `results/final_full_image_comparison_sigma25.png` | original and processed native images |
| `results/final_detail_crops_sigma25.png` | high-gradient crops plus flat-region artifact audit |
| `results/quality_runtime_tradeoff.png` | fair complete-call quality/runtime comparison |
| `results/final_run_manifest.json` | seeds, image roles, method settings, and selection protocol |

No friend report, drafting prompt, or unrelated exploratory output is included.
