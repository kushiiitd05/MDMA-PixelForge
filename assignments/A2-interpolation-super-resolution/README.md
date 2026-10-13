# A2 — Interpolation-Based Super-Resolution · Kush Tokas · 2023296 · CSE 340

**Deliverables:** `report/A2_Interpolation_SR_Report.pdf` (4 pages, editable `.docx` beside it) and `A2_Interpolation_SR.ipynb` (executed, for the TA demo).

## What the study does
Four 512×512 grayscale images from the Assignment 1 dataset are downsampled to 256², 128² and 64² by **one fixed procedure**
(Gaussian pyramid, σ = 1 per ×2 stage, half-pixel lattice, symmetric borders), then each level is enlarged **directly back to 512²**
with interpolation of order 0–5 plus Lanczos-3. Every reconstruction is scored with PSNR and SSIM against the original, and every
method is timed. Section 2 of the report adds our own decoder on the same inputs and metrics: a compact CNN on the low-resolution
grid followed by an exact projection that forces the output to reduce back to the input image.

## Run it
```bash
cd A2-interpolation-super-resolution
pip install -r ../A1-fast-exact-nlm/requirements.txt torch   # numpy, scipy, scikit-image, matplotlib, ... + torch
# 1. required interpolation study: regenerates results/quality.csv, results/runtime.csv, figures/  (~25 s)
python code/run_assignment.py
# 2. extension demo: loads the shipped weights (SHA-256 checked) and reproduces the stored scores  (~1 min, needs torch)
python code/run_extension_demo.py
# 3. reconstruction figure used on page 3 of the report (optional)
python code/make_recon_figure.py
# 4. rebuild the notebook / report (optional; the report needs Google Chrome for PDF printing)
python make_notebook.py
python build_report.py
```
Steps 1–3 run on CPU/MPS on the laptop. Nothing is downloaded; every input is inside this folder.

## Folder contents
| Path | What it holds |
|---|---|
| `report/` | `A2_Interpolation_SR_Report.pdf` (4 pages) and `.docx` |
| `A2_Interpolation_SR.ipynb` | Executed TA notebook: convention checks, tables, figures, live cells |
| `code/a2_ops.py` | Downsampling (1) and every interpolation kernel |
| `code/a2_eval.py` | Loading, grayscale, PSNR/SSIM, region masks, timing |
| `code/a2_neighborhood_min.py` | Local-range overshoot (ringing measure) |
| `code/run_assignment.py` | Regenerates every interpolation CSV and figure |
| `code/sr_extension.py` | Our decoder: exact P1 operator, projection x̂ = B y + P F(y), the DirectLR network |
| `code/run_extension_demo.py` | Checks weight hashes and reproduces the stored assignment-image scores |
| `code/make_recon_figure.py` | Builds the reconstruction figure (Figure 6) |
| `data/` | The four selected USC-SIPI images (unchanged copies) |
| `figures/` | fig1 originals + downsampled, fig2 crops per image, fig3 full reconstructions, fig4–5 quality/runtime plots, fig6 error maps, fig7 step edge, fig8 reconstruction grid |
| `results/quality.csv` | PSNR, SSIM, region PSNR, overshoot for every image × method × factor |
| `results/runtime.csv` | Median runtime per method, factor and implementation track |
| `results/downsampler_joint.csv`, `results/downsampler_transfer.csv` | Downsampling comparison (Table 5) and filter transfer functions |
| `results/environment.json`, `results/lr/` | Environment record with image hashes; the downsampled PNGs |
| `results/extension_demo.csv` | Output of the demo run |
| `extension/weights/` | The selected checkpoints (2×, 4×, 8×) |
| `extension/model_manifest.json` | Which checkpoint is used per scale, with SHA-256, training pool, seed, selection rule |
| `extension/results/` | Canonical per-image table, role table, paired statistics, latency records, frozen role selection, policy lock, coverage ledger, trade-off plots |
| `extension/figures/` | 8×/4×/2× method-comparison crops and error maps |
| `research_code/` | Training, evaluation, latency and selection scripts (not needed for the demo) |
| `MANIFEST.txt` | SHA-256 (16 hex) of every file in this folder |
| `assignment2.pdf` | The assignment brief |

## Protocol (frozen before any score was computed)
- **Images:** 4.2.03 Mandrill, 7.1.02 Jet, 7.1.03 Tank, 4.2.06 Sailboat — chosen by a content rule over the 20 natural 512² images.
- **Grayscale:** BT.601 luma (PIL `L`).
- **Downsampling:** Gaussian σ=1 per ×2 stage, recursive 512→256→128→64, half-pixel lattice (LR pixel *i* covers HR block [s·i, s·i+s)), symmetric borders.
- **Upsampling map:** x = (u+0.5)/s − 0.5, so nearest neighbour equals the s×s block copy of the prelude.
- **Metrics:** PSNR with R=255; SSIM with a Gaussian window σ=1.5, data range 255; both on the full frame after clipping.
- **Runtime:** median of warm single-thread calls, excluding file I/O and metric computation.

## Extension in one paragraph
x̂ = B y + P F(y), with B = Dᵀ(DDᵀ)⁻¹ and P = I − BD from the exact downsampling operator, and F a 12-layer, 96-channel CNN on the
low-resolution grid with a bicubic skip (~0.95 M weights). Because P is an orthogonal projector, ‖x̂ − x‖² = ‖F(y) − x‖² − ‖(I−P)(F(y) − x)‖²:
the correction can never make the prediction worse, and the output reproduces the low-resolution input. Training used 345 images that
exclude all four test images; checkpoints and the four reported roles (highest PSNR, highest SSIM, lowest latency, best quality–latency
trade-off) were selected on four held-out validation images and frozen before any test image was scored. Claim scope: best measured
choices among the evaluated pipelines for this downsampler on this hardware — not a general state-of-the-art claim.

## TA demo talking points
1. Notebook §1: the assignment's 2×2 example → 4×4 block copy, plus a hand-computed bilinear value.
2. Notebook §2: the coordinate convention is checked against OpenCV.
3. Tables 1–3: quality rises with order, but the gap shrinks with the factor (+1.09 dB at 2×, +0.51 dB at 8×).
4. Figure 7 (step edge): nearest = staircase, bilinear = blur, order ≥ 3 = overshoot/ringing.
5. Notebook §9: the projection makes D x̂ = y (residual ≈ 0 before clipping), and Table 4 shows why the promoted method differs per scale.
6. Live cells: change `name`, `s`, `method` or `role` and rerun.
