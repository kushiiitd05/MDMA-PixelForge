<div align="center">

# 🖼️ MDMA · PixelForge

### Digital Image Processing — CSE340 · IIIT Delhi · Monsoon 2026

*Faster denoising, sharper upscaling, and SIFT without a single Gaussian.*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-SciPy-013243?logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-extension-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebooks-executed-F37626?logo=jupyter&logoColor=white)

</div>

---

## ✨ Highlights

| | Work | One-line result |
|---|---|---|
| **A1** | [Fast exact Non-Local Means](assignments/A1-fast-exact-nlm) | Same output as direct NLM (max diff **2.3e-13**) at **17.26× speed**, plus two quality extensions (spatial-PCA D5, selective FusionV5) |
| **A2** | [Interpolation-based super-resolution](assignments/A2-interpolation-super-resolution) | Orders 0–5 + Lanczos-3 at 2×/4×/8×, PSNR/SSIM/runtime for every cell, plus our own CNN decoder with an **exact back-projection** |
| **Mid-sem** | [ZG-ND-SIFT](midsem/ZG-ND-SIFT) | SIFT with **zero Gaussian kernels**: nonlinear diffusion scale-space, Hessian detector, Bartlett windows |

---

## 🧪 Assignment 1 — Fast Exact Non-Local Means

> *Implement a more efficient Bilateral Filter or Non-Local Means; report PSNR, SSIM and runtime on four USC-SIPI images.*

- `fast_exact_nlm` reorders the computation: displacement-wise SSD + separable running-box sums, **O(N S²)** instead of **O(N S² K²)**.
- Output-equivalent to the direct version, so the speed-up is free.
- Research extensions: **D5** (spatially weighted PCA patches + KD-tree retrieval + exact rerank) and **FusionV5** (sends only the hardest pixels to D5).
- On the native-resolution confirmation set, D5 and FusionV5 beat fast exact NLM in **both PSNR and SSIM at σ = 10, 25, 50**.

```bash
cd assignments/A1-fast-exact-nlm
conda env create -f environment.yml && conda activate dip-efficient-nlm   # or: pip install -r requirements.txt
jupyter lab Efficient_NLM_Demo.ipynb                                      # quick equivalence demo
python code/best4_final_experiment.py                                     # full experiment -> results_reproduced/
```
📄 Report: [`report/Efficient_NLM_Report.pdf`](assignments/A1-fast-exact-nlm/report/Efficient_NLM_Report.pdf)

## 🔍 Assignment 2 — Interpolation-Based Super-Resolution

- One fixed downsampler (Gaussian pyramid, σ = 1 per ×2 stage), then direct enlargement back to 512² with interpolation orders 0–5 and Lanczos-3.
- Every reconstruction is scored (PSNR, SSIM, region PSNR, ringing overshoot) and timed.
- Quality rises with order, but the gain shrinks with the factor: **+1.09 dB at 2×, +0.51 dB at 8×**.
- **Extension:** a compact CNN on the low-res grid, followed by an exact projection x̂ = B y + P F(y), so the output always reduces back to the input. Weights are shipped and SHA-256 checked.

```bash
cd assignments/A2-interpolation-super-resolution
python code/run_assignment.py        # interpolation study (~25 s)
python code/run_extension_demo.py    # CNN + projection extension (needs torch)
```
📄 Report: [`report/`](assignments/A2-interpolation-super-resolution/report) · 📓 [`A2_Interpolation_SR.ipynb`](assignments/A2-interpolation-super-resolution/A2_Interpolation_SR.ipynb)

## 🌀 Mid-sem — ZG-ND-SIFT (Zero-Gaussian Nonlinear-Diffusion SIFT)

| Stage | Classroom SIFT | ZG-ND-SIFT |
|---|---|---|
| Scale-space | Gaussian blur | Perona–Malik **nonlinear diffusion** (rational conductivity) |
| Detector | Difference of Gaussians | scale-normalised **Hessian determinant** (KAZE-style) |
| Orientation window | circular Gaussian | **radial Bartlett** taper |
| Descriptor window | Gaussian | **separable 2-D Bartlett** taper |

Everything else (26-neighbour extrema, Taylor refinement, edge rejection, 36-bin orientation, 128-D descriptor) matches the classroom algorithm.
📓 [`ZG_ND_SIFT.ipynb`](midsem/ZG-ND-SIFT/ZG_ND_SIFT.ipynb) · 📄 [report](midsem/ZG-ND-SIFT/2023296_Kush_Tokas_ZG_ND_SIFT_Report.pdf)

---

## 📚 Course material

| Folder | Contents |
|---|---|
| [`lectures/`](lectures) | Lectures 1–3, Chapter 4 (parts 1–2), convolution theorem, compressed-sensing concepts |
| [`notes/`](notes) | Full-course learning roadmap (`master_roadmap.md`) and a Lecture 3 / Chapter 4 revision cheat sheet |

Textbook: R. C. Gonzalez & R. E. Woods, *Digital Image Processing*, 4th ed., Pearson. Not included; buy or borrow a copy.

## 🗂️ Layout

```
MDMA-PixelForge/
├── assignments/
│   ├── A1-fast-exact-nlm/                 code · notebook · data · results · report
│   └── A2-interpolation-super-resolution/ code · notebook · extension (weights) · figures · report
├── midsem/ZG-ND-SIFT/                     notebook + report
├── lectures/                              course slides
└── notes/                                 roadmap + cheat sheet
```

<div align="center">

**Kush Tokas** · 2023296 · IIIT Delhi · Team **MDMA**

<sub>Test images: USC-SIPI Image Database. Slides belong to their instructors.</sub>

</div>
