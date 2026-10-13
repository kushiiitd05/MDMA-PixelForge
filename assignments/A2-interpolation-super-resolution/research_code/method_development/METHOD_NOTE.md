# Method note — consistent null-space prediction for the frozen A2 degradation

Status: **working draft, updated as stages finish.** Every number is from `method_development/results/` and was produced in this workspace.

## 0. Assignment constraints kept (assignment2.pdf)
- **Same inputs as the required interpolation study.** Same 4 images, the frozen P1 downsampling procedure throughout, 256/128/64 → 512.
- **Same evaluation.** PSNR/SSIM against the 512² greyscale original, plus runtime.
- **Required interpolation results are untouched** (`submission/`). This work is a separately labelled extension.
- **It runs on the Mac** (no external GPU), so it can be demonstrated to TAs.
- **Lanczos / bicubic protocols never enter the method comparison.** They remain sensitivity studies.

## 1. Observation model (M0, `operators.py`, `test_operators.py`)
- **Operator.** y = D x with D(X) = A X Aᵀ.
  - A is the exact 1-D matrix of the implemented P1 reduction: σ = 1 Gaussian taps, every ×2 stage, half-pixel lattice, symmetric borders.
  - A is read from impulse responses of `a2_ops.make_lr`.
  - Observations are float (unquantised), as in every earlier experiment. The 8-bit case is studied separately.
- **Gram inverse.** K = A Aᵀ = Q diag(l) Qᵀ, so (DDᵀ + λI)⁻¹R = Q[(QᵀRQ) ⊘ (l lᵀ + λ)]Qᵀ. No 2-D matrix is ever formed.
- **Right inverse and projector.** B = Dᵀ(DDᵀ)⁻¹ and P = I − BD.
- **Identity checks** on the real operators at 2×/4×/8×, for both P1 and L3:

| Check | Worst value |
|---|---|
| D vs `make_lr` | ≤ 4e-13 |
| adjoint identity | ≤ 7e-16 |
| DB − I | ≤ 5e-14 |
| DP | ≤ 2e-12 |
| P² − P | ≤ 2e-12 |
| projection error identity | ≤ 5e-16 |

- **Conditioning.** cond(DDᵀ) for P1 is 35 / 118 / 160 (2× / 4× / 8×). For L3 it is ≈ 4.

## 2. Consistency (M1, `run_m1_consistency.py`)
- **Euclidean consistency.** C_y(z) = z + B(y − Dz) = By + Pz. For clean y and Dx = y: ‖C_y(z) − x‖² = ‖z − x‖² − ‖(I−P)(z − x)‖².
- **Existing IBP is oblique.** It back-projects with bicubic U, so its limit is z + U(DU)⁻¹(y − Dz).

Measured at P1 8× (mean of 4 assignment images, dB):

| Predictor | none | IBP20 | IBP limit | exact C_y |
|---|---|---|---|---|
| bicubic | 23.68 | 24.87 | 24.88 | **24.92** |
| quintic spline | 23.83 | 24.91 | 24.92 | **24.95** |
| HAT-L (pretrained) | 24.28 | 25.16 | 25.17 | **25.19** |
| LapSRN (P1, dev16) | 25.19 | 25.23 | 25.23 | **25.24** |

- **Never worse.** Exact C_y did not increase MSE in any of 112 per-image cases. It is ≥ IBP20 everywhere (+0.02–0.05 dB at 4×/8×, equal at 2×).
- **Much faster.** 0.7–2.3 ms on CPU vs 52–72 ms for IBP20. SSIM is also marginally higher.
- **Clipping breaks exact consistency slightly.** RMS residual after clipping is 0.01 / 0.03 / 0.07 grey levels (2× / 4× / 8×). Before clipping it is ≤ 7e-14.
- **8-bit observation** y = round(y): hard consistency costs 0.04–0.10 dB (the base-error floor). A regularised λ selected on dev images gave no benefit at 8× (λ = 0 was chosen).
- **Error split before correction.** Networks carry 18–26% of their squared error in the observable subspace; LapSRN carries 1%. After correction, all remaining error is null-space.

## 3. Trainable estimator (M2, `m2_models.py`, `run_m2.py`)
- **Model.** x̂ = By + P F_θ(y). F_θ is an LR-space CNN with pixel shuffle plus a bicubic skip.
- **Splits, fixed before training:**
  - train: misc images except val / fresh / assignment, plus Vid4 GT luma frames
  - val: 4 dev images, used for all selection
  - **fresh**: 12 non-overlapping 512² crops of the three 1024² misc images, never used for selection
  - assignment: 4 images, historical comparison
- **Rules.** Pre-registration and its amendment: `results/M2_preregistration.json`.

Controlled comparison (4k iterations, batch 16, crop 128, MSE, same seed), fresh-set PSNR with consistent output:

| Variant | params | fresh | assignment | device ms (net+proj) |
|---|---|---|---|---|
| direct w64 d8, projection at inference | 296k | 27.125 | 25.195 | 1.43 |
| direct w64 d8, projection in training | 296k | 26.972 | 25.019 | 1.42 |
| progressive (3 × ×2 stages), projection at inference | 244k | 27.162 | 25.247 | 3.55 |
| progressive, projection in training | 244k | 27.127 | 25.132 | 3.47 |

- **Progressive vs direct:** +0.04 dB at 2.5× runtime. Below the 0.05 dB claim threshold, so direct is kept.
- **Projection in training scores lower.** The earlier "cond 160 amplifies gradients" explanation is **retracted**: ‖P‖₂ = 1 and ∇_z L = Pᵀ∇_x̂ L. Hypotheses under test are in `run_ctrain_gap.py`.

Baselines on the same splits (`run_m2_baselines.py`), fresh set:

| Baseline | none | exact C_y |
|---|---|---|
| quintic spline | 25.818 | 26.899 |
| IBP20 | — | 26.845 |
| LapSRN (P1) | 27.018 | 27.121 |
| SwinIR-M | 26.241 | 27.085 |
| **HAT-L** | 26.272 | **27.135** |

### M2b capacity / budget sweep (selection by val)

| Config | val | fresh | assignment | device ms |
|---|---|---|---|---|
| w64 d8, 4k | 24.253 | 27.125 | 25.195 | 1.43 |
| w64 d8, 16k | 24.354 | 27.238 | 25.398 | 1.43 |
| w96 d12, 4k | 24.254 | 27.152 | 25.229 | 2.98 |
| **w96 d12, 16k (val best)** | **24.468** | **27.281** | **25.502** | 2.99 |
| w128 d16 plain, 4k / 16k | 23.886 | 26.879 | 24.924 | 6.1–6.6 |
| w96 d12, 16k + ×8 TTA (ensemble, 8× cost) | 24.592 | 27.341 | 25.603 | ~8 × 3.0 |

- **Undertrained at 4k:** 4k → 16k adds +0.10 to +0.21 dB on val.
- **The deep plain stack does not optimise** (stuck at spline level). Residual blocks are being tested (M2d).

### M2c where the remaining error is (val only)
- **97% of remaining error power lies above the LR Nyquist frequency 1/16.**
- **Strong edges carry only 14% of squared error**, and there the direct model already beats HAT-L + C_y.
- **85% lies in regions whose bicubic(y) gradient is weak:** fine texture removed by σ_eff ≈ 4.6.
- **Implication.** The conditional-mean detail for stochastic texture is small, so large gains from an edge-only module are unlikely. Capacity and training were tested first (M2d).

### M2d capacity / budget / modules (original pool; val selection)

| Config | val | fresh | assignment |
|---|---|---|---|
| plain w96 d12, 16k (val best, original pool) | **24.468** | 27.281 | 25.502 |
| plain w96 d12, 64k | 24.437 | 27.255 | 25.512 |
| residual w96 b8, 16k / 64k | 24.384 / 24.406 | 27.223 / 27.243 | 25.379 / 25.476 |
| residual + pooled context, 16k | 24.413 | 27.228 | 25.381 |
| residual + 16-ch full-res refinement, 16k | 24.350 | 27.237 | 25.396 |

- No gain from 64k iterations, depth or either module on the original pool. The model is not undertrained beyond 16k; the pool looks like the limit.

### M2e data: original pool vs +DIV2K
- **Split, declared before training:** DIV2K_valid_HR 0801–0880 → training; 0881–0900 → `div2k_eval20`, centre 512² luma crops, never trained on or used for selection.

| Model (projection at output) | val | fresh | assignment | div2k_eval20 |
|---|---|---|---|---|
| original pool, w96 d12 16k | 24.468 | 27.281 | 25.502 | 22.419 |
| +DIV2K, w96 d12 16k | 24.524 | 27.334 | 25.604 | 22.587 |
| **+DIV2K, w96 d12 64k (val best overall)** | **24.561** | **27.350** | **25.661** | **22.641** |
| +DIV2K, residual b8 64k | 24.552 | 27.310 | 25.591 | 22.596 |

- **Baselines on div2k_eval20** (exact projection): spline 21.871, LapSRN 22.238, SwinIR-M 22.236, HAT-L 22.344.
  - SwinIR/HAT may have used DIV2K validation images upstream, so their scores here are not a clean hold-out.
- **Conclusion:** more diverse data helps (+0.17 dB at equal budget), and with it the longer budget helps too. The model was data-limited.

### M2f consistency-in-training gap: controlled loss experiment
- **Loss:** L_λ = ‖P(z−x)‖² + λ‖(I−P)(z−x)‖². Same init, minibatch sequence, optimiser and budget (+DIV2K, 16k).

| λ | val | fresh | assignment | div2k_eval20 |
|---|---|---|---|---|
| 0 (projected loss) | 24.265 | 27.204 | 25.307 | 22.338 |
| 0.1 | 24.534 | 27.317 | 25.609 | 22.571 |
| 1 (= plain MSE) | 24.558 | 27.332 | 25.616 | 22.597 |

- **Supervising the observable component helps** (λ = 1 beats λ = 0 on 29/32 held-out images); 10% weight recovers most of the gap.
- **Ruled out:** float precision, and a border-only deficit.
- **Correction:** the raw-output intensity range is not evidence, because PDᵀ = 0.
- **Run-to-run noise:** two λ = 1 / MSE runs of the same configuration differ by 0.034 dB on val (MPS non-determinism). Differences < 0.05 dB are not claimed.

### Latency (idle, one architecture per fresh process, 3 rotations × 100 warm runs; `latency_isolated.json`)
- **Path timed:** host LR → MPS, network, exact projection, clip, MPS → host.

| Method (8×, 64² → 512²) | median ms | × bicubic |
|---|---|---|
| torch bicubic (same path) | 0.67 | 1.0 |
| direct w64 d8 | 2.38 | 3.5 |
| **direct w96 d12** | **3.85** | **5.7** |
| residual b8 / +context / +full-res refinement | 5.27 / 5.35 / 5.93 | 7.8 / 7.9 / 8.8 |
| SwinIR-M + exact projection (native ×8, untiled) | 312 | ≈ 460 |
| HAT-L + exact projection (×4 then ×2; 64-px tiling forced by 8 GB) | 20 089 | — (tiling inflates; not a fair speed ratio) |

- **The earlier single-process timings are superseded.** One architecture measured 3.6–11.5 ms depending on process state.
- **Cold setup** (operator eigendecomposition) ≈ 1.1 s once per scale; not included in per-image times.

### Data audits
- **Vid4:** all 171 frames (calendar 41, city 34, foliage 49, walk 47) are training-only; no evaluation split contains video.
- **Sampling:** with uniform per-image sampling, ~62% of +DIV2K minibatches come from 4 video sequences → group-balanced sampling test (M2g).
- **7.1.xx siblings:** not near-duplicates of the test frames (8–18 dB PSNR between them).

### M2g sampling: uniform vs group-balanced (each Vid4 sequence = one source), 2 seeds, 16k

| Sampling (mean of 2 seeds) | val | fresh | assignment | div2k_eval20 |
|---|---|---|---|---|
| uniform per image | 24.544 | 27.334 | 25.606 | 22.604 |
| group-balanced | **24.623** | 27.348 | 25.614 | 22.644 |

- **Val improves on both seeds** (+0.10, +0.06), so group-balanced is adopted by the selection rule.
- **Held-out gains are below the 0.05 dB claim threshold** (48/64 per-image wins).

## 4. Final candidate (`run_final_candidate.py`, `final_report.py` → `results/FINAL_report.md`)
- **Configuration.** x̂ = By + P F_θ(y) with F_θ = plain LR-space CNN, width 96, depth 12, pixel shuffle ×8, bicubic skip (970k params).
  - Plain MSE, group-balanced sampling, +DIV2K 0801–0880, 64k iterations, batch 16, crop 128.
  - Exact Euclidean projection at inference.
  - 2 seeds; reported numbers are the mean over seeds.

8×, frozen P1 (mean per-image PSNR / SSIM):

| Method | fresh (12) | div2k_eval20 (20) | assignment (4) |
|---|---|---|---|
| quintic spline + exact projection | 26.899 / 0.594 | 21.871 / 0.547 | 24.945 / 0.583 |
| LapSRN (P1) + exact projection | 27.121 / 0.604 | 22.238 / 0.569 | 25.235 / 0.593 |
| SwinIR-M + exact projection | 27.085 / 0.602 | 22.236* / 0.565 | 25.160 / 0.590 |
| HAT-L + exact projection | 27.135 / 0.604 | 22.344* / 0.571 | 25.190 / 0.591 |
| **ours** | **27.406 / 0.616** | **22.791 / 0.590** | **25.708 / 0.603** |
| ours + ×8 TTA (8× cost, reference) | 27.460 / 0.618 | — | 25.776 / 0.605 |

\*may have seen DIV2K validation images upstream.

**Paired per-image ΔPSNR vs HAT-L + exact projection** (bootstrap 95% CI over images):

| Split | ΔPSNR | CI95 | ours better |
|---|---|---|---|
| fresh | +0.27 | [0.12, 0.43] | 10/12 |
| div2k_eval20 | +0.45 | [0.18, 0.78] | 18/20 |
| assignment | +0.52 | [0.05, 1.23] | 3/4 |

- Against SwinIR-M, LapSRN and spline the differences are larger. On fresh, ours is better than LapSRN on all 12 images.
- The fresh CI is optimistic: its 12 crops come from 3 source images.

**Latency** (host to host, MPS, idle, isolated processes):

| Method | Median |
|---|---|
| **ours** | **3.85 ms (5.7× bicubic)** |
| SwinIR-M + projection | 313 ms |
| HAT-L + projection | 20.1 s (tiled on 8 GB; not a fair ratio) |

**Selection hygiene and limits:**
- Every choice was made on val. The assignment images were used repeatedly for reporting during development, so treat assignment numbers as historical; fresh and div2k_eval20 are the held-out evidence.
- 8× only so far; single degradation (P1, float observation); one hardware platform.
- **Claim scope:** best in our tested setting (8×, frozen P1, these splits, Apple M2 MPS, 15 Sep 2026). Not a general SOTA claim.

**Closest prior art and difference:**
- Null-space / data-consistent networks: Schwab et al. 2019.
- Consistency Enforcing Module: Bahat & Michaeli, CVPR 2020.
- Range–null-space diffusion: DDNM, ICLR 2023.
- LR-space sub-pixel CNNs: ESPCN.
- Our difference: an exact, separable, eigenbasis Euclidean projection for the half-pixel, symmetric-border P1 operator (0.6 ms), used with a compact LR-space predictor. The measured facts:
  - the observable-component supervision effect (λ experiment)
  - that data diversity, not depth, was the limit
  - the resulting accuracy/latency point vs pretrained transformers under the same exact projection

## 5. Next
- 4× and 2× versions of the final recipe.
- M3: course-Laplacian vs per-stage consistency and error propagation.
- M4 codec (predicted-error coding with projected corrections) only after M3.

## 6. SRD bounded comparison (spec §8) — result: no improvement, branch study stopped
- **Setup.** All runs start from `final_bal_s20260915` (val 24.694), 16k continuation updates, identical minibatches.
  - LR generated per full image, 8-px real halo.
  - Source-unit sampling: 110 units, video mass 3.64%.
  - Adam 5e-5 cosine; MSE.

| Run | params | val after fit | val during continuation (1k / 8k / 16k) | val-selected state |
|---|---|---|---|---|
| A core continuation | 969,760 | — | 24.652 / 24.611 / 24.614 | start checkpoint (24.694) |
| B fixed-stencil branch | 977,632 | 24.688 | 24.652 / 24.607 / 24.611 | after fit (24.688) |
| C full SRD | 979,179 | 24.690 | 24.658 / 24.611 / 24.612 | after fit (24.690) |

- **Every continuation lowered val** (≈ −0.08 dB). The fitted heads were neutral (−0.004 to −0.006 dB).
- **Neither branch beat the unchanged core on val**, so under the pre-set rule the second-seed repeat is skipped and the **completed core stays deployed**.
- **Not diagnosed:** why continuation under the new crop/halo policy and learning rate lowers val. No held-out metric was used for any decision.

## 7. Final campaign (15–16 Sep 2026): 2×/4×/8×, four roles, frozen selection
- **Training.** DirectLR w96 d12, 64k updates, MSE, group-balanced sampling, 2 seeds per scale. 8× on Mac (MPS), 4×/2× on GPU (CUDA).
  - Pool rule fixed before 4×/2×: adding BSD68 gave ΔPSNR +0.039 dB, ΔSSIM +0.0009 → pool `bsd`. This is within earlier run-to-run spread.
  - `run_final_scales.py`, `decide_pool.py`, `results/final_scales/`.
- **Matched-degradation reference.** Official SwinIR-M weights fine-tuned on P1: same images, sampler and validation selection; L1, 20k updates, one seed (`run_swinir_p1.py`).
  - The 2× run was restarted from scratch on a freed GPU; both runs gave identical validation at update 1k.
- **Evaluation.** `final_eval.py` scores every pipeline on val / assignment / fresh / DIV2K-20 / Urban100-c (100 genuine HR, luma, centre 512²).
  - Scored on the GPU; classical val rows cross-checked against the Mac to 1.7e-8 dB.
  - Latency: `bench_final_latency.py` on the Mac, isolated processes, 5 rotations (raised from 3 after a bimodal MPS state).
- **Selection.** Policy and scoring code hash-locked (`results/final_eval/policy_lock.json`, 22:05) before any external image was scored for new pipelines.
  - Roles frozen at 00:52 from validation + latency only (`select_roles.py`, `roles_frozen.json`).
- **Winners** (overall; +P = exact projection):
  - Highest PSNR / SSIM at every scale: SwinIR-M fine-tuned on P1 (+P).
  - Fastest: torch bicubic.
  - Promoted trade-off: bicubic+P at 2× and 4×; OURS (seed 7, raw) at 8×. OURS exceeds the 10× bicubic budget at 4× (10.4× raw) and 2× (33×).
- **Urban100-c, OURS PSNR-best +P.**
  - vs quintic+P: +3.01 / +1.39 / +0.66 dB (2×/4×/8×).
  - vs HAT-L+P: −0.04 [−0.33, 0.26] / +0.52 [0.32, 0.76] / +0.20 [0.10, 0.31].
  - vs SwinIR-M P1-ft+P: −2.17 / −1.31 / −0.82 dB, at ~100× (8×) to ~50× (4×) less latency.
- **Report.** `submission/build_report.py` builds 4 pages (A1 design) + DOCX. The downsampler sensitivity page comes from `collect_downsampler_joint.py` (existing B1/B2/B8/B9 records).
