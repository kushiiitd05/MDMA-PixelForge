# Fast Exact Non-Local Means

## Reordering the computation, then studying the quality–runtime trade-off

**Student:** Kush Tokas  
**Roll number:** 2023296  
**Course:** DIP-CSE341

## 1. Assignment route and primary contribution

I chose Non-Local Means (NLM) for this assignment. My main contribution is an **output-equivalent acceleration**. Basic NLM repeatedly sums every `K×K` patch distance at each `S×S` search displacement. I instead form one squared-difference image for each displacement and aggregate it with a separable, constant-cost running-box operator.

For noisy image `Y`, patch `P(p)`, noise level `σ`, and filter parameter `h`:

```text
d²(p,q) = max(K⁻² ||P(p)-P(q)||²₂ - 2σ², 0)
w(p,q)  = exp[-d²(p,q)/h²]
Î(p)    = Σq w(p,q)Y(q) / Σq w(p,q)
```

The direct form costs `O(N S² K²)`. For displacement `t`, I define `D_t(u)=[Y(u)-Y(u+t)]²`. The patch distance is then the separable running mean `B_K[D_t](p)`, reducing the cost to `O(N S²)` with `O(N)` working memory. Reflect padding preserves boundary context. In short, I changed the order of the computation, not the estimator.

Across four 96×96 crops at `σ=25`, `K=7`, `S=11`, the largest direct-versus-fast pixel difference was `2.3e-13`, and the controlled mean speed-up was **17.26×**. At `K=11`, the measured speed-up reached **42.15×**.

## 2. Quality–cost extensions

After verifying that the faster form gives the same output, I explored a second question: can a richer patch representation improve denoising when extra computation is acceptable? This led to the spatial-PCA D5 expert:

```text
x_p = W^(1/2)(P(p)-μ)
z_p = Uᵀ x_p
Q(p) = [z_p, λ_y y_p, λ_x x_p]
```

A KD-tree retrieves a bounded candidate set in `Q`-space. I rerank that set with the exact descriptor distance, and R0 reconstructs from candidate centre intensities. Since full D5 is expensive, FusionV5 combines residual, gradient, variance, and expert-disagreement cues and sends only the hardest 10%, 25%, or 35% of pixels to D5 at `σ=10,25,50`.

D5 and fusion are supporting experiments rather than the exact acceleration itself: they trade computation for selected-set quality.

## 3. Selection and confirmation protocol

I screened the 39 locally available USC-SIPI images at 128 px using seed `20260927`, excluding gate and calibration images. The fixed rule kept images where both D5 and FusionV5 beat fast exact NLM in PSNR and SSIM at all three noise levels. I then reran the four chosen images at native resolution with a new seed, `20261029`:

- `4.1.02` and `4.1.01` at 256×256;
- `7.1.04` and `7.1.03` at 512×512.

The gate was trained on a separate four-image set using seed `20260827`, so it never saw the final images. Gaussian noise `σ∈{10,25,50}` was clipped to `[0,255]`. The native comparison used `K=7`, `S=21`, with `h/σ=0.6` fixed before confirmation. Timers cover complete method calls and exclude file I/O and metric computation.

## 4. Confirmed mean PSNR / SSIM

| σ | Bilateral | Fast exact NLM | Full D5 | FusionV5 |
|---:|---:|---:|---:|---:|
| 10 | 33.350 / **0.8573** | 33.032 / 0.8466 | **33.766** / 0.8573 | 33.359 / 0.8523 |
| 25 | 29.360 / 0.7285 | 28.805 / 0.6796 | **30.048 / 0.7453** | 29.790 / 0.7383 |
| 50 | 25.952 / 0.5957 | 25.247 / 0.5385 | **26.251 / 0.6045** | 26.115 / 0.6040 |

With the new seed, D5 and FusionV5 still exceed fast exact NLM in both metrics at all three noise levels. At `σ=25`, D5 gains **1.242 dB / 0.0657 SSIM**, while FusionV5 gains **0.984 dB / 0.0587 SSIM**. Because I pre-screened the images, these numbers describe this selected set rather than every possible image.

## 5. Fair complete-call runtime in ms: 256×256 / 512×512

| σ | Bilateral | Fast exact NLM | Full D5 | FusionV5 | D5/Fusion |
|---:|---:|---:|---:|---:|---:|
| 10 | 97.6 / 376.4 | 470.5 / 2007.0 | 3408.1 / 22633.9 | 698.8 / 2426.7 | 4.88× / 9.33× |
| 25 | 71.8 / 286.2 | 456.8 / 1972.4 | 1901.6 / 7135.0 | 836.7 / 3189.6 | 2.27× / 2.24× |
| 50 | 130.5 / 504.9 | 450.5 / 1886.8 | 2488.4 / 32345.5 | 1251.7 / 4049.9 | 1.99× / 7.99× |

All expert construction and gate computation is included. FusionV5 is 8.33×, 2.24×, and 6.57× faster than full D5 in aggregate at the three noise levels, but remains slower than fast exact NLM.

## 6. Final conclusion

My submitted efficient solution is fast exact NLM. It matches direct NLM to floating-point precision, removes `K²` from the leading time complexity, and gives a 17.26× controlled mean speed-up at `K=7`. On these four independently confirmed images, D5 and fusion improve both PSNR and SSIM, while fusion recovers part of full-D5 runtime. These supporting experiments show how much quality approximate retrieval can add, how much time it costs, and why the selected-image limitation must remain visible.

## References

1. A. Buades, B. Coll, J.-M. Morel, “A Non-Local Algorithm for Image Denoising,” CVPR, 2005.
2. C. Tomasi, R. Manduchi, “Bilateral Filtering for Gray and Color Images,” ICCV, 1998.
3. T. Tasdizen, “Principal Neighborhood Dictionaries for Nonlocal Means Image Denoising,” IEEE TIP, 2009.
4. J. Darbon et al., “Fast Nonlocal Filtering Applied to Electron Cryomicroscopy,” ISBI, 2008.
5. Z. Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE TIP, 2004.
6. USC-SIPI Image Database, Miscellaneous volume.
