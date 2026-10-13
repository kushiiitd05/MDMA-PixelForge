# ZG-ND-SIFT — Zero-Gaussian Nonlinear-Diffusion SIFT

**CSE340 Digital Image Processing · Mid-semester examination**

SIFT rebuilt with **no Gaussian kernel anywhere**. The three places where classroom SIFT uses a Gaussian, plus the Gaussian-derived DoG detector, are each replaced with a published non-Gaussian alternative:

| # | Stage | Classroom SIFT | ZG-ND-SIFT |
|---|---|---|---|
| 1 | Scale-space | Gaussian G(x, y, σ) | Perona–Malik nonlinear diffusion, rational (G2) conductivity, κ from the raw gradient |
| 2 | Detector | Difference of Gaussians | scale-normalised Hessian determinant σ⁴(LxxLyy − Lxy²) (KAZE) |
| 3 | Orientation window | circular Gaussian, σ = 1.5 × scale | radial Bartlett (triangular) taper |
| 4 | Descriptor window | Gaussian, σ = half region | separable 2-D Bartlett taper |

The rest of the pipeline is the classroom algorithm:
- 26-neighbour 3-D extrema;
- quadratic Taylor localisation;
- trace²/det edge rejection;
- 36-bin orientation histogram with the 80 % rule;
- a 4×4×8 = 128-D descriptor (normalise → clip 0.2 → renormalise).

Test image: `skimage.data.camera()` (512×512), used as provided.

## Run

```bash
pip install numpy scipy scikit-image matplotlib jupyterlab
jupyter lab ZG_ND_SIFT.ipynb      # Run All; prints its own total runtime at the end
```

## Files

| File | What |
|---|---|
| `ZG_ND_SIFT.ipynb` | Executed notebook: every stage, six required figures + two zoomed supplementary figures |
| `2023296_Kush_Tokas_ZG_ND_SIFT_Report.pdf` | Submitted report |
