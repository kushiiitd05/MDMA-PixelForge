"""Generate and execute A2_Interpolation_SR.ipynb (TA demonstration notebook).

Usage (from submission/, eda311 first on PATH):  python make_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Assignment 2 — Interpolation-Based Super-Resolution\n"
       "Four 512×512 grayscale images → Gaussian-pyramid downsampling to 256/128/64 → interpolation back to 512 with "
       "orders 0–5 (+ cubic convolution, Lanczos) → PSNR, SSIM, runtime.\n\n"
       "All operators live in `code/a2_ops.py`. Metrics and timing live in `code/a2_eval.py`."),
    code("import sys, time\nsys.path.insert(0, 'code')\nimport numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "import a2_ops as ops, a2_eval as ev, run_assignment as ra\n"
         "pd.set_option('display.precision', 3)\nimages = ra.load()\nlist(images)"),
    md("## 1. Prelude: nearest neighbour on the PDF example, plus a hand-checkable bilinear value"),
    code("toy = np.array([[40., 120.], [180., 220.]])\n"
         "print('nearest 2x:\\n', ops.upsample(toy, 2, 'nearest', 'centered'))\n"
         "bl = ops.upsample(toy, 2, 'bilinear', 'centered')\n"
         "hand = .75*.75*40 + .75*.25*120 + .25*.75*180 + .25*.25*220   # output (1,1) -> input (0.25, 0.25)\n"
         "print('bilinear 2x:\\n', bl.round(2)); print('hand value at (1,1):', hand)"),
    md("## 2. Coordinate convention check\n"
       "LR pixel *i* covers HR block [s·i, s·i+s), so output *u* maps to x=(u+0.5)/s−0.5. "
       "Nearest neighbour must equal block copy, and bilinear must match OpenCV."),
    code("import cv2\nr = np.random.default_rng(0).uniform(0, 255, (64, 64))\n"
         "for s in (2, 4, 8):\n"
         "    assert np.array_equal(ops.upsample(r, s, 'nearest', 'centered'), np.kron(r, np.ones((s, s))))\n"
         "    assert np.allclose(ops.upsample(r, s, 'bilinear', 'centered'), cv2.resize(r.astype(np.float32), (64*s, 64*s), interpolation=cv2.INTER_LINEAR), atol=1e-3)\n"
         "print('convention checks passed')"),
    md("## 3. Downsampling (same procedure for every image): Gaussian σ=1 per ×2 stage, recursive"),
    code("lrs = ra.make_lrs(images)\nfig, ax = plt.subplots(4, 4, figsize=(10, 10))\n"
         "for r_, (n, im) in enumerate(images.items()):\n"
         "    ax[r_, 0].imshow(im, cmap='gray', vmin=0, vmax=255); ax[r_, 0].set_title(n, fontsize=8)\n"
         "    for c_, s in enumerate((2, 4, 8), 1):\n"
         "        ax[r_, c_].imshow(lrs[(n, s)], cmap='gray', vmin=0, vmax=255, interpolation='nearest'); ax[r_, c_].set_title(f'{512//s}x{512//s}', fontsize=8)\n"
         "[a.axis('off') for a in ax.ravel()]; plt.tight_layout(); plt.show()"),
    md("## 4. Methods"),
    code("pd.DataFrame([(m, ops.METHODS[m][0], ops.METHODS[m][1]) for m in ra.METHODS], columns=['key', 'family', 'order/degree'])"),
    md("## 5. Super-resolve everything and score (PSNR, SSIM vs original 512×512)"),
    code("t0 = time.perf_counter(); q = ra.quality(images, lrs); print(f'{len(q)} reconstructions scored in {time.perf_counter()-t0:.1f}s')\n"
         "q.pivot_table(index='label', columns='scale', values=['psnr', 'ssim'], aggfunc='mean').reindex(ra.LABELS.values(), level=0)"),
    code("q.pivot_table(index=['image', 'label'], columns='scale', values='psnr').round(2)"),
    md("## 6. Runtime (warm, single thread, median; interpolation only)"),
    code("rt = ra.runtime(lrs)\nrt.pivot_table(index=['track', 'label'], columns='scale', values='ms_median').round(3)"),
    md("## 7. Figures"),
    code("for f in ('fig1_originals_and_lr', 'fig4_quality_vs_scale', 'fig5_runtime_vs_psnr', 'fig6_error_maps', 'fig7_step_edge_profile'):\n"
         "    pass\nra.fig_quality_vs_scale(q); ra.fig_runtime_vs_quality(q, rt); ra.fig_artifacts(images, lrs); ra.fig_sr_grids(images, lrs)\n"
         "from IPython.display import Image, display\n"
         "for f in ('fig4_quality_vs_scale', 'fig5_runtime_vs_psnr', 'fig2_sr_crops_4.2.06', 'fig6_error_maps', 'fig7_step_edge_profile'):\n"
         "    display(Image(filename=f'figures/{f}.png', width=900))"),
    md("## 8. Live demo cell — change image / scale / method"),
    code("name, s, method = '4.2.03.tiff', 4, 'spline3'\n"
         "lr = lrs[(name, s)]\nt0 = time.perf_counter(); out = ops.upsample(lr, s, method, 'centered'); dt = (time.perf_counter()-t0)*1e3\n"
         "sc = ev.score(images[name], out)\n"
         "fig, ax = plt.subplots(1, 3, figsize=(12, 4))\n"
         "ax[0].imshow(lr, cmap='gray', vmin=0, vmax=255, interpolation='nearest'); ax[0].set_title(f'input {lr.shape}')\n"
         "ax[1].imshow(np.clip(out, 0, 255), cmap='gray', vmin=0, vmax=255); ax[1].set_title(f'{method} {s}x  PSNR {sc[\"psnr\"]:.2f}  SSIM {sc[\"ssim\"]:.3f}  {dt:.1f} ms')\n"
         "ax[2].imshow(images[name], cmap='gray', vmin=0, vmax=255); ax[2].set_title('original')\n"
         "[a.axis('off') for a in ax]; plt.show()"),
    md("## 9. Research extension (separate from the required study): consistent null-space SR\n"
       "x̂ = B y + P F(y). F is a compact LR-space CNN (DirectLR w96 d12 + bicubic skip), and B, P come from the exact P1 operator. "
       "Weights are listed in `extension/model_manifest.json` (SHA-256 checked). Roles were selected on validation images only; see report §6–7."),
    code("import json, subprocess\nroles = pd.read_csv('extension/results/role_table.csv')\n"
         "roles[roles.scope == 'overall'][['scale', 'role', 'method_id', 'val_psnr', 'val_ssim', 'latency_median_ms', 'x_torch_bicubic', 'assignment_psnr', 'assignment_ssim', 'urban100c_psnr', 'urban100c_ssim']]"),
    code("r = subprocess.run([sys.executable, 'code/run_extension_demo.py'], capture_output=True, text=True)\nprint(r.stdout[-3000:], r.stderr[-2000:])\nassert r.returncode == 0"),
    code("from IPython.display import Image, display\ndisplay(Image(filename='extension/results/fig_tradeoff_val.png', width=950))\n"
         "display(Image(filename='extension/figures/ext_crops_x8.png', width=950))"),
    md("### Live extension cell — change image / scale / role"),
    code("import sr_extension as X\nname, s, role = '4.2.06.tiff', 8, 'psnr'\n"
         "net, entry = X.load_model(s, role)\nlr = lrs[(name, s)]\nX.super_resolve(lr, s, net)\n"
         "t0 = time.perf_counter(); out = X.super_resolve(lr, s, net, entry['projection']); dt = (time.perf_counter() - t0) * 1e3\n"
         "base = ops.upsample(lr, s, 'spline5', 'centered')\n"
         "fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))\n"
         "for a, (t, im) in zip(ax, (('quintic B-spline %.2f dB' % ev.score(images[name], base)['psnr'], base), (f\"{entry['method_id']}  {ev.score(images[name], out)['psnr']:.2f} dB / {ev.score(images[name], out)['ssim']:.3f}  {dt:.1f} ms\", out), ('original', images[name]))):\n"
         "    a.imshow(np.clip(im, 0, 255), cmap='gray', vmin=0, vmax=255); a.set_title(t, fontsize=8); a.axis('off')\n"
         "plt.show(); print('LR residual RMS (grey levels):', X.lr_residual(out, lr, s))"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (eda311)", "language": "python"}})
ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
out = ROOT / "A2_Interpolation_SR.ipynb"
nbf.write(nb, out)
print("wrote", out)
