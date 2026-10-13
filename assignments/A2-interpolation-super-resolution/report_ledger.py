"""Coverage ledger for the report: (family, status, artefact relative to A2_upscale/, conclusion). Each row summarises a completed record."""

LEDGER = [
    ("Interpolation orders / kernels", "completed", "orders 0–5, Keys a=−0.5/−0.75, Mitchell, Lanczos-3/4 (§3; pilot A1_main_table)", "quality rises with order, gap shrinks with scale; kernel must be named"),
    ("Anti-aliasing / smoothing choice", "completed (sensitivity)", "B1 prefilters, B2 paths, B8 Lanczos downscaler, B9 transfer functions", "lighter anti-alias raises interpolation PSNR; P1 stays frozen for the study"),
    ("Direct vs recursive enlargement", "completed", "B3 cascades (216 sequences), WGL-RP multiscale", "linear cascades ≤0.02 dB; bilinear cascade −0.18 dB"),
    ("Gaussian / Laplacian analysis", "completed", "A0 calibration, C Laplacian codec, B9-T1", "cascaded σ=1 equals single σ_eff; Laplacian round trip exact"),
    ("IBP and exact projection", "completed", "B4, M1, B10 oblique", "projection never increased MSE; IBP-20 = oblique consistent projection"),
    ("Edge-aware / adaptive interpolation", "completed (negative)", "B4 JBU, B4b NEDI, oracle router", "positive-weight adaptive methods lose on anti-aliased input"),
    ("Pretrained SR (35-model zoo)", "completed", "D1 (P1/BIC/L3), M2 baselines, this study", "under P1 raw pretrained nets score below IBP-20; exact projection adds ≈0.85–0.9 dB to SwinIR-M / HAT-L at 8× (fresh, assignment)"),
    ("GAN / perceptual SR", "completed (fidelity scope)", "D1 Real-ESRGAN, Real-HAT-GAN", "lower PSNR by design (perception–distortion); out of final scope"),
    ("Task-trained networks", "completed", "M2–M2g, final_scales", "compact LR-space CNN + projection is the deployed family"),
    ("Internal self-similarity / ZSSR", "completed (negative)", "D4 ZSSR", "8× 24.60 dB at ~80 s per image; below the trained core"),
    ("Width / depth / context / refinement", "completed", "M2b, M2d", "no gain beyond w96 d12 on the pool; pooled context and HR refinement neutral"),
    ("Data diversity and source balancing", "completed", "M2e (+DIV2K), M2g, BSD68 pool test", "data was the limiting factor; group-balanced sampling adopted"),
    ("Projected vs ordinary loss", "completed", "M2f λ experiment", "plain MSE (λ=1) best; λ=0 −0.29 dB val"),
    ("SRD spatial residual decoder", "completed (rejected)", "run_srd.py, SRD.jsonl", "no validation gain; not deployed"),
    ("Kernel-mapping / fine-tuned pretrained on P1", "prototype negative / not run", "sota_kmp prototype; run_sota_finetune.py written, not executed", "unmeasured beyond prototype; would narrow the pretrained gap"),
    ("WGL-RP learned residual pyramid", "completed (rejected on rule R4)", "PROPOSAL_WGL_RP.md, F_wglrp_acceptance.json", "cascade beat single shot but failed acceptance"),
    ("Haar / compression", "completed (measured bytes)", "Track C, C4/C5 RD, C7 learned codecs", "downscale+SR is a poor codec; prediction saves 10–13% bits in a Haar codec"),
    ("Projected residual codec (M4)", "mathematics only", "method note §5", "future work; no byte measurement"),
]
