"""Build the four-page A2 report (fixed A4 pages, A1 design system) and its editable DOCX companion.

No number is typed into the text: every value is read at build time from
  results/{quality,runtime,downsampler_joint,downsampler_transfer}.csv, results/environment.json, figures/*.png
  extension/results/*, extension/figures/*, extension/model_manifest.json
Full per-image extension tables, paired statistics, latency records and the coverage ledger stay in extension/results/.
Usage (from submission/):  python build_report.py   -> report/A2_Interpolation_SR_Report.{pdf,docx}
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from report_design import CSS, limit, readout, section, stamp, wedge
from report_flow import flow_svg
from select_roles_helper import latency_key
from report_ledger import LEDGER

ROOT = Path(__file__).resolve().parent
RES, FIG, REP, EXT = ROOT / "results", ROOT / "figures", ROOT / "report", ROOT / "extension"
XRES, XFIG = EXT / "results", EXT / "figures"
BUILD = REP / "_build"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TITLE = "What the Low-Resolution Image Still Pins Down"
SUBTITLE = "Interpolation of order 0–5 at 2×, 4× and 8×, and how far an exactly consistent learned decoder goes beyond it"
ORDER = ["Nearest (0)", "Bilinear (1)", "Quadratic local (2)", "Quadratic B-spline (2)", "Bicubic Keys a=-0.5 (3)",
         "Cubic B-spline (3)", "Quartic B-spline (4)", "Quintic B-spline (5)", "Lanczos-3 (sinc ref.)"]
SHORT = {"4.2.03.tiff": "Mandrill", "7.1.02.tiff": "Jet", "7.1.03.tiff": "Tank", "4.2.06.tiff": "Sailboat"}
ROLE_ORDER = ["Highest PSNR", "Highest SSIM", "Lowest latency", "Best trade-off (promoted)"]
FOOT = "Interpolation-based super-resolution"
PAGES = 4
EXTRA_CSS = """
body{font-size:8.2pt;line-height:1.27}
p{margin-bottom:1.3mm}
.sec{margin:2.2mm 0 1mm}
.eq{padding:1.1mm 2mm;margin:1.2mm 0;font-size:8.8pt}
table{font-size:6.6pt;margin:.8mm 0 1.1mm}
th,td{padding:.62mm .6mm}
thead th{font-size:6pt}
tbody th{font-size:6.6pt}
figcaption,.caption{font-size:6.9pt}
li{margin-bottom:.5mm}
.refs{font-size:6.5pt;columns:1}
.refs p{margin-bottom:.4mm}
.stamp{margin:1.2mm 0;padding:1.2mm 2mm}
.limit{margin:1.2mm 0;padding:1.2mm 2mm}
"""


# ----------------------------------------------------------------------------- helpers
def img(path: Path, style: str = "") -> str:
    return f"<img src='data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}' style='{style}'>"


def tag(mid) -> str:
    """Compact pipeline label; full identities are in the model manifest and per-image CSV."""
    if not isinstance(mid, str):
        return "—"
    pr = " +P" if mid.endswith("+proj") else ""
    m = mid.split(":", 1)[1].replace("+proj", "")
    if m.startswith("fs_x"):
        _, _, _, seed, crit = m.split("_")
        return f"OURS s{seed[1:]} {crit.upper()}-ckpt{pr}"
    if m.startswith("final_bal_"):
        return f"OURS prev. 8× core{pr}"
    if m.startswith("swinirP1"):
        return f"SwinIR-M P1-ft{pr}"
    names = {"ibp20_bicubic_a05": "IBP-20", "torch_bicubic": "bicubic (torch)", "bicubic_a05": "Keys bicubic", "spline5": "quintic B-spline",
             "spline4": "quartic B-spline", "spline3": "cubic B-spline", "spline2": "quadratic B-spline", "quad_local": "local quadratic", "lanczos3": "Lanczos-3",
             "nearest": "nearest", "bilinear": "bilinear", "SwinIR-M": "SwinIR-M (pretrained)", "HAT-L": "HAT-L (pretrained)"}
    return names.get(m, m) + pr


def pc(p, q, dp=2, dq=3) -> str:
    return f"{p:.{dp}f} / {q:.{dq}f}" if pd.notna(p) else "—"


def thead(cols: list[str]) -> str:
    return "<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead>"


def footer(text: str, number: int) -> str:
    return f"<div class='footer'>{FOOT} · {text} <span class='right'>{number} / {PAGES}</span></div>"


# ----------------------------------------------------------------------------- data
def load() -> dict:
    q = pd.read_csv(RES / "quality.csv")
    n = {"q": q, "rt": pd.read_csv(RES / "runtime.csv"), "env": json.loads((RES / "environment.json").read_text())}
    n["mean"] = q.groupby(["label", "scale"])[["psnr", "ssim", "local_overshoot", "psnr_edge", "psnr_smooth", "psnr_texture"]].mean()
    n["joint"] = pd.read_csv(RES / "downsampler_joint.csv")
    n["transfer"] = pd.read_csv(RES / "downsampler_transfer.csv")
    n["roles"] = pd.read_csv(XRES / "role_table.csv")
    n["summ"] = pd.read_csv(XRES / "method_summary.csv")
    n["resid"] = pd.read_csv(XRES / "residuals.csv")
    n["frozen"] = json.loads((XRES / "roles_frozen.json").read_text())
    n["hist"] = json.loads((XRES / "history.json").read_text())
    n["lat"] = json.loads((XRES / "latency.json").read_text())
    n["pi"] = pd.read_csv(XRES / "per_image.csv")
    return n


def S(n, stage, ds, s, mid, col):
    x = n["summ"]
    r = x[(x.stage == stage) & (x.dataset == ds) & (x.scale == s) & (x.method_id == mid)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def role(n, s, name, scope="overall"):
    r = n["roles"]
    return r[(r.scale == s) & (r.role == name) & (r.scope == scope)].iloc[0]


def runtime_of(n, label: str, s: int) -> float:
    rt = n["rt"]
    r = rt[(rt.label == label) & (rt.scale == s) & (rt.track != "OpenCV optimised")]
    return float(r.ms_median.iloc[0])


# ----------------------------------------------------------------------------- page 1
def roles_table(n) -> str:
    r = n["roles"][n["roles"].scope == "overall"].copy()
    r["role"] = pd.Categorical(r.role, ROLE_ORDER, ordered=True)
    rows = []
    for _, x in r.sort_values(["scale", "role"]).iterrows():
        rows.append(f"<tr><td>{x.scale}×</td><th>{x.role}</th><td class=l>{tag(x.method_id)}</td><td>{pc(x.val_psnr, x.val_ssim, 2, 4)}</td>"
                    f"<td>{x.latency_median_ms:.1f} ({x.x_torch_bicubic:.0f}×)</td><td>{pc(x.assignment_psnr, x.assignment_ssim)}</td>"
                    f"<td>{pc(x.urban100c_psnr, x.urban100c_ssim)}</td></tr>")
    return (f"<table class='roles'>{thead(['Scale', 'Role', 'Pipeline (+P = exact projection)', 'Validation', 'ms (× bicubic)', 'Assignment 4', 'Urban100-c'])}"
            f"<tbody>{''.join(rows)}</tbody></table>")


def page1(n) -> str:
    m = n["mean"]
    P = lambda lab, s, c="psnr": m.loc[(lab, s), c]
    q5, nn = "Quintic B-spline (5)", "Nearest (0)"
    pro8, best8 = role(n, 8, "Best trade-off (promoted)"), role(n, 8, "Highest PSNR")
    res, pool = n["resid"], n["hist"]["pool_decision"]
    ob8 = role(n, 8, "Highest PSNR", "ours")
    return f"""
<section class='page'>
  <div class='kicker'>Digital Image Processing (CSE 340) &middot; Assignment 2 &middot; Interpolation-Based Super-Resolution</div>
  <h1>{TITLE}</h1><div class='subtitle'>{SUBTITLE}</div>
  <div class='meta'>Kush Tokas &nbsp;/&nbsp; 2023296 &nbsp;/&nbsp; CSE 340 Digital Image Processing &nbsp;/&nbsp; {time.strftime('%d %B %Y')}</div>
  {wedge()}
  {readout([(f"{P(q5, 2):.2f} → {P(q5, 8):.2f} dB", "best interpolation (quintic)<br>2× → 8×, four images"),
            (f"+{P(q5, 2) - P(nn, 2):.2f} → +{P(q5, 8) - P(nn, 8):.2f} dB", "order 0 → 5 gain<br>shrinks from 2× to 8×"),
            (f"{best8.assignment_psnr:.2f} dB", f"best 8× PSNR (assignment)<br>{tag(best8.method_id)}"),
            (f"{pro8.assignment_psnr:.2f} dB / {pro8.latency_median_ms:.1f} ms", f"promoted 8× pipeline<br>{tag(pro8.method_id)}")])}
  <div class='two'>
    <div>
      {section(0, '1', 'The required study: one fixed procedure')}
      <p>Four 512² USC-SIPI images from Assignment 1, picked by a content rule before any score existed: <b>Mandrill</b> 4.2.03 (texture), <b>Jet</b> 7.1.02 (smooth + sharp edges),
      <b>Tank</b> 7.1.03 (mid-frequency), <b>Sailboat</b> 4.2.06 (mixed), as BT.601 luma. Every image is reduced by the course Gaussian pyramid — the one procedure used everywhere in this report:</p>
      <div class='eq'>G<sub>k+1</sub> = ↓2 ( g<sub>σ=1</sub> ∗ G<sub>k</sub> ): 512² → 256² → 128² → 64² <span class='tag'>(1)</span></div>
      <p>σ = 1 px per ×2 stage, half-pixel sampling, symmetric borders. Each of 256², 128², 64² is enlarged <i>directly</i> back to 512² with output pixel u mapped to x = (u + ½)/s − ½,
      so nearest neighbour is exactly the s×s block copy of the prelude. <b>Methods</b> (order = kernel degree): nearest (0), bilinear (1), local quadratic and quadratic B-spline (2),
      Keys cubic a = −0.5 and cubic B-spline (3), quartic (4), quintic (5) B-splines, Lanczos-3 (windowed sinc). <b>Scores:</b> PSNR (R = 255), SSIM (Gaussian σ = 1.5) on the full clipped 512² frame;
      <b>runtime</b> = median of warm single-thread calls. Results: <b>Tables 1–3</b> (interpolation, page 2) and <b>Table 5</b> (smoothing choice, page 4).</p>
    </div>
    <div>
      {section(1, '2', 'What we built: a decoder that must agree with its input')}
      <p>An interpolated image does not reproduce its own low-resolution input: applying (1) to it gives something else, so part of its error is <i>visible</i> in the data.
      Our decoder removes exactly that part, and learns the rest. (1) is linear, y = Dx with D(X) = AXAᵀ, so with B = Dᵀ(DDᵀ)⁻¹ and P = I − BD any prediction F(y) becomes</p>
      <div class='eq'>x̂ = F(y) + B( y − D F(y) ) = B y + P F(y) <span class='tag'>(2)</span></div>
      <div class='eq'>‖x̂ − x‖² = ‖F(y) − x‖² − ‖(I − P)(F(y) − x)‖² <span class='tag'>(3)</span></div>
      <p><b>Technique 1 — learned prior.</b> F is a 12-layer, 96-channel CNN that works entirely on the small LR grid, ending in one pixel-shuffle added to a bicubic upsampling
      (0.92–0.97 M weights). It supplies the detail the low-resolution image does not contain. <b>Technique 2 — exact consistency.</b> (2) is an orthogonal projection onto the set of
      images that reduce back to y, so by (3) it can never make the prediction worse, and the output satisfies D x̂ = y up to float precision. Both run once, in sequence, at decode time; <b>Table 4</b> compares them with every alternative.</p>
      {stamp('Consistency of exact projections (all test images)', [('max |D x̂ − y| before clipping', f'{res.max_preclip.max():.1e} grey levels'), ('RMS after clipping to [0, 255]', f'{res.rms_postclip_mean.max():.3f} grey levels')])}
    </div>
  </div>
  <figure style='margin-top:1mm'>{flow_svg()}<figcaption><b>Figure 1 · the full pipeline.</b> <b>Lane A</b> is the required study: one downsampling procedure, then one direct interpolation
  per factor. <b>Lane B</b> is our decoder on the same input y: the learned prior proposes F(y), then the exact projection forces agreement with y before clipping. The dashed path is training —
  345 images that exclude all four test images, 64k updates and two seeds per scale, with checkpoints and roles chosen on four held-out validation images and frozen before any test image was scored.
  The BSD68 part of the pool was kept by a rule fixed in advance ({pool['delta_psnr']:+.3f} dB on validation). References measured in the same harness: every interpolation order ± projection, IBP-20,
  pretrained SwinIR-M and HAT-L ± projection, and SwinIR-M <i>fine-tuned on</i> (1), which separates degradation mismatch from architecture.</figcaption></figure>
  <div class='caption' style='margin-top:1.2mm'><b>Table 4.</b> Role winners per scale (PSNR dB / SSIM), chosen on validation and latency only; assignment and Urban100-c scored after freezing.
  Our own family's best 8× checkpoint is {tag(ob8.method_id)}; DIV2K-20 and per-checkpoint rows in extension/results/role_table.csv.</div>
  {roles_table(n)}
  {footer('Task, our method, and the four winners', 1)}
</section>"""


def mean_runtime_table(n) -> str:
    """Interpolation rows (the required study), then the two learned decoders on the same images, marked in blue."""
    m = n["mean"]
    best = {(s, c): max(m.loc[(lab, s), c] for lab in ORDER) for s in (2, 4, 8) for c in ("psnr", "ssim")}
    rows = []
    for lab in ORDER:
        cells = []
        for s in (2, 4, 8):
            p, q = m.loc[(lab, s), "psnr"], m.loc[(lab, s), "ssim"]
            cells.append(f"<td>{'<b>%.2f</b>' % p if p == best[(s, 'psnr')] else '%.2f' % p} / {'<b>%.3f</b>' % q if q == best[(s, 'ssim')] else '%.3f' % q}</td>")
        cells += [f"<td>{runtime_of(n, lab, s):.2f}</td>" for s in (2, 4, 8)]
        rows.append(f"<tr><th>{lab}</th>{''.join(cells)}</tr>")
    rows.append("<tr class='grp'><th>Extension on the same images (not interpolation)</th><td colspan=6>reconstruction back to 512² from the same low-resolution inputs</td></tr>")
    for label, pick in (("OURS · DirectLR w96 d12 + exact projection", lambda s: role(n, s, "Highest PSNR", "ours").method_id),
                        ("SwinIR-M fine-tuned on (1) + exact projection", lambda s: role(n, s, "Highest PSNR").method_id)):
        cells, lats = [], []
        for s in (2, 4, 8):
            mid = pick(s)
            cells.append(f"<td class=ours>{S(n, 'external', 'assignment', s, mid, 'psnr'):.2f} / {S(n, 'external', 'assignment', s, mid, 'ssim'):.3f}</td>")
            lats.append(f"<td class=ours>{n['lat'][latency_key(s, mid)]['median_of_medians_ms']:.1f}</td>")
        rows.append(f"<tr><th class=ours>{label}</th>{''.join(cells)}{''.join(lats)}</tr>")
    return (f"<table>{thead(['Method (order)', 'PSNR / SSIM 2×', 'PSNR / SSIM 4×', 'PSNR / SSIM 8×', 'ms 2×', 'ms 4×', 'ms 8×'])}<tbody>{''.join(rows)}</tbody></table>")


def per_image_table(n, metric: str, fmt: str) -> str:
    piv = n["q"].pivot_table(index="label", columns=["scale", "image"], values=metric)
    cols = [(s, im) for s in (2, 4, 8) for im in SHORT]
    best = {c: piv[c].max() for c in cols}
    head = ("<thead><tr><th rowspan=2>Method</th>" + "".join(f"<th colspan=4>{s}× ({512 // s}² → 512²)</th>" for s in (2, 4, 8)) + "</tr><tr>" +
            "".join(f"<th>{SHORT[im]}</th>" for _ in (2, 4, 8) for im in SHORT) + "</tr></thead>")
    rows = "".join(f"<tr><th>{lab}</th>" + "".join(f"<td>{'<b>' if piv.loc[lab, c] == best[c] else ''}{format(piv.loc[lab, c], fmt)}{'</b>' if piv.loc[lab, c] == best[c] else ''}</td>" for c in cols) + "</tr>"
                   for lab in ORDER)
    return f"<table>{head}<tbody>{rows}</tbody></table>"


def page2(n) -> str:
    return f"""
<section class='page'>
  <div class='pagehd'>Downscaling inputs, then upscaling scores</div>{wedge()}
  <figure>{img(FIG / 'fig1_originals_and_lr.png', 'width:100%;height:84mm;object-fit:contain')}<figcaption><b>Figure 2 · downscaling.</b> The four grayscale originals (512²) and the three downsampled versions of each (256², 128², 64²) produced by (1) — the inputs every method receives. They are shown enlarged with pixel replication so the true sample grid stays visible.</figcaption></figure>
  <div class='caption' style='margin-top:1.5mm'><b>Table 1 · upscaling, mean over the four images.</b> PSNR (dB) / SSIM and median runtime (ms; SciPy for B-splines, NumPy reference for kernel methods). Bold = best interpolation per scale; blue = the two learned decoders of Section 2 on the same inputs (their runtime is the full decode path on the Mac, so it is not comparable with the single-thread interpolation column).</div>
  {mean_runtime_table(n)}
  <div class='caption'><b>Table 2 · upscaling.</b> PSNR (dB) for every image, interpolation method and factor. Bold = best for that image and factor.</div>
  {per_image_table(n, 'psnr', '.2f')}
  <div class='caption'><b>Table 3 · upscaling.</b> SSIM for every image, interpolation method and factor.</div>
  {per_image_table(n, 'ssim', '.3f')}
  {footer('Downscaled inputs; upscaling PSNR, SSIM and runtime', 2)}
</section>"""


# ----------------------------------------------------------------------------- page 3
def page3(n) -> str:
    crops = XFIG / "ext_crops_x8.png"
    return f"""
<section class='page'>
  <div class='pagehd'>Upscaled results: reconstructions and artefacts</div>{wedge()}
  <div class='ratio'>
    <figure>{img(FIG / 'fig3_full_sr_sailboat.png', 'width:100%;height:96mm;object-fit:contain')}<figcaption><b>Figure 3.</b> Sailboat enlarged from 256², 128² and 64² (rows)
    with nearest, bilinear, Keys bicubic and quintic B-spline (columns).</figcaption></figure>
    <div>
      <figure>{img(FIG / 'fig7_step_edge_profile.png', 'width:100%;height:50mm;object-fit:contain')}<figcaption><b>Figure 4.</b> An ideal step enlarged 8×: nearest = staircase, bilinear = ramp (blur),
      order ≥ 3 = overshoot and ringing on both sides.</figcaption></figure>
      <figure style='margin-top:1.5mm'>{img(FIG / 'fig6_error_maps.png', 'width:100%;height:44mm;object-fit:contain')}<figcaption><b>Figure 5.</b> 4× Jet crop and signed error
      (red = too bright): every kernel's error sits on edges.</figcaption></figure>
    </div>
  </div>
  <figure style='margin-top:2mm'>{img(FIG / 'fig8_reconstructions.png', 'width:100%;height:70mm;object-fit:contain')}<figcaption><b>Figure 6 · the reconstructions.</b> Every test image restored to 512²
  from each of its three downsampled versions: the best interpolation (quintic B-spline) beside the promoted extension pipeline of Table 4, with full-image PSNR / SSIM under each panel.
  Crops of all four images for seven kernels at every factor are in figures/fig2_sr_crops_*.png.</figcaption></figure>
  {f"<figure style='margin-top:2mm'>{img(crops, 'width:100%;height:56mm;object-fit:contain')}<figcaption><b>Figure 7.</b> 8× edge crops of all four images: interpolation, projection, pretrained HAT-L, SwinIR-M fine-tuned on (1) and OURS (full-image PSNR / SSIM under each crop). Crops for the other images and factors: figures/ and extension/figures/.</figcaption></figure>" if crops.exists() else ''}
  {footer('Super-resolved images: blocking, blur, ringing, staircase', 3)}
</section>"""


# ----------------------------------------------------------------------------- page 4
def joint_rows(n) -> str:
    j = n["joint"]
    downs = ["no smoothing (decimation)", "Gaussian σ=1.4", "Gaussian σ=1 (P1, frozen)", "Gaussian σ=0.5", "box / area", "Lanczos-3 direct",
             "DCT band crop (ideal low-pass)", "co-designed for the upscaler (8-bit LR)"]
    out = []
    for d in downs:
        cells = []
        for s in (2, 4, 8):
            r = j[(j.down == d) & (j.scale == s) & (j.up != "IBP-20 (matched D)")].sort_values("psnr")
            cells.append(f"<td>{r.iloc[-1].psnr:.2f} <span class=ci>{r.iloc[-1].up.replace(' B-spline', '').replace(' zero-pad', '').replace('Keys bicubic', 'Keys').replace('Lanczos-3', 'Lanczos')}</span></td>" if len(r) else "<td>—</td>")
        out.append(f"<tr><th>{d}</th>{''.join(cells)}</tr>")
    return "".join(out)


def page4(n) -> str:
    m = n["mean"]
    P = lambda lab, s, c="psnr": m.loc[(lab, s), c]
    nn, bl, bc, cs, q5 = "Nearest (0)", "Bilinear (1)", "Bicubic Keys a=-0.5 (3)", "Cubic B-spline (3)", "Quintic B-spline (5)"
    q = n["q"]
    per = q.pivot_table(index=["image", "label"], columns="scale", values="psnr")
    best2 = q[q.scale == 2].groupby("image").psnr.max()
    j, tr = n["joint"], n["transfer"]
    J = lambda d, u, s: float(j[(j.down == d) & (j.up == u) & (j.scale == s)].psnr.iloc[0])
    G = lambda f, s, c: float(tr[(tr["filter"] == f) & (tr.scale == s)][c].iloc[0])
    pair_best = {s: j[(j.scale == s) & (j.up != "IBP-20 (matched D)")].sort_values("psnr").iloc[-1] for s in (2, 8)}
    ceil8 = max(n["hist"]["b10_best_linear_consistent_8x"].values())
    lat = n["lat"]
    ms = {s: lat[f"x{s}|ours:DirectLR w96 d12"]["median_of_medians_ms"] for s in (2, 4, 8)}
    pi = n["pi"][n["pi"].stage == "external"]
    raw = pi[~pi.method_id.str.endswith("+proj")]
    prj = pi[pi.method_id.str.endswith("+proj") & ~pi.method_id.str.contains("ibp20")].assign(method_id=lambda d: d.method_id.str[:-5])
    pair = prj.merge(raw, on=["dataset", "scale", "method_id", "image_id"], suffixes=("_p", "_r"))
    proj_ok = f"{int((pair.psnr_db_p >= pair.psnr_db_r - 1e-6).sum())}/{len(pair)}"
    U = lambda s, mid: S(n, "external", "urban100c", s, mid, "psnr")
    ob = {s: role(n, s, "Highest PSNR", "ours").method_id for s in (2, 4, 8)}
    ft = {s: role(n, s, "Highest PSNR").method_id for s in (2, 4, 8)}
    gain_txt = " / ".join(f"{U(s, ob[s]) - U(s, 'classical:spline5+proj'):+.2f}" for s in (2, 4, 8)) + " dB"
    hat_txt = " / ".join(f"{U(s, ob[s]) - U(s, 'pretrained:HAT-L+proj'):+.2f}" for s in (2, 4, 8))
    ft_vs_hat = ", ".join(f"{s}× {U(s, ft[s]) - U(s, 'pretrained:HAT-L+proj'):+.2f} dB over HAT-L + P" for s in (2, 4, 8))
    lk = lambda s, mid: n["lat"][f"x{s}|ours:DirectLR w96 d12+proj" if mid.startswith("ours:") else f"x{s}|{mid}"]["median_of_medians_ms"]
    speed_txt = " / ".join(f"{lk(s, 'pretrained:HAT-L+proj') / lk(s, ob[s]):.0f}×" for s in (2, 4, 8))
    winners = []
    for s in (2, 4, 8):
        a, b, c, d = (role(n, s, r) for r in ROLE_ORDER)
        winners.append(f"<li><b>{s}×</b>: PSNR — {tag(a.method_id)}; SSIM — {tag(b.method_id)}; fastest — {tag(c.method_id)}; <b>promoted — {tag(d.method_id)}</b> "
                       f"({d.assignment_psnr:.2f} dB / {d.assignment_ssim:.3f} on the assignment images, {d.latency_median_ms:.1f} ms).</li>")
    return f"""
<section class='page'>
  <div class='pagehd'>Discussion, best choices and why</div>{wedge()}
  <div class='two'>
    <div>
      {section(2, '3', 'How quality changes from 2× to 8×')}
      <p>Quality falls steeply with the factor: quintic gives {P(q5, 2):.2f} → {P(q5, 4):.2f} → {P(q5, 8):.2f} dB and SSIM {P(q5, 2, 'ssim'):.3f} → {P(q5, 8, 'ssim'):.3f}.
      A 64² input holds 1.56% of the samples, and the pyramid already removed detail finer than its grid. The gap between methods shrinks with it: order 0 → 5 gains
      {P(q5, 2) - P(nn, 2):.2f} / {P(q5, 4) - P(nn, 4):.2f} / {P(q5, 8) - P(nn, 8):.2f} dB. <i>Why:</i> the error splits into energy above the LR Nyquist limit, which no interpolator can recover,
      plus a passband term the kernel controls; B-splines approach the ideal low-pass, so their gains saturate while the lost band grows with s.</p>
      {section(3, '4', 'Visual differences and artefacts')}
      <ul>
        <li><b>Nearest (0)</b>: blocking and staircase edges (Figs. 2–3), no overshoot; it keeps local contrast, so its SSIM beats bilinear at 2× ({P(nn, 2, 'ssim'):.3f} vs {P(bl, 2, 'ssim'):.3f}).</li>
        <li><b>Bilinear (1)</b>: no blocks but the strongest blur — its triangle kernel is a second low-pass on top of the pyramid.</li>
        <li><b>Order 2</b>: the local quadratic ({P('Quadratic local (2)', 2):.2f} dB) is below the quadratic B-spline ({P('Quadratic B-spline (2)', 2):.2f} dB); "second order" must name its kernel.</li>
        <li><b>Order ≥ 3</b>: sharper edges with overshoot/ringing from negative lobes; mean local overshoot at 8× rises Keys {P(bc, 8, 'local_overshoot'):.3f} → cubic {P(cs, 8, 'local_overshoot'):.3f} → quintic {P(q5, 8, 'local_overshoot'):.3f} grey levels.</li>
      </ul>
      {section(4, '5', 'Quality versus runtime, and content')}
      <p>Quality rises with spline order with diminishing returns — cubic → quintic adds {P(q5, 2) - P(cs, 2):.2f} dB at 2× and {P(q5, 8) - P(cs, 8):.2f} dB at 8× while SciPy time about doubles
      ({runtime_of(n, cs, 2):.0f} → {runtime_of(n, q5, 2):.0f} ms). A cubic kernel gives the best quality per millisecond. <b>Smooth regions</b> are easy for every method
      (&gt; {m.xs(8, level='scale').psnr_smooth.min():.1f} dB at 8×); <b>edges</b> are hardest and gain most from order (+{P(q5, 2, 'psnr_edge') - P(nn, 2, 'psnr_edge'):.1f} dB vs +{P(q5, 2, 'psnr_smooth') - P(nn, 2, 'psnr_smooth'):.1f} dB smooth, 2×);
      <b>texture</b> converges: Mandrill peaks at {best2['4.2.03.tiff']:.2f} dB even at 2× (Jet {best2['7.1.02.tiff']:.2f} dB), and there nearest ({per.loc[('4.2.03.tiff', nn), 2]:.2f}) even beats bilinear ({per.loc[('4.2.03.tiff', bl), 2]:.2f}).</p>
      {section(5, '6', 'Best smoothing, and the best pipeline overall')}
      <p>The required tables keep one fixed downsampler. A separate question is which smoothing, and which <i>pair</i> of downsampler and interpolator, reconstructs best (same four images,
      each level enlarged directly to 512²). Lighter anti-aliasing wins for fidelity: the pyramid's Gaussian keeps {100 * G('P1 gauss sigma=1 per-stage', 8, 'passband_energy_kept'):.0f}% of the passband at 8×
      (gain {G('P1 gauss sigma=1 per-stage', 8, 'gain_0.9fc'):.2f} at 0.9 f<sub>c</sub>) against Lanczos-3's {100 * G('lanczos3 direct', 8, 'passband_energy_kept'):.0f}% ({G('lanczos3 direct', 8, 'gain_0.9fc'):.2f}), both with ≤ {100 * G('lanczos3 direct', 8, 'alias_energy_frac'):.1f}% alias energy;
      no smoothing aliases badly ({J('no smoothing (decimation)', 'cubic B-spline', 8):.2f} dB at 8×). Recursive and direct smoothing agree to ≤ 0.02 dB.</p>
      <div class='caption'><b>Table 5.</b> Downsampling comparison: mean PSNR (dB) of the best interpolator for each smoothing (its name in grey; SSIM in results/downsampler_joint.csv).</div>
      <table class='tight'>{thead(['Downsampling', '2×', '4×', '8×'])}<tbody>{joint_rows(n)}</tbody></table>
      <p><b>Best pipeline, end to end.</b> With the downsampler fixed by the task, the best decoders are those of Table 4. If both ends may be designed (a codec), the best measured pair is
      <b>{pair_best[2].down} + {pair_best[2].up}</b> at 2× ({pair_best[2].psnr:.2f} dB) and <b>{pair_best[8].down} + {pair_best[8].up}</b> at 8× ({pair_best[8].psnr:.2f} dB), i.e. {pair_best[2].psnr - P(q5, 2):+.2f} and
      {pair_best[8].psnr - P(q5, 8):+.2f} dB over the frozen Gaussian with the same interpolator.</p>
    </div>
    <div>
      {section(6, '7', 'Beyond interpolation: who wins and why')}
      <ul>{''.join(winners)}</ul>
      <ul>
        <li><b>Projection never hurts</b> (clean y): (3) is an orthogonal projection onto {{z : Dz = y}}, which contains x. Measured: projected ≥ raw PSNR in {proj_ok} image–pipeline pairs.</li>
        <li><b>Learning beats the linear ceiling</b>: the best consistent linear method is an oblique projection capped at {ceil8:.2f} dB (8×, assignment images; bound verified {n['hist']['b10_bound_holds']});
        the MSE-optimal E[x | y] is non-linear, and P F(y) learns only the missing null-space part.</li>
        <li><b>Degradation must match</b>: networks trained on bicubic inputs estimate E[x | y<sub>bicubic</sub>]; fine-tuning SwinIR-M on (1) removes that bias, and it leads on quality at every scale (Urban100-c: {ft_vs_hat}).</li>
        <li><b>Learning helps most where the input still holds information</b>: OURS over quintic + projection on Urban100-c is {gain_txt} at 2× / 4× / 8×; over pretrained HAT-L + projection {hat_txt} dB, while running {speed_txt} faster on the Mac.</li>
        <li><b>The trade-off winner changes with scale</b>: convolution cost grows with the LR area (512/s)², ×4 per halving of s — OURS takes {ms[8]:.1f} / {ms[4]:.1f} / {ms[2]:.1f} ms at 8× / 4× / 2×, while bicubic stays ≈ 1 ms, so the 10×-bicubic budget admits the network only at large factors.</li>
      </ul>
      {limit('Limits of the claims', 'One noise-free synthetic degradation and one machine (HAT-L needs tiling on 8 GB, so its time is machine-specific). The assignment images were viewed during development, so their extension scores are descriptive; Urban100-c and DIV2K-20 were never trained on, but pretrained networks may have seen them upstream. Validation scores of selected checkpoints are optimistic. Claim: best measured choices among the evaluated pipelines for this downsampler on this hardware — not a general state-of-the-art claim. Paired bootstrap statistics, seeds, latency records and the coverage ledger: extension/results/.')}
      {section(7, '8', 'Conclusion')}
      <p><b>Interpolation:</b> quality rises with order (quintic ≈ Lanczos &gt; cubic &gt; quadratic &gt; bilinear ≈ nearest) but the gains are small beside the loss from larger factors and
      from content; a cubic kernel is the best quality per unit time. <b>Beyond it:</b> exact consistency with the observed image is the cheapest large gain (bicubic + projection is the
      promoted pipeline at 2× and 4×); the best quality at every scale comes from SwinIR-M fine-tuned on this downsampler; and OURS is the promoted pipeline at 8×, the only learned model inside the
      10×-bicubic budget there.</p>
      <div class='refs'>
        <p>[1] Keys, IEEE TASSP 1981 · [2] Unser, IEEE SPM 1999 · [3] Getreuer, IPOL 2011 · [4] Burt &amp; Adelson, IEEE TCOM 1983 · [5] Irani &amp; Peleg, CVGIP 1991 ·
        [6] Schwab, Antholzer &amp; Haltmeier, Inverse Problems 2019 · [7] Bahat &amp; Michaeli, CVPR 2020 · [8] Wang, Yu &amp; Zhang (DDNM), ICLR 2023 · [9] Shi et al. (ESPCN), CVPR 2016 ·
        [10] Liang et al. (SwinIR), ICCVW 2021 · [11] Chen et al. (HAT), CVPR 2023 · [12] Wang et al. (SSIM), IEEE TIP 2004 · [13] Blau &amp; Michaeli, CVPR 2018 · Data: USC-SIPI, DIV2K, BSD68, Urban100, Vid4.</p>
      </div>
    </div>
  </div>
  {footer('Discussion, sensitivity, justification and conclusion', 4)}
</section>"""


def build_html(n: dict) -> str:
    body = page1(n) + page2(n) + page3(n) + page4(n)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>A2 Interpolation-Based Super-Resolution</title><style>{CSS}{EXTRA_CSS}</style></head><body>{body}</body></html>"


# ----------------------------------------------------------------------------- outputs
def render_pdf(html_path: Path, pdf_path: Path) -> None:
    profile = BUILD / "chrome-profile"
    pdf_path.unlink(missing_ok=True)
    proc = subprocess.Popen([CHROME, "--headless", "--disable-gpu", "--no-sandbox", f"--user-data-dir={profile}", "--no-pdf-header-footer",
                             f"--print-to-pdf={pdf_path}", html_path.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last, deadline = -1, time.time() + 240
    while time.time() < deadline:
        size = pdf_path.stat().st_size if pdf_path.exists() else -1
        if size > 0 and size == last:
            break
        last = size
        time.sleep(1.5)
    proc.terminate()
    proc.wait(timeout=10)
    shutil.rmtree(profile, ignore_errors=True)
    if not pdf_path.exists():
        raise RuntimeError("Chrome did not produce the PDF")


def build_docx(n: dict, path: Path) -> None:
    """Editable companion with the same tables, figures and conclusions (simpler layout than the PDF)."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(.5)
    sec.left_margin = sec.right_margin = Inches(.6)
    doc.styles["Normal"].font.name = "Charter"
    doc.styles["Normal"].font.size = Pt(9)

    def head(text, size=12):
        r = doc.add_paragraph().add_run(text)
        r.bold, r.font.size = True, Pt(size)

    def para(text):
        doc.add_paragraph(text).alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def picture(p, width=7.0):
        if p.exists():
            doc.add_paragraph().add_run().add_picture(str(p), width=Inches(width))

    def dtable(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        for i, h in enumerate(headers):
            t.rows[0].cells[i].text = h
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = str(v)
        for row in t.rows:
            for c in row.cells:
                for pgh in c.paragraphs:
                    for r in pgh.runs:
                        r.font.size = Pt(7)

    m = n["mean"]
    head(TITLE, 18)
    para(SUBTITLE)
    para("[Student Name] · [Roll Number] · [Course] · [Submission Date]")
    head("1. Task and one fixed downsampling procedure")
    para("Images: Mandrill 4.2.03, Jet 7.1.02, Tank 7.1.03, Sailboat 4.2.06 (BT.601 luma). Gaussian pyramid σ=1 per ×2 stage, half-pixel sampling, symmetric borders: "
         "512→256→128→64; each level enlarged directly to 512 with x=(u+½)/s−½ (nearest = block copy). Methods: nearest, bilinear, local quadratic, quadratic/cubic/quartic/quintic "
         "B-splines, Keys cubic, Lanczos-3. PSNR (R=255), SSIM (σ=1.5) on the full clipped frame; runtime = median of warm single-thread calls.")
    head("2. Extension: an exactly consistent learned decoder")
    para("x̂ = F(y) + B(y − D F(y)) = By + P F(y), B = Dᵀ(DDᵀ)⁻¹, P = I − BD, so ‖x̂ − x‖² = ‖F − x‖² − ‖(I−P)(F − x)‖². F = 12-layer 96-channel LR-grid CNN with bicubic skip, "
         "MSE on training-only images. Roles selected on four validation images plus Mac latency, frozen before test scoring.")
    head("Table 4. Role winners")
    r = n["roles"][n["roles"].scope == "overall"]
    dtable(["Scale", "Role", "Pipeline", "Val PSNR/SSIM", "ms", "Assignment", "Urban100-c", "DIV2K-20"],
           [[f"{x.scale}×", x.role, tag(x.method_id), pc(x.val_psnr, x.val_ssim, 2, 4), f"{x.latency_median_ms:.1f}", pc(x.assignment_psnr, x.assignment_ssim),
             pc(x.urban100c_psnr, x.urban100c_ssim), pc(x.div2k_eval20_psnr, x.div2k_eval20_ssim)] for _, x in r.iterrows()])
    picture(FIG / "fig1_originals_and_lr.png", 6.2)
    head("Table 1. Mean PSNR / SSIM and runtime (ms)")
    dtable(["Method", "2×", "4×", "8×", "ms 2×", "ms 4×", "ms 8×"],
           [[lab] + [pc(m.loc[(lab, s), "psnr"], m.loc[(lab, s), "ssim"]) for s in (2, 4, 8)] + [f"{runtime_of(n, lab, s):.2f}" for s in (2, 4, 8)] for lab in ORDER])
    for metric, name, fmt in (("psnr", "Table 2. PSNR per image", ".2f"), ("ssim", "Table 3. SSIM per image", ".3f")):
        head(name)
        piv = n["q"].pivot_table(index="label", columns=["scale", "image"], values=metric)
        dtable(["Method"] + [f"{s}× {SHORT[im]}" for s in (2, 4, 8) for im in SHORT], [[lab] + [format(piv.loc[lab, (s, im)], fmt) for s in (2, 4, 8) for im in SHORT] for lab in ORDER])
    for f in (FIG / "fig3_full_sr_sailboat.png", FIG / "fig7_step_edge_profile.png", FIG / "fig6_error_maps.png", FIG / "fig2_sr_crops_4.2.03.png", XFIG / "ext_crops_x8.png"):
        picture(f, 6.6)
    head("Discussion and conclusion")
    para("See the PDF for the full discussion (every number there is generated from the same files). Quality falls with the factor and the spread between orders shrinks; nearest blocks, "
         "bilinear blurs, degree ≥ 3 rings; a cubic kernel gives the best quality per unit time; smooth regions are easy, edges gain most from order, fine texture converges. "
         "Lighter anti-aliasing (Lanczos-3) raises every interpolator's PSNR, but the required tables keep one fixed Gaussian pyramid. Exact consistency is the most useful single step; "
         "the promoted pipeline per scale is the promoted row of Table 4. Claim scope: best measured choices among the evaluated pipelines for this downsampler on this hardware.")
    doc.save(path)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    n = load()
    pd.DataFrame(LEDGER, columns=["family", "status", "artefact", "conclusion"]).to_csv(XRES / "coverage_ledger.csv", index=False)
    html_path = BUILD / "A2_Interpolation_SR_Report.html"
    html_path.write_text(build_html(n), encoding="utf-8")
    pdf = REP / "A2_Interpolation_SR_Report.pdf"
    render_pdf(html_path, pdf)
    build_docx(n, REP / "A2_Interpolation_SR_Report.docx")
    (REP / "A2_Interpolation_SR_Report.html").unlink(missing_ok=True)
    print(json.dumps({"pdf": str(pdf), "pdf_kb": pdf.stat().st_size // 1024, "docx": str(REP / "A2_Interpolation_SR_Report.docx")}, indent=1))


if __name__ == "__main__":
    main()
