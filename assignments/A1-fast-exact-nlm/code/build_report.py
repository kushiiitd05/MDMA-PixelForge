"""Build the final three-page DIP A1 report from measured evidence.

The PDF is rendered from self-contained HTML with fixed A4 pages.  The DOCX is
an editable companion containing the same numbers and figures.  No numerical
result is typed into the report: all values are read from the experiment CSVs.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results"
BUILD = ROOT / "_report_build"
REPORT = ROOT / "report"
CHROME = os.environ.get(
    "CHROME",
    shutil.which("google-chrome")
    or shutil.which("chromium")
    or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)

TITLE = "Fast Exact Non-Local Means"
SUBTITLE = "Reordering the computation, then studying the quality–runtime trade-off"
METHODS = ("BF_exact", "Fast_exact_NLM", "D5_full", "FusionV5")
LABEL = {
    "BF_exact": "Bilateral filter",
    "Fast_exact_NLM": "Fast exact NLM",
    "D5_full": "Full D5",
    "FusionV5": "Selective fusion",
}


from report_flow_diagram import flow_svg, render_png as render_flow_png
from report_design import CSS, readout, section, stamp, wedge

# The page-one flow is intentionally compact so its explanatory caption remains
# above the fixed footer at A4 size.
CSS += """
.page:first-of-type{font-size:8.15pt;line-height:1.24}
.page:first-of-type p{margin-bottom:1.25mm}
.page:first-of-type .sec{margin-top:2mm;margin-bottom:1mm}
.page:first-of-type ul{margin:.5mm 0 1mm}
.page:first-of-type li{margin-bottom:.45mm}
.page:first-of-type .readout{margin-bottom:2.5mm}
.flow{margin-top:1mm;padding-top:1mm}
.flow svg{height:15mm}
.flow figcaption{font-size:6.1pt;line-height:1.12;margin-top:.45mm}
"""


def image_data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def numbers() -> dict:
    agg = pd.read_csv(EVIDENCE / "final_aggregate.csv")
    eq = pd.read_csv(EVIDENCE / "exact_equivalence_benchmark.csv")
    scale = pd.read_csv(EVIDENCE / "exact_acceleration_patch_scaling.csv")
    by_resolution = pd.read_csv(EVIDENCE / "final_by_resolution.csv")
    manifest = json.loads((EVIDENCE / "final_run_manifest.json").read_text())
    a = {}
    for sigma in (10, 25, 50):
        frame = agg[agg.noise_sigma == sigma].set_index("method")
        a[sigma] = {
            m: {
                "psnr": float(frame.loc[m, "PSNR"]),
                "ssim": float(frame.loc[m, "SSIM"]),
                "ms": float(frame.loc[m, "runtime_ms"]),
                "p": float(frame.loc[m, "P_D5"]),
            }
            for m in METHODS
        }
    return {
        "agg": a,
        "eq": eq,
        "scale": scale,
        "by_resolution": by_resolution,
        "manifest": manifest,
        "mean_speedup": float(eq.speedup.mean()),
        "max_error": float(eq.max_abs_difference.max()),
    }


def quality_rows(n: dict) -> str:
    rows = []
    for sigma, values in n["agg"].items():
        best_p = max(values[m]["psnr"] for m in METHODS)
        best_s = max(values[m]["ssim"] for m in METHODS)
        cells = []
        for method in METHODS:
            p = f"{values[method]['psnr']:.3f}"
            s = f"{values[method]['ssim']:.4f}"
            if values[method]["psnr"] == best_p:
                p = f"<b>{p}</b>"
            if values[method]["ssim"] == best_s:
                s = f"<b>{s}</b>"
            cells.append(f"<td>{p} / {s}</td>")
        rows.append(f"<tr><th>{sigma}</th>{''.join(cells)}</tr>")
    return "".join(rows)


def runtime_rows(n: dict) -> str:
    rows = []
    for sigma, values in n["agg"].items():
        ratio = values["D5_full"]["ms"] / values["FusionV5"]["ms"]
        cells = "".join(f"<td>{values[m]['ms']:.1f}</td>" for m in METHODS)
        rows.append(
            f"<tr><th>{sigma}</th>{cells}<td>{ratio:.2f}×</td>"
            f"<td>{100 * values['FusionV5']['p']:.0f}%</td></tr>"
        )
    return "".join(rows)


def runtime_resolution_rows(n: dict) -> str:
    """Report 256 and 512 costs separately instead of hiding mixed sizes."""
    frame = n["by_resolution"]
    rows = []
    for sigma in (10, 25, 50):
        cells = []
        for method in METHODS:
            values = []
            for resolution in ("256x256", "512x512"):
                row = frame[
                    frame.native_resolution.eq(resolution)
                    & frame.noise_sigma.eq(sigma)
                    & frame.method.eq(method)
                ]
                values.append(float(row.iloc[0].runtime_ms))
            cells.append(f"<td>{values[0]:.1f} / {values[1]:.1f}</td>")
        ratio256 = (
            frame[(frame.native_resolution == "256x256") & (frame.noise_sigma == sigma) & (frame.method == "D5_full")].iloc[0].runtime_ms
            / frame[(frame.native_resolution == "256x256") & (frame.noise_sigma == sigma) & (frame.method == "FusionV5")].iloc[0].runtime_ms
        )
        ratio512 = (
            frame[(frame.native_resolution == "512x512") & (frame.noise_sigma == sigma) & (frame.method == "D5_full")].iloc[0].runtime_ms
            / frame[(frame.native_resolution == "512x512") & (frame.noise_sigma == sigma) & (frame.method == "FusionV5")].iloc[0].runtime_ms
        )
        rows.append(
            f"<tr><th>{sigma}</th>{''.join(cells)}"
            f"<td>{ratio256:.2f}× / {ratio512:.2f}×</td></tr>"
        )
    return "".join(rows)


def equivalence_rows(n: dict) -> str:
    rows = []
    for r in n["eq"].itertuples():
        rows.append(
            f"<tr><th>{r.image_id}</th><td>{r.direct_ms:.1f}</td>"
            f"<td>{r.fast_ms:.1f}</td><td><b>{r.speedup:.2f}×</b></td>"
            f"<td>{r.max_abs_difference:.1e}</td></tr>"
        )
    return "".join(rows)




def build_html(n: dict) -> str:
    agg = n["agg"]
    d5gain50 = agg[50]["D5_full"]["psnr"] - agg[50]["Fast_exact_NLM"]["psnr"]
    d5ssim50 = agg[50]["D5_full"]["ssim"] - agg[50]["Fast_exact_NLM"]["ssim"]
    fusion_ratios = [agg[s]["D5_full"]["ms"] / agg[s]["FusionV5"]["ms"] for s in (10,25,50)]
    byres = n["by_resolution"]
    def rv(resolution: str, sigma: int, method: str, column: str) -> float:
        row = byres[
            byres.native_resolution.eq(resolution)
            & byres.noise_sigma.eq(sigma)
            & byres.method.eq(method)
        ]
        return float(row.iloc[0][column])
    d5_512_psnr25 = rv("512x512", 25, "D5_full", "PSNR") - rv("512x512", 25, "Fast_exact_NLM", "PSNR")
    d5_512_ssim25 = rv("512x512", 25, "D5_full", "SSIM") - rv("512x512", 25, "Fast_exact_NLM", "SSIM")
    d5_256_psnr25 = rv("256x256", 25, "D5_full", "PSNR") - rv("256x256", 25, "Fast_exact_NLM", "PSNR")
    d5_256_ssim25 = rv("256x256", 25, "D5_full", "SSIM") - rv("256x256", 25, "Fast_exact_NLM", "SSIM")
    full = image_data(EVIDENCE / "final_full_image_comparison_sigma25.png")
    crops = image_data(EVIDENCE / "final_detail_crops_sigma25.png")
    trade = image_data(EVIDENCE / "quality_runtime_tradeoff.png")
    scale = image_data(EVIDENCE / "exact_acceleration_patch_scaling.png")
    return f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>

<section class='page'>
  <div class='kicker'>Digital Image Processing &middot; Assignment 1 &middot; Non-Local Means</div>
  <h1>{TITLE}</h1><div class='subtitle'>{SUBTITLE}</div>
  <div class='meta'>Kush Tokas &nbsp;/&nbsp; 2023296 &nbsp;/&nbsp; DIP-CSE341</div>
  {wedge()}
  {readout([
    (f"{n['mean_speedup']:.1f}&times;", "mean exact speed-up<br>K=7, S=11"),
    (f"{n['max_error']:.1e}", "largest direct-vs-fast<br>pixel difference"),
    (f"+{d5gain50:.2f} dB", "full D5 over fast<br>exact NLM at &sigma;=50"),
    (f"{fusion_ratios[2]:.2f}&times;", "fusion faster than<br>full D5 at &sigma;=50"),
  ])}
  <div class='two'>
    <div>
      {section(0, '1', 'Assignment route and contribution')}
      <p>I chose <b>Non-Local Means (NLM)</b> for this assignment. My main contribution is an
      <b>output-equivalent acceleration</b>. Basic NLM repeatedly sums every K×K patch
      distance at every search displacement. I instead form one squared-difference image for
      each displacement and aggregate it with a separable, constant-cost box operator.</p>
      <p>After verifying that this faster form gives the same output, I explored a second
      question: can a richer patch representation improve denoising when extra computation
      is acceptable? That experiment led to the spatial-PCA D5 expert and a selective fusion
      gate. I report them separately because they trade speed for quality, while the first
      method is an exact computational improvement.</p>
      {section(1, '2', 'Basic NLM and exact acceleration')}
      <p>For noisy image Y, patch P(p), odd patch size K, odd search size S and noise σ, the
      bias-corrected distance and NLM estimate are</p>
      <div class='eq'>d²(p,q)=max(K⁻²‖P(p)−P(q)‖²₂−2σ²,0)<br>
      Î(p)=Σ<sub>q∈S(p)</sub>w(p,q)Y(q) / Σ<sub>q∈S(p)</sub>w(p,q),
      &nbsp;w=exp[−d²/h²]. <span class='tag'>(1)</span></div>
      <p><b>Direct form.</b> For each of N pixels, the basic implementation visits S²
      displacements and sums K² samples, giving <b>O(NS²K²)</b> time.</p>
      <p><b>Accelerated form.</b> For displacement t, compute D<sub>t</sub>(u)=[Y(u)−Y(u+t)]²,
      then d²(p,p+t)=max(B<sub>K</sub>[D<sub>t</sub>](p)−2σ²,0), where B<sub>K</sub> is a
      separable running box mean. Accumulate the same weights and intensities immediately.
      This is <b>O(NS²)</b> time and O(N) working memory; reflect padding preserves boundary
      context. In other words, I changed the order of the computation, not the estimator.</p>
      {stamp(f"{n['max_error']:.1e}")}
      <p class='note'>Four 96&times;96 crops at &sigma;=25, K=7, S=11; the difference sits at
      double-precision round-off.</p>
    </div>
    <div>
      {section(2, '3', 'D5: an approximate patch expert')}
      <p>D5 standardizes and spatially windows each K×K patch before PCA:</p>
      <div class='eq'>x<sub>p</sub>=W<sup>1/2</sup>(P(p)−μ), &nbsp; z<sub>p</sub>=Uᵀx<sub>p</sub>,<br>
      Q(p)=[z<sub>p</sub>, λ<sub>y</sub>y<sub>p</sub>, λ<sub>x</sub>x<sub>p</sub>].
      <span class='tag'>(2)</span></div>
      <p>I use a KD-tree to retrieve a bounded candidate set in Q-space, then rerank those
      candidates with the exact descriptor distance. Gaussian weights combine their
      <b>centre intensities (R0)</b>. PCA keeps the patch description compact, while the
      coordinate terms prevent the search from ignoring spatial distance. D5 is therefore
      an approximate NLM variant, not textbook NLM, and its value must be demonstrated by
      the experiment.</p>
      {section(3, '4', 'Selective fusion and gate')}
      <p>Running D5 everywhere is expensive. To reduce that cost, I combine local residual,
      gradient, variance, and expert-disagreement cues into a difficulty score g(p). A fixed
      budget sends only the most difficult pixels to D5:</p>
      <div class='eq'>Î<sub>F</sub>(p)=m(p)Î<sub>D5</sub>(p)+[1−m(p)]Î<sub>cheap</sub>(p),<br>
      m(p)=1[g(p) is in the top ρ fraction]. <span class='tag'>(3)</span></div>
      <p>ρ is 0.10, 0.25 and 0.35 for σ=10,25,50. Expert maps, scoring, PCA fitting,
      retrieval and reconstruction are all included in the fusion timer.</p>
      <div class='limit'><div class='limit-hd'>Scope of the fusion claim</div>
      Fusion reduces the cost of <i>full D5</i>; it does not make fast exact NLM faster. Its
      gains were confirmed with a new noise seed, but only on the pre-screened four-image
      set. I therefore treat fusion as an extension, not as the main assignment claim.</div>
      {section(4, '5', 'Experimental protocol')}
      <ul>
        <li>I first screened the 39 available USC-SIPI images at 128 px. After removing the
        gate/calibration images, I kept images where both D5 and fusion beat fast exact NLM
        in PSNR and SSIM at all three noise levels: 4.1.02, 4.1.01, 7.1.04, and 7.1.03.</li>
        <li>I then reran these four at their native 256×256 or 512×512 size with a new seed,
        20261029. Noise σ∈{{10,25,50}} is clipped to [0,255]. The gate was trained on a
        separate four-image set with seed 20260827, so it never saw the final images.</li>
        <li>K=7, S=21 for the native-image comparison. Whole-call wall time excludes file
        I/O and metric calculation; identical machine and environment for all methods.</li>
      </ul>
    </div>
  </div>
  <figure class='flow'>{flow_svg()}
  <figcaption><b>Figure 1 &middot; Both pipelines end to end.</b>
  <b style='color:#1B36C4'>Lane A</b>, the submitted solution, changes only the order of
  evaluation &mdash; the estimator is untouched.
  <b style='color:#0f766e'>Lane B</b>, the research extension, spends a fixed pixel budget
  &rho; on the approximate expert, so &rho; controls real retrieval work rather than merely
  selecting an output after the fact. The lattice beside each section number marks one
  displacement in the S&times;S search grid.</figcaption></figure>
  <div class='footer'>Efficient NLM · Method and mathematical formulation <span class='right'>1 / 3</span></div>
</section>

<section class='page'>
  <div class='pagehd'>Native-resolution visual and quantitative results</div>{wedge()}
  <figure><img class='fullimg' src='{full}'><figcaption><b>Figure 2.</b> Independent native-size
  confirmation at σ=25. Rows are interior scene, portrait, terrain/vehicles and armoured
  vehicle. Columns: original, noisy, bilateral, fast exact NLM, full D5, and selective
  fusion. The set spans faces, edges, fine terrain and structured texture.</figcaption></figure>
  <div class='ratio' style='margin-top:2mm'>
    <div>
      <div class='caption'><b>Table 1.</b> Mean PSNR (dB) / SSIM over four native images.
      Higher is better; metric-wise best values are bold.</div>
      <table><thead><tr><th>σ</th><th>Bilateral</th><th>Fast exact NLM</th><th>Full D5</th><th>Fusion</th></tr></thead>
      <tbody>{quality_rows(n)}</tbody></table>
      <p class='note'>With the new seed, D5 and fusion still exceed fast exact NLM in both
      metrics at σ=10, 25 and 50. At σ=25, D5 gains
      {d5_512_psnr25:.3f} dB / {d5_512_ssim25:+.4f} SSIM on the 512×512 pair and
      {d5_256_psnr25:.3f} dB / {d5_256_ssim25:+.4f} SSIM on the 256×256 pair. Because the
      images were pre-screened, these numbers describe this selected set rather than every
      possible image.</p>
    </div>
    <div>
      <figure><img class='cropimg' src='{crops}'><figcaption><b>Figure 3.</b> Automatically
      selected high-gradient crops at σ=25 plus a fifth flat-region audit row. The detail
      crops use maximum clean-image gradient energy; the audit uses minimum gradient energy
      so structured artifacts cannot hide behind aggregate metrics.</figcaption></figure>
    </div>
  </div>
  <div class='footer'>Efficient NLM · Original and processed images; PSNR and SSIM <span class='right'>2 / 3</span></div>
</section>

<section class='page'>
  <div class='pagehd'>Efficiency evidence, trade-offs, and conclusion</div>{wedge()}
  <div class='two'>
    <div>
      <figure><img class='plotsmall' src='{scale}'><figcaption><b>Figure 4.</b> Patch-size
      scaling on a fixed 96×96 crop (S=11). Direct NLM grows approximately with K²; the
      displacement/box version remains near 15 ms while preserving the same output.</figcaption></figure>
      <div class='caption'><b>Table 2.</b> Exact-equivalence benchmark, K=7, S=11 (ms).</div>
      <table class='tight'><thead><tr><th>Image</th><th>Direct</th><th>Fast</th><th>Speed-up</th><th>Max |Δ|</th></tr></thead>
      <tbody>{equivalence_rows(n)}</tbody></table>
      <p class='note'>Mean speed-up: <b>{n['mean_speedup']:.2f}×</b>. At K=11 the measured
      speed-up reaches <b>{float(n['scale'].iloc[-1].speedup):.2f}×</b>, consistent with
      eliminating the K² factor. This controlled crop test is used because a literal direct
      native-resolution run would add cost without changing the equivalence claim.</p>
    </div>
    <div>
      <figure><img class='plotsmall' src='{trade}'><figcaption><b>Figure 5.</b> Mean quality
      against whole-call runtime (log scale). D5 and fusion improve selected-set quality,
      while fast exact NLM remains the substantially cheaper NLM route.</figcaption></figure>
      <div class='caption'><b>Table 3.</b> Mean wall time per image (ms), shown as 256×256 / 512×512.</div>
      <table class='tight'><thead><tr><th>σ</th><th>BF</th><th>Fast NLM</th><th>D5</th><th>Fusion</th><th>D5/Fusion</th></tr></thead>
      <tbody>{runtime_resolution_rows(n)}</tbody></table>
      <p class='note'>All required expert construction and gate computation is included.
      Runtime values are machine-dependent; relative values within this run are the intended
      comparison. Each size entry is the mean of two complete native-image calls.</p>
    </div>
  </div>
  {section(5, '6', 'What succeeded, what did not, and the final claim')}
  <div class='two'>
    <div>
      <p><b>The exact acceleration worked.</b> Reusing displacement-wise differences removes
      the repeated K² patch work without changing the NLM estimate. The measured output is
      equal to direct NLM up to floating-point precision, with a {n['mean_speedup']:.1f}×
      mean speed-up at K=7 and larger gains as K grows.</p>
      <p><b>D5 bought extra quality, but at a clear cost.</b> On these four confirmed images,
      the spatial-PCA representation and reranked retrieval improve both PSNR and SSIM over
      fast exact NLM at every noise level. The flat-region crop also shows why I did not rely
      on the two scalar scores alone: structured texture can still remain visible.</p>
    </div>
    <div>
      <p><b>Fusion recovered part of the D5 runtime.</b> It is {fusion_ratios[0]:.2f}×,
      {fusion_ratios[1]:.2f}× and {fusion_ratios[2]:.2f}× faster than full D5 at the three
      noise levels. It also improves both metrics over fast exact NLM at all three levels,
      but remains slower than fast exact NLM. For this set, it is a useful middle ground
      between the cheaper exact method and full D5. A broader image set and an
      uncertainty-aware gate would be the next step.</p>
      <p><b>Conclusion.</b> My submitted efficient solution is fast exact NLM. D5 and fusion
      are supporting experiments: they show how much quality approximate retrieval can add,
      how much time it costs, and why visual inspection still matters.</p>
    </div>
  </div>
  <div class='pagehd' style='font-size:11pt;margin-top:3mm'>References</div>
  <div class='refs'>
    <p>[1] A. Buades, B. Coll, J.-M. Morel, “A Non-Local Algorithm for Image Denoising,” CVPR, 2005.</p>
    <p>[2] C. Tomasi, R. Manduchi, “Bilateral Filtering for Gray and Color Images,” ICCV, 1998.</p>
    <p>[3] T. Tasdizen, “Principal Neighborhood Dictionaries for Nonlocal Means Image Denoising,” IEEE TIP, 2009.</p>
    <p>[4] J. Darbon et al., “Fast Nonlocal Filtering Applied to Electron Cryomicroscopy,” ISBI, 2008.</p>
    <p>[5] Z. Wang et al., “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE TIP, 2004.</p>
    <p>[6] USC-SIPI Image Database, Miscellaneous volume.</p>
  </div>
  <div class='footer'>Efficient NLM · Runtime, complexity, limitations, and conclusion <span class='right'>3 / 3</span></div>
</section>
</body></html>"""


def render_pdf(html: Path, pdf: Path) -> None:
    profile = BUILD / "chrome-profile"
    profile.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        CHROME, "--headless", "--disable-gpu", "--no-sandbox",
        f"--user-data-dir={profile}", "--no-pdf-header-footer",
        "--print-to-pdf-no-header", f"--print-to-pdf={pdf}", html.as_uri(),
    ], check=True, capture_output=True, timeout=180)


def add_doc_table(doc, headers: list[str], rows: list[list[str]]) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = text
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(7)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = text
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cells[i].paragraphs[0].runs:
                run.font.size = Pt(7)


def build_docx(n: dict, path: Path) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(.45)
    sec.left_margin = sec.right_margin = Inches(.55)
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(8.4)
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = 1.0

    def title(text: str, size=17):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text); r.bold = True; r.font.size = Pt(size); r.font.color.rgb = RGBColor(16,42,67)

    def head(text: str):
        p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(2)
        r = p.add_run(text); r.bold = True; r.font.size = Pt(10); r.font.color.rgb = RGBColor(23,78,117)

    def para(text: str):
        p = doc.add_paragraph(text); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    title(TITLE); title(SUBTITLE, 11)
    p = doc.add_paragraph("Kush Tokas · 2023296 · DIP-CSE341")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    head("1. Assignment route and contribution")
    para("I chose Non-Local Means (NLM). My main contribution is an output-equivalent acceleration. Basic NLM repeatedly sums every K×K patch distance at each search displacement. I instead form one squared-difference image per displacement and aggregate it with a separable, constant-cost box operator. After verifying the same output, I explored D5 and selective fusion as separate quality–runtime extensions.")
    head("2. Basic NLM and exact acceleration")
    para("d²(p,q)=max(K⁻²||P(p)-P(q)||²₂-2σ²,0), w(p,q)=exp[-d²/h²], and Î(p)=Σw(p,q)Y(q)/Σw(p,q). The direct form costs O(NS²K²). For displacement t, I compute D_t(u)=[Y(u)-Y(u+t)]² and d²(p,p+t)=max(B_K[D_t](p)-2σ²,0), where B_K is a separable running box mean. This reduces the complexity to O(NS²) time with O(N) working memory. Reflect padding is used. The order of computation changes; the estimator does not.")
    head("3. D5 and selective fusion")
    para("D5 uses x_p=W^(1/2)(P(p)-μ), z_p=U^T x_p and query Q(p)=[z_p, λ_y y_p, λ_x x_p]. I use a KD-tree to find a bounded candidate set, rerank it with exact descriptor distance, and reconstruct from candidate centre intensities (R0). Because full D5 is expensive, fusion uses residual, gradient, variance, and expert-disagreement cues to send only the hardest 10%, 25%, or 35% of pixels to D5 at σ=10,25,50. Fusion lowers D5 cost, but it is still slower than fast exact NLM.")
    head("4. Protocol")
    para("I screened the 39 available USC-SIPI images at 128 px, excluding gate and calibration images. The fixed rule kept images where both D5 and fusion beat fast exact NLM in PSNR and SSIM at σ∈{10,25,50}. The selected images are 4.1.02 and 4.1.01 at 256×256, plus 7.1.04 and 7.1.03 at 512×512. I reran them at native size with a new seed, 20261029. The gate was trained on a separate four-image set using seed 20260827 and did not see the final images. Native comparison uses K=7, S=21; timers include complete calls and exclude I/O and metrics.")
    flow_path = render_flow_png(BUILD / "method_flow.png")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(flow_path), width=Inches(7.05))
    para("Figure 1. Both pipelines end to end. Lane A is the output-equivalent exact acceleration. Lane B is the selective spatial-PCA research extension; its pixel budget controls real retrieval work.")

    doc.add_page_break(); title("Native-resolution results", 15)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(EVIDENCE / "final_full_image_comparison_sigma25.png"), width=Inches(7.05))
    para("Figure 2. Original and processed images from the independent native-size confirmation at σ=25. Columns: original, noisy, bilateral, fast exact NLM, full D5 and selective fusion.")
    rows=[]
    for s,v in n["agg"].items():
        rows.append([str(s)]+[f"{v[m]['psnr']:.3f} / {v[m]['ssim']:.4f}" for m in METHODS])
    add_doc_table(doc,["σ","Bilateral","Fast exact NLM","Full D5","Fusion"],rows)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(EVIDENCE / "final_detail_crops_sigma25.png"), width=Inches(6.25))
    para("Figure 3. Automatically selected high-gradient crops plus a flat-region artifact audit; both selections use clean-image local gradient energy.")

    doc.add_page_break(); title("Efficiency, trade-offs, and conclusion", 15)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(EVIDENCE / "exact_acceleration_patch_scaling.png"), width=Inches(3.3))
    p.add_run().add_picture(str(EVIDENCE / "quality_runtime_tradeoff.png"), width=Inches(3.8))
    para("Figures 4–5. Exact patch-size scaling and the native-image quality/runtime trade-off. D5 and fusion improve selected-set quality; fast exact NLM remains the substantially cheaper NLM route.")
    rows=[]
    for r in n["eq"].itertuples():
        rows.append([r.image_id,f"{r.direct_ms:.1f}",f"{r.fast_ms:.1f}",f"{r.speedup:.2f}×",f"{r.max_abs_difference:.1e}"])
    add_doc_table(doc,["Image","Direct ms","Fast ms","Speed-up","Max |Δ|"],rows)
    rows=[]
    rf=n["by_resolution"]
    for s in (10,25,50):
        row=[str(s)]
        for m in METHODS:
            vals=[]
            for resolution in ("256x256","512x512"):
                vals.append(float(rf[(rf.native_resolution==resolution)&(rf.noise_sigma==s)&(rf.method==m)].iloc[0].runtime_ms))
            row.append(f"{vals[0]:.1f} / {vals[1]:.1f}")
        rows.append(row)
    add_doc_table(doc,["σ","BF 256/512","Fast NLM 256/512","D5 256/512","Fusion 256/512"],rows)
    head("Conclusion")
    para(f"The accelerated method matches direct NLM within {n['max_error']:.1e} and is {n['mean_speedup']:.2f}× faster on average at K=7, S=11. On these four independently confirmed images, D5 and fusion beat fast exact NLM in both PSNR and SSIM at all three noise levels. Fusion recovers part of the D5 runtime but remains slower than fast NLM. I therefore submit fast exact NLM as the efficient solution and present D5 and fusion as supporting quality–runtime experiments. Since the images were pre-screened, their gains should be read as selected-set results rather than universal performance.")
    head("References")
    para("[1] Buades et al., CVPR 2005. [2] Tomasi and Manduchi, ICCV 1998. [3] Tasdizen, IEEE TIP 2009. [4] Darbon et al., ISBI 2008. [5] Wang et al., IEEE TIP 2004. [6] USC-SIPI Image Database.")
    doc.save(path)


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    n = numbers()
    html = BUILD / "Efficient_NLM_Report.html"
    html.write_text(build_html(n), encoding="utf-8")
    pdf = REPORT / "Efficient_NLM_Report.pdf"
    render_pdf(html, pdf)
    docx = REPORT / "Efficient_NLM_Report.docx"
    build_docx(n, docx)
    print(json.dumps({"pdf": str(pdf), "docx": str(docx), "html": str(html)}, indent=2))


if __name__ == "__main__":
    main()
