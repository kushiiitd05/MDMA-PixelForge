"""Two-lane pipeline diagram for the report: the required interpolation study and our decoder, end to end.

Lane A is what the assignment asks for. Lane B is our method: the same low-resolution input goes through a learned
LR-grid CNN and is then corrected by the exact projection, so the two techniques (learned prior, exact consistency)
appear in the order they run. The training path is drawn once, dashed, because it never touches the test images.
"""
from __future__ import annotations

from report_design import AMBER, GREY_SOFT, GREY_TEXT, INK, MONO, SANS, ULTRA

W, H = 980, 206


def _box(x, y, w, h, title, sub="", color=INK, fill="#FFFFFF", dash=False, rx=3) -> str:
    stroke = f" stroke-dasharray='3 2'" if dash else ""
    t = (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{color}' stroke-width='0.9'{stroke}/>"
         f"<text x='{x + w / 2}' y='{y + (13 if sub else h / 2 + 3.2)}' text-anchor='middle' font-family={SANS!r} font-size='8.3' font-weight='600' fill='{INK}'>{title}</text>")
    if sub:
        for i, line in enumerate(sub.split("|")):
            t += (f"<text x='{x + w / 2}' y='{y + 24 + 8.6 * i}' text-anchor='middle' font-family={MONO!r} font-size='7' fill='{GREY_TEXT}'>{line}</text>")
    return t


def _arrow(x1, y1, x2, y2, label="", color=GREY_SOFT, dash=False) -> str:
    d = " stroke-dasharray='3 2'" if dash else ""
    s = f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='0.9' marker-end='url(#ah)'{d}/>"
    if label:
        s += (f"<text x='{(x1 + x2) / 2}' y='{min(y1, y2) - 3.5}' text-anchor='middle' font-family={MONO!r} font-size='6.6' fill='{GREY_TEXT}'>{label}</text>")
    return s


def flow_svg() -> str:
    p = [f"<svg viewBox='0 0 {W} {H}' width='100%' xmlns='http://www.w3.org/2000/svg'>",
         f"<defs><marker id='ah' viewBox='0 0 8 8' refX='7' refY='4' markerWidth='5' markerHeight='5' orient='auto'>"
         f"<path d='M0,0 L8,4 L0,8 z' fill='{GREY_SOFT}'/></marker></defs>"]
    # lane labels
    p.append(f"<text x='0' y='9' font-family={SANS!r} font-size='7.4' font-weight='600' letter-spacing='1.4' fill='{GREY_SOFT}'>A · REQUIRED STUDY</text>")
    p.append(f"<text x='0' y='103' font-family={SANS!r} font-size='7.4' font-weight='600' letter-spacing='1.4' fill='{ULTRA}'>B · OUR DECODER (SAME INPUT, SAME METRICS)</text>")
    # lane A
    p.append(_box(0, 16, 104, 40, "Original 512²", "grayscale|4 images"))
    p.append(_arrow(104, 36, 150, 36, "Gaussian σ=1, ↓2"))
    p.append(_box(150, 16, 116, 40, "256² · 128² · 64²", "same procedure|for every image"))
    p.append(_arrow(266, 36, 322, 36, "order 0–5, Lanczos"))
    p.append(_box(322, 16, 124, 40, "Interpolated 512²", "one direct step|per factor"))
    p.append(_arrow(446, 36, 500, 36))
    p.append(_box(500, 16, 118, 40, "PSNR · SSIM · ms", "vs the original|Tables 1–3"))
    # shared LR hand-off into lane B
    p.append(_arrow(208, 56, 208, 112, "same y", dash=False))
    # lane B
    p.append(_box(150, 112, 116, 44, "LR input y", "64² / 128² / 256²", color=ULTRA))
    p.append(_arrow(266, 134, 318, 134))
    p.append(_box(318, 112, 132, 44, "1 · Learned prior", "12 × (3×3 conv, 96)|on the LR grid", color=ULTRA, fill="#F7F8FC"))
    p.append(_arrow(450, 134, 500, 134, "s² channels"))
    p.append(_box(500, 112, 132, 44, "pixel shuffle ⊕ bicubic", "raw prediction F(y)|0.92–0.97 M weights", color=ULTRA, fill="#F7F8FC"))
    p.append(_arrow(632, 134, 682, 134))
    p.append(_box(682, 112, 150, 44, "2 · Exact projection", "x̂ = F + B(y − D F)|D x̂ = y, never worse", color=ULTRA, fill="#F7F8FC"))
    p.append(_arrow(832, 134, 872, 134, "clip"))
    p.append(_box(872, 112, 108, 44, "Output 512²", "same metrics|Table 4", color=ULTRA))
    # training path (dashed, never test images)
    p.append(_box(0, 158, 190, 44, "Training (never the 4 test images)", "345 images: USC-SIPI, Vid4,|DIV2K, BSD68 · 128² crops", color=AMBER, fill="#FDF8F0", dash=True))
    p.append(_arrow(190, 180, 318, 180, "y = D x, MSE on F(y)", color=AMBER, dash=True))
    p.append(_box(318, 158, 180, 44, "64k updates, 2 seeds", "Adam 2·10⁻⁴ cosine|batch 16", color=AMBER, fill="#FDF8F0", dash=True))
    p.append(_arrow(498, 180, 600, 180, "checkpoints", color=AMBER, dash=True))
    p.append(_box(600, 158, 232, 44, "Validation selection (4 held-out images)", "best val PSNR / SSIM · latency|roles frozen before test scoring", color=AMBER, fill="#FDF8F0", dash=True))
    p.append(f"<line x1='716' y1='158' x2='716' y2='157' stroke='{AMBER}' stroke-width='0.9' stroke-dasharray='3 2' marker-end='url(#ah)'/>")
    p.append("</svg>")
    return "".join(p)
