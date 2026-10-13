"""Method flow diagram for the Efficient NLM report.

Two lanes, because the report makes two separate claims and the reader must not
conflate them:

  Lane A  the submitted efficient solution -- an exact restructuring of basic
          NLM in which the patch size K leaves the leading cost term;
  Lane B  the research extension -- selective routing of a small pixel budget to
          the approximate spatial-PCA expert.

Emitted as inline SVG so it stays vector-sharp in the printed PDF, and
rasterised through headless Chrome for the DOCX, which cannot embed SVG
reliably across Word versions.
"""

from __future__ import annotations


from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

W, H = 1000, 158
BOX_W, GAP = 148, 22
STEP = BOX_W + GAP

LANE_A = [
    ("Noisy image Y", ["native 512x512"]),
    ("Displacement loop", ["t over the S x S", "search grid (S = 21)"]),
    ("Squared difference", ["Dₜ(u) =", "[Y(u) − Y(u+t)]²"]),
    ("Separable box mean", ["Bₖ[Dₜ] via running sum", "O(1) per pixel, any K"]),
    ("Bias-corrected weight", ["d² = max(Bₖ − 2σ², 0)", "w = exp(−d²/h²)"]),
    ("Accumulate → Î(p)", ["Σ w·Y(u+t) / Σ w", "O(N S²) total"]),
]

LANE_B = [
    ("Noisy image Y", ["same input"]),
    ("Cheap experts", ["Gaussian + exact", "bilateral estimates"]),
    ("Difficulty cues", ["residual, gradient,", "variance, disagreement"]),
    ("Score g(p) → mask", ["m(p) = top-ρ fraction", "ρ = 10 / 25 / 35 %"]),
    None,  # split node, drawn separately
    ("Blend → Îₓ(p)", ["m·Î₅ + (1−m)·Îᶜʰᵉᵃᵖ", "one fused output"]),
]


def _box(x: float, y: float, w: float, h: float, fill: str, stroke: str) -> str:
    return (f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='4' "
            f"fill='{fill}' stroke='{stroke}' stroke-width='1.1'/>")


def _label(cx: float, y: float, title: str, lines: list[str], title_fill: str) -> str:
    out = [f"<text x='{cx:.1f}' y='{y + 15:.1f}' text-anchor='middle' font-size='10.4' "
           f"font-weight='700' fill='{title_fill}'>{title}</text>"]
    for i, line in enumerate(lines):
        out.append(f"<text x='{cx:.1f}' y='{y + 29 + i * 11.4:.1f}' text-anchor='middle' "
                   f"font-size='9.1' fill='#3d5570'>{line}</text>")
    return "".join(out)


def _arrow(x1: float, y1: float, x2: float, y2: float, colour: str) -> str:
    """Line plus an explicit polygon head.

    SVG `marker-end` is not honoured by every rasteriser (PyMuPDF drops it), and
    a flow diagram whose arrows lose their direction is worse than no diagram,
    so the head is drawn as real geometry instead.
    """
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size, half = 6.0, 2.7
    bx, by = x2 - ux * size, y2 - uy * size
    head = (f"{x2:.1f},{y2:.1f} {bx + px * half:.1f},{by + py * half:.1f} "
            f"{bx - px * half:.1f},{by - py * half:.1f}")
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{bx:.1f}' y2='{by:.1f}' "
            f"stroke='{colour}' stroke-width='1.4'/>"
            f"<polygon points='{head}' fill='{colour}'/>")


def _text_block(x: float, y: float, lines: list[str], size: float, fill: str,
                weight: str = "400", leading: float = 11.0) -> str:
    """One <text> element per line. SVG has no automatic wrapping, so any
    multi-line copy has to be broken explicitly at authoring time."""
    return "".join(
        f"<text x='{x:.1f}' y='{y + i * leading:.1f}' font-size='{size}' "
        f"font-weight='{weight}' fill='{fill}'>{line}</text>"
        for i, line in enumerate(lines))


def _lane(nodes: list, top: float, height: float, fill: str, stroke: str,
          accent: str, title_fill: str) -> str:
    parts = []
    for i, node in enumerate(nodes):
        x = i * STEP
        cx = x + BOX_W / 2
        if node is None:
            # Split node: expensive D5 branch above, cheap branch below.
            half = (height - 6) / 2
            parts.append(_box(x, top, BOX_W, half, "#fdf3e3", "#e2b775"))
            parts.append(f"<text x='{cx:.1f}' y='{top + 12:.1f}' text-anchor='middle' "
                         f"font-size='9.6' font-weight='700' fill='#8a5a12'>"
                         f"D5 expert &#183; ρ of pixels</text>")
            parts.append(f"<text x='{cx:.1f}' y='{top + 22.5:.1f}' text-anchor='middle' "
                         f"font-size='8.5' fill='#7a5416'>PCA → KD-tree → rerank → R0</text>")
            parts.append(_box(x, top + half + 6, BOX_W, half, fill, stroke))
            parts.append(f"<text x='{cx:.1f}' y='{top + half + 18:.1f}' text-anchor='middle' "
                         f"font-size='9.6' font-weight='700' fill='{title_fill}'>"
                         f"cheap estimate</text>")
            parts.append(f"<text x='{cx:.1f}' y='{top + half + 28.5:.1f}' text-anchor='middle' "
                         f"font-size='8.5' fill='#3d5570'>remaining 1−ρ of pixels</text>")
        else:
            title, lines = node
            parts.append(_box(x, top, BOX_W, height, fill, stroke))
            parts.append(_label(cx, top, title, lines, title_fill))

    mid = top + height / 2
    for i in range(len(nodes) - 1):
        x1 = i * STEP + BOX_W
        x2 = (i + 1) * STEP
        if nodes[i + 1] is None:
            half = (height - 6) / 2
            parts.append(_arrow(x1 + 2, mid, x2 - 4, top + half / 2, accent))
            parts.append(_arrow(x1 + 2, mid, x2 - 4, top + half + 6 + half / 2, accent))
        elif nodes[i] is None:
            half = (height - 6) / 2
            parts.append(_arrow(x1 + 2, top + half / 2, x2 - 4, mid, accent))
            parts.append(_arrow(x1 + 2, top + half + 6 + half / 2, x2 - 4, mid, accent))
        else:
            parts.append(_arrow(x1 + 2, mid, x2 - 4, mid, accent))
    return "".join(parts)


def flow_svg(standalone: bool = False) -> str:
    """`standalone=True` stamps absolute px dimensions so rasterisers that
    ignore a percentage width still honour the aspect ratio."""
    size = f"width='{W}' height='{H}'" if standalone else "width='100%'"
    blue, teal = "#1B36C4", "#0f766e"

    lane_a = _lane(LANE_A, 16, 56, "#eef4fb", "#b7cbe4", blue, "#123b66")
    lane_b = _lane(LANE_B, 95, 56, "#edf7f5", "#a9d2cc", teal, "#0f5c55")

    title_a = _text_block(0, 9, [
        "A &#183; Submitted efficient solution &#8212; exact displacement-wise NLM, "
        "output identical to direct NLM"], 10.2, blue, "700")

    title_b = _text_block(0, 88, [
        "B &#183; Research extension &#8212; selective fusion routing a fixed pixel budget "
        "to the approximate spatial-PCA expert"], 10.2, teal, "700")

    return f"""<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg'
 font-family='Arial, Helvetica, sans-serif' {size}>
<rect width='{W}' height='{H}' fill='white'/>
{title_a}{lane_a}
{title_b}{lane_b}
</svg>"""


def render_png(out_path: Path, scale: int = 3) -> Path:
    """Rasterise the SVG for DOCX embedding (Word cannot be relied on for SVG)."""
    import pymupdf

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(stream=flow_svg(standalone=True).encode(), filetype="svg")
    doc[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale)).save(out_path)
    return out_path


if __name__ == "__main__":
    path = render_png(Path("_report_build") / "method_flow.png")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
