"""Design system for the A2 report — the A1 "Photometric" system, carried over so both reports read as one series.

Step wedge as the page rule (a tone-reproduction target), Charter for text, Menlo for every measured quantity,
and rationed colour: ultramarine marks an exactly verified claim, amber marks a limitation. The section glyph is a
5x5 sampling lattice with one cell filled; in A2 it reads as "one LR sample among s x s HR pixels", and the filled
cell advances section by section so the motif doubles as an index.
"""
from __future__ import annotations

INK = "#0E1116"
PAPER = "#FFFFFF"
WEDGE = ("#1F2430", "#4A515E", "#878E9A", "#C4C9D0", "#E8EBEF")
GREY_TEXT = "#4A515E"
GREY_SOFT = "#878E9A"
RULE = "#D8DCE2"
ULTRA = "#1B36C4"   # exactly verified claim only
AMBER = "#C77800"   # limitation only

SERIF = "'Charter','Iowan Old Style','Palatino Linotype',Georgia,serif"
SANS = "'Futura','Avenir Next','Gill Sans','Helvetica Neue',sans-serif"
MONO = "'Menlo','SF Mono','DejaVu Sans Mono',monospace"

GLYPH_CELLS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 0), (3, 1), (1, 3), (0, 4), (2, 0), (2, 4), (0, 2)]


def wedge(height_mm: float = 1.5) -> str:
    cells = "".join(f"<div style='flex:1;background:{c}'></div>" for c in WEDGE)
    return f"<div style='display:flex;height:{height_mm}mm;width:100%;margin:0 0 3mm'>{cells}</div>"


def grid_glyph(index: int, size: int = 20) -> str:
    col, row = GLYPH_CELLS[index % len(GLYPH_CELLS)]
    dots = []
    for r in range(5):
        for c in range(5):
            on = (c, r) == (col, row)
            dots.append(f"<circle cx='{c * 5 + 2.5}' cy='{r * 5 + 2.5}' r='{1.5 if on else 0.85}' fill='{ULTRA if on else '#C4C9D0'}'/>")
    return f"<svg viewBox='0 0 25 25' width='{size}' height='{size}' style='flex:none'>{''.join(dots)}</svg>"


def section(index: int, number: str, title: str) -> str:
    return f"<div class='sec'>{grid_glyph(index)}<div><div class='eyebrow'>Section {number}</div><h2>{title}</h2></div></div>"


def readout(items: list[tuple[str, str]]) -> str:
    cells = "".join(f"<div class='ro'><div class='ro-v'>{v}</div><div class='ro-l'>{l}</div></div>" for v, l in items)
    return f"<div class='readout' style='grid-template-columns:repeat({len(items)},1fr)'>{cells}</div>"


def stamp(title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<div class='stamp-row'><span>{a}</span><b>{b}</b></div>" for a, b in rows)
    return f"<div class='stamp'><div class='stamp-hd'>{title}</div>{body}</div>"


def limit(title: str, html: str) -> str:
    return f"<div class='limit'><div class='limit-hd'>{title}</div>{html}</div>"


def page(body: str, footer: str, number: int, total: int, heading: str | None = None) -> str:
    hd = f"<div class='pagehd'>{heading}</div>{wedge()}" if heading else ""
    return f"<section class='page'>{hd}{body}<div class='footer'>{footer} <span class='right'>{number} / {total}</span></div></section>"


CSS = f"""
@page {{ size: A4; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; color:{INK}; background:{PAPER}; font-family:{SERIF}; font-size:8.6pt; line-height:1.3; -webkit-font-smoothing:antialiased; }}
.page {{ width:210mm; height:297mm; padding:11mm 12.5mm 12mm; position:relative; page-break-after:always; overflow:hidden; }}
.page:last-child {{ page-break-after:auto; }}
.kicker {{ font-family:{SANS}; font-size:7.6pt; font-weight:600; color:{ULTRA}; letter-spacing:.16em; text-transform:uppercase; margin:0 0 1.6mm; }}
h1 {{ font-family:{SERIF}; font-weight:700; font-size:21pt; line-height:1.06; margin:0 0 1.4mm; letter-spacing:-.2pt; }}
.subtitle {{ font-family:{SERIF}; font-style:italic; font-size:10.6pt; color:{GREY_TEXT}; margin:0 0 1.8mm; }}
.meta {{ font-family:{MONO}; font-size:7.4pt; color:{GREY_SOFT}; margin:0 0 2.5mm; }}
.pagehd {{ font-family:{SERIF}; font-weight:700; font-size:14.5pt; margin:0 0 1mm; }}
.readout {{ display:grid; gap:0; border-top:.5mm solid {INK}; border-bottom:.2mm solid {RULE}; margin:0 0 3.2mm; }}
.ro {{ padding:1.8mm 2.5mm 1.9mm 0; border-right:.2mm solid {RULE}; }}
.ro:last-child {{ border-right:0; }}
.ro-v {{ font-family:{MONO}; font-size:12.5pt; font-weight:600; color:{INK}; line-height:1; letter-spacing:-.3pt; }}
.ro-l {{ font-family:{SANS}; font-size:6.8pt; color:{GREY_SOFT}; margin-top:1.3mm; letter-spacing:.05em; line-height:1.25; }}
.sec {{ display:flex; gap:2.2mm; align-items:flex-start; margin:3mm 0 1.3mm; }}
.sec:first-child {{ margin-top:0; }}
.eyebrow {{ font-family:{SANS}; font-size:6.6pt; font-weight:600; color:{GREY_SOFT}; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.5mm; }}
h2 {{ font-family:{SERIF}; font-size:11pt; font-weight:700; margin:0; line-height:1.12; }}
h3 {{ font-family:{SANS}; font-size:7.2pt; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:{INK}; margin:2.4mm 0 1mm; }}
p {{ margin:0 0 1.6mm; text-align:justify; hyphens:auto; }}
ul {{ padding-left:4mm; margin:1mm 0 1.5mm; }}
li {{ margin-bottom:.8mm; }}
b {{ font-weight:700; }}
.num {{ font-family:{MONO}; font-size:.94em; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:5.5mm; }}
.ratio {{ display:grid; grid-template-columns:1.15fr .85fr; gap:4.5mm; }}
.eq {{ border-left:.5mm solid {ULTRA}; background:#F7F8FC; padding:1.5mm 2.4mm; text-align:center; margin:1.6mm 0; font-family:{SERIF}; font-size:9.2pt; line-height:1.45; }}
.eq .tag {{ float:right; font-family:{MONO}; font-size:7.4pt; color:{GREY_SOFT}; }}
.stamp {{ border:.35mm solid {ULTRA}; padding:1.7mm 2.2mm; margin:1.8mm 0; font-family:{MONO}; }}
.stamp-hd {{ font-family:{SANS}; font-size:6.6pt; font-weight:600; color:{ULTRA}; letter-spacing:.18em; text-transform:uppercase; margin-bottom:1.2mm; }}
.stamp-row {{ display:flex; justify-content:space-between; font-size:7.6pt; color:{GREY_TEXT}; line-height:1.5; }}
.stamp-row b {{ color:{INK}; }}
.limit {{ border-left:.5mm solid {AMBER}; background:#FDF8F0; padding:1.6mm 2.2mm; margin:1.7mm 0; }}
.limit-hd {{ font-family:{SANS}; font-size:6.6pt; font-weight:600; color:{AMBER}; letter-spacing:.18em; text-transform:uppercase; margin-bottom:1mm; }}
figure {{ margin:0; }}
figure img {{ width:100%; display:block; }}
figcaption,.caption {{ font-size:7.2pt; line-height:1.25; color:{GREY_TEXT}; margin-top:1mm; text-align:justify; }}
figcaption b,.caption b {{ font-family:{SANS}; font-size:6.8pt; font-weight:600; letter-spacing:.09em; text-transform:uppercase; color:{INK}; }}
table {{ width:100%; border-collapse:collapse; font-family:{MONO}; font-size:6.9pt; margin:1mm 0 1.4mm; }}
th,td {{ padding:.85mm .8mm; text-align:center; border:0; border-bottom:.2mm solid {RULE}; }}
thead th {{ font-family:{SANS}; font-size:6.3pt; font-weight:600; color:{GREY_SOFT}; letter-spacing:.06em; text-transform:uppercase; border-bottom:.4mm solid {INK}; }}
tbody th {{ font-family:{MONO}; font-size:6.9pt; color:{INK}; font-weight:600; text-align:left; }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:.4mm solid {INK}; }}
td b {{ color:{ULTRA}; }}
td.ours, th.ours {{ color:{ULTRA}; font-weight:700; }}
td.l {{ text-align:left; }}
td {{ white-space:nowrap; }}
table.text td {{ white-space:normal; }}
tr.grp td, tr.grp th {{ background:#F4F5F7; font-family:{SANS}; font-size:6.4pt; letter-spacing:.08em; text-transform:uppercase; color:{GREY_TEXT}; text-align:left; }}
table.roles td, table.roles th {{ padding:.4mm .5mm; white-space:nowrap; }}
table.tight td, table.tight th {{ padding:.42mm .55mm; }}
table.tight td {{ white-space:normal; }}
table.text {{ font-family:{SERIF}; font-size:7.2pt; }}
table.text td, table.text th {{ text-align:left; vertical-align:top; padding:.9mm 1.2mm; }}
tbody th {{ white-space:nowrap; }}
table.text tbody th {{ white-space:normal; font-family:{SERIF}; font-size:7.2pt; }}
.ci {{ color:{GREY_SOFT}; font-size:6.1pt; white-space:nowrap; }}
thead th {{ white-space:normal; }}
.note {{ font-size:7.2pt; line-height:1.27; color:{GREY_TEXT}; }}
.footer {{ position:absolute; left:12.5mm; right:12.5mm; bottom:5mm; font-family:{SANS}; font-size:6.6pt; color:{GREY_SOFT}; letter-spacing:.06em; border-top:.2mm solid {RULE}; padding-top:1.2mm; }}
.right {{ float:right; font-family:{MONO}; letter-spacing:0; }}
.refs {{ font-size:7.1pt; line-height:1.26; columns:2; column-gap:5.5mm; }}
.refs p {{ text-align:left; break-inside:avoid; margin-bottom:1mm; }}
"""
