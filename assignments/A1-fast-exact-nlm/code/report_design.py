"""Design system for the A1 report: "Photometric" with search-grid motifs.

The direction is taken from the subject rather than from academic-template
convention. Three decisions carry it:

Step wedge instead of a gradient rule. A photographic step wedge is the
reference target used to check that an imaging system reproduces tone
correctly. This report is about reproducing an estimator exactly, so the
header rule is a real five-step grey ramp rather than decoration.

Charter for body text. Matthew Carter drew Charter for output devices that
could not render fine detail, so it stays legible under exactly the kind of
degradation this paper measures. It is also simply a better print face at
8.6 pt than the default sans.

Monospace for every measured quantity. Numbers produced by the experiment are
set in Menlo so they read as instrument output and never blend into prose.

Colour is rationed deliberately: ultramarine marks the exact-acceleration
claim, amber marks the honest limitation, and nothing else is coloured. When
two claims of different epistemic status sit on one page, colour is the
fastest way to keep the reader from merging them.

The search-grid glyph beside each section heading is a 5x5 displacement
lattice with one cell filled; the filled cell advances section by section, so
the motif is an index rather than an ornament. Its legend appears once.
"""

from __future__ import annotations

# Photometric palette.
INK = "#0E1116"
PAPER = "#FFFFFF"
WEDGE = ("#1F2430", "#4A515E", "#878E9A", "#C4C9D0", "#E8EBEF")
GREY_TEXT = "#4A515E"
GREY_SOFT = "#878E9A"
RULE = "#D8DCE2"
ULTRA = "#1B36C4"   # exact-method claim only
AMBER = "#C77800"   # honest limitation only

SERIF = "'Charter','Iowan Old Style','Palatino Linotype',Georgia,serif"
SANS = "'Futura','Avenir Next','Gill Sans','Helvetica Neue',sans-serif"
MONO = "'Menlo','SF Mono','DejaVu Sans Mono',monospace"

# Filled cell (col, row) per section, walking the lattice as an index.
GLYPH_CELLS = [(0, 2), (1, 1), (2, 0), (3, 1), (4, 2), (2, 4)]


def wedge(height_mm: float = 1.5) -> str:
    """Five-step grey ramp used as the page rule."""
    cells = "".join(
        f"<div style='flex:1;background:{c}'></div>" for c in WEDGE)
    return (f"<div style='display:flex;height:{height_mm}mm;width:100%;"
            f"margin:0 0 3mm'>{cells}</div>")


def grid_glyph(index: int, size: int = 20) -> str:
    """5x5 displacement lattice with the section's cell filled."""
    col, row = GLYPH_CELLS[index % len(GLYPH_CELLS)]
    dots = []
    for r in range(5):
        for c in range(5):
            on = (c, r) == (col, row)
            dots.append(
                f"<circle cx='{c * 5 + 2.5}' cy='{r * 5 + 2.5}' "
                f"r='{1.5 if on else 0.85}' fill='{ULTRA if on else '#C4C9D0'}'/>")
    return (f"<svg viewBox='0 0 25 25' width='{size}' height='{size}' "
            f"style='flex:none'>{''.join(dots)}</svg>")


def section(index: int, number: str, title: str) -> str:
    """Grid glyph + tracked eyebrow numeral + serif title."""
    return (
        f"<div class='sec'>{grid_glyph(index)}"
        f"<div><div class='eyebrow'>Section {number}</div>"
        f"<h2>{title}</h2></div></div>")


def readout(items: list[tuple[str, str]]) -> str:
    """Measured headline values, set as instrument output on hairlines."""
    cells = "".join(
        f"<div class='ro'><div class='ro-v'>{value}</div>"
        f"<div class='ro-l'>{label}</div></div>" for value, label in items)
    return f"<div class='readout'>{cells}</div>"


def stamp(max_error: str) -> str:
    """The signature element: a calibration stamp for the equivalence claim."""
    return (
        "<div class='stamp'>"
        "<div class='stamp-hd'>Equivalence check</div>"
        f"<div class='stamp-row'><span>max |&Delta;| vs direct NLM</span>"
        f"<b>{max_error}</b></div>"
        "<div class='stamp-row'><span>verdict</span><b>OUTPUT-IDENTICAL</b></div>"
        "</div>")


CSS = f"""
@page {{ size: A4; margin: 0; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; color:{INK}; background:{PAPER}; font-family:{SERIF};
        font-size:8.7pt; line-height:1.32; -webkit-font-smoothing:antialiased; }}
.page {{ width:210mm; height:297mm; padding:11mm 12.5mm 10mm; position:relative;
         page-break-after:always; overflow:hidden; }}
.page:last-child {{ page-break-after:auto; }}

.kicker {{ font-family:{SANS}; font-size:7.6pt; font-weight:600; color:{ULTRA};
           letter-spacing:.16em; text-transform:uppercase; margin:0 0 1.6mm; }}
h1 {{ font-family:{SERIF}; font-weight:700; font-size:21pt; line-height:1.06;
      margin:0 0 1.4mm; letter-spacing:-.2pt; }}
.subtitle {{ font-family:{SERIF}; font-style:italic; font-size:10.6pt;
             color:{GREY_TEXT}; margin:0 0 1.8mm; }}
.meta {{ font-family:{MONO}; font-size:7.4pt; color:{GREY_SOFT}; margin:0 0 2.5mm; }}
.pagehd {{ font-family:{SERIF}; font-weight:700; font-size:14.5pt; margin:0 0 1mm; }}

.readout {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0;
            border-top:.5mm solid {INK}; border-bottom:.2mm solid {RULE};
            margin:0 0 3.4mm; }}
.ro {{ padding:1.8mm 2.5mm 1.9mm 0; border-right:.2mm solid {RULE}; }}
.ro:last-child {{ border-right:0; }}
.ro-v {{ font-family:{MONO}; font-size:13pt; font-weight:600; color:{INK};
         line-height:1; letter-spacing:-.3pt; }}
.ro-l {{ font-family:{SANS}; font-size:6.9pt; color:{GREY_SOFT}; margin-top:1.3mm;
         letter-spacing:.05em; line-height:1.25; }}

.sec {{ display:flex; gap:2.2mm; align-items:flex-start; margin:3.2mm 0 1.4mm; }}
.sec:first-child {{ margin-top:0; }}
.eyebrow {{ font-family:{SANS}; font-size:6.6pt; font-weight:600; color:{GREY_SOFT};
            letter-spacing:.18em; text-transform:uppercase; margin-bottom:.5mm; }}
h2 {{ font-family:{SERIF}; font-size:11pt; font-weight:700; margin:0; line-height:1.12; }}

p {{ margin:0 0 1.7mm; text-align:justify; hyphens:auto; }}
ul {{ padding-left:4mm; margin:1mm 0 1.6mm; }}
li {{ margin-bottom:.9mm; }}
b {{ font-weight:700; }}
.num {{ font-family:{MONO}; font-size:.94em; }}

.two {{ display:grid; grid-template-columns:1fr 1fr; gap:5.5mm; }}
.ratio {{ display:grid; grid-template-columns:1.12fr .88fr; gap:4.5mm; }}

.eq {{ border-left:.5mm solid {ULTRA}; background:#F7F8FC; padding:1.6mm 2.4mm;
       text-align:center; margin:1.7mm 0; font-family:{SERIF}; font-size:9.3pt;
       line-height:1.4; }}
.eq .tag {{ float:right; font-family:{MONO}; font-size:7.4pt; color:{GREY_SOFT}; }}

.stamp {{ border:.35mm solid {ULTRA}; padding:1.7mm 2.2mm; margin:1.8mm 0;
          font-family:{MONO}; }}
.stamp-hd {{ font-family:{SANS}; font-size:6.6pt; font-weight:600; color:{ULTRA};
             letter-spacing:.18em; text-transform:uppercase; margin-bottom:1.2mm; }}
.stamp-row {{ display:flex; justify-content:space-between; font-size:7.8pt;
              color:{GREY_TEXT}; line-height:1.5; }}
.stamp-row b {{ color:{INK}; }}

.limit {{ border-left:.5mm solid {AMBER}; background:#FDF8F0; padding:1.7mm 2.2mm;
          margin:1.8mm 0; }}
.limit-hd {{ font-family:{SANS}; font-size:6.6pt; font-weight:600; color:{AMBER};
             letter-spacing:.18em; text-transform:uppercase; margin-bottom:1mm; }}

figure {{ margin:0; }}
figure img {{ width:100%; display:block; }}
figcaption,.caption {{ font-size:7.3pt; line-height:1.25; color:{GREY_TEXT};
                       margin-top:1mm; text-align:justify; }}
figcaption b,.caption b {{ font-family:{SANS}; font-size:6.9pt; font-weight:600;
                           letter-spacing:.09em; text-transform:uppercase; color:{INK}; }}
.fullimg {{ width:100%; height:107mm; object-fit:contain; }}
.cropimg {{ width:100%; height:81mm; object-fit:contain; }}
.plotsmall {{ width:100%; height:55mm; object-fit:contain; }}
.flow {{ margin-top:2mm; border-top:.2mm solid {RULE}; padding-top:2.2mm; }}
.flow svg {{ width:100%; display:block; }}

table {{ width:100%; border-collapse:collapse; font-family:{MONO}; font-size:7.3pt;
         margin:1.1mm 0 1.5mm; }}
th,td {{ padding:1.25mm 1mm; text-align:center; border:0;
         border-bottom:.2mm solid {RULE}; }}
thead th {{ font-family:{SANS}; font-size:6.7pt; font-weight:600; color:{GREY_SOFT};
            letter-spacing:.07em; text-transform:uppercase;
            border-bottom:.4mm solid {INK}; }}
tbody th {{ font-family:{MONO}; font-size:7.3pt; color:{INK}; font-weight:600; }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:.4mm solid {INK}; }}
td b {{ color:{ULTRA}; }}

.note {{ font-size:7.3pt; line-height:1.28; color:{GREY_TEXT}; }}
.footer {{ position:absolute; left:12.5mm; right:12.5mm; bottom:5mm;
           font-family:{SANS}; font-size:6.6pt; color:{GREY_SOFT};
           letter-spacing:.06em; border-top:.2mm solid {RULE}; padding-top:1.2mm; }}
.right {{ float:right; font-family:{MONO}; letter-spacing:0; }}
.refs {{ font-size:7.3pt; line-height:1.28; columns:2; column-gap:5.5mm; }}
.refs p {{ text-align:left; break-inside:avoid; margin-bottom:1.1mm; }}
"""
