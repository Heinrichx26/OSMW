"""Build Fig. 1 from an SVG source and convert it to vector PDF."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SVG_PATH = RESULTS / "fig1_mechanism.svg"
PDF_PATH = RESULTS / "fig1_mechanism.pdf"
WIDTH = 540
HEIGHT = 440
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"


def configure_fonts() -> None:
    global FONT_REGULAR, FONT_BOLD
    windows_fonts = Path("C:/Windows/Fonts")
    regular = windows_fonts / "arial.ttf"
    bold = windows_fonts / "arialbd.ttf"
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("FigureArial", str(regular)))
        pdfmetrics.registerFont(TTFont("FigureArial-Bold", str(bold)))
        FONT_REGULAR = "FigureArial"
        FONT_BOLD = "FigureArial-Bold"


TEXT = [
    {
        "x": 270,
        "y": 38,
        "text": "Same channel, task-selected workload",
        "size": 21,
        "weight": "bold",
        "anchor": "middle",
        "fill": "#111111",
        "family": "Arial, Helvetica, sans-serif",
    },
    {"x": 50, "y": 92, "text": "fixed channel", "size": 17, "weight": "bold", "fill": "#111111"},
    {
        "x": 198,
        "y": 90,
        "text": "dressed T^n channel",
        "size": 13.5,
        "weight": "bold",
        "fill": "#111111",
    },
    {"x": 198, "y": 114, "text": "chosen before task", "size": 10.5, "fill": "#5f6368"},
    {"x": 352, "y": 90, "text": "operator SRE", "size": 13, "weight": "bold", "fill": "#8a1538"},
    {"x": 352, "y": 114, "text": "= n log(4/3)", "size": 13.5, "weight": "bold", "fill": "#8a1538"},
    {"x": 50, "y": 162, "text": "measured task", "size": 10.5, "weight": "bold", "fill": "#5f6368"},
    {"x": 198, "y": 162, "text": "selected law", "size": 10.5, "weight": "bold", "fill": "#5f6368"},
    {
        "x": 334,
        "y": 162,
        "text": "fixed-error workload",
        "size": 10.5,
        "weight": "bold",
        "fill": "#5f6368",
    },
    {"x": 50, "y": 214, "text": "local density", "size": 15.2, "weight": "bold", "fill": "#111111"},
    {"x": 198, "y": 212, "text": "n local Pauli columns", "size": 12.8, "fill": "#111111"},
    {"x": 334, "y": 208, "text": "D_eta = n", "size": 12.8, "weight": "bold", "fill": "#8a1538"},
    {"x": 334, "y": 232, "text": "term K_eta = 1", "size": 9.6, "fill": "#5f6368"},
    {"x": 50, "y": 292, "text": "Pauli string", "size": 15.2, "weight": "bold", "fill": "#111111"},
    {"x": 198, "y": 290, "text": "one global column", "size": 12.8, "fill": "#111111"},
    {
        "x": 334,
        "y": 286,
        "text": "K_eta = ceil(eta 2^n)",
        "size": 12.8,
        "weight": "bold",
        "fill": "#8a1538",
    },
    {"x": 334, "y": 310, "text": "support grows as 2^n", "size": 9.6, "fill": "#5f6368"},
    {"x": 50, "y": 370, "text": "all-Pauli task", "size": 15.2, "weight": "bold", "fill": "#111111"},
    {"x": 198, "y": 368, "text": "uniform columns", "size": 12.8, "fill": "#111111"},
    {
        "x": 334,
        "y": 364,
        "text": "K_eta = ceil(eta 4^n)",
        "size": 12.8,
        "weight": "bold",
        "fill": "#8a1538",
    },
    {"x": 334, "y": 388, "text": "all-Pauli endpoint", "size": 9.6, "fill": "#5f6368"},
]


def svg_text_attrs(item: dict[str, object]) -> str:
    attrs = [
        f'x="{item["x"]}"',
        f'y="{item["y"]}"',
        f'font-size="{item["size"]}"',
        f'fill="{item["fill"]}"',
        f'font-family="{item.get("family", "Arial, Helvetica, sans-serif")}"',
    ]
    if item.get("weight"):
        attrs.append(f'font-weight="{item["weight"]}"')
    if item.get("anchor"):
        attrs.append(f'text-anchor="{item["anchor"]}"')
    return " ".join(attrs)


def build_svg() -> str:
    rows = [180, 258, 336]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect x="0" y="0" width="540" height="440" fill="#ffffff"/>',
        '<line x1="28" y1="54" x2="512" y2="54" stroke="#2f3136" stroke-width="1.4"/>',
        '<rect x="34" y="76" width="472" height="62" fill="#ffffff" stroke="#3c4043" stroke-width="1"/>',
    ]
    for y in rows:
        parts.extend(
            [
                f'<rect x="34" y="{y}" width="472" height="62" fill="#ffffff" stroke="#3c4043" stroke-width="0.85"/>',
                f'<rect x="34" y="{y}" width="6" height="62" fill="#8a1538"/>',
            ]
        )
    parts.extend(
        [
            '<text '
            + svg_text_attrs(item)
            + ">"
            + html.escape(str(item["text"]))
            + "</text>"
            for item in TEXT
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def parse_color(value: str) -> colors.Color:
    return colors.HexColor(value)


def svg_number(value: str | None, default: float = 0.0) -> float:
    if not value:
        return default
    match = re.match(r"[-+]?\d*\.?\d+", value)
    return float(match.group(0)) if match else default


def svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    root = ET.parse(svg_path).getroot()
    width = svg_number(root.get("width"), WIDTH)
    height = svg_number(root.get("height"), HEIGHT)
    pdf = canvas.Canvas(str(pdf_path), pagesize=(width, height))

    def y_pdf(y: float) -> float:
        return height - y

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "rect":
            x = svg_number(elem.get("x"))
            y = svg_number(elem.get("y"))
            w = svg_number(elem.get("width"))
            h = svg_number(elem.get("height"))
            fill = elem.get("fill", "none")
            stroke = elem.get("stroke", "none")
            stroke_width = svg_number(elem.get("stroke-width"), 1.0)
            pdf.setLineWidth(stroke_width)
            if fill != "none":
                pdf.setFillColor(parse_color(fill))
            if stroke != "none":
                pdf.setStrokeColor(parse_color(stroke))
            pdf.rect(
                x,
                y_pdf(y + h),
                w,
                h,
                stroke=1 if stroke != "none" else 0,
                fill=1 if fill != "none" else 0,
            )
        elif tag == "line":
            pdf.setStrokeColor(parse_color(elem.get("stroke", "#000000")))
            pdf.setLineWidth(svg_number(elem.get("stroke-width"), 1.0))
            pdf.line(
                svg_number(elem.get("x1")),
                y_pdf(svg_number(elem.get("y1"))),
                svg_number(elem.get("x2")),
                y_pdf(svg_number(elem.get("y2"))),
            )
        elif tag == "text":
            text = "".join(elem.itertext())
            x = svg_number(elem.get("x"))
            y = svg_number(elem.get("y"))
            size = svg_number(elem.get("font-size"), 12.0)
            weight = elem.get("font-weight", "")
            font = FONT_BOLD if weight == "bold" else FONT_REGULAR
            pdf.setFont(font, size)
            pdf.setFillColor(parse_color(elem.get("fill", "#111111")))
            anchor = elem.get("text-anchor", "start")
            if anchor == "middle":
                pdf.drawCentredString(x, y_pdf(y), text)
            elif anchor == "end":
                pdf.drawRightString(x, y_pdf(y), text)
            else:
                pdf.drawString(x, y_pdf(y), text)
    pdf.showPage()
    pdf.save()


def main() -> None:
    configure_fonts()
    RESULTS.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    svg_to_pdf(SVG_PATH, PDF_PATH)
    print(f"svg={SVG_PATH}")
    print(f"pdf={PDF_PATH}")


if __name__ == "__main__":
    main()
