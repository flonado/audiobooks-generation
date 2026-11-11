"""PDF розділу з кольоровою розміткою мовців (з data/speakers/<книга>/NN_*.json) для ручної перевірки.

    python tools/speakers_to_pdf.py data/speakers/<книга>/02_Hitagi_Crab_002.json [-o out.pdf]

Кожна репліка підсвічена кольором мовця й має маленьку мітку «Ім'я·джерело»: R = правило, L2 = модель (шар 2),
L3 = модель (шар 3), C = виправлення шару 3; «?!» = сумнівна (є у review.json). Зліва номер абзацу.
"""
import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

FONTS = Path(r"C:\Windows\Fonts")
COLORS = {"Araragi": "#cfe3ff", "Hanekawa": "#ffd9a8", "Senjogahara": "#e3d0ff", "Hachikuji": "#c9f2c7",
          "Oshino": "#f7ef9a", "Karen": "#ffc9d9", "Tsukihi": "#ffc9d9", "Shinobu": "#f6e0b8", "Other": "#dddddd"}
UNKNOWN = "#ffb3b3"
NARRATION = "#eaf2ff"  # оповідь від першої особи = Арараґі (світліший відтінок його кольору)
SOURCE = {"rule": "R", "llm:2": "L2", "llm:2?": "L2?", "llm:3": "L3", "check": "C", "check:alt": "ЧЕРГ", "llm:unknown": "L2?"}


def source(by: str | None) -> str:
    if not by:
        return ""
    return SOURCE.get(by) or SOURCE.get(by.split(":")[0], by)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()
    data = json.loads(a.json.read_text(encoding="utf-8"))
    out = a.output or a.json.with_suffix(".pdf")

    for name, f in (("B", "georgia.ttf"), ("B-Bold", "georgiab.ttf"), ("B-Italic", "georgiai.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
    pdfmetrics.registerFontFamily("B", normal="B", bold="B-Bold", italic="B-Italic", boldItalic="B-Bold")
    body = ParagraphStyle("body", fontName="B", fontSize=10, leading=17, spaceAfter=4, leftIndent=9 * mm, firstLineIndent=-9 * mm)
    h1 = ParagraphStyle("h1", fontName="B-Bold", fontSize=15, leading=20, spaceAfter=3 * mm)
    small = ParagraphStyle("small", fontName="B", fontSize=8.5, leading=13)

    def clean(t: str) -> str:
        return escape(t.replace("─", "—").replace("\u3000", " "))

    story = [Paragraph(clean(data["title"]), h1)]
    legend = "  ".join(f'<font backColor="{c}">&nbsp;{n}&nbsp;</font>' for n, c in COLORS.items() if n in ("Araragi", "Hanekawa", "Senjogahara", "Hachikuji", "Oshino"))
    story += [Paragraph(f'{legend}  <font backColor="{UNKNOWN}">&nbsp;сумнівно / без мітки&nbsp;</font>  <font backColor="{NARRATION}">&nbsp;оповідь = Арараґі&nbsp;</font>', small),
              Paragraph("Мітка після репліки: ім'я · джерело (R = правило, L2 / L3 = модель, C = виправлення правилом, ЧЕРГ (червоним) = заповнено чергуванням, перевірте, ?! = сумнівна). "
                        "Зліва номер абзацу: за ним зручно писати мені, де помилка.", small), Spacer(1, 4 * mm)]
    for i, p in enumerate(data["paragraphs"]):
        parts = [f'<font size="7" color="#888888">{i}&nbsp;&nbsp;</font>']
        for s in p["segs"]:
            if s["kind"] != "quote":
                parts.append(f'<font backColor="{NARRATION}">{clean(s["text"])}</font>' if s["text"].strip() else clean(s["text"]))
                continue
            who, flag = s.get("speaker"), s.get("flag")
            text = f"“{clean(s['text'])}”" + ("" if not s.get("open") else "…")
            if who in (None, "narrator") and who is not None:
                parts.append(f'<font backColor="{NARRATION}">{text}</font> <font size="6" color="#777777">Арараґі-оповідь·{source(s.get("by"))}</font>')
                continue
            color = UNKNOWN if (not who or flag) else COLORS.get(who, "#dddddd")
            tag = f'{who or "?"}·{source(s.get("by"))}' + (" ?!" if flag or not who else "")
            tagcolor = "#c00000" if s.get("by") == "check:alt" else "#555555"
            parts.append(f'<font backColor="{color}">{text}</font> <font size="6" color="{tagcolor}">{escape(tag)}</font>')
        story.append(Paragraph("".join(parts), body))
    SimpleDocTemplate(str(out), pagesize=A4, title=data["title"], leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=16 * mm).build(story)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
