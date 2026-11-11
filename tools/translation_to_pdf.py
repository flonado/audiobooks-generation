"""Збирає PDF з папки перекладених txt-розділів (перший рядок файлу — назва розділу, далі абзаци по рядку).

    python tools/translation_to_pdf.py data/translations/<книга> [-o out.pdf] [--title "Назва"]
"""
import argparse
import sys
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A5
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape

FONTS = Path(r"C:\Windows\Fonts")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--title")
    a = ap.parse_args()

    for name, f in (("Body", "georgia.ttf"), ("Body-Bold", "georgiab.ttf"), ("Body-Italic", "georgiai.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")

    title = a.title or a.folder.name.replace("_", " ")
    out = a.output or a.folder.with_suffix(".pdf")
    body = ParagraphStyle("body", fontName="Body", fontSize=10, leading=14.5, alignment=TA_JUSTIFY,
                          firstLineIndent=5 * mm, spaceAfter=1.5, hyphenationLang=None)
    h1 = ParagraphStyle("h1", fontName="Body-Bold", fontSize=17, leading=22, alignment=TA_CENTER, spaceAfter=10 * mm)
    cover = ParagraphStyle("cover", parent=h1, fontSize=22, leading=28)

    def footer(canv, doc):
        canv.setFont("Body", 8)
        canv.drawCentredString(A5[0] / 2, 9 * mm, str(doc.page))

    doc = BaseDocTemplate(str(out), pagesize=A5, title=title, leftMargin=16 * mm, rightMargin=16 * mm,
                          topMargin=16 * mm, bottomMargin=18 * mm)
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                                                             leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
                                      onPage=footer)])
    story = [Spacer(1, 60 * mm), Paragraph(escape(title), cover), PageBreak()]
    files = sorted(a.folder.glob("*.txt"))
    for i, f in enumerate(files):
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        story.append(Paragraph(escape(lines[0]), h1))
        story += [Paragraph(escape(l), body) for l in lines[1:]]
        if i < len(files) - 1:
            story.append(PageBreak())
    doc.build(story)
    print(f"{out}  ({len(files)} розділів, {out.stat().st_size // 1024} КБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
