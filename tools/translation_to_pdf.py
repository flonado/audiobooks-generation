"""Збирає PDF з папки перекладених txt-розділів (перший рядок файлу — назва розділу, далі абзаци по рядку).

    python tools/translation_to_pdf.py data/translations/<книга> [-o out.pdf] [--title "Назва"]
        [--cover обкладинка.jpg] [--inserts ілюстрація1.jpg ілюстрація2.jpg ...]
        [--before "файл-розділу.txt=картинка1.jpg,картинка2.jpg" ...]

--cover ставиться перед титульною сторінкою, --inserts (повносторінкові картинки) після неї, перед текстом,
--before вставляє картинки перед вказаним розділом (за іменем файлу), по одній на сторінку.
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
from reportlab.platypus import BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape

FONTS = Path(r"C:\Windows\Fonts")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--title")
    ap.add_argument("--cover", type=Path, help="картинка обкладинки перед титульною сторінкою")
    ap.add_argument("--before", nargs="*", default=[], metavar="ФАЙЛ=КАРТИНКА,КАРТИНКА",
                    help="картинки перед розділом: ім'я файлу розділу, '=', картинки через кому")
    ap.add_argument("--inserts", type=Path, nargs="*", default=[], help="повносторінкові картинки після титульної")
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
    text_tpl = PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                                                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
                            onPage=footer)
    # сторінки з картинками: без полів і без номера
    img_tpl = PageTemplate(id="img", frames=[Frame(0, 0, A5[0], A5[1], leftPadding=0, rightPadding=0,
                                                   topPadding=0, bottomPadding=0)])
    # перша сторінка береться з першого шаблону: з обкладинкою це «картинка»
    doc.addPageTemplates([img_tpl, text_tpl] if a.cover else [text_tpl, img_tpl])

    def picture(path: Path):
        """Картинка на всю сторінку зі збереженням пропорцій (по центру)."""
        from PIL import Image as PILImage
        w, h = PILImage.open(path).size
        k = min(A5[0] / w, A5[1] / h)
        return [Spacer(1, (A5[1] - h * k) / 2 - 0.5), Image(str(path), width=w * k, height=h * k)]

    story = []
    if a.cover:
        story += picture(a.cover) + [NextPageTemplate("p"), PageBreak()]
    story += [Spacer(1, 60 * mm), Paragraph(escape(title), cover)]
    for img in a.inserts:
        story += [NextPageTemplate("img"), PageBreak()] + picture(img)
    story += [NextPageTemplate("p"), PageBreak()]
    before = {k: [Path(x) for x in v.split(",")] for k, v in (item.split("=", 1) for item in a.before)}
    files = sorted(a.folder.glob("*.txt"))
    for i, f in enumerate(files):
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        story.append(Paragraph(escape(lines[0]), h1))
        story += [Paragraph(escape(l), body) for l in lines[1:]]
        if i < len(files) - 1:
            for img in before.get(files[i + 1].name, []):
                story += [NextPageTemplate("img"), PageBreak()] + picture(img)
            story += [NextPageTemplate("p"), PageBreak()]
    doc.build(story)
    print(f"{out}  ({len(files)} розділів, {out.stat().st_size // 1024} КБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
