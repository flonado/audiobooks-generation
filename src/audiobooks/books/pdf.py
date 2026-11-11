import re
from pathlib import Path

from .base import Chapter


def load_pdf(path: Path) -> list[Chapter]:
    """Every page with text becomes one chapter (PDFs have no reliable chapter structure)."""
    from pypdf import PdfReader  # optional dependency

    chapters = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        raw = page.extract_text() or ""
        raw = re.sub(r"^\s*\d+\s*$", "", raw, flags=re.MULTILINE)  # standalone page numbers
        raw = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw)               # words broken by a hyphen
        text = " ".join(raw.split())
        if text:
            chapters.append(Chapter(f"Сторінка {number}", text))
    return chapters
