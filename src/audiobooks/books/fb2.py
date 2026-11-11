import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .base import Chapter

NS = {"f": "http://www.gribuser.ru/xml/fictionbook/2.0"}


def _text(elem: ET.Element) -> str:
    return " ".join("".join(elem.itertext()).split())


def load_fb2(path: Path) -> list[Chapter]:
    """
    Every FB2 section that holds paragraphs directly, in document order.
    Ignores notes bodies and empty sections.
    """
    root = ET.parse(path).getroot()
    chapters: list[Chapter] = []

    def walk(section: ET.Element):
        paragraphs = [_text(p) for p in section.findall("f:p", NS)]
        paragraphs = [p for p in paragraphs if p]
        if paragraphs:
            title_el = section.find("f:title", NS)
            title = _text(title_el) if title_el is not None else f"Без назви {len(chapters) + 1}"
            chapters.append(Chapter(title, "\n".join(paragraphs)))
        for sub in section.findall("f:section", NS):
            walk(sub)

    for body in root.findall("f:body", NS):
        if body.get("name") == "notes":
            continue
        for section in body.findall("f:section", NS):
            walk(section)
    return chapters
