import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from .base import Chapter

MIN_CHAPTER_CHARS = 500  # shorter spine items are covers, blank pages, contents
OPF_NS = {"o": "http://www.idpf.org/2007/opf"}
CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}


class _ParagraphParser(HTMLParser):
    """Collects <p> paragraphs and the first heading of an XHTML page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.heading = ""
        self._buf: list[str] | None = None
        self._in_heading = False
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "p":
            self._buf = []
        elif tag in ("h1", "h2", "h3", "h4") and not self.heading:
            self._in_heading = True
            self._buf = []

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1
        elif tag == "p" and self._buf is not None:
            text = " ".join("".join(self._buf).split())
            if text:
                self.paragraphs.append(text)
            self._buf = None
        elif tag in ("h1", "h2", "h3", "h4") and self._in_heading:
            self.heading = " ".join("".join(self._buf or []).split())
            self._in_heading = False
            self._buf = None

    def handle_data(self, data):
        if not self._skip and self._buf is not None:
            self._buf.append(data)


def _opf_path(z: zipfile.ZipFile) -> str:
    container = ET.fromstring(z.read("META-INF/container.xml"))
    return container.find(".//c:rootfile", CONTAINER_NS).get("full-path")


def _toc_labels(z: zipfile.ZipFile, opf: str, manifest: dict[str, str]) -> list[tuple[str, str]]:
    """[(spine href, label)] from the EPUB3 nav document; empty if there is none."""
    base = posixpath.dirname(opf)
    for href in manifest.values():
        full = posixpath.normpath(posixpath.join(base, href))
        if full.endswith(("toc.xhtml", "nav.xhtml")) and full in z.namelist():
            data = z.read(full).decode("utf-8", "replace")
            nav = re.search(r'<nav[^>]*epub:type="toc".*?</nav>', data, re.S)
            if nav:
                return [
                    (posixpath.basename(h.split("#")[0]), " ".join(re.sub(r"<[^>]+>", "", t).split()))
                    for h, t in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', nav.group(0), re.S)
                ]
    return []


def _story_name(label: str) -> str:
    """'CHAPTER ONE HITAGI CRAB' -> 'Hitagi Crab'."""
    return re.sub(r"^CHAPTER\s+\w+\s+", "", label, flags=re.I).title()


def load_epub(path: Path) -> list[Chapter]:
    """
    Spine items with enough text become chapters. Title = story name from the
    table of contents + the page heading (e.g. "Hitagi Crab 001").
    """
    with zipfile.ZipFile(path) as z:
        opf = _opf_path(z)
        base = posixpath.dirname(opf)
        root = ET.fromstring(z.read(opf))
        manifest = {i.get("id"): i.get("href") for i in root.find("o:manifest", OPF_NS)}
        spine = [manifest[i.get("idref")] for i in root.find("o:spine", OPF_NS)]
        toc = _toc_labels(z, opf, manifest)

        # Story = the latest TOC entry that starts at or before this spine item.
        toc_index = {}
        for href, label in toc:
            for pos, item in enumerate(spine):
                if posixpath.basename(item) == href:
                    toc_index[pos] = label

        chapters: list[Chapter] = []
        story = ""
        for pos, href in enumerate(spine):
            label = toc_index.get(pos, "")
            if label.upper().startswith("CHAPTER"):
                story = _story_name(label)
            html = z.read(posixpath.normpath(posixpath.join(base, href))).decode("utf-8", "replace")
            parser = _ParagraphParser()
            parser.feed(html)
            if sum(len(p) for p in parser.paragraphs) < MIN_CHAPTER_CHARS:
                continue
            heading = parser.heading
            if heading.isdigit() and story:
                title = f"{story} {heading}"
            else:
                title = heading or f"{story} {len(chapters) + 1}".strip()
            chapters.append(Chapter(title, "\n".join(parser.paragraphs)))
    return chapters
