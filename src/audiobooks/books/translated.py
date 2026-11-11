"""
Переклади лежать у data/translations/<книга>/NN_<назва>.txt.
Формат файлу: перший рядок - назва розділу, далі абзаци (кожен в окремому рядку).
"""
from pathlib import Path

from .base import Chapter


def load_translated(directory: Path) -> list[Chapter]:
    chapters = []
    for file in sorted(directory.glob("*.txt")):
        lines = [ln.strip() for ln in file.read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 2:
            chapters.append(Chapter(lines[0], "\n".join(lines[1:])))
    return chapters
