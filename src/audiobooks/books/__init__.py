from pathlib import Path

from ..paths import BOOKS_DIR, TRANSLATIONS_DIR
from .base import Chapter
from .epub import load_epub
from .fb2 import load_fb2
from .pdf import load_pdf
from .translated import load_translated

__all__ = ["Chapter", "resolve_book", "load_book"]


def resolve_book(name: str) -> Path:
    """
    Accepts a path, a file name from books/, or the name of a translation folder
    from data/translations/.
    """
    for candidate in (Path(name), BOOKS_DIR / name, TRANSLATIONS_DIR / name):
        if candidate.exists():
            return candidate
    raise SystemExit(f"Книгу '{name}' не знайдено ні як шлях, ні в {BOOKS_DIR}, ні в {TRANSLATIONS_DIR}.")


def load_book(name: str) -> list[Chapter]:
    path = resolve_book(name)
    if path.is_dir():
        return load_translated(path)
    loaders = {".fb2": load_fb2, ".epub": load_epub, ".pdf": load_pdf}
    try:
        return loaders[path.suffix.lower()](path)
    except KeyError:
        raise SystemExit(f"Непідтримуваний формат: {path.suffix} (підтримуються fb2, epub, pdf)")
