"""Єдине місце, де визначені директорії проєкту."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BOOKS_DIR = ROOT / "books"                  # вхідні книги (.fb2, .epub, .pdf)
CONFIGS_DIR = ROOT / "configs"              # yaml-конфіги збирання .m4b

DATA_DIR = ROOT / "data"                    # усе згенероване / завантажене (не в git)
CACHE_DIR = DATA_DIR / "cache"              # кеш шматків аудіо (cache/gemini)
MODELS_DIR = DATA_DIR / "models"            # локальні моделі (piper, ukrainian-tts, styletts2, lapa)
OUTPUT_DIR = DATA_DIR / "output"            # озвучені розділи (output/gemini, output/local, ...)
TRANSLATIONS_DIR = DATA_DIR / "translations"  # переклади по розділах (<книга>/NN_назва.txt)
AUDIOBOOKS_DIR = DATA_DIR / "audiobooks"    # готові .m4b
