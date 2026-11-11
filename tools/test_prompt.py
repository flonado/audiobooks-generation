"""
Перевірка, чи шматок відхиляється через нашу інструкцію диктору (Gemini).
Запускає той самий шматок двома промптами (по 1 запиту на кожен), нічого не кешує.

  python tools/test_prompt.py book.fb2 -c 6 --chunk 4 --model gemini-3.8-flash-tts
"""
import argparse
import os
import sys

from google import genai

from audiobooks.books import load_book
from audiobooks.text import prepare_text
from audiobooks.tts import gemini


def attempt(client, chunk, model, voice, style, label):
    try:
        data = gemini.generate_chunk(client, chunk, model, voice, style, max_attempts=1)
        print(f"[{label}] OK ({len(data)} байт)")
    except gemini.FatalApiError as e:
        raise SystemExit(f"Зупинено: {e}")
    except RuntimeError:
        print(f"[{label}] ВІДХИЛЕНО (причину дивіться у рядку 'failed' вище)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("-c", "--chapter", type=int, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--model", default=gemini.DEFAULT_MODEL)
    ap.add_argument("--voice", default="Aoede")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    title, body = load_book(args.book)[args.chapter - 1]
    chunks = gemini.split_text_into_chunks(prepare_text(title, body))
    print(f"Розділ {args.chapter}, шматок {args.chunk}/{len(chunks)}, модель {args.model}\n")
    chunk = chunks[args.chunk - 1]
    attempt(client, chunk, args.model, args.voice, gemini.DEFAULT_STYLE, "поточна інструкція")
    attempt(client, chunk, args.model, args.voice, "Read in a neutral tone.", "мінімальна інструкція")


if __name__ == "__main__":
    main()
