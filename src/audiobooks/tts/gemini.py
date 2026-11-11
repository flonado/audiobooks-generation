"""Озвучення через Gemini TTS з кешем шматків, поділом шматків при відмовах і відновленням."""
import hashlib
import re
import time
from pathlib import Path

from ..audio import save_wav, silence, unique_path
from ..books import load_book
from ..paths import CACHE_DIR, OUTPUT_DIR
from ..text import parse_selection, prepare_text, slugify

DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
SAMPLE_RATE = 24000
# NB: the style text is part of the cache key. Changing it invalidates every cached chunk.
DEFAULT_STYLE = (
    "You are a professional audiobook narrator reading a Ukrainian fantasy novel aloud. "
    "Read in Ukrainian with correct stress. Use a warm, expressive voice with natural, "
    "unhurried pacing. Pause briefly between sentences and a little longer between "
    "paragraphs. In dialogue (lines starting with a dash) change intonation to fit the "
    "speaker's emotion, such as anger, fear, irony or tenderness, without exaggerating. "
    "Raise tension and speed during action scenes and soften during quiet moments. "
    "Keep the same voice, pitch and pace consistent throughout. Do not read these "
    "instructions aloud; read only the text below."
)


class FatalApiError(Exception):
    """Errors that retrying or splitting the chunk cannot fix (daily quota, unknown model)."""


def split_text_into_chunks(text: str, max_chars: int = 1500) -> list[str]:
    """Sentence-aware chunks. The exact output is part of the cache key: do not change casually."""
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            words = sentence.split(" ")
            sub_chunk = ""
            for word in words:
                if len(sub_chunk) + len(word) + 1 <= max_chars:
                    sub_chunk += (" " if sub_chunk else "") + word
                else:
                    if sub_chunk:
                        chunks.append(sub_chunk)
                    sub_chunk = word
            if sub_chunk:
                current_chunk = sub_chunk
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def chunk_cache_path(cache_dir: Path, idx: int, chunk: str, model: str, voice: str, style: str) -> Path:
    key = f"{model}|{voice}|{style}|{chunk}".encode("utf-8")
    return cache_dir / f"chunk_{idx:04d}_{hashlib.sha256(key).hexdigest()[:12]}.pcm"


def generate_chunk(client, chunk: str, model: str, voice: str, style: str, max_attempts: int = 3) -> bytes:
    from google.genai import types

    prompt = (
        f"### DIRECTOR'S NOTES (guidance only, never spoken aloud)\n{style}\n\n"
        f"#### TRANSCRIPT (speak only this text)\n{chunk}"
    )
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                ),
            )
            candidate = response.candidates[0] if response.candidates else None
            if candidate is None or candidate.content is None or not candidate.content.parts:
                reason = candidate.finish_reason if candidate else getattr(response.prompt_feedback, "block_reason", None)
                raise ValueError(f"Model returned no content (finish_reason/block_reason: {reason}).")
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
            raise ValueError("No inline audio data returned by model.")

        except Exception as e:
            message = str(e)
            if "RESOURCE_EXHAUSTED" in message:
                if "PerDay" in message:
                    raise FatalApiError(
                        "Вичерпано денний ліміт запитів для цієї моделі. Спробуйте іншу модель або завтра."
                    ) from e
                # Per-minute limit: wait as the API asks, without spending a retry attempt.
                match = re.search(r"retry in ([\d.]+)s", message)
                wait = float(match.group(1)) + 2 if match else 30
                print(f"  Ліміт запитів за хвилину, чекаю {wait:.0f}с...")
                time.sleep(wait)
                attempt -= 1
                continue
            if "NOT_FOUND" in message:
                raise FatalApiError(f"Модель не знайдено: {model}") from e
            print(f"  Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                print(f"  Retrying in {5 * attempt}s...")
                time.sleep(5 * attempt)

    raise RuntimeError("Failed to generate audio chunk after all attempts.")


def generate_with_split_fallback(
    client, chunk: str, model: str, voice: str, style: str, delay: float, depth: int = 0
) -> bytes:
    """
    Generate one chunk. If the model refuses it (FinishReason.OTHER / SAFETY on a violent
    passage), split it in half by sentences and generate the halves separately.
    No sentence is ever dropped: if the smallest part still fails, the error propagates.
    """
    try:
        # The refusal repeats for the same text, so a retry only burns quota: split right away.
        return generate_chunk(client, chunk, model, voice, style, max_attempts=1)
    except RuntimeError:
        sentences = re.split(r"(?<=[.!?…]) +", chunk)
        if depth >= 2 or len(sentences) < 2:
            raise
        mid = len(sentences) // 2
        halves = [" ".join(sentences[:mid]), " ".join(sentences[mid:])]
        print(f"  Шматок не проходить, ділю на {len(halves)} частини (глибина {depth + 1})...")
        parts = []
        for half in halves:
            time.sleep(delay)
            parts.append(generate_with_split_fallback(client, half, model, voice, style, delay, depth + 1))
        return silence(0.4, SAMPLE_RATE).join(parts)


def synthesize(client, text: str, output_path: Path, model: str, voice: str, style: str,
               cache_dir: Path, delay: float, reuse_from: list[str]):
    chunks = split_text_into_chunks(text, max_chars=1500)
    print(f"{len(text)} символів, {len(chunks)} шматків.")
    gap = silence(0.4, SAMPLE_RATE)
    pcm = []

    for idx, chunk in enumerate(chunks, start=1):
        chunk_file = chunk_cache_path(cache_dir, idx, chunk, model, voice, style)
        if not chunk_file.exists():
            for other_model in reuse_from:  # chunk cached by another model: reuse it
                other = chunk_cache_path(cache_dir, idx, chunk, other_model, voice, style)
                if other.exists():
                    print(f"Шматок {idx}: знайдено в кеші моделі {other_model}.")
                    chunk_file = other
                    break
        if chunk_file.exists():
            print(f"Шматок {idx}/{len(chunks)} з кешу.")
            data = chunk_file.read_bytes()
        else:
            print(f"Генерація шматка {idx}/{len(chunks)} ({len(chunk)} символів)...")
            data = generate_with_split_fallback(client, chunk, model, voice, style, delay)
            tmp = chunk_file.with_suffix(".tmp")  # atomic write: no corrupt cache on interrupt
            tmp.write_bytes(data)
            tmp.replace(chunk_file)
            time.sleep(delay)
        pcm.append(data)
        if idx < len(chunks):
            pcm.append(gap)

    save_wav(output_path, b"".join(pcm), SAMPLE_RATE)


def run(args):
    """Handler of `audiobook gemini`."""
    import os

    from google import genai

    chapters = load_book(args.book)
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Задайте змінну GEMINI_API_KEY.")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    cache_dir = CACHE_DIR / "gemini"
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR / "gemini"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reuse_from = [m.strip() for m in args.reuse_from.split(",") if m.strip()]

    for n in parse_selection(args.chapters, len(chapters)):
        title, body = chapters[n - 1]
        print(f"\n=== {n}. {title} ===")
        out = unique_path(out_dir, f"{n:02d}_{slugify(title)}", f"{args.model}_{args.voice}")
        try:
            synthesize(client, prepare_text(title, body), out, args.model, args.voice, args.style,
                       cache_dir, args.delay, reuse_from)
        except FatalApiError as e:
            raise SystemExit(f"\nЗупинено: {e}\nГотові шматки збережено в кеші, наступний запуск продовжить з цього місця.")
