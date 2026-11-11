"""Локальне озвучення (без API і лімітів): piper, ukrainian-tts, styletts2."""
import sys
import time
from pathlib import Path

from ..audio import save_wav, silence, unique_path
from ..books import load_book
from ..paths import OUTPUT_DIR
from ..text import parse_selection, prepare_text, slugify, spell_numbers, split_sentences

PAUSE_SECONDS = 0.35
ENGINES = ("piper", "ukrainian-tts", "styletts2", "higgs3")
DEFAULT_VOICES = {"piper": "tetiana", "ukrainian-tts": "tetiana", "styletts2": "Тетяна Гончарова",
                  "higgs3": "(без референсу)"}


def make_engine(engine: str, voice: str, speed: float = 1.0, **tuning):
    """Engines are imported lazily: each one lives in its own virtual environment."""
    if engine == "piper":
        from .piper_engine import PiperEngine as cls
    elif engine == "ukrainian-tts":
        from .ukrainian_tts_engine import UkrainianTtsEngine as cls
    elif engine == "styletts2":
        from .styletts2_engine import StyleTTS2Engine as cls
    elif engine == "higgs3":
        from .higgs_engine import HiggsEngine as cls
    else:
        raise SystemExit(f"Невідомий рушій '{engine}'. Доступні: {', '.join(ENGINES)}")
    return cls(voice, speed, **tuning)


def synthesize_text(eng, text: str, max_chars: int | None = None, label: str = "фрагмент") -> bytes:
    pieces = split_sentences(spell_numbers(text), max_chars or eng.max_chars)
    gap = None
    pcm = []
    start = time.time()
    for i, piece in enumerate(pieces, start=1):
        pcm.append(eng.synth(piece))
        gap = gap or silence(PAUSE_SECONDS, eng.sample_rate)
        if i < len(pieces):
            pcm.append(gap)
        print(f"\r  {label} {i}/{len(pieces)}  ({time.time() - start:.0f}с)", end="", flush=True)
    print()
    return b"".join(pcm)


def _tuning(args) -> dict:
    return {"noise_scale": args.noise_scale, "noise_w": args.noise_w, "volume": args.volume}


def run(args):
    """Handler of `audiobook local`."""
    sys.stdout.reconfigure(encoding="utf-8")
    chapters = load_book(args.book)
    voice = args.voice or DEFAULT_VOICES[args.engine]
    eng = make_engine(args.engine, voice, args.speed, **_tuning(args))
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR / "local"

    for n in parse_selection(args.chapters, len(chapters)):
        title, body = chapters[n - 1]
        print(f"\n=== {n}. {title} ===")
        pcm = synthesize_text(eng, prepare_text(title, body), args.max_chars)
        out = unique_path(out_dir, f"{n:02d}_{slugify(title)}_{args.engine}_{slugify(voice)}", "v")
        save_wav(out, pcm, eng.sample_rate)


def run_sample(args):
    """Handler of `audiobook sample`: one excerpt through the chosen engine(s)/voice(s), to compare by ear."""
    sys.stdout.reconfigure(encoding="utf-8")
    chapters = load_book(args.book)
    title, body = chapters[args.chapter - 1]
    if args.text:
        text = args.text
    else:
        text = prepare_text(title, body)[: args.chars]
        text = text[: text.rfind(" ")] if " " in text else text
    print(f"Зразок: {len(text)} символів.")
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR / "samples"

    engines = [args.engine] if args.engine else list(ENGINES)
    for engine_name in engines:
        voices = [args.voice] if args.voice else _all_voices(engine_name)
        for voice in voices:
            print(f"\n[{engine_name} / {voice}]")
            try:
                eng = make_engine(engine_name, voice, args.speed, **_tuning(args))
                pcm = synthesize_text(eng, text, args.max_chars, "шматок")
                save_wav(out_dir / f"{engine_name}_{slugify(voice)}.wav", pcm, eng.sample_rate)
            except ImportError as e:
                print(f"  Пропущено: рушій не встановлено в цьому venv ({e}).")
                break
            except Exception as e:  # noqa: BLE001  (a broken voice must not stop the whole comparison)
                print(f"  Помилка: {e}")


def _all_voices(engine: str) -> list[str]:
    if engine == "piper":
        from .piper_engine import VOICES
        return list(VOICES)
    if engine == "ukrainian-tts":
        from .ukrainian_tts_engine import VOICES
        return list(VOICES)
    if engine == "higgs3":
        from .higgs_engine import available_voices
        return available_voices()
    from .styletts2_engine import available_voices
    return available_voices()
