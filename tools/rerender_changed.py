"""Переозвучує в готовому WAV лише ті фрагменти, де змінились наголоси (StyleTTS2).

    python tools/rerender_changed.py <книга> <розділ> <готовий.wav> --voice "Марічка Штирбулова" --old-stress stress_overrides.yaml

Готовий WAV розрізається на фрагменти за паузами (0,35 с точної тиші між фрагментами), фрагменти порівнюються з
тим, що дало б озвучення за поточними configs/stress*.yaml. --old-stress — які файли наголосів діяли, коли робили WAV
(решта вважається новою); змінені фрагменти озвучуються заново, решта лишається як була. Результат пишеться в новий файл
(…_v2) поруч, оригінал не змінюється.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from audiobooks.audio import save_wav, silence, unique_path
from audiobooks.books import load_book
from audiobooks.paths import CONFIGS_DIR
from audiobooks.text import spell_numbers, split_sentences, prepare_text
from audiobooks.tts.local import PAUSE_SECONDS, make_engine
from audiobooks.tts.styletts2_engine import _accent


def split_by_gaps(pcm: np.ndarray, sr: int, gap_s: float) -> list[np.ndarray]:
    """Розрізає за серіями точних нулів довжиною не менше 0,9·gap (паузи між фрагментами)."""
    need = int(gap_s * sr * 0.9)
    zero = pcm == 0
    segs, start, i, n = [], 0, 0, len(pcm)
    while i < n:
        if zero[i]:
            j = i
            while j < n and zero[j]:
                j += 1
            if j - i >= need:
                segs.append(pcm[start:i])
                start = j
            i = j
        else:
            i += 1
    segs.append(pcm[start:])
    return segs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("chapter", type=int)
    ap.add_argument("wav", type=Path)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--old-stress", nargs="+", default=["stress_overrides.yaml"], help="файли наголосів з configs/, що діяли для цього WAV")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    title, body = load_book(a.book)[a.chapter - 1]
    eng = make_engine("styletts2", a.voice, a.speed)
    pieces = split_sentences(spell_numbers(prepare_text(title, body)), eng.max_chars)

    pcm, sr = sf.read(a.wav, dtype="int16")
    segs = split_by_gaps(pcm, sr, PAUSE_SECONDS)
    print(f"Фрагментів у тексті {len(pieces)}, у WAV {len(segs)}")
    if len(segs) != len(pieces):
        print("Кількість не збігається, розрізати WAV за паузами не вдалося. Нічого не змінено.")
        return 1

    new_over = eng.overrides
    old_over = {}
    for name in a.old_stress:
        for k, v in (yaml.safe_load((CONFIGS_DIR / name).read_text(encoding="utf-8")) or {}).items():
            old_over[str(k).lower()] = _accent(v)
    changed = []
    for i, p in enumerate(pieces):
        eng.overrides = new_over
        now = eng._prepare(p)
        eng.overrides = old_over
        before = eng._prepare(p)
        if now != before:
            changed.append(i)
    eng.overrides = new_over
    print(f"Змінились наголоси у {len(changed)} з {len(pieces)} фрагментів: {[i + 1 for i in changed]}")
    if a.dry_run or not changed:
        return 0

    gap = np.frombuffer(silence(PAUSE_SECONDS, sr), dtype=np.int16)
    out = []
    for i, seg in enumerate(segs):
        if i in changed:
            seg = np.frombuffer(eng.synth(pieces[i]), dtype=np.int16)
            print(f"\r  переозвучено {changed.index(i) + 1}/{len(changed)}", end="", flush=True)
        out += [seg, gap] if i < len(segs) - 1 else [seg]
    print()
    target = unique_path(a.wav.parent, a.wav.stem, "v")
    save_wav(target, np.concatenate(out).tobytes(), sr)
    print(f"Збережено: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
