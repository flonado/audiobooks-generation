r"""
Створює голос Higgs 3 (клонування в контексті): wav-зразок (моно, 24 кГц) + його текст у data/models/higgs3/voices.
Навчання немає: рушій кодує зразок кодеком і ставить його в промпт. Найкраще 6-14 с чистого мовлення одного мовця,
текст має збігатися із записаним дослівно (з ним схожість помітно вища).

  .venv-higgs\Scripts\python.exe tools\make_higgs_voice.py "Арараґі" data\voice_examples\araragi_v2\wavs\araragi_v2_002.wav --text "Любите ви свою матір…"
  .venv-higgs\Scripts\python.exe tools\make_higgs_voice.py "Сенджоґахара" clip.wav --from-csv data\voice_examples\senjogahara_ep3\metadata.csv
  .venv-higgs\Scripts\python.exe tools\make_higgs_voice.py "Хачікуджі" a.wav b.wav --from-csv m.csv   # кілька кліпів склеюються
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from audiobooks.tts.higgs_engine import SAMPLE_RATE, VOICES_DIR

MAX_SEC = 14.0  # стільки секунд зразка бере рушій (довший = повільніша генерація)


def csv_text(csv_path: Path, wav: Path) -> str:
    """metadata.csv у форматі «wavs/x.wav|текст|мовець» (як у наборах для навчання)."""
    for row in csv.reader(csv_path.read_text(encoding="utf-8").splitlines(), delimiter="|"):
        if row and Path(row[0]).name == wav.name:
            return row[1].strip()
    raise SystemExit(f"У {csv_path} немає рядка для {wav.name}")


def load_mono(path: Path) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("wavs", nargs="+", type=Path)
    ap.add_argument("--text", help="дослівний текст зразка (для кількох кліпів — на всі разом)")
    ap.add_argument("--from-csv", type=Path, help="узяти текст кліпів з metadata.csv")
    ap.add_argument("--gap", type=float, default=0.25, help="пауза між склеєними кліпами, с")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if not args.text and not args.from_csv:
        raise SystemExit("Потрібен --text або --from-csv.")
    texts = [args.text] if args.text else [csv_text(args.from_csv, w) for w in args.wavs]
    gap = np.zeros(int(args.gap * SAMPLE_RATE), dtype=np.float32)
    parts, used, total = [], [], 0.0
    for i, w in enumerate(args.wavs):
        clip = load_mono(w)
        if used and total + len(clip) / SAMPLE_RATE > MAX_SEC:
            print(f"  {w.name} пропущено: разом було б довше за {MAX_SEC:.0f} с (рушій бере лише стільки)")
            continue
        parts += [clip, gap]
        total += len(clip) / SAMPLE_RATE + args.gap
        used.append(i)
    y = np.concatenate(parts[:-1])
    y = y / max(np.abs(y).max(), 1e-6) * 0.9
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    out = VOICES_DIR / f"{args.name}.wav"
    sf.write(out, y, SAMPLE_RATE, subtype="PCM_16")
    text = args.text if args.text else " ".join(texts[i] for i in used)
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"Збережено {out}  {len(y) / SAMPLE_RATE:.1f} с")
    if len(y) / SAMPLE_RATE > MAX_SEC:
        print(f"  Увага: зразок довший за {MAX_SEC:.0f} с, рушій обріже запис, а текст лишиться повним: візьміть коротший кліп.")


if __name__ == "__main__":
    main()
