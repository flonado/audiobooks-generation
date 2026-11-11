"""
Міряє висоту голосу (F0) для кожного голосу StyleTTS2 на одній фразі,
щоб звузити вибір (наприклад, голос підлітка). Висота не замінює прослуховування.
  .venv-styletts\\Scripts\\python.exe tools\\styletts2_pitch_scan.py
"""
import sys

import librosa
import numpy as np

from audiobooks.tts.styletts2_engine import SAMPLE_RATE, StyleTTS2Engine, available_voices

PHRASE = ("Я не боюся, — сказав Рудько і підвів голову. "
          "Тримайся ближче до мене, ми майже прийшли до табору.")


def median_f0(pcm: bytes) -> float:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    f0, voiced, _ = librosa.pyin(audio, fmin=70, fmax=400, sr=SAMPLE_RATE)
    f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else float("nan")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for voice in available_voices():
        pcm = StyleTTS2Engine(voice).synth(PHRASE)  # reloads the model per voice: slow but simple
        rows.append((median_f0(pcm), voice))
        print(f"{rows[-1][0]:6.0f} Гц  {voice}", flush=True)
    print("\nВід найнижчого до найвищого:")
    for f0, name in sorted(rows):
        print(f"{f0:6.0f} Гц  {name}")


if __name__ == "__main__":
    main()
