r"""
Готує розділи до збирання .m4b: обрізає гучний «хлопок» наприкінці файлів Gemini (після тиші в останні ~0.2 с рівень
стрибає майже до 0 дБ) і додає паузу між розділами. Оригінали не змінюються: чисті копії лягають у --out-dir,
а конфіг для `audiobook m4b` пишеться поруч із вхідним.

  .venv-styletts\Scripts\python.exe tools\clean_chapter_edges.py configs\audiobook_1-20.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from audiobooks.paths import OUTPUT_DIR, ROOT


def db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def trim_end_burst(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Прибирає гучний сплеск наприкінці (до 0.5 с), якщо перед ним є тиша; повертає (звук, скільки секунд обрізано).
    Файл, що закінчується тишею (StyleTTS), не чіпає."""
    n = int(0.02 * sr)
    frames = len(y) // n
    level = [db(y[i * n:(i + 1) * n]) for i in range(frames)]
    i = frames
    while i > 0 and level[i - 1] > -45:  # гучний хвіст
        i -= 1
    loud = frames - i
    quiet = 0
    while i - quiet > 0 and level[i - quiet - 1] <= -45:
        quiet += 1
    if 0 < loud <= 25 and quiet >= 3:  # гучний хвіст після тиші, коротший за 0.5 с
        return y[: i * n], (len(y) - i * n) / sr
    return y, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--pause", type=float, default=1.0, help="тиша після кожного розділу, с")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR / "m4b_clean"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ch in cfg["chapters"]:
        src = ROOT / ch["file"]
        y, sr = sf.read(src, dtype="float32")
        y = y if y.ndim == 1 else y.mean(1)
        y, cut = trim_end_burst(y, sr)
        # обидві частини розділу рівно затухають і закінчуються тишею
        fade = int(0.05 * sr)
        y = y.copy()
        y[-fade:] *= np.linspace(1, 0, fade)
        y = np.concatenate([y, np.zeros(int(args.pause * sr), dtype=np.float32)])
        dst = out / (src.stem + "_clean.wav")
        sf.write(dst, y, sr, subtype="PCM_16")
        ch["file"] = dst.relative_to(ROOT).as_posix()
        print(f"{ch['title']:14s} обрізано {cut:.2f} с  → {dst.name}")
    cfg["output"] = cfg["output"].replace(".m4b", "_clean.m4b")
    new_cfg = cfg_path.with_name(cfg_path.stem + "_clean.yaml")
    new_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("Конфіг:", new_cfg)


if __name__ == "__main__":
    main()
