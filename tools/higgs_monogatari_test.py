r"""
Проба Higgs 3: по одній репліці Сенджоґахари, Арараґі, Ханекави й Хачікуджі з українського перекладу «Моногатарі».
Один раз завантажує модель, далі перемикає референсні голоси (data/models/higgs3/voices).

  .venv-higgs\Scripts\python.exe tools\higgs_monogatari_test.py [--dtype fp32] [--only Арараґі] [--tags "<|emotion:contentment|>"]
Результат: output/higgs_test/<персонаж>[_<мітка>].wav
"""
import argparse
import sys
import time

from audiobooks.audio import save_wav
from audiobooks.paths import OUTPUT_DIR
from audiobooks.tts.higgs_engine import HiggsEngine

LINES = {
    "Сенджоґахара": "Ви справді дріб'язкова людина, якщо вас хвилюють такі незначні зауваження.",
    "Арараґі": "Якщо так сказати, то так, але я думав, що дівчина може трохи краще знати обставини іншої дівчини.",
    "Ханекава": "Але хіба ти не знаєш Сенджоґахару краще за мене, Арараґі? Навіщо ти запитуєш мене?",
    "Хачікуджі": "Що ви таке, коли ваші мрії не збуваються?",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", help="лише цей персонаж")
    ap.add_argument("--tags", default="", help="теги стилю на початку, напр. <|emotion:contentment|>")
    ap.add_argument("--label", default="", help="суфікс імені файлу")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no-ref", action="store_true", help="без референсу (власний голос моделі)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t0 = time.time()
    eng = HiggsEngine(device=args.device, dtype=args.dtype, tags=args.tags, seed=args.seed, temperature=args.temperature)
    print(f"Модель завантажено за {time.time() - t0:.0f} с")
    out_dir = OUTPUT_DIR / "higgs_test"
    for name, text in LINES.items():
        if args.only and name != args.only:
            continue
        eng.set_voice("(без референсу)" if args.no_ref else name)
        t = time.time()
        pcm = eng.synth(text)
        secs = len(pcm) / 2 / eng.sample_rate
        print(f"{name}: {secs:.1f} с звуку за {time.time() - t:.0f} с (x{(time.time() - t) / secs:.1f} від реального часу)")
        suffix = "_".join(p for p in (args.label, "безреф" if args.no_ref else "") if p)
        save_wav(out_dir / f"{name}{'_' + suffix if suffix else ''}.wav", pcm, eng.sample_rate)


if __name__ == "__main__":
    main()
