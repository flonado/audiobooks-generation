"""
audiobook <команда>:
  chapters   показати розділи книги (fb2/epub/pdf або папки з перекладом)
  translate  перекласти розділи англійська -> українська (Lapa LLM)
  gemini     озвучити розділи через Gemini TTS (потрібен GEMINI_API_KEY)
  local      озвучити розділи локально (piper / ukrainian-tts / styletts2)
  sample     короткий зразок для порівняння голосів
  speakers   розмітити, хто говорить (шар 1: правила)
  m4b        зібрати .m4b з готових WAV за YAML-конфігом
"""
import argparse
import sys


def _add_selection(p: argparse.ArgumentParser):
    p.add_argument("book", help="файл із books/, шлях, або назва папки з data/translations/")
    p.add_argument("-c", "--chapters", required=True, help="номери розділів з `audiobook chapters`: 3 | 2-5 | 1,4,7-8")


def _add_local_tuning(p: argparse.ArgumentParser):
    p.add_argument("--speed", type=float, default=1.0, help="швидкість (piper, styletts2)")
    p.add_argument("--noise-scale", type=float, help="piper: менше = рівніший голос (за замовчуванням 0.667)")
    p.add_argument("--noise-w", type=float, help="piper: менше = рівніший ритм (за замовчуванням 0.8)")
    p.add_argument("--volume", type=float, default=0.85, help="piper: гучність, <1 залишає запас від кліпінгу")
    p.add_argument("--max-chars", type=int, help="макс. довжина фрагмента (за замовчуванням залежить від рушія)")


def build_parser() -> argparse.ArgumentParser:
    from .tts.gemini import DEFAULT_MODEL, DEFAULT_STYLE
    from .tts.local import ENGINES

    ap = argparse.ArgumentParser(prog="audiobook", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chapters", help="показати розділи книги")
    p.add_argument("book")

    p = sub.add_parser("translate", help="переклад англійська -> українська (Lapa LLM)")
    _add_selection(p)
    p.add_argument("--first-chars", type=int, help="тест: перекласти лише перші ~N символів розділу")
    p.add_argument("--ctx", type=int, default=8192, help="контекст llama-server")
    p.add_argument("--gpu-layers", type=int, default=99, help="шарів на GPU (0 = лише CPU)")
    p.add_argument("--out-dir", help="куди зберегти (за замовчуванням data/translations/<книга>)")

    p = sub.add_parser("gemini", help="озвучення через Gemini TTS")
    _add_selection(p)
    p.add_argument("--model", default=DEFAULT_MODEL, help="модель TTS; ліміти рахуються окремо для кожної моделі")
    p.add_argument("--reuse-from", default="", help="моделі (через кому), чий кеш можна використати")
    p.add_argument("--voice", default="Aoede")
    p.add_argument("--style", default=DEFAULT_STYLE, help="інструкція для диктора (входить у ключ кешу!)")
    p.add_argument("--delay", type=float, default=2.0, help="пауза між запитами, с")
    p.add_argument("--out-dir")

    p = sub.add_parser("local", help="локальне озвучення")
    _add_selection(p)
    p.add_argument("--engine", choices=ENGINES, required=True)
    p.add_argument("--voice", help="голос (за замовчуванням залежить від рушія)")
    _add_local_tuning(p)
    p.add_argument("--out-dir")

    p = sub.add_parser("sample", help="короткий зразок для порівняння голосів")
    p.add_argument("book")
    p.add_argument("--chapter", type=int, default=1, help="з якого розділу брати початок")
    p.add_argument("--chars", type=int, default=700)
    p.add_argument("--text", help="свій текст замість уривка з книги")
    p.add_argument("--engine", choices=ENGINES, help="лише цей рушій (інакше всі, що встановлені в цьому venv)")
    p.add_argument("--voice", help="лише цей голос")
    _add_local_tuning(p)
    p.add_argument("--out-dir")

    p = sub.add_parser("speakers", help="розмітка мовців в оригіналі (шар 1: правила; --llm: шар 2)")
    _add_selection(p)
    p.add_argument("--llm", action="store_true", help="шар 2: модель розбирає репліки, яких не визначили правила")
    p.add_argument("--validate", action="store_true", help="шар 3: перевірка узгодженості, сумнівне → review.json")
    p.add_argument("--verify", action="store_true", help="шар 3, крок 2: модель перевіряє сумнівні репліки з широким контекстом")
    p.add_argument("--present", action="store_true", help="шар 2: спершу з'ясувати, хто є в сцені (повільніше, для цих текстів не допомогло)")
    p.add_argument("--ctx", type=int, default=8192, help="контекст llama-server")
    p.add_argument("--gpu-layers", type=int, default=99)
    p.add_argument("--max-windows", type=int, help="тест: лише перші N вікон розділу")
    p.add_argument("--out-dir")

    p = sub.add_parser("gender", help="перевірка роду в українському перекладі (за оригіналом і cast)")
    _add_selection(p)
    p.add_argument("--ctx", type=int, default=8192, help="контекст llama-server")
    p.add_argument("--gpu-layers", type=int, default=99)
    p.add_argument("--translation-dir", help="папка з перекладом (за замовчуванням data/translations/<книга>)")
    p.add_argument("--densest", type=int, default=0, metavar="N", help="тест: лише N вікон з найбільшою кількістю слів, залежних від роду")

    p = sub.add_parser("gender-quotes", help="рід усередині лапок: хто говорить (модель, кілька етапів)")
    _add_selection(p)
    p.add_argument("--ctx", type=int, default=8192, help="контекст llama-server")
    p.add_argument("--gpu-layers", type=int, default=99)
    p.add_argument("--paragraphs", metavar="A-B", help="лише абзаци з A до B (не включно) в кожному розділі")

    p = sub.add_parser("m4b", help="зібрати .m4b")
    p.add_argument("config", nargs="?", default="configs/audiobook.yaml")
    return ap


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)

    if args.command == "chapters":
        from .books import load_book
        for i, (title, text) in enumerate(load_book(args.book), start=1):
            print(f"{i:3}. {title}  ({len(text)} символів)")
    elif args.command == "translate":
        from .translate import lapa
        lapa.run(args)
    elif args.command == "gemini":
        from .tts import gemini
        gemini.run(args)
    elif args.command == "local":
        from .tts import local
        local.run(args)
    elif args.command == "sample":
        from .tts import local
        local.run_sample(args)
    elif args.command == "speakers":
        from . import speakers
        (speakers.run_verify if args.verify else speakers.run_validate if args.validate else speakers.run_llm if args.llm else speakers.run)(args)
    elif args.command == "gender":
        from . import gender
        gender.run(args)
    elif args.command == "gender-quotes":
        from . import gender_quotes
        gender_quotes.run(args)
    elif args.command == "m4b":
        from . import m4b
        m4b.run(args)


if __name__ == "__main__":
    main()
