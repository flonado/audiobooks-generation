# audiobooks

Книга (fb2 / epub / pdf) → переклад (Lapa LLM, за потреби) → озвучення → `.m4b`.

## Структура

```
books/                 вхідні книги (.fb2, .epub, .pdf)            [не в git]
configs/               yaml-конфіги збирання .m4b, glossary/ для перекладу
src/audiobooks/
  books/               читання форматів: fb2, epub, pdf, папка з перекладом
  translate/lapa.py    переклад через llama.cpp (llama-server) + Lapa LLM
  tts/gemini.py        озвучення Gemini TTS (кеш, поділ шматків, відновлення)
  tts/local.py         локальні рушії: piper_engine, ukrainian_tts_engine, styletts2_engine
  m4b.py               збирання .m4b (ffmpeg, глави)
  cli.py               команда `audiobook`
tools/                 допоміжні скрипти (діагностика промптів, висота голосів)
data/                  усе згенероване й завантажене                [не в git]
  cache/               gemini/ (шматки аудіо), translate/ (шматки перекладу)
  models/              piper/, ukrainian-tts/, styletts2/, lapa/, llama.cpp/
  output/              gemini/, local/, samples/ (озвучені розділи)
  translations/        <книга>/NN_<назва>.txt (переклад по розділах)
  audiobooks/          готові .m4b
```

## Середовища

Рушії мають несумісні залежності, тому живуть в окремих venv. Пакет ставиться в кожен як `-e . --no-deps`.

| venv | призначення |
|---|---|
| системний Python | gemini, m4b, chapters, translate (потрібні `pyyaml`, `google-genai`, `imageio-ffmpeg`) |
| `.venv-tts` | `local --engine piper` / `ukrainian-tts` |
| `.venv-styletts` | `local --engine styletts2` |
| `.venv-higgs` | `local --engine higgs3` (Higgs Audio v3, клонування за зразком; ваги в `data/models/higgs3/`, на CPU ~8× повільніше за реальний час; голоси: `tools/make_higgs_voice.py`, проба: `tools/higgs_monogatari_test.py`) |

## Команди

```powershell
audiobook chapters "<книга>"                                  # розділи й номери
audiobook translate "<книга>.epub" -c 1 --first-chars 1800   # тест: перші ~1800 символів
audiobook translate "<книга>.epub" -c 1-3                    # переклад розділів
audiobook gemini <книга або папка перекладу> -c 4 --model gemini-3.8-flash-tts
.venv-tts\Scripts\audiobook local <книга> -c 4 --engine piper --voice tetiana
.venv-styletts\Scripts\audiobook local <папка перекладу> -c 1 --engine styletts2 --voice "Тетяна Гончарова"
.venv-tts\Scripts\audiobook sample <книга> --chapter 3        # порівняння голосів
audiobook m4b configs\audiobook.yaml                          # збирання .m4b
```

`<книга>` — файл із `books/`, шлях, або назва папки з `data/translations/` (тоді береться переклад).
