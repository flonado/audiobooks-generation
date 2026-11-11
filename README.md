# audiobooks

Book (fb2 / epub / pdf) → translation (Lapa LLM, optional) → speech synthesis → `.m4b`.

## Layout

```
books/                 input books (.fb2, .epub, .pdf)                 [not in git]
configs/               yaml configs for building .m4b, glossary/ for translation
src/audiobooks/
  books/               format readers: fb2, epub, pdf, folder of translated chapters
  translate/lapa.py    translation via llama.cpp (llama-server) + Lapa LLM
  tts/gemini.py        Gemini TTS synthesis (cache, chunk splitting, resume)
  tts/local.py         local engines: piper_engine, ukrainian_tts_engine, styletts2_engine
  m4b.py               .m4b assembly (ffmpeg, chapters)
  cli.py               the `audiobook` command
tools/                 helper scripts (prompt diagnostics, voice pitch, voice cloning)
data/                  everything generated or downloaded              [not in git]
  cache/               gemini/ (audio chunks), translate/ (translation chunks)
  models/              piper/, ukrainian-tts/, styletts2/, higgs3/, lapa/, llama.cpp/
  output/              gemini/, local/, samples/ (synthesized chapters)
  translations/        <book>/NN_<title>.txt (translation, one file per chapter)
  audiobooks/          finished .m4b files
```

## Environments

The engines have incompatible dependencies, so each lives in its own venv. The package is installed into each one as `-e . --no-deps`.

| venv | purpose |
|---|---|
| system Python | gemini, m4b, chapters, translate (needs `pyyaml`, `google-genai`, `imageio-ffmpeg`) |
| `.venv-tts` | `local --engine piper` / `ukrainian-tts` |
| `.venv-styletts` | `local --engine styletts2` |
| `.venv-higgs` | `local --engine higgs3` (Higgs Audio v3, voice cloning from a sample; weights in `data/models/higgs3/`; on CPU about 8× slower than real time; voices: `tools/make_higgs_voice.py`, quick test: `tools/higgs_monogatari_test.py`) |

The Higgs weights are licensed for non-commercial use only, and a voice may be cloned only with the speaker's permission.

## Commands

```powershell
audiobook chapters "<book>"                                  # chapters and their numbers
audiobook translate "<book>.epub" -c 1 --first-chars 1800   # test: first ~1800 characters
audiobook translate "<book>.epub" -c 1-3                    # translate chapters
audiobook gemini <book or translation folder> -c 4 --model gemini-3.8-flash-tts
.venv-tts\Scripts\audiobook local <book> -c 4 --engine piper --voice tetiana
.venv-styletts\Scripts\audiobook local <translation folder> -c 1 --engine styletts2 --voice "Тетяна Гончарова"
.venv-tts\Scripts\audiobook sample <book> --chapter 3        # compare voices
audiobook m4b configs\audiobook.yaml                          # assemble the .m4b
```

`<book>` is a file from `books/`, a path, or the name of a folder under `data/translations/` (in which case the translation is used).

Set `GEMINI_API_KEY` in the environment before using the `gemini` command.
