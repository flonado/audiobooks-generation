r"""
Перекладає розділ з англійського, а рід беруть з російського видання; російський текст моделі не показується.
Варіант rug5 (за замовчуванням, найкращий з перевірених):
  1. tools/dialogue_labels.py: рід мовця й співрозмовника для кожної репліки (за російським і англійським абзацами,
     чергування по всій розмові); лишаються лише мітки з поріг впевненості margin >= 2;
  2. переклад Lapa з позначкою [speaker: female; listener: male] перед кожною реплікою та правилами про рід усіх персонажів;
  3. tools/gender_fix.py: «я/ти + слово з родом» за мітками, ролі дівчат (баскетболістка, капітанка), лапки «…», імена за глосарієм.
Результат (не в data/translations): data/output/gender_exp/<книга>_ch<NN>_<варіант>.txt (готовий текст), *_raw.txt (до перевірки),
*_changes.txt (що змінила перевірка) і *_review.txt (англійський | український абзаци для ручного огляду).

  python tools\translate_rug.py 2 3              # книга 2, розділ 3, rug5
  python tools\translate_rug.py 2 1 rug3         # старіші варіанти (rug2, rug3)
"""
import importlib.util
import json
import sys
from pathlib import Path

from audiobooks import gender as g
from audiobooks.books import load_book
from audiobooks.paths import DATA_DIR, OUTPUT_DIR
from audiobooks.speakers import load_notes
from audiobooks.translate import lapa
from dialogue_labels import confident, label_turns
from gender_fix import fix_paragraph, normalize
from ru_speaker_gender import propagate_genders

TOOLS = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gx = _load("gender_experiment")
STEMS = {"1": "Monogatari Series v01 - BAKEMONOGATARI Part 1 [Vertical] [LuCaZ]",
         "2": "Monogatari Series v02 - BAKEMONOGATARI Part 2 [Vertical] [LuCaZ]"}
ALIGN = {"1": "ru_align.json", "2": "ru_align_v2.json"}
NAMES = {"Araragi": "masc", "Oshino": "masc", "Hanekawa": "femn", "Senjogahara": "femn", "Hachikuji": "femn", "Kanbaru": "femn",
         "Sengoku": "femn", "Karen": "femn", "Tsukihi": "femn", "Shinobu": "femn"}
RUG5_RULE = _load("gender_exp2").RUG4_RULE


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    book, n = sys.argv[1], int(sys.argv[2])
    variant = sys.argv[3] if len(sys.argv) > 3 else "rug5"
    stem = STEMS[book]
    en = [l for l in load_book(stem + ".epub")[n - 1][1].splitlines() if l.strip()]
    ru = json.loads((DATA_DIR / "gender_eval" / ALIGN[book]).read_text(encoding="utf-8"))[str(n)]
    notes = load_notes(stem)
    genders = g.cast_genders(notes["cast"])
    glossary = lapa.load_glossary(stem + ".epub")
    labels = {}
    if variant == "rug5":
        labels = confident(label_turns(en, ru, NAMES))  # лише переконливі мітки: хибні гірші за відсутні
        word = {"masc": "male", "femn": "female"}
        tags = [f"[speaker: {word[labels[i]['speaker']]}; listener: {word[labels[i]['addressee']]}]" if i in labels else ""
                for i in range(len(en))]
        cast = ", ".join(f"{k} ({'male' if genders[k] == 'masc' else 'female'})" for k in notes["cast"] if genders.get(k))
        prompt = lapa.PROMPT.replace("Output only the translation.", "Output only the translation." + RUG5_RULE.format(cast=cast))
    else:
        alt = propagate_genders(en, ru)
        tags = [{"masc": "[speaker: male]", "femn": "[speaker: female]"}.get(alt.get(i), "") for i in range(len(en))]
        prompt = gx.prompt_for(variant, notes, genders)
    with lapa.LlamaServer(ctx=6144, gpu_layers=24) as server:
        uk, _ = gx.translate_window(server, variant, en, tags, glossary, prompt, (), ru if variant.startswith("ru") and variant != "rug5" else ())
    out = OUTPUT_DIR / "gender_exp"
    stem_out = f"v{book}_ch{n:02d}_{variant}"
    (out / f"{stem_out}_raw.txt").write_text("\n".join(uk) + "\n", encoding="utf-8")
    final, log = [], []
    for i, t in enumerate(uk):
        t = normalize(t)
        if variant == "rug5":
            l = labels.get(i, {})
            t, changes = fix_paragraph(t, l.get("speaker"), l.get("addressee"), i in labels)
            log += [f"[{i}] {old} → {new} ({who})" for old, new, who in changes]
        final.append(t)
    (out / f"{stem_out}.txt").write_text("\n".join(final) + "\n", encoding="utf-8")
    (out / f"{stem_out}_changes.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    review = [f"[{i}] {tags[i] or '(без позначки)'}\nEN: {en[i]}\nUK: {final[i]}\n" for i in range(len(en))]
    (out / f"{stem_out}_review.txt").write_text("\n".join(review), encoding="utf-8")
    print(f"готово: {len(final)} абзаців, програмна перевірка змінила {len(log)} слів")


if __name__ == "__main__":
    main()
