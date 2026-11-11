r"""
Експеримент: переклад двох найгірших шматків розділу 2 книги 2 з позначками мовця й співрозмовника
(tools/dialogue_labels.py) і/або з програмною перевіркою роду (tools/gender_fix.py).
Еталон: ручні мітки мовця й співрозмовника у GOLD (я їх склав після читання розділу).
Помилка = зміна, яку gender_fix внесе в абзац, якщо дати йому ЕТАЛОННІ мітки.

  python tools\gender_exp2.py base rug4          # base: поточний переклад (з v2_ch02_rug3.txt), rug4: новий переклад
"""
import importlib.util
import json
import sys
from pathlib import Path

from audiobooks import gender as g
from audiobooks.books import load_book
from audiobooks.paths import OUTPUT_DIR
from audiobooks.speakers import load_notes
from audiobooks.translate import lapa
from dialogue_labels import label_turns
from gender_fix import fix_paragraph, normalize

TOOLS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gx", TOOLS / "gender_experiment.py")
gx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gx)

STEM = "Monogatari Series v02 - BAKEMONOGATARI Part 2 [Vertical] [LuCaZ]"
NAMES = {"Araragi": "masc", "Oshino": "masc", "Hanekawa": "femn", "Senjogahara": "femn", "Hachikuji": "femn", "Kanbaru": "femn",
         "Sengoku": "femn", "Karen": "femn", "Tsukihi": "femn", "Shinobu": "femn"}
M, F = "masc", "femn"
WINDOWS = {"A": (122, 142), "B": (214, 234)}
GOLD = {  # абзац розділу 2: (мовець, співрозмовник)
    124: (F, M), 125: (M, F), 126: (F, M), 127: (M, F), 128: (F, M), 129: (M, F), 133: (M, F), 134: (F, M), 135: (M, F),
    140: (F, M), 141: (M, F),
    215: (F, M), 216: (M, F), 217: (F, M), 218: (M, F), 223: (F, M), 227: (F, M), 228: (M, F), 229: (F, M), 230: (M, F), 233: (F, M),
}
RUG4_RULE = """
- Some input lines start with a tag in square brackets, for example [speaker: female; listener: male]. It gives the gender of the person who \
speaks the quoted words and of the person being spoken to. Use it for Ukrainian grammatical gender: past-tense verbs, adjectives, participles \
and nouns that refer to the speaker (after «я», «мене») follow the SPEAKER's gender (я зробила / я зробив, я була рада / я був радий); words that \
refer to the listener (after «ти», «тебе») follow the LISTENER's gender (ти зробив / ти зробила, ти не капітан / ти не капітанка). Never copy a \
tag into the output. Lines without a tag: decide from the context.
- Forms of address: the characters are friends, classmates and siblings, so by default they use the informal «ти». Use the formal «ви» only where \
the English is clearly polite or distant (a stranger, a much older person, a teacher, deliberately mocking formality).
- Keep the English formatting: spoken lines go in «» (never start a line with a dash).
- Grammatical gender applies to EVERY person mentioned, not only the speaker. Characters: {cast}. Every noun, adjective, participle, pronoun and \
verb that refers to a female character must be feminine, and to a male character masculine. This includes roles and professions: for girls use \
feminine forms (однокласниця, баскетболістка, капітанка, суперниця, учениця) and past-tense verbs (вона мала, вона підійшла)."""


def errors(paragraphs: list[str], first: int, labels: dict | None = None) -> list[str]:
    """Помилки роду в абзацах first..: gender_fix з ЕТАЛОННИМИ мітками (або з переданими)."""
    labels = labels if labels is not None else GOLD
    out = []
    for k, t in enumerate(paragraphs):
        i = first + k
        sp, ad = labels.get(i, (None, None))
        _, ch = fix_paragraph(normalize(t), sp, ad, i in labels)
        out += [f"[{i}] {old} → {new} ({who})" for old, new, who in ch]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    variants = sys.argv[1:] or ["base", "rug4"]
    en = [l for l in load_book(STEM + ".epub")[1][1].splitlines() if l.strip()]
    ru = json.loads((OUTPUT_DIR.parent / "gender_eval" / "ru_align_v2.json").read_text(encoding="utf-8"))["2"]
    pred = label_turns(en, ru, NAMES)
    notes = load_notes(STEM)
    genders = g.cast_genders(notes["cast"])
    glossary = lapa.load_glossary(STEM + ".epub")
    cast = ", ".join(f"{n} ({'male' if genders[n] == 'masc' else 'female'})" for n in notes["cast"] if genders.get(n))
    prompt = lapa.PROMPT.replace("Output only the translation.", "Output only the translation." + RUG4_RULE.format(cast=cast))
    base = (OUTPUT_DIR / "gender_exp" / "v2_ch02_rug3.txt").read_text(encoding="utf-8").splitlines()
    results = {}
    server_ctx = None
    for v in variants:
        for w, (a, b) in WINDOWS.items():
            if v == "base":
                uk = base[a:b]
            else:
                if server_ctx is None:
                    server_ctx = lapa.LlamaServer(ctx=6144, gpu_layers=24)
                    server_ctx.__enter__()
                tags = []
                for i in range(a, b):
                    p = pred.get(i)
                    tags.append(f"[speaker: {'female' if p['speaker'] == F else 'male'}; listener: {'female' if p['addressee'] == F else 'male'}]" if p else "")
                print(f"[{v}] вікно {w}: {a}-{b}", flush=True)
                uk, _ = gx.translate_window(server_ctx, v, en[a:b], tags, glossary, prompt, (), ())
                (OUTPUT_DIR / "gender_exp" / f"v2_ch02_{v}_{w}.txt").write_text("\n".join(uk) + "\n", encoding="utf-8")
            results[(v, w)] = uk
    if server_ctx:
        server_ctx.__exit__(None, None, None)
    print("\n=== помилки роду за еталонними мітками (до / після програмної перевірки з передбаченими мітками) ===")
    for v in variants:
        for w, (a, b) in WINDOWS.items():
            uk = results[(v, w)]
            before = errors(uk, a)
            fixed = []
            for k, t in enumerate(uk):
                i = a + k
                p = pred.get(i, {})
                fixed.append(fix_paragraph(normalize(t), p.get("speaker"), p.get("addressee"), i in pred)[0])
            after = errors(fixed, a)
            print(f"{v:5s} вікно {w}: {len(before)} → {len(after)}")
            for e in before:
                print("     ", e)
            if after:
                print("      після:", after)


if __name__ == "__main__":
    main()
