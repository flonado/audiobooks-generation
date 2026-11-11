r"""
Експеримент: чи допомагає зміна промпту перекладача ставити правильний рід (дієслова, прикметники при «я» та інше).
Бере вікна абзаців із діалогами, де чергуються герої різної статі (еталон роду мовця: data/speakers), перекладає їх
Lapa з різними варіантами промпту та рахує помилки роду. Нічого не записує в data/translations.

  python tools\gender_experiment.py --variants base cast tags both --windows dev
Варіанти: base (поточний промпт), cast (рід персонажів у правилах), tags (перед рядком мовець у дужках), both.
Вікна: dev, test або all (див. WINDOWS). Результат: data/output/gender_exp/<варіант>_<вікно>.txt і report.txt.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from ru_speaker_gender import propagate_genders, speaker_gender
from audiobooks import gender as g
from audiobooks import gender_quotes as gq
from audiobooks.books import load_book
from audiobooks.paths import CACHE_DIR, OUTPUT_DIR
from audiobooks.speakers import SPEAKERS_DIR, load_notes
from audiobooks.text import slugify
from audiobooks.translate import lapa

STEM = "Monogatari Series v01 - BAKEMONOGATARI Part 1 [Vertical] [LuCaZ]"
EPUB = STEM + ".epub"
# (розділ, перший абзац, після останнього): густі діалоги, де чергуються герої різної статі
GOLD_FILE = Path(__file__).resolve().parents[1] / "data" / "gender_eval" / "gold_quotes.json"
WINDOWS = {
    "dev": [(2, 0, 24), (2, 92, 116)],
    "test": [(2, 44, 68), (2, 68, 92)],
}
# «важкі»: репліки без слів автора, мовця розмітка не знає (розд. 10, 12, 14); еталон: data/gender_eval/gold_quotes.json
WINDOWS["hard"] = [(14, 190, 230), (10, 415, 455), (12, 240, 280)]
WINDOWS["small"] = [(10, 436, 456), (12, 255, 280)]  # 10 хв на варіант: тут були реальні помилки
WINDOWS["all"] = WINDOWS["dev"] + WINDOWS["test"]

CAST_RULE = """
- Grammatical gender matters. Ukrainian past-tense verbs, adjectives and participles must agree with the gender of the person they describe, \
which English does not show. Characters: {cast}. A female speaker says «я зробила», «я була рада»; a male speaker says «я зробив», «я був радий». \
Narration in the first person ("I") is spoken by Araragi (male)."""
TAG_RULE = """
- Each input line starts with a tag in square brackets: it says who speaks the quoted words of that line (or that the line is narration). \
Use it to choose masculine or feminine forms for the speaker's own "I" (я зробив / я зробила) and for other words about the speaker. \
Never copy the tag into the output."""
RU_RULE = """
- Before the English text you get a Russian reference translation of the same lines. Russian past-tense verbs show the speaker's gender \
(сказал = male, сказала = female). Use the Russian ONLY to choose masculine or feminine forms in Ukrainian (я зробив / я зробила, \
він / вона). The Ukrainian must be translated from the English text, never from the Russian."""
RU2_RULE = RU_RULE + """
- The Russian is NOT a model for wording or formatting. Keep the English formatting: spoken lines go in «» (never start a line with a dash), \
and keep the English tone of address: use «ти» between friends and peers, «ви» only if the English is clearly formal. Do not add, drop \
or reorder anything because of the Russian."""
RUG_RULE = """
- Some input lines start with a tag [speaker: female] or [speaker: male]: the gender of the person who speaks the quoted words of that line. \
Use it to choose masculine or feminine forms for the speaker's own "I" (я зробив / я зробила) and for other words about the speaker. \
Lines without a tag: decide from the context. Never copy a tag into the output, and keep everything else exactly as you would without tags."""
RUG2_RULE = RUG_RULE + """
- Forms of address: the characters are friends, classmates and siblings, so by default they use the informal «ти». Use the formal «ви» \
only where the English is clearly polite or distant (a stranger, a much older person, a teacher, deliberately mocking formality)."""
RUG3_RULE = RUG2_RULE + """
- Grammatical gender applies to EVERY person mentioned, not only the speaker. Characters: {cast}. Every noun, adjective, participle, pronoun \
and verb that refers to a female character must be feminine, and to a male character masculine. This includes roles and professions: \
for girls use feminine forms (однокласниця, баскетболістка, гравчиня, героїня, учениця, капітанка), and past-tense verbs (вона мала, \
вона підійшла). Check this for plural groups too (однокласниці for girls)."""
AUTO_RULE = """
- Work out who speaks each line from the surrounding lines (dialogue usually alternates between two characters). Start every output line \
with a tag in square brackets naming the speaker of the quoted words in that line (one of: {names}), or [narration] if the line has no \
quoted speech, and then the Ukrainian translation of the line. Choose masculine or feminine forms for the speaker's own "I" by the \
speaker's gender. The tags are removed afterwards."""
SPEAKER_PROMPT = """Below are numbered lines from a novel. The narrator ("I") is Araragi (male). Characters: {cast}.
For each line say who speaks the quoted words in it, working out the speaker from the surrounding lines (dialogue usually alternates \
between two characters). A line without quoted speech is "narration".

Previous lines (context only):
{ctx}

Lines:
{lines}

Answer with one line per number in the form "N: Name" where Name is one of {names} or narration. No other text."""


def tag_for(paragraph_segs: list[dict], genders: dict[str, str]) -> str:
    who = []
    for s in paragraph_segs:
        if s["kind"] == "quote":
            sp = s.get("speaker")
            sp = "Araragi" if sp == "narrator" else sp
            if sp and sp not in who:
                who.append(sp)
    name = lambda w: f"{w} ({'female' if genders.get(w) == 'femn' else 'male'})"
    if not who:
        return "[narration by Araragi (male)]"
    return "[" + "; ".join(name(w) for w in who) + "]"


def prompt_for(variant: str, notes: dict, genders: dict[str, str]) -> str:
    extra = ""
    if variant in ("cast", "both", "auto", "ctx", "auto2"):
        cast = ", ".join(f"{n} ({'male' if genders[n] == 'masc' else 'female'})" for n in notes["cast"] if genders.get(n))
        extra += CAST_RULE.format(cast=cast)
    if variant in ("tags", "both", "auto2"):
        extra += TAG_RULE
    if variant == "rug":
        extra += RUG_RULE
    if variant == "rug2":
        extra += RUG2_RULE
    if variant == "rug3":
        cast = ", ".join(f"{n} ({'male' if genders[n] == 'masc' else 'female'})" for n in notes["cast"] if genders.get(n))
        extra += RUG3_RULE.format(cast=cast)
    if variant == "ru":
        extra += RU_RULE
    if variant in ("ru2", "ru3", "ru4"):
        extra += RU2_RULE
    if variant == "auto":
        extra += AUTO_RULE.format(names=", ".join(n for n in notes["cast"] if n != "Other"))
    return lapa.PROMPT.replace("Output only the translation.", "Output only the translation." + extra)


def ru_hint(r: str, variant: str) -> str:
    """Рядок російської підказки: ru3 без тире на початку, ru4 з репліками у «» (щоб модель не переймала тире)."""
    r = r[:350]
    if variant == "ru3":
        return re.sub(r"^[—–-]\s*", "", r)
    if variant == "ru4" and re.match(r"^[—–-]\s*", r):
        return "«" + re.sub(r"^[—–-]\s*", "", r).rstrip() + "»"
    return r


def translate_window(server, variant, paragraphs, tags, glossary, prompt, preceding=(), ru=()) -> tuple[list[str], list[str]]:
    """(переклад без міток, сирі відповіді моделі). preceding: англійські абзаци перед вікном (контекст для ctx)."""
    cache_dir = CACHE_DIR / "gender_exp" / variant
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = []
    lines = [f"{t} {p}" if t else p for p, t in zip(paragraphs, tags)]
    pos = 0
    for group in lapa.group_paragraphs(lines):
        this = prompt
        if variant == "ctx":
            ctx = "\n".join((list(preceding) + list(paragraphs[:pos]))[-6:]).replace("{", "{{").replace("}", "}}")
            this = prompt.replace("English text:\n{text}", "Previous lines, for context only (do NOT translate them):\n" + ctx +
                                  "\n\nEnglish text to translate:\n{text}")
        if variant in ("ru", "ru2", "ru3", "ru4"):
            ref = "\n".join(f"{i + 1}. {ru_hint(r, variant)}" for i, r in enumerate(ru[pos:pos + len(group)])).replace("{", "{{").replace("}", "}}")
            this = prompt.replace("English text:\n{text}", "Russian reference (for the gender of speakers only; numbered like the English lines):\n" +
                                  ref + "\n\nEnglish text to translate:\n{text}")
        pos += len(group)
        lapa.PROMPT = this
        glossary_text = lapa._glossary_block(glossary, "\n".join(group))
        key = hashlib.sha256(f"{this}|{glossary_text}|{chr(10).join(group)}".encode()).hexdigest()[:16]
        f = cache_dir / f"{key}.json"
        if f.exists():
            res = json.loads(f.read_text(encoding="utf-8"))
        else:
            t = time.time()
            res = [lapa.fix_lookalikes(x) for x in lapa.translate_group(server, group, glossary_text)]
            f.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            print(f"    група {len(group)} абзаців, {sum(map(len, group))} символів: {time.time() - t:.0f} с", flush=True)
        raw += res
    return [re.sub(r"^\s*\[[^\]]*\]\s*", "", x) for x in raw], raw


def predict_speakers(server, notes, genders, paragraphs, preceding) -> list[str]:
    """Прохід 1 для auto2: модель сама визначає мовця кожного абзацу з контексту. Повертає імена з cast або 'narration'."""
    cast = ", ".join(f"{n} ({'male' if genders[n] == 'masc' else 'female'})" for n in notes["cast"] if genders.get(n))
    names = ", ".join(n for n in notes["cast"] if n != "Other")
    out = []
    cache_dir = CACHE_DIR / "gender_exp" / "speakers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(paragraphs), 20):
        chunk = paragraphs[s:s + 20]
        ctx = "\n".join((list(preceding) + list(paragraphs[:s]))[-6:])
        prompt = SPEAKER_PROMPT.format(cast=cast, names=names, ctx=ctx, lines="\n".join(f"{i}: {p}" for i, p in enumerate(chunk)))
        f = cache_dir / (hashlib.sha256(prompt.encode()).hexdigest()[:16] + ".txt")
        if f.exists():
            answer = f.read_text(encoding="utf-8")
        else:
            answer = server.complete(prompt, max_tokens=400, temperature=0.0)
            f.write_text(answer, encoding="utf-8")
        found = {int(m.group(1)): m.group(2) for m in re.finditer(r"^\s*(\d+)\s*[:.]\s*(\w+)", answer, re.M)}
        out += [found.get(i, "narration") for i in range(len(chunk))]
    return out


def tag_name(raw_line: str, cast: list[str]) -> str | None:
    m = re.match(r"^\s*\[([^\]]*)\]", raw_line)
    if not m:
        return None
    word = m.group(1).strip().lower()
    return next((c for c in cast if word.startswith(c.lower())), "narration" if word.startswith("narr") else None)


def past_gender(word: str) -> str | None:
    """Рід дієслова минулого часу за закінченням (-в, -вся, -ла, -лася). Усе інше (тих, невідомі слова) не рахуємо."""
    low = word.lower()
    if not re.search(r"(в|вся|ла|лася|міг|ліг|біг|ніс)$", low):
        return None
    gen = g.loose_verb_gender(low)
    return gen if gen in ("masc", "femn") else None


def evaluate(uk: list[str], en: list[str], first: int, chapter: int, notes: dict, genders: dict, names: dict, gold: dict):
    """Помилки роду: (а) «я + дієслово» у репліках проти мовця з розмітки; (б) правила gender.py: ім'я/слова автора/займенники."""
    narr = genders.get(notes["narrator"], "masc")
    cands, skipped = gq.candidates(uk, en)
    checked = wrong = 0
    bad = []
    for c in cands:
        who = gold.get((first + c["par"], c["quote"]))
        gen = genders.get(who or "", "")
        if not gen:
            continue
        for v in c["verbs"]:
            vg = past_gender(v["old"])
            if vg is None:
                continue
            checked += 1
            if vg != gen:
                wrong += 1
                bad.append(f"[{first + c['par']}] «{v['old']}» у репліці {who} ({gen}): {c['uk'][:70]}")
    fixes = g.check_chapter(notes, names, en, uk, narr)
    rule_bad = [f"[{first + f['par']}] {f['old']} → {f['new']} ({f['rule']}: {f['who']})" for f in fixes]
    return {"checked": checked, "wrong": wrong, "rule_errors": len(rule_bad), "skipped": len(skipped), "bad": bad, "rule_bad": rule_bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["base"], choices=["base", "cast", "tags", "both", "auto", "ctx", "auto2", "ru", "ru2", "ru3", "ru4", "rug", "rug2", "rug3"])
    ap.add_argument("--windows", default="dev", choices=list(WINDOWS))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--gpu-layers", type=int, default=99)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    notes = load_notes(STEM)
    genders = g.cast_genders(notes["cast"])
    names = g.uk_names(STEM, genders)
    glossary = lapa.load_glossary(EPUB)
    chapters = load_book(EPUB)
    slug = slugify(STEM)[:60]
    out_dir = OUTPUT_DIR / "gender_exp"
    out_dir.mkdir(parents=True, exist_ok=True)
    RU = json.loads((GOLD_FILE.parent / "ru_align.json").read_text(encoding="utf-8")) if (GOLD_FILE.parent / "ru_align.json").exists() else {}
    manual = json.loads(GOLD_FILE.read_text(encoding="utf-8")) if GOLD_FILE.exists() else {}
    summary = []
    with lapa.LlamaServer(ctx=args.ctx, gpu_layers=args.gpu_layers) as server:
        for variant in args.variants:
            prompt = prompt_for(variant, notes, genders)
            tot = {"checked": 0, "wrong": 0, "rule_errors": 0, "skipped": 0}
            spk = {"total": 0, "right": 0}
            details = []
            t0 = time.time()
            for n, a, b in WINDOWS[args.windows]:
                en_all = [l for l in chapters[n - 1][1].splitlines() if l.strip()]
                en = en_all[a:b]
                segs = json.loads(next((SPEAKERS_DIR / slug).glob(f"{n:02d}_*.json")).read_text(encoding="utf-8"))["paragraphs"][a:b]
                preceding = en_all[max(0, a - 6):a]
                tags = []
                alt = propagate_genders(en_all, RU[str(n)]) if variant in ("rug2", "rug3") else {}
                predicted = predict_speakers(server, notes, genders, en, preceding) if variant == "auto2" else None
                for i, sg in enumerate(segs):
                    if variant == "auto2":
                        sp = predicted[i]
                        tags.append("[narration by Araragi (male)]" if sp == "narration" or sp not in genders else
                                    f"[{sp} ({'female' if genders[sp] == 'femn' else 'male'})]")
                    elif variant in ("rug2", "rug3"):
                        tags.append({"masc": "[speaker: male]", "femn": "[speaker: female]"}.get(alt.get(a + i), ""))
                    elif variant == "rug":  # рід мовця з російського абзацу; сам російський текст моделі не показується
                        gg = speaker_gender(RU[str(n)][a + i])
                        tags.append({"masc": "[speaker: male]", "femn": "[speaker: female]"}.get(gg, ""))
                    elif variant not in ("tags", "both"):
                        tags.append("")
                    elif (sp := manual.get(str(n), {}).get(str(a + i))):  # ручний еталон має перевагу над розміткою шарів
                        tags.append(f"[{sp} ({'female' if genders.get(sp) == 'femn' else 'male'})]")
                    else:
                        tags.append(tag_for(sg["segs"], genders))
                print(f"[{variant}] розділ {n}, абзаци {a}-{b}", flush=True)
                uk, raw = translate_window(server, variant, en, tags, glossary, prompt, preceding, RU[str(n)][a:b] if variant in ("ru", "ru2", "ru3", "ru4") else ())
                pred = predicted if variant == "auto2" else [tag_name(r, list(notes["cast"])) for r in raw] if variant == "auto" else None
                if pred and len(pred) == len(en):
                    for par, sp in manual.get(str(n), {}).items():
                        i = int(par) - a
                        if 0 <= i < len(pred):
                            spk["total"] += 1
                            spk["right"] += pred[i] == sp
                if len(uk) != len(en):
                    print(f"  УВАГА: абзаців {len(uk)} замість {len(en)}")
                    continue
                (out_dir / f"{variant}_{n:02d}_{a}-{b}.txt").write_text("\n".join(uk) + "\n", encoding="utf-8")
                gold = gq._gold(slug, n)
                for par, sp in manual.get(str(n), {}).items():  # ручний еталон: усі репліки абзацу належать мовцю
                    for q in range(1, 7):
                        gold[(int(par), q)] = sp
                r = evaluate(uk, en, a, n, notes, genders, names, gold)
                for k in tot:
                    tot[k] += r[k]
                details += [f"  розд.{n} {a}-{b}: " + s for s in r["bad"] + r["rule_bad"]]
            line = (f"{variant:5s} «я+дієслово» у репліках: помилок {tot['wrong']} з {tot['checked']}; "
                    f"помилок за правилами (ім'я, слова автора, займенники): {tot['rule_errors']}; "
                    f"абзаців не зіставлено {tot['skipped']}; "
                    + (f"мовець визначений правильно {spk['right']} з {spk['total']}; " if spk["total"] else "")
                    + f"{(time.time() - t0) / 60:.1f} хв")
            print(line, flush=True)
            summary.append(line + "\n" + "\n".join(details))
    (out_dir / f"report_{args.windows}.txt").write_text("\n\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
