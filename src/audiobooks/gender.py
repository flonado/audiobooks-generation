"""
Перевірка роду в українському перекладі (англійська не має роду дієслів, тож модель-перекладач часто помиляється:
«Ошіно засміялася»).

Правила (без моделі), кожне виправляє лише слово, яке явно суперечить відомому роду; решта тексту не чіпається:
  name   ім'я з cast + дієслово минулого часу («Ошіно засміялася і сказала»), також «— сказала Ханекава»;
  я      «я» в оповіді (поза лапками) = оповідач, його рід береться з cast (narrator);
  en     займенники він/вона/його/її… за англійським абзацом: якщо там усі he/him/his (або усі she/her), а в
         перекладі займенник іншого роду і в абзаці немає it, займенник міняється, разом із дієсловами при ньому;
  model  (--model) дієслова без підмета-імені: модель за англійським реченням каже, хто виконує дію.
Рід персонажів: configs/speakers/<книга>.yaml (cast: «male…» / «female…»), українські імена: configs/glossary.
Результат: data/translations/<книга>/_gender/NN_report.txt (усі зміни з контекстом) і виправлена копія NN_*.txt.
Оригінал перекладу не змінюється.
"""
import re
from pathlib import Path

import yaml

from .books import load_book, resolve_book
from .paths import ROOT, TRANSLATIONS_DIR
from .speakers import GLOSSARY_DIR, load_notes
from .text import parse_selection, slugify

GENDER_LOG = ROOT / "data" / "cache" / "gender_llm.log"
WORD = re.compile(r"[\w'’ʼ]+(?:-[\w'’ʼ]+)*", re.UNICODE)
CASES = ("nomn", "gent", "datv", "accs", "ablt", "loct", "voct")

BE = {"був": "masc", "була": "femn", "було": "neut", "були": "plur"}
PRONOUNS = {"він": "masc", "вона": "femn", "його": "masc", "її": "femn", "йому": "masc", "їй": "femn",
            "ним": "masc", "нею": "femn", "нього": "masc", "неї": "femn", "нім": "masc", "ній": "femn"}
PRONOUN_SWAP = {"він": "вона", "його": "її", "йому": "їй", "ним": "нею", "нього": "неї", "нім": "ній"}
PRONOUN_SWAP.update({v: k for k, v in list(PRONOUN_SWAP.items())})
SUBJECT_PRONOUNS = {"він", "вона"}

# слова, на яких ланцюжок «підмет → дієслова» обривається (інший підмет або підрядне речення)
STOP = {"він", "вона", "воно", "вони", "я", "ми", "ти", "ви", "що", "чи", "як", "коли", "бо", "щоб", "якщо", "поки",
        "хоча", "який", "яка", "яке", "які", "де", "куди", "звідки", "тому", "поскільки", "ніби", "наче", "доки",
        "але", "а", "проте", "однак", "зате", "це", "то", "його", "її", "їх", "мій", "твій", "наш", "ваш", "свій"}
STOP_LEMMAS = {"який", "котрий", "хто", "що", "це", "цей", "той"}
BREAK = re.compile(r"[.!?…;:—–─«»“”„\"]|\s-\s")
IRREGULAR = {"могла": "міг", "змогла": "зміг", "допомогла": "допоміг", "лягла": "ліг", "несла": "ніс"}
IRREGULAR_BACK = {v: k for k, v in IRREGULAR.items()}
SENT_END = re.compile(r"[.!?…;]")
TAG_BEFORE = re.compile(r"[»”\"]\s*[,.!?…]?\s*[—–─-]+\s*$")

EN_MASC = re.compile(r"\b(?:he|him|his|himself)\b", re.I)
EN_FEM = re.compile(r"\b(?:she|her|hers|herself)\b", re.I)
EN_IT = re.compile(r"\b(?:it|its|itself)\b", re.I)

_morph = None


def morph():
    global _morph
    if _morph is None:
        try:
            import pymorphy3
        except ImportError:
            raise SystemExit("Потрібен pymorphy3: pip install pymorphy3 pymorphy3-dicts-uk")
        _morph = pymorphy3.MorphAnalyzer(lang="uk")
    return _morph


def _is_past_verb(p) -> bool:
    return p.tag.POS == "VERB" and "past" in p.tag and p.tag.gender in ("masc", "femn", "neut")


def loose_verb_gender(word: str) -> str | None:
    """Як verb_gender, але приймає дієслівний розбір і тоді, коли словник ставить першим прикметник з тим самим балом
    («чула» → чулий/чути). Лише для слова, що стоїть упритул до підмета."""
    g = verb_gender(word)
    if g:
        return g
    parses = morph().parse(word)
    if parses and parses[0].tag.POS in ("ADJF", "ADJS"):
        for p in parses[1:4]:
            if _is_past_verb(p) and p.score >= parses[0].score * 0.9:
                return str(p.tag.gender)
    return None


def verb_gender(word: str) -> str | None:
    """masc / femn / neut / plur для дієслова минулого часу, інакше None."""
    low = word.lower()
    if low in BE:
        return BE[low]
    parses = morph().parse(word)
    if parses:
        p0 = parses[0]
        if _is_past_verb(p0):
            return str(p0.tag.gender)
        if p0.tag.POS == "VERB" and "past" in p0.tag and p0.tag.number == "plur":
            return "plur"
    return None


# ---------------------------------------------------------------------------------------------------------------
# Зміна форми слова
def _restore_case(src: str, new: str) -> str:
    if src.isupper() and len(src) > 1:
        return new.upper()
    return new[:1].upper() + new[1:] if src[:1].isupper() else new


def _check_verb(word: str, gender: str) -> bool:
    return any(_is_past_verb(p) and p.tag.gender == gender for p in morph().parse(word))


def inflect_verb(word: str, target: str) -> str | None:
    """Минулий час: рід masc/femn. Правило за закінченням, результат перевіряється морфологією."""
    low = word.lower()
    if low in BE:
        return _restore_case(word, {"masc": "був", "femn": "була"}[target])
    table = IRREGULAR if target == "masc" else IRREGULAR_BACK
    if low in table:
        return _restore_case(word, table[low])
    refl = low[-2:] if low.endswith(("ся", "сь")) else ""
    core = low[:-2] if refl else low
    cands = []
    if target == "femn":
        if core.endswith("шов"):
            cands.append(core[:-3] + "шла")
        elif core.endswith("в"):
            cands.append(core[:-1] + "ла")
        cands.append(core + "ла")
    else:
        if core.endswith("шла"):
            cands.append(core[:-3] + "шов")
        elif core.endswith("ла"):
            stem = core[:-2]
            cands.append(stem + "в" if stem[-1:] in "аеиіоуяюєї" else stem)
            cands.append(stem)
    for c in cands:
        new = c + refl
        if _check_verb(new, target):
            return _restore_case(word, new)
    return None


def inflect_pronoun(word: str) -> str | None:
    new = PRONOUN_SWAP.get(word.lower())
    return _restore_case(word, new) if new else None


# ---------------------------------------------------------------------------------------------------------------
# Дані про персонажів
def cast_genders(cast: dict) -> dict[str, str]:
    out = {}
    for name, desc in cast.items():
        d = str(desc).strip().lower()
        out[name] = "masc" if d.startswith("male") else "femn" if d.startswith("female") else ""
    return out


def uk_names(book_stem: str, genders: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Українське імʼя (називний) -> (персонаж з cast, рід). З глосарія: беремо останнє слово значення."""
    path = GLOSSARY_DIR / f"{book_stem}.yaml"
    glossary = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    out = {}
    for en, uk in glossary.items():
        if en in genders and genders[en]:
            out[str(uk).split()[-1]] = (en, genders[en])
    if "Meme" in glossary or "Oshino" in genders:
        out.setdefault("Меме", ("Oshino", genders.get("Oshino", "masc")))
    return out


CLOSERS = {"«": "»", "“": "”", "„": "”", '"': '"'}


def quote_spans(text: str) -> list[tuple[int, int]]:
    """Місця прямої мови: «…», “…”, „…”, \"…\"; незакрита лапка тягнеться до кінця абзацу."""
    spans, start, closer = [], None, None
    for i, ch in enumerate(text):
        if start is None:
            if ch in CLOSERS:
                start, closer = i, CLOSERS[ch]
        elif ch == closer or (closer == "”" and ch == "“"):
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def in_quote(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos <= b for a, b in spans)


# ---------------------------------------------------------------------------------------------------------------
# Правила
def _fix(par, m, kind, rule, who, expected, text):
    if kind == "verb":
        new = inflect_verb(m.group(), expected)
    else:
        new = inflect_pronoun(m.group())
    a, b = m.start(), m.end()
    ctx = "…" + text[max(0, a - 70):a] + "[" + text[a:b] + "]" + text[b:b + 70] + "…"
    return {"par": par, "start": a, "end": b, "old": m.group(), "new": new, "who": who, "rule": rule, "context": ctx}


SPEECH_LEMMAS = {"сказати", "запитати", "спитати", "відповісти", "додати", "зауважити", "промовити", "пробурмотіти", "вигукнути",
                 "прошепотіти", "заперечити", "погодитися", "продовжити", "нагадати", "заявити", "пояснити", "підтвердити",
                 "відрізати", "буркнути", "мовити", "перебити", "запротестувати", "поцікавитися", "відказати", "вимовити"}
SPEECH_LEAD = {"як", "що", "коли", "бо"}


def _speech_then_name(text: str, matches: list, j: int, who_gender: tuple[str, str], par: int) -> list[dict]:
    """«…, сказала Ошіно»: дієслово мовлення, за яким підмет-ім'я (без тире). Ім'я після дієслова буває й додатком
    («Сенджоґахара запитала Ошіно»), тому дієслово має починати власну частину: після коми, тире, лапки, крапки чи «як/що»."""
    who, gen = who_gender
    prev = matches[j - 1]
    if text[prev.end():matches[j].start()].strip():
        return []
    vg = verb_gender(prev.group())
    if vg not in ("masc", "femn") or vg == gen or morph().parse(prev.group())[0].normal_form not in SPEECH_LEMMAS:
        return []
    k, head = j - 2, text[:prev.start()]
    if k >= 0:
        low = matches[k].group().lower()
        if low in ADVERBS or any(p.tag.POS == "ADVB" for p in morph().parse(low)):
            head, k = text[:matches[k].start()], k - 1
    stripped = head.rstrip()
    if stripped and stripped[-1] not in ',—–─»”".!?…' and not (k >= 0 and matches[k].group().lower() in SPEECH_LEAD):
        return []
    return [_fix(par, prev, "verb", "tag-name", who, gen, text)]


ADVERBS = {"б", "би", "не", "ні", "ніколи", "нічого", "нікого", "ніде", "навіть", "просто", "ще", "вже", "лише", "тільки",
           "також", "теж", "ж", "майже", "ледве", "завжди", "давно", "часто", "аж", "таки", "вдруге", "знову", "справді"}


def chain(text: str, matches: list, j: int, expected: str, par: int, rule: str, who: str, skip_names: set[str],
          comma_stop: bool = False) -> list[dict]:
    """Дієслова минулого часу, що йдуть за підметом matches[j] (до кінця речення, іншого підмета чи підрядного).
    comma_stop: перевіряти лише першу частину до коми (для «я»: далі може бути чужа репліка без лапок)."""
    fixes, nonverb = [], 0
    for k in range(j + 1, len(matches)):
        gap = text[matches[k - 1].end():matches[k].start()]
        w = matches[k].group()
        if BREAK.search(gap) or (comma_stop and "," in gap) or ("," in gap and w.lower() in ("і", "й")):
            break
        if w.lower() in STOP or w in skip_names:
            break
        g = verb_gender(w) or (loose_verb_gender(w) if k == j + 1 else None)
        if g == "plur" or g == "neut":
            break
        if g:
            nonverb = 0
            if g != expected:
                fixes.append(_fix(par, matches[k], "verb", rule, who, expected, text))
            continue
        top = morph().parse(w)[:1]
        after_prep = any(p.tag.POS == "PREP" for p in morph().parse(matches[k - 1].group())[:2])
        if top and (top[0].normal_form in STOP_LEMMAS or (top[0].tag.POS == "NOUN" and top[0].tag.case == "nomn" and not after_prep)):
            break
        if w.lower() in ADVERBS:  # прислівники й частки не вводять нового підмета і не рахуються у ліміт
            continue
        nonverb += 1
        if nonverb > 3:
            break
    return fixes


def _object_position(matches: list, text: str, j: int) -> bool:
    """Ім'я стоїть після дієслова, прийменника чи іменника («лоб Хачікуджі»), або після «як»: це не підмет."""
    if j == 0 or BREAK.search(text[matches[j - 1].end():matches[j].start()]):
        return False
    prev = matches[j - 1].group()
    if prev.lower() == "як":
        return True
    top = morph().parse(prev)[:1]
    return bool(top) and top[0].tag.POS in ("PREP", "VERB", "INFN", "NOUN")


def _antecedent_nearby(text: str, matches: list, j: int, gender: str, names: dict) -> bool:
    """У тому ж реченні перед займенником є іменник його (поточного) роду: займенник може бути до нього, а не до персонажа."""
    for k in range(j - 1, max(-1, j - 9), -1):
        if SENT_END.search(text[matches[k].end():matches[k + 1].start()]):
            return False
        if matches[k].group() in names:
            continue
        if any(q.tag.POS == "NOUN" and str(q.tag.gender) == gender and q.tag.number != "plur"
               for q in morph().parse(matches[k].group())[:3]):
            return True
    return False


def check_paragraph(par: int, text: str, en: str, names: dict[str, tuple[str, str]], narrator_gender: str) -> list[dict]:
    matches = list(WORD.finditer(text))
    spans = quote_spans(text)
    fixes = []
    skip = set(names)
    # абзац із прямою мовою, де не названо жодного персонажа: хто говорить, невідомо, тому «я» й займенники не чіпаємо
    unsure = (any(c in text for c in '«»“”„"') or text.lstrip().startswith(("—", "–", "─")))         and not any(m.group() in names for m in matches)

    for j, m in enumerate(matches):
        w = m.group()
        # name + дієслова
        if w in names and not in_quote(m.start(), spans) and not _object_position(matches, text, j):
            who, g = names[w]
            fixes += chain(text, matches, j, g, par, "name", who, skip)
        if w in names and j > 0 and not in_quote(m.start(), spans):
            fixes += _speech_then_name(text, matches, j, names[w], par)
        # авторські слова після репліки: «…», — сказала Ханекава / зауважила я (тире буває й «─»)
        subj = (names[w] if w in names else ("narrator", narrator_gender) if w.lower() == "я" and narrator_gender else None)
        if subj and j > 0 and not in_quote(m.start(), spans):
            who, g = subj
            prev = matches[j - 1]
            vg = verb_gender(prev.group())
            before = text[matches[j - 2].end():prev.start()] if j > 1 else text[:prev.start()]
            if vg in ("masc", "femn") and text[prev.end():m.start()].strip() == "" and TAG_BEFORE.search(before):
                if vg != g:
                    fixes.append(_fix(par, prev, "verb", "tag", who, g, text))
                # мовець цієї репліки відомий із авторських слів: «я + дієслово» усередині неї узгоджується з його родом
                for a, b in spans:
                    if b < prev.start() and re.fullmatch(r"\s*[,.!?…]?\s*[—–─-]+\s*", text[b + 1:prev.start()]):
                        for t, q in enumerate(matches):
                            if a < q.start() < b and q.group().lower() == "я":
                                fixes += chain(text, matches, t, g, par, "tag-quote", who, skip, comma_stop=True)
        # «я» в оповіді
        if w.lower() == "я" and narrator_gender and not unsure and not in_quote(m.start(), spans):
            fixes += chain(text, matches, j, narrator_gender, par, "я", "narrator", skip, comma_stop=True)
            # «подумала я»: дієслово безпосередньо перед «я»
            if j > 0 and text[matches[j - 1].end():m.start()].strip() == "":
                pv = loose_verb_gender(matches[j - 1].group())
                if pv in ("masc", "femn") and pv != narrator_gender:
                    fixes.append(_fix(par, matches[j - 1], "verb", "я-перед", "narrator", narrator_gender, text))

    # займенники за англійським абзацом
    m_cnt, f_cnt, it_cnt = len(EN_MASC.findall(en)), len(EN_FEM.findall(en)), len(EN_IT.findall(en))
    if it_cnt == 0 and (m_cnt > 0) != (f_cnt > 0) and not unsure:
        expected = "masc" if m_cnt else "femn"
        for j, m in enumerate(matches):
            low = m.group().lower()
            if (low in PRONOUNS and PRONOUNS[low] != expected and not in_quote(m.start(), spans)
                    and not _antecedent_nearby(text, matches, j, PRONOUNS[low], names)):
                fixes.append(_fix(par, m, "pron", "en", "he" if m_cnt else "she", expected, text))
                if low in SUBJECT_PRONOUNS:
                    fixes += chain(text, matches, j, expected, par, "en", "he" if m_cnt else "she", skip)
    unique = {}
    for f in fixes:
        unique.setdefault((f["start"], f["end"]), f)
    return sorted(unique.values(), key=lambda f: f["start"])


def apply_fixes(uk: list[str], fixes: list[dict]) -> list[str]:
    out = list(uk)
    for f in sorted((f for f in fixes if f["new"]), key=lambda f: (f["par"], -f["start"])):
        p = out[f["par"]]
        out[f["par"]] = p[:f["start"]] + f["new"] + p[f["end"]:]
    return out


def check_chapter(notes: dict, names: dict, en: list[str], uk: list[str], narrator_gender: str) -> list[dict]:
    fixes = []
    for i, (e, u) in enumerate(zip(en, uk)):
        fixes += check_paragraph(i, u, e, names, narrator_gender)
    return fixes


def run(args) -> None:
    book_path = resolve_book(args.book)
    notes = load_notes(book_path.stem)
    genders = cast_genders(notes["cast"])
    names = uk_names(book_path.stem, genders)
    narrator_gender = genders.get(notes.get("narrator", ""), "")
    slug = slugify(book_path.stem)[:60]
    tr_dir = Path(args.translation_dir) if args.translation_dir else TRANSLATIONS_DIR / slug
    out_dir = tr_dir / "_gender"
    out_dir.mkdir(parents=True, exist_ok=True)
    chapters = load_book(args.book)
    files = {int(f.name[:2]): f for f in sorted(tr_dir.glob("[0-9][0-9]_*.txt"))}
    print(f"Імена: {', '.join(f'{k}={v[1]}' for k, v in names.items())}; оповідач: {notes.get('narrator')} ({narrator_gender})")
    total = 0
    for n in parse_selection(args.chapters, len(chapters)):
        f = files.get(n)
        if not f:
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        title, uk = lines[0], [ln for ln in lines[1:] if ln.strip()]
        en = [ln for ln in chapters[n - 1][1].splitlines() if ln.strip()]
        if len(en) != len(uk):
            print(f"{n}: абзаців в оригіналі {len(en)}, у перекладі {len(uk)}: пропускаю", flush=True)
            continue
        fixes = check_chapter(notes, names, en, uk, narrator_gender)
        (out_dir / f.name).write_text("\n".join([title, *apply_fixes(uk, fixes)]) + "\n", encoding="utf-8")
        by_rule = {}
        for x in fixes:
            by_rule[x["rule"]] = by_rule.get(x["rule"], 0) + 1
        head = f"{n}. {title}: невідповідностей роду {len(fixes)} {by_rule}, не вдалося виправити {sum(1 for x in fixes if not x['new'])}"
        rep = [head, ""]
        for x in fixes:
            rep.append(f"[{x['par']}] {x['old']} -> {x['new'] or '??? (не вдалося)'}   ({x['rule']}: {x['who']})\n    {x['context']}")
        (out_dir / f"{f.stem}_report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
        total += len(fixes)
        print(head, flush=True)
    print(f"Разом: {total}. Звіти й виправлені копії: {out_dir}")
