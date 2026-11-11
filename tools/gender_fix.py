r"""
Програмна перевірка роду в українському абзаці за відомими родами мовця й співрозмовника (з tools/dialogue_labels.py):
  - «я» + дієслово/прикметник/дієприкметник у репліці → рід мовця; «я» в оповіді → оповідач (чоловік);
  - «ти» + дієслово/прикметник/іменник-роль у репліці → рід співрозмовника;
  - чоловічі назви ролей дівчат (баскетболіст, капітан…) → жіночі, якщо в абзаці йдеться про дівчину.
Змінюється лише слово, що суперечить роду; решта тексту не чіпається. Повертає (новий текст, список змін).
"""
import re

from audiobooks import gender as g

PARTICLES = {"не", "б", "би", "ж", "же", "вже", "все", "просто", "лише", "тільки", "ще", "ніколи", "справді", "дуже", "теж", "також",
             "завжди", "майже", "навіть", "вам", "тобі", "мені", "собі", "йому", "їй", "тебе", "мене", "ним", "нею", "її", "його"}
WORD = re.compile(r"[\w'’ʼ]+", re.U)
STOP = re.compile(r"[.,;:!?…—–«»„“”]")
# чоловіча назва ролі -> жіноча (лише ті, що в цих книгах стосуються дівчат; «третьокурсник» не включено: так називає себе Арараґі)
ROLES = {
    "баскетболіст": "баскетболістка", "капітан": "капітанка", "спортсмен": "спортсменка", "суперник": "суперниця",
    "другокурсник": "другокурсниця", "першокурсник": "першокурсниця", "обманщик": "обманщиця", "п’ятикласник": "п’ятикласниця",
    "п'ятикласник": "п’ятикласниця", "коханець": "коханка", "президент": "президентка", "гравець": "гравчиня",
    "новачок": "новачка",
}
FEMALE_NAMES = ("Камбару", "Хачікуджі", "Сенджоґахар", "Ханекав", "Карен", "Цукіхі", "Сенґоку", "Шінобу", "Ґаен", "Надеко")
MALE_NAMES = ("Арараґі", "Ошіно", "Койомі")


NAME_FIX = (("Хачікіджі", "Хачікуджі"), ("Арарґі", "Арараґі"), ("Канбару", "Камбару"), ("Суруга", "Суруґа"))


def normalize(text: str) -> str:
    """Імена за глосарієм і лапки «…» (вкладені „…“): модель то пише прямі лапки, то кучеряві."""
    for old, new in NAME_FIX:
        text = text.replace(old, new)
    out, depth = [], 0
    for ch in text:
        if ch in "«„":
            out.append("«" if depth == 0 else "„")
            depth += 1
        elif ch == "“":
            # англійська відкривна лапка: на рівні 0 — зовнішня, на рівні 1 — вкладена; на рівні ≥2 це наше закриття вкладеної
            if depth == 0:
                out.append("«")
                depth = 1
            elif depth == 1:
                out.append("„")
                depth = 2
            else:
                depth -= 1
                out.append("»" if depth == 0 else "“")
        elif ch in '»”' or (ch == '"' and depth > 0):
            depth = max(depth - 1, 0)
            out.append("»" if depth == 0 else "“")
        elif ch == '"':
            out.append("«")
            depth = 1
        else:
            out.append(ch)
    text = "".join(out)
    text = re.sub(r"(?<![\w’'])['‘](?=\w)", "„", text)
    return re.sub(r"(?<=[\w.!?…,])['’](?![\w])", "“", text)


def inflect_gender(word: str, gender: str) -> str | None:
    """Те саме слово (відмінок, число) у потрібному роді; None, якщо не вдалось."""
    low = word.lower()
    if g.morph().parse(low)[0].tag.POS == "VERB" or g.loose_verb_gender(low):
        return g.inflect_verb(word, gender)
    p = adj_parse(low) or g.morph().parse(low)[0]
    if p.tag.POS not in ("ADJF", "ADJS", "PRTF", "PRTS", "NOUN"):
        return None
    want = {gender}
    if p.tag.case:
        want.add(str(p.tag.case))
    if p.tag.number:
        want.add(str(p.tag.number))
    form = p.inflect(want)
    if not form:  # «Бідна» (ADJF compb): береться лема
        lemma = g.morph().parse(p.normal_form)[0]
        form = lemma.inflect(want)
    return g._restore_case(word, form.word) if form else None


def is_noun_lemma(word: str) -> bool:
    """Слово — іменник у називному («людина»); pymorphy.uk читає такі слова ще й як присвійні прикметники («людин»)."""
    low = word.lower().replace("’", "'").replace("ʼ", "'")
    return any(q.tag.POS == "NOUN" and "Surn" not in q.tag and q.normal_form == low for q in g.morph().parse(low)[:6])


def adj_parse(word: str):
    """Розбір слова як прикметника/дієприкметника (серед перших розборів), якщо воно не іменник у називному."""
    low = word.lower().replace("’", "'").replace("ʼ", "'")
    if is_noun_lemma(low) or g.morph().parse(low)[0].tag.POS in ("VERB", "INFN", "ADVB", "PRCL", "CONJ", "PREP", "NPRO", "NUMR"):
        return None
    return next((q for q in g.morph().parse(low)[:6] if q.tag.POS in ("ADJF", "ADJS", "PRTS", "PRTF")), None)


def word_gender(word: str) -> str | None:
    """masc/femn для слова, що має рід у цьому контексті (минулий час, короткий/повний прикметник, дієприкметник)."""
    low = word.lower().replace("’", "'").replace("ʼ", "'")
    if g.loose_verb_gender(low) in ("masc", "femn") and re.search(r"(в|вся|ла|лася|міг|ліг|біг|ніс)$", low):
        return g.loose_verb_gender(low)
    p = adj_parse(low)
    if p and p.tag.gender in ("masc", "femn") and "plur" not in p.tag:
        return str(p.tag.gender)
    return None


def _target_after(text: str, start: int, target: str, changes: list, who: str):
    """Після займенника (позиція start): перше слово з родом упродовж 3 слів у межах речення — до роду target."""
    seen = 0
    for m in WORD.finditer(text, start):
        if STOP.search(text[start:m.start()]):
            return
        w = m.group()
        if w.lower() in PARTICLES:
            continue
        seen += 1
        if seen > 5:
            return
        cur = word_gender(w)
        if cur and _modifies_noun(text, m.end()):
            cur = None
        if cur and seen > 3 and not re.search(r"(в|вся|ла|лася|міг|ліг)$", w.lower()):
            cur = None  # далеко від «я/ти» шукаємо лише дієслова, не прикметники
        if cur and cur != target:
            new = inflect_gender(w, target) if w.lower() not in g.IRREGULAR else None
            if new and new.lower() != w.lower():
                changes.append((m.start(), m.end(), w, new, who))
            return
        if cur:
            return
        low = w.lower()
        if low in ROLES and ROLES[low] != low and target == "femn":
            changes.append((m.start(), m.end(), w, _fem_role(w), who + ":роль"))
            return
        start = m.end()


def _modifies_noun(text: str, pos: int) -> bool:
    """Прикметник/дієприкметник стоїть перед іменником чи іменем, з яким узгоджується («зайнята людина», «старшу Сенджоґахару»)."""
    m = WORD.search(text, pos)
    if not m or STOP.search(text[pos:m.start()]):
        return False
    if m.group()[0].isupper():
        return True
    return any(p.tag.POS == "NOUN" for p in g.morph().parse(m.group().lower())[:3])


def _fem_role(word: str) -> str:
    """Чоловіча назва ролі → жіноча в тому ж відмінку."""
    p = g.morph().parse(word.lower())[0]
    lemma = p.normal_form
    fem = ROLES.get(lemma)
    if not fem:
        return word
    q = g.morph().parse(fem)[0]
    want = {str(p.tag.case)} if p.tag.case else set()
    if p.tag.number:
        want.add(str(p.tag.number))
    form = q.inflect(want) if want else q
    return g._restore_case(word, form.word) if form else word


def fix_paragraph(uk: str, speaker: str | None, addressee: str | None, in_dialogue: bool) -> tuple[str, list]:
    """speaker/addressee: masc/femn/None. Поза лапками «я» — оповідач (masc)."""
    spans = g.quote_spans(uk)
    nested = [(m.start(), m.end()) for m in re.finditer("„[^“]*“", uk)]
    changes: list = []
    for m in WORD.finditer(uk):
        w = m.group().lower()
        inside = g.in_quote(m.start(), spans)
        if any(a <= m.start() < b for a, b in nested):
            continue
        if w == "я":
            tgt = speaker if inside else "masc"
            if tgt:
                _target_after(uk, m.end(), tgt, changes, "я")
        elif w == "мене" and (speaker if inside else "masc"):
            # «вважала мене нестерпною»: після дієслова, одразу прикметник в орудному (або короткий); «частина мене була зворушена» не чіпаємо
            tgt = speaker if inside else "masc"
            prev = [x for x in WORD.finditer(uk, 0, m.start())][-1:] 
            nxt = WORD.search(uk, m.end())
            if prev and nxt and not STOP.search(uk[m.end():nxt.start()]):
                pv = g.morph().parse(prev[0].group().lower())[0]
                pp = g.morph().parse(nxt.group().lower())[0]
                if pv.tag.POS == "VERB" and pv.normal_form not in ("бути", "стати") and (
                        (pp.tag.POS == "ADJF" and pp.tag.case == "ablt") or pp.tag.POS in ("ADJS", "PRTS"))                         and pp.tag.gender in ("masc", "femn") and str(pp.tag.gender) != tgt and not _modifies_noun(uk, nxt.end()):
                    new = inflect_gender(nxt.group(), tgt)
                    if new and new.lower() != nxt.group().lower():
                        changes.append((nxt.start(), nxt.end(), nxt.group(), new, "мене"))
        elif w == "ти" and not inside:
            # оповідач говорить сам до себе («Бідна ти, я пожалів себе»): прикметник перед «ти» — чоловічого роду
            prev = [x for x in WORD.finditer(uk, 0, m.start())][-1:]
            if prev and not STOP.search(uk[prev[0].end():m.start()]):
                pp = adj_parse(prev[0].group())
                if pp and pp.tag.POS in ("ADJF", "ADJS") and pp.tag.gender == "femn" and not _modifies_noun(uk, m.end()):
                    new = inflect_gender(prev[0].group(), "masc")
                    if new:
                        changes.append((prev[0].start(), prev[0].end(), prev[0].group(), new, "ти(оповідач)"))
        elif w == "ти" and inside and addressee:
            _target_after(uk, m.end(), addressee, changes, "ти")
    # ролі дівчат: репліка чоловіка до дівчини, або в абзаці названо дівчину і немає «він»
    female_ctx = (in_dialogue and speaker == "masc" and addressee == "femn") or (
        any(n in uk for n in FEMALE_NAMES) and not re.search(r"\bвін\b|\bйого\b", uk))
    if female_ctx:
        for m in WORD.finditer(uk):
            low = m.group().lower()
            p = g.morph().parse(low)[0]
            if p.normal_form in ROLES and ROLES[p.normal_form] != p.normal_form and p.tag.POS == "NOUN":
                already = any(c[0] == m.start() for c in changes)
                if not already and not (in_dialogue and speaker == "femn" and not any(n in uk for n in FEMALE_NAMES)):
                    changes.append((m.start(), m.end(), m.group(), _fem_role(m.group()), "роль"))
    out = uk
    for a, b, old, new, who in sorted(changes, reverse=True):
        out = out[:a] + new + out[b:]
    return out, [(old, new, who) for _, _, old, new, who in sorted(changes)]
