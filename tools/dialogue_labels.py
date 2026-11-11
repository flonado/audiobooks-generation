r"""
Рід мовця й співрозмовника для кожної репліки розділу, виведений для ВСІЇ серії реплік разом.
Діалог двох людей чергується: репліки 0, 2, 4… належать одній особі, 1, 3, 5… іншій. Рід кожної з двох особ
визначається за свідченнями з усіх реплік серії:
  - російське видання: дієслова минулого часу з «я»/без підмета в репліці = рід мовця; з «ты» = рід співрозмовника;
  - англійський абзац: слова автора «Hachikuji said», звертання «Hachikuji,»/«Mister Araragi,» (співрозмовник).
Обираємо поєднання (рід особи A, рід особи B), що збігається з найбільшою вагою свідчень. Російський текст
у модель-перекладач не передається: повертаються лише роди.

  labels = label_turns(en_paragraphs, ru_paragraphs, cast_genders)   # {номер абзацу: {"speaker": "masc"/"femn", "addressee": ...}}
"""
import re
from itertools import product

from ru_speaker_gender import OTHER_SUBJECT, WORD, morph, speech_part

RU_SPLIT = re.compile(r"[,.!?;…]+")
SAY = r"(?:said|asked|replied|answered|added|murmured|whispered|continued|conceded|offered|objected|muttered|snapped|declared|admitted|warned|exclaimed|shouted)"


def _past_gender(word: str) -> str | None:
    p = morph().parse(word)[0]
    t = p.tag
    if p.score < 0.4 or "sing" not in t or t.gender not in ("masc", "femn"):
        return None
    if (t.POS in ("VERB", "INFN") and "past" in t) or t.POS == "ADJS":
        return str(t.gender)
    return None


def ru_evidence(ru: str) -> tuple[list[str], list[str]]:
    """(роди мовця, роди співрозмовника) за російською реплікою."""
    sp, ad = [], []
    for clause in RU_SPLIT.split(speech_part(ru)):
        words = WORD.findall(clause)
        low = [w.lower() for w in words]
        third = (set(OTHER_SUBJECT) - {"ты"}) | {"ему", "ей", "тебе"}
        for w in words:
            g = _past_gender(w)
            if not g:
                continue
            if "ты" in low:
                ad.append(g)
            elif "я" in low or not (set(low) & third):
                sp.append(g)
    return sp, ad


def is_turn(en: str) -> bool:
    return "“" in en or en[:1] == '"'


def en_evidence(en: str, names: dict[str, str]) -> tuple[list[str], list[str]]:
    """(рід мовця за словами автора, рід співрозмовника за звертанням) з англійського абзацу."""
    sp, ad = [], []
    outside = re.sub(r"“[^”]*”", " ", en)
    for name, g in names.items():
        if re.search(rf"\b{name}\b (?:\w+ )?{SAY}\b|\b{SAY} {name}\b|\bsaid {name}\b", outside):
            sp.append(g)
    for quote in re.findall(r"“([^”]*)”", en):
        for name, g in names.items():
            if re.search(rf"(?:^|, )(?:Mister |Miss )?{name}[,.!?…]|, (?:Mister |Miss )?{name}$", quote.strip()):
                ad.append(g)
    return sp, ad


MIN_MARGIN = 2


def confident(labels: dict, min_margin: int = MIN_MARGIN) -> dict:
    """Лише мітки, у яких свідчення переконливо визначають паритет серії; решта лишається без позначки й без виправлень."""
    return {i: l for i, l in labels.items() if l.get("margin", 0) >= min_margin}


def label_turns(en: list[str], ru: list[str], names: dict[str, str], gap: int = 5) -> dict[int, dict[str, str]]:
    """names: {"Hachikuji": "femn", "Araragi": "masc", …} для звертань і слів автора.

    Розмова = репліки, між якими не більше gap абзаців. Двоє співрозмовників мають по одному роду на всю розмову.
    Усередині розмови суцільні серії реплік (без абзаців оповіді між ними) чергуються суворо, а після абзацу оповіді
    говорити може і той самий, тому паритет кожної серії підбирається окремо за свідченнями."""
    turns = [i for i, e in enumerate(en) if is_turn(e)]
    convs, cur = [], []
    for t in turns:
        if cur and t - cur[-1] > gap:
            convs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        convs.append(cur)
    out = {}
    for conv in convs:
        runs, run = [], []
        for t in conv:
            if run and t - run[-1] > 1:
                runs.append(run)
                run = []
            run.append(t)
        runs.append(run)
        ev = {}
        for i in conv:
            rs, ra = ru_evidence(ru[i])
            es, ea = en_evidence(en[i], names)
            ev[i] = [("s", g, 1) for g in rs] + [("a", g, 1) for g in ra] + [("s", g, 3) for g in es] + [("a", g, 2) for g in ea]

        def run_score(pair, run, parity):
            score = 0
            for k, i in enumerate(run):
                for kind, g, w in ev[i]:
                    who = (k + parity) % 2 if kind == "s" else 1 - (k + parity) % 2
                    score += w if pair[who] == g else -w
            return score

        best_pair, best_total, best_parity, second_total = None, 0, None, 0
        for pair in product(("masc", "femn"), repeat=2):
            total, parities, last = 0, [], None
            for run in runs:
                sc = [run_score(pair, run, p) for p in (0, 1)]
                if sc[0] == sc[1]:  # без свідчень: відповідає той, хто не говорив останнім
                    p = 0 if last is None else (1 if last == 0 else 0)
                else:
                    p = 0 if sc[0] > sc[1] else 1
                total += max(sc)
                parities.append(p)
                last = (len(run) - 1 + p) % 2
            if total > best_total:
                second_total = best_total
                best_pair, best_total, best_parity = pair, total, parities
            elif total > second_total:
                second_total = total
        if best_pair is None:
            continue
        same = best_pair[0] == best_pair[1]
        for run, p in zip(runs, best_parity):
            # впевненість серії: на скільки свідчення за цей паритет переважають протилежний; однакові роди в розмові
            # (мовець = співрозмовник) майже завжди ознака збою, бо в книзі майже скрізь говорять чоловік із жінкою
            margin = 0 if same else abs(run_score(best_pair, run, 0) - run_score(best_pair, run, 1))
            for k, i in enumerate(run):
                out[i] = {"speaker": best_pair[(k + p) % 2], "addressee": best_pair[1 - (k + p) % 2],
                          "conf": best_total, "margin": margin}
    return out
