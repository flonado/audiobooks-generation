r"""
Рід мовця за російським абзацом (російське минулого часу має рід, англійське ні): дієслова минулого часу й короткі
прикметники в самій репліці (слова автора після тире не враховуються). Повертає "masc", "femn" або None.
Російський текст ніколи не показується моделі-перекладачу: їй лишається лише позначка роду, тож переклад
лишається з англійського.
"""
import re

_morph = None


def morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer(lang="ru")
    return _morph


WORD = re.compile(r"[А-Яа-яЁё-]+")


def speech_part(ru: str) -> str:
    """Репліка без слів автора: у «— Репліка, — сказала вона. — Далі» беремо 1-й і 3-й відрізки."""
    if not re.match(r"^\s*[—–-]", ru):
        return ru
    parts = re.split(r"\s[—–]\s", re.sub(r"^\s*[—–-]\s*", "", ru))
    return " ".join(p for i, p in enumerate(parts) if i % 2 == 0)


OTHER_SUBJECT = {"ты", "вы", "он", "она", "оно", "они", "тебя", "тебе", "тобой", "вас", "вам", "его", "её", "ему", "ей",
                 "арараги", "арараги-кун", "сендзёгахара", "хачикудзи", "ошино", "ханэкава", "цундере-тян", "кто", "что"}


def speaker_gender(ru: str) -> str | None:
    """Лише дієслова з «я» або без явного підмета в тому ж реченні-уривку: «Ты трогал?» стосується співрозмовника."""
    counts = {"masc": 0, "femn": 0}
    for clause in re.split(r"[,.!?;…]+", speech_part(ru)):
        words = WORD.findall(clause)
        low = {w.lower() for w in words}
        if low & OTHER_SUBJECT and "я" not in low:
            continue
        for w in words:
            p = morph().parse(w)[0]
            tag = p.tag
            if p.score < 0.4:
                continue
            if tag.POS in ("VERB", "INFN") and "past" in tag and tag.gender in counts and "sing" in tag:
                counts[str(tag.gender)] += 1
            elif tag.POS == "ADJS" and tag.gender in counts:
                counts[str(tag.gender)] += 1
    if counts["masc"] == counts["femn"]:
        return None
    return "masc" if counts["masc"] > counts["femn"] else "femn"


OPP = {"masc": "femn", "femn": "masc"}


def _quote_only(t: str) -> bool:
    return t[:1] in "“\"" and t.rstrip()[-1:] in "”\"" and t.count("“") <= 1


def propagate_genders(en: list[str], ru: list[str], reach: int = 2) -> dict[int, str]:
    """Рід мовця для абзаців розділу: з російського абзацу, а для суцільних реплік без нього — за чергуванням у серії
    реплік (сусід через 1 абзац протилежного роду, через 2 такого самого). Точність на розмітці ≈88%, покриття ≈54%."""
    lab = {i: g for i in range(len(en)) if (g := speaker_gender(ru[i]))}
    out = dict(lab)
    for i in range(len(en)):
        if i in lab or not _quote_only(en[i]):
            continue
        best = None
        for d in range(1, reach + 1):
            for j in (i - d, i + d):
                if j in lab and all(_quote_only(en[k]) for k in range(min(i, j), max(i, j) + 1)):
                    best = best or (lab[j] if d % 2 == 0 else OPP[lab[j]])
            if best:
                break
        if best:
            out[i] = best
    # у абзаців без реплік російські дієслова стосуються третіх осіб, а не мовця: позначку не ставимо
    return {i: g for i, g in out.items() if en[i][:1] in "“\""}
