"""Дрібні текстові утиліти, спільні для всіх команд."""
import re


def prepare_text(title: str, body: str) -> str:
    """
    Make paragraphs end with punctuation so sentence-aware chunking treats
    each paragraph as its own sentence, and prepend the chapter title.
    """
    lines = [title] + body.split("\n")
    fixed = [ln if re.search(r'[.!?…:;»"”)]$', ln) else ln + "." for ln in lines]
    return " ".join(fixed)


def parse_selection(spec: str, total: int) -> list[int]:
    """'3' -> [3], '2-5' -> [2,3,4,5], '1,4,7-8' -> [1,4,7,8] (1-based)."""
    picked = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            picked.extend(range(int(a), int(b) + 1))
        elif part:
            picked.append(int(part))
    bad = [n for n in picked if not 1 <= n <= total]
    if bad:
        raise SystemExit(f"Немає таких розділів: {bad}. Доступно 1-{total}.")
    return picked


def slugify(title: str) -> str:
    """Safe file-name fragment that keeps Cyrillic letters."""
    return re.sub(r"[^\w-]+", "_", title).strip("_")


def split_sentences(text: str, max_chars: int) -> list[str]:
    """Group sentences into pieces up to max_chars."""
    sentences = re.split(r"(?<=[.!?…]) +", text)
    pieces, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            pieces.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        pieces.append(current)
    return pieces


_ONES = ["нуль", "один", "два", "три", "чотири", "п'ять", "шість", "сім", "вісім", "дев'ять",
         "десять", "одинадцять", "дванадцять", "тринадцять", "чотирнадцять", "п'ятнадцять",
         "шістнадцять", "сімнадцять", "вісімнадцять", "дев'ятнадцять"]
_TENS = ["", "", "двадцять", "тридцять", "сорок", "п'ятдесят", "шістдесят", "сімдесят", "вісімдесят", "дев'яносто"]
_HUNDREDS = ["", "сто", "двісті", "триста", "чотириста", "п'ятсот", "шістсот", "сімсот", "вісімсот", "дев'ятсот"]


def _number_words(n: int) -> str:
    """Ukrainian cardinal (nominative, masculine) for 0..999999; larger numbers are left as digits."""
    if n >= 1_000_000:
        return str(n)
    if n >= 1000:
        t, rest = divmod(n, 1000)
        if t % 100 in range(11, 20) or t % 10 not in (1, 2, 3, 4):
            unit = "тисяч"
        else:
            unit = "тисяча" if t % 10 == 1 else "тисячі"
        head = _number_words(t).replace("один", "одна").replace("два", "дві") if t % 10 in (1, 2) and t % 100 not in range(11, 20) else _number_words(t)
        return " ".join(x for x in (head, unit, _number_words(rest) if rest else "") if x)
    parts = []
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10])
        n %= 10
    if n or not parts:
        parts.append(_ONES[n])
    return " ".join(parts)


def spell_numbers(text: str) -> str:
    """Replace digit groups with words: local TTS engines silently drop digits ('Розділ 4' -> 'Розділ')."""
    return re.sub(r"\d+", lambda m: _number_words(int(m.group())), text)
