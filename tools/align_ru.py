r"""
Зіставляє абзаци англійського оригіналу з абзацами російського видання (books/1.Bakemonogatari_Joukan_nopic.epub), щоб
російський абзац можна було дати перекладачеві підказкою про рід мовця (російське минулого часу має рід, англійське ні).
Алгоритм: послідовне вирівнювання (динамічне програмування) з кроками 1:1, 1:2, 2:1, 1:0, 0:1; ціна кроку залежить від
відношення довжин, чи починається абзац як репліка, і від збігу чисел.

  python tools\align_ru.py          →  data/gender_eval/ru_align.json: {"10": ["рос. абзац для EN-абзацу 0", ...], ...}
"""
import json
import math
import re
import sys
from pathlib import Path

from audiobooks.books import load_book
from audiobooks.paths import DATA_DIR

# книга: (англійське видання, російське видання, {розділ російського видання: англійські розділи з 1}, файл результату)
BOOKS = {
    "1": ("Monogatari Series v01 - BAKEMONOGATARI Part 1 [Vertical] [LuCaZ]", "1.Bakemonogatari_Joukan_nopic.epub",
          {0: range(1, 9), 1: range(9, 18)}, "ru_align.json"),
    "2": ("Monogatari Series v02 - BAKEMONOGATARI Part 2 [Vertical] [LuCaZ]", "2.Bakemonogatari_Chuukan_nopic.epub",
          {0: range(1, 10), 1: range(10, 17)}, "ru_align_v2.json"),
}
RATIO = 0.8        # середнє відношення довжин рос/англ
SKIP = 2.6         # ціна пропуску абзацу в одній з мов
BAND = 150         # наскільки індекси можуть розійтися
NUM = re.compile(r"\d+")


def is_dialogue_en(t: str) -> bool:
    return t[:1] in "“\"«"


def is_dialogue_ru(t: str) -> bool:
    return t[:1] in "—–-«\""


def cost(en: list[str], ru: list[str]) -> float:
    le, lr = sum(len(x) for x in en), sum(len(x) for x in ru)
    c = abs(math.log((lr + 8) / ((le + 8) * RATIO)))
    # діалог на початку першого абзацу має збігатися
    if is_dialogue_en(en[0]) != is_dialogue_ru(ru[0]):
        c += 1.2
    ne, nr = NUM.findall(" ".join(en)), NUM.findall(" ".join(ru))
    if ne or nr:
        c += 0.0 if ne == nr else 0.5
    return c + (0.4 if len(en) + len(ru) > 2 else 0.0)


def align(en: list[str], ru: list[str]) -> list[tuple[range, range]]:
    n, m = len(en), len(ru)
    INF = 1e18
    best = {(0, 0): (0.0, None)}
    for i in range(n + 1):
        for j in range(max(0, i - BAND), min(m, i + BAND) + 1):
            if (i, j) not in best:
                continue
            base = best[(i, j)][0]
            for di, dj in ((1, 1), (1, 2), (2, 1), (1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                c = SKIP if di == 0 or dj == 0 else cost(en[i:ni], ru[j:nj])
                if base + c < best.get((ni, nj), (INF, None))[0]:
                    best[(ni, nj)] = (base + c, (i, j))
    path, cur = [], (n, m)
    while best[cur][1] is not None:
        prev = best[cur][1]
        path.append((range(prev[0], cur[0]), range(prev[1], cur[1])))
        cur = prev
    return path[::-1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    stem, ru_name, parts, out_name = BOOKS[sys.argv[1] if len(sys.argv) > 1 else "1"]
    en_book, ru_book = load_book(stem + ".epub"), load_book(ru_name)
    out = {}
    for ru_idx, chapters in parts.items():
        ru = [l for l in ru_book[ru_idx][1].splitlines() if l.strip()]
        en, owner = [], []  # абзаци всіх англійських розділів частини підряд, і до якого розділу належить кожен
        for n in chapters:
            lines = [l for l in en_book[n - 1][1].splitlines() if l.strip()]
            en += lines
            owner += [(n, k) for k in range(len(lines))]
        path = align(en, ru)
        for n in chapters:
            out[str(n)] = [""] * sum(1 for o in owner if o[0] == n)
        stat = {"1:1": 0, "інші": 0, "без пари": 0}
        for ei, ri in path:
            if not len(ri):
                stat["без пари"] += 1
                continue
            stat["1:1" if len(ei) == 1 and len(ri) == 1 else "інші"] += 1
            text = " ".join(ru[j] for j in ri)
            for i in ei:
                n, k = owner[i]
                out[str(n)][k] = text
        print(f"частина {ru_idx + 1}: англ. {len(en)}, рос. {len(ru)}, кроків {len(path)}: {stat}")
    p = DATA_DIR / "gender_eval" / out_name
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("збережено", p)


if __name__ == "__main__":
    main()
