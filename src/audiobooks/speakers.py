"""
Розмітка «хто говорить» в оригіналі книги (шар 1: правила без моделі).

Абзаци оригіналу й перекладу збігаються 1:1, тому мітки, зняті з англійського тексту, потім переносяться на переклад.
Результат: data/speakers/<книга>/NN.json. Кожен абзац — список сегментів:
  {"kind": "narration" | "quote", "text": ..., "speaker": імʼя | "narrator" | null, "by": звідки мітка, "hint": ...}
`speaker == null` — репліка, яку правила не змогли визначити (її розбирають наступні шари).
"""
import json
import re
from collections import Counter
from pathlib import Path

from .books import load_book, resolve_book
from .paths import ROOT
from .text import parse_selection, slugify

SPEAKERS_DIR = ROOT / "data" / "speakers"
GLOSSARY_DIR = ROOT / "configs" / "glossary"
NARRATOR = "narrator"
PROTAGONIST = "Araragi"  # оповідь від першої особи: «I said» = Арараґі
ALIASES = {"Meme": "Oshino", "Mèmè": "Oshino"}

VERBS = (r"said|asked|replied|answered|added|muttered|whispered|shouted|exclaimed|cried|continued|murmured|retorted|"
         r"demanded|explained|declared|told|responded|called|snapped|sighed|remarked|suggested|insisted|groaned|"
         r"mumbled|commented|stated|announced|admitted|interrupted|protested|grumbled|yelled|screamed|answered|"
         r"inquired|began|finished|persisted|offered|pointed out|went on|corrected|conceded|objected|joked|teased")
ADVERB = r"(?:\w+ly\s+)?"
OPEN, CLOSE = "“", "”"


def load_names(book: str) -> list[str]:
    """Імена персонажів = односкладові ASCII-ключі глосарія (Araragi, Hanekawa, ...)."""
    import yaml
    path = GLOSSARY_DIR / f"{Path(book).stem}.yaml"
    names = []
    if path.exists():
        for k in yaml.safe_load(path.read_text(encoding="utf-8")) or {}:
            if k.isascii() and " " not in k and k not in ALIASES:
                names.append(k)
    return names


def split_segments(paragraph: str) -> list[dict]:
    """Розбиває абзац на репліки (в “ ”) і оповідь. Репліка, що не закрилась до кінця абзацу, має open=True."""
    segs, pos = [], 0
    while pos < len(paragraph):
        i = paragraph.find(OPEN, pos)
        if i < 0:
            break
        if i > pos:
            segs.append({"kind": "narration", "text": paragraph[pos:i]})
        j = paragraph.find(CLOSE, i + 1)
        if j < 0:
            segs.append({"kind": "quote", "text": paragraph[i + 1:], "open": True})
            return segs
        segs.append({"kind": "quote", "text": paragraph[i + 1:j]})
        pos = j + 1
    if pos < len(paragraph):
        segs.append({"kind": "narration", "text": paragraph[pos:]})
    return segs


def _subject(word: str, names: list[str]) -> tuple[str | None, str | None]:
    """(speaker, hint): імʼя персонажа, «I» = головний герой, he/she = лише підказка про стать."""
    word = ALIASES.get(word, word)
    if word in names:
        return word, None
    if word == "I":
        return PROTAGONIST, None
    if word.lower() in ("he", "she"):
        return None, word.lower()
    return None, None


def _attribution_after(text: str, names: list[str]) -> tuple[str | None, str | None]:
    name = "|".join(map(re.escape, [*names, *ALIASES, "I", "he", "she", "He", "She"]))
    head = text.lstrip(" ,.")[:90]
    m = re.match(rf"({name})\s+{ADVERB}(?:{VERBS})\b", head) or re.match(rf"(?:{VERBS})\s+({name})\b", head)
    return _subject(m.group(1), names) if m else (None, None)


def _attribution_before(text: str, names: list[str]) -> tuple[str | None, str | None]:
    """«she said, “...”»: підмет із дієсловом мовлення в кінці попередньої оповіді (останнє речення)."""
    name = "|".join(map(re.escape, [*names, *ALIASES, "I", "he", "she", "He", "She"]))
    tail = re.split(r"(?<=[.!?])\s+", text.strip())[-1][-120:]
    m = re.search(rf"(?:^|\W)({name})\s+{ADVERB}(?:{VERBS})\b[^.!?“”]{{0,50}}[,:]?\s*$", tail)
    return _subject(m.group(1), names) if m else (None, None)


def label_paragraph(segs: list[dict], names: list[str]) -> None:
    quotes = [i for i, s in enumerate(segs) if s["kind"] == "quote"]
    for s in segs:
        if s["kind"] == "narration":
            s["speaker"], s["by"] = NARRATOR, "narration"
    for n, i in enumerate(quotes):
        s = segs[i]
        s["speaker"], s["by"], s["hint"] = None, None, None
        after = segs[i + 1]["text"] if i + 1 < len(segs) and segs[i + 1]["kind"] == "narration" else ""
        before = segs[i - 1]["text"] if i > 0 and segs[i - 1]["kind"] == "narration" else ""
        who, hint = _attribution_after(after, names)
        by = "after"
        if not who and not hint:
            who, hint = _attribution_before(before, names)
            by = "before"
        # «“A,” she said, “B.”»: друга частина розірваної репліки (мала літера, попередня закінчується комою)
        if n > 0 and not who:
            p = segs[quotes[n - 1]]
            if p["text"].rstrip().endswith(",") and s["text"][:1].islower():
                who, by = p["speaker"], "split"
        # «“A,” she said. “B.”»: після підпису тієї ж репліки наступна репліка абзацу належить тому самому мовцеві
        if n > 0 and not who and quotes[n - 1] == i - 2:
            p = segs[i - 2]
            if p.get("by") == "rule:after" and p.get("speaker"):
                who, by = p["speaker"], "same"
        if who:
            s["speaker"], s["by"] = who, f"rule:{by}"
        elif hint:
            s["hint"] = hint


def label_chapter(paragraphs: list[str], names: list[str]) -> list[dict]:
    out, prev_open = [], None
    for p in paragraphs:
        segs = split_segments(p)
        label_paragraph(segs, names)
        # абзац, що починається з “ після абзацу з незакритою реплікою, — продовження тієї ж репліки
        if prev_open and segs and segs[0]["kind"] == "quote" and not segs[0].get("speaker"):
            segs[0]["speaker"], segs[0]["by"] = prev_open, "rule:continued"
        quotes = [s for s in segs if s["kind"] == "quote"]
        prev_open = quotes[-1].get("speaker") if quotes and quotes[-1].get("open") else None
        out.append({"segs": segs})
    return out


def run(args) -> None:
    book_name = args.book
    book_path = resolve_book(book_name)
    names = load_names(book_path.stem)
    slug = slugify(book_path.stem)[:60]
    out_dir = Path(args.out_dir) if args.out_dir else SPEAKERS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    chapters = load_book(book_name)
    print(f"Персонажі з глосарія: {', '.join(names) or '—'}")
    total = Counter()
    for n in parse_selection(args.chapters, len(chapters)):
        title, text = chapters[n - 1]
        labeled = label_chapter(text.splitlines(), names)
        (out_dir / f"{n:02d}_{slugify(title)}.json").write_text(
            json.dumps({"title": title, "paragraphs": labeled}, ensure_ascii=False, indent=1), encoding="utf-8")
        q = Counter("known" if s.get("speaker") else "unknown" for p in labeled for s in p["segs"] if s["kind"] == "quote")
        total += q
        print(f"{n:2}. {title:22} реплік {sum(q.values()):4}, визначено правилами {q['known']:4}, невизначено {q['unknown']:4}")
    print(f"Разом: визначено {total['known']} з {sum(total.values())} реплік; результат у {out_dir}")


# ---------------------------------------------------------------------------------------------------------------
# Шар 2: модель розбирає репліки, яких не визначили правила (вікнами по кілька абзаців із контекстом).
# ---------------------------------------------------------------------------------------------------------------
SPEAKER_NOTES_DIR = ROOT / "configs" / "speakers"
LLM_LOG = ROOT / "data" / "cache" / "speakers_llm.log"
WINDOW_CHARS = 2200       # скільки тексту абзаців, що розмічаються, у одному запиті
CONTEXT_PARAGRAPHS = 6    # абзаців перед вікном (уже з мітками), для контексту
CONTEXT_CHARS = 260       # довжина одного контекстного абзацу

PROMPT_L2 = """You are annotating a novel for an audiobook: for each quotation, decide who says it.

Story notes:
{story}

Cast (the narrator is Araragi, and "I" in the narration means Araragi):
{cast}

Rules:
- Quotations are marked [P.N] where P is the paragraph number and N the quote number inside it. A marker like [12.1=Hanekawa] already has a known speaker (use it as context).
- Dialogue usually alternates between two people; a quote that sounds like a reply answers the previous one.
- "he"/"she" in a tag tell the gender; "I" in a tag is Araragi.
- If you really cannot tell, answer Unknown.

Text:
{text}

Answer with a JSON object that gives the speaker for each of these quotations: {targets}
"""


PROMPT_PRESENT = """Below is a passage of a novel narrated by Araragi ("I").

Story notes:
{story}

Cast:
{cast}

Text:
{text}

Which characters from the cast are physically present in this scene and take part in it (speak, or are spoken to directly)? Characters who are only mentioned, remembered or discussed are NOT present. Answer with a JSON object {{"present": [...]}}.
"""


def load_notes(book_stem: str) -> dict:
    import yaml
    path = SPEAKER_NOTES_DIR / f"{book_stem}.yaml"
    if not path.exists():
        raise SystemExit(f"Немає нотаток для розмітки: {path} (cast + story).")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _render(idx: int, segs: list[dict], truncate: int = 0) -> str:
    out, q = [], 0
    for s in segs:
        if s["kind"] == "quote":
            q += 1
            tag = f"[{idx}.{q}={s['speaker']}]" if s.get("speaker") else f"[{idx}.{q}]"
            out.append(f"{tag}{OPEN}{s['text']}{CLOSE}")
        else:
            out.append(s["text"])
    text = "".join(out)
    return (text[:truncate] + "…") if truncate and len(text) > truncate else text


def _windows(paras: list[dict]) -> list[tuple[int, int]]:
    """Вікна [a, b) послідовних абзаців; у вікно беруться лише такі, що закінчуються на абзаці з невизначеною реплікою."""
    wins, i, n = [], 0, len(paras)
    while i < n:
        if not any(s["kind"] == "quote" and not s.get("speaker") for s in paras[i]["segs"]):
            i += 1
            continue
        j, size = i, 0
        while j < n and (j == i or size + sum(len(s["text"]) for s in paras[j]["segs"]) <= WINDOW_CHARS):
            size += sum(len(s["text"]) for s in paras[j]["segs"])
            j += 1
        wins.append((i, j))
        i = j
    return wins


def _gender(cast: dict, name: str) -> str | None:
    d = str(cast.get(name, "")).lower()
    return "she" if d.startswith("female") else "he" if d.startswith("male") else None


def apply_llm_answers(answers: dict, paras: list[dict], a: int, b: int, cast: dict) -> int:
    done = 0
    for key, name in answers.items():
        m = re.fullmatch(r"(\d+)\.(\d+)", key)
        if not m:
            continue
        pi, qn = int(m.group(1)), int(m.group(2))
        name = ALIASES.get(name, name)
        if not (a <= pi < b) or (name not in cast and name != "Unknown"):
            continue
        quotes = [s for s in paras[pi]["segs"] if s["kind"] == "quote"]
        if not 1 <= qn <= len(quotes) or quotes[qn - 1].get("speaker"):
            continue
        s = quotes[qn - 1]
        s["llm"] = name
        if name == "Unknown":
            s["by"] = "llm:unknown"
            continue
        conflict = s.get("hint") and _gender(cast, name) not in (None, s["hint"])
        s["speaker"], s["by"] = name, "llm:2?" if conflict else "llm:2"
        if conflict:
            s["note"] = f"підказка '{s['hint']}' не збігається зі статтю {name}"
        done += 1
    return done


def _targets(paras: list[dict], a: int, b: int) -> list[str]:
    return [f"{i}.{n}" for i in range(a, b)
            for n, s in enumerate((s for s in paras[i]["segs"] if s["kind"] == "quote"), start=1) if not s.get("speaker")]


def _present(server, notes: dict, cast: dict, cast_text: str, text: str) -> list[str]:
    """Хто з персонажів бере участь у сцені: Арараґі (оповідач) є завжди, згадані в розмові — ні."""
    schema = {"type": "object", "required": ["present"], "additionalProperties": False,
              "properties": {"present": {"type": "array", "items": {"enum": list(cast)}}}}
    reply = server.complete(PROMPT_PRESENT.format(story=notes["story"].strip(), cast=cast_text, text=text),
                            max_tokens=80, temperature=0.1,
                            extra={"response_format": {"type": "json_object", "schema": schema}})
    try:
        names = json.loads(reply)["present"]
    except (ValueError, KeyError):
        names = list(cast)
    return list(dict.fromkeys([PROTAGONIST, *names]))


def run_llm(args) -> None:
    from .translate.lapa import LlamaServer
    book_path = resolve_book(args.book)
    notes = load_notes(book_path.stem)
    cast = notes["cast"]
    cast_text = "\n".join(f"- {n}: {d}" for n, d in cast.items())
    slug = slugify(book_path.stem)[:60]
    out_dir = Path(args.out_dir) if args.out_dir else SPEAKERS_DIR / slug
    files = {int(f.name[:2]): f for f in sorted(out_dir.glob("[0-9][0-9]_*.json"))}
    import time
    with LlamaServer(ctx=args.ctx, gpu_layers=args.gpu_layers) as server:
        for n in parse_selection(args.chapters, max(files, default=0)):
            f = files.get(n)
            if not f:
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            paras = data["paragraphs"]
            wins = _windows(paras)
            if args.max_windows:
                wins = wins[:args.max_windows]
            print(f"{n}. {data['title']}: {len(wins)} вікон", flush=True)
            start = time.time()
            for k, (a, b) in enumerate(wins, start=1):
                before = [f"{i}: {_render(i, paras[i]['segs'], CONTEXT_CHARS)}" for i in range(max(0, a - CONTEXT_PARAGRAPHS), a)]
                text = "\n".join(before + [f"{i}: {_render(i, paras[i]['segs'])}" for i in range(a, b)])
                targets = _targets(paras, a, b)
                prompt = PROMPT_L2.format(story=notes["story"].strip(), cast=cast_text, text=text, targets=", ".join(targets))
                present = _present(server, notes, cast, cast_text, text) if args.present else list(cast)
                allowed = [*present, "Unknown"]
                schema = {"type": "object", "additionalProperties": False, "required": targets,
                          "properties": {t: {"enum": allowed} for t in targets}}
                reply = server.complete(prompt, max_tokens=60 + 25 * len(targets), temperature=0.1,
                                        extra={"response_format": {"type": "json_object", "schema": schema}})
                try:
                    answers = json.loads(reply)
                except ValueError:
                    answers = {}
                got = apply_llm_answers(answers, paras, a, b, cast)
                with open(LLM_LOG, "a", encoding="utf-8") as log:
                    log.write("\n".join([f"=== розділ {n}, абзаци {a}-{b - 1}", prompt, f"присутні: {present}", "--- відповідь", reply, "", ""]))
                f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  вікно {k}/{len(wins)} (абзаци {a}-{b - 1}): визначено {got}, {time.time() - start:.0f} с", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Шар 3: перевірка розмітки правилами узгодженості; що лишилось сумнівним — у review.json для ручного розгляду.
# ---------------------------------------------------------------------------------------------------------------
def _quotes(paras: list[dict], i: int) -> list[dict]:
    return [s for s in paras[i]["segs"] if s["kind"] == "quote"]


def _dialogue_runs(paras: list[dict]) -> list[list[int]]:
    """Серії послідовних абзаців, у кожному з яких є репліки."""
    runs, cur = [], []
    for i in range(len(paras)):
        if _quotes(paras, i):
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _says_name(text: str, name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\b", text))


def _bare_chains(paras: list[dict]) -> list[list[int]]:
    """Серії послідовних абзаців, у кожному з яких рівно одна репліка й жодної оповіді («голий» діалог)."""
    chains, cur = [], []
    for i, p in enumerate(paras):
        qs = [x for x in p["segs"] if x["kind"] == "quote"]
        bare = len(qs) == 1 and qs[0].get("speaker") != NARRATOR  # одна репліка в абзаці; оповідь поруч — зазвичай лише підпис
        if bare:
            cur.append(i)
        elif cur:
            chains.append(cur)
            cur = []
    if cur:
        chains.append(cur)
    return chains


def _fill_alternation(paras: list[dict]) -> Counter:
    """Діалог двох: у «голій» серії репліки йдуть по черзі. Серію з порушеним чергуванням (або з мовцем, якого
    у цьому діалозі нема) перезаповнюю за якорем — відомим мовцем безпосередньо перед серією (або після неї)."""
    stats = Counter()
    llm = lambda s: str(s.get("by", "")).startswith("llm")
    scenes: list[list[int]] = []
    for run in _dialogue_runs(paras):  # серії, розділені кількома абзацами оповіді, — одна сцена (ті самі учасники)
        if scenes and run[0] - scenes[-1][-1] <= 4:
            scenes[-1] += run
        else:
            scenes.append(list(run))
    for run in scenes:
        quotes = [s for i in run for s in _quotes(paras, i)]
        counts = Counter(s["speaker"] for s in quotes if s.get("speaker") and s["speaker"] not in (PROTAGONIST, NARRATOR)
                         and (s["by"].startswith("rule") or s["by"].startswith("check") or s["by"] in ("llm:2", "llm:3")))
        others = [n for n, c in counts.items() if c >= 2]
        if len(others) != 1:
            continue
        pair = (PROTAGONIST, others[0])
        other = lambda x: pair[1] if x == pair[0] else pair[0]
        for chain in _bare_chains(paras):
            if not set(chain) <= set(run) or len(chain) < 2:
                continue
            seq = [_quotes(paras, i)[0] for i in chain]
            suspect = any(llm(s) and s.get("speaker") not in pair for s in seq) or any(
                (llm(a) or llm(b)) and a.get("speaker") == b.get("speaker") for a, b in zip(seq, seq[1:]))
            if not suspect:
                continue
            before = _quotes(paras, chain[0] - 1)[-1:] if chain[0] > 0 else []
            after = _quotes(paras, chain[-1] + 1)[:1] if chain[-1] + 1 < len(paras) else []
            # якір: відома (не від моделі) мітка всередині серії, інакше мовець перед серією, інакше після неї
            first = None
            for k, s in enumerate(seq):
                if not llm(s) and s.get("speaker") in pair and not s.get("flag"):
                    first = s["speaker"] if k % 2 == 0 else other(s["speaker"])
                    break
            if first is None and before and before[0].get("speaker") in pair and not before[0].get("flag"):
                first = other(before[0]["speaker"])
            elif first is None and after and after[0].get("speaker") in pair and not after[0].get("flag"):
                last = other(after[0]["speaker"])
                first = last if len(seq) % 2 else other(last)
            if first is None:
                continue
            fill = [first if k % 2 == 0 else other(first) for k in range(len(seq))]
            if after and after[0].get("speaker") in pair and not after[0].get("flag") and fill[-1] == after[0]["speaker"]:
                continue  # якір після серії суперечить заповненню: нічого не вигадую
            if any(not llm(s) and s.get("speaker") != f for s, f in zip(seq, fill) if s.get("speaker")):
                continue  # правило/перевірка вже визначили інакше
            for s, f in zip(seq, fill):
                if s.get("speaker") != f:
                    s["was"], s["speaker"], s["by"] = s.get("speaker"), f, "check:alt"
                    s.pop("flag", None)
                    s.pop("candidates", None)
                    stats["заповнено чергуванням"] += 1
    return stats


def validate_chapter(paras: list[dict], cast: dict) -> dict:
    """Позначає й за змоги виправляє мітки моделі; повертає лічильники."""
    stats = Counter()
    for p in paras:
        qs = [x for x in p["segs"] if x["kind"] == "quote"]
        for n, q in enumerate(qs):
            if q.get("by", "").startswith("rule") or q.get("speaker") == NARRATOR or not q["text"][:1].islower():
                continue
            # «“A,” she said, “b.”»: друга частина розірваної репліки — лише якщо між ними коротка авторська ремарка
            i = p["segs"].index(q)
            gap = len(p["segs"][i - 1]["text"]) if i > 0 and p["segs"][i - 1]["kind"] == "narration" else 0
            prev_comma = n > 0 and qs[n - 1]["text"].rstrip().endswith(",") and gap <= 60
            if not prev_comma:  # слова в лапках всередині оповіді («“haunted house”»), а не репліка
                q["was"], q["speaker"], q["by"] = q.get("speaker"), NARRATOR, "check:fragment"
                q.pop("flag", None)
                stats["цитований фрагмент → оповідь"] += 1
    for run in _dialogue_runs(paras):
        quotes = [(i, s) for i in run for s in _quotes(paras, i)]
        trusted = {s["speaker"] for _, s in quotes if s.get("speaker") and not str(s.get("by", "")).startswith("llm")}
        trusted |= {s["speaker"] for _, s in quotes if s.get("by") == "llm:2"}
        others = {x for x in trusted if x != PROTAGONIST}
        for i, s in quotes:
            if not str(s.get("by", "")).startswith("llm") or not s.get("speaker"):
                continue
            who, flag = s["speaker"], None
            if _says_name(s["text"], who) and who != PROTAGONIST:
                flag = "name_in_quote"     # про себе в третій особі не говорять: імовірніше, її обговорюють
            elif s["by"] == "llm:2?":
                flag = "gender"
            elif who != PROTAGONIST and others - {who} and who not in trusted:
                flag = "outlier"           # у серії говорить інша людина, а ця названа лише моделлю
            elif who != PROTAGONIST and len(others) == 1 and who not in others \
                    and any(_says_name(t["text"], who) for _, t in quotes):
                flag = "outlier"
            if not flag:
                continue
            stats[f"позначено:{flag}"] += 1
            neighbours = {paras_q["speaker"] for j in (i - 1, i + 1) if 0 <= j < len(paras)
                          for paras_q in _quotes(paras, j) if paras_q.get("speaker")}
            candidates = sorted(others - {who})
            if len(candidates) == 1 and candidates[0] not in neighbours:
                s["was"], s["speaker"], s["by"] = who, candidates[0], f"check:{flag}"
                s.pop("note", None)
                stats["виправлено"] += 1
            else:
                s["flag"] = flag
                s["candidates"] = candidates
    stats += _fill_alternation(paras)
    # дві сусідні однорядкові репліки одного мовця: у діалозі так не буває, хтось із них помилковий
    for i in range(len(paras) - 1):
        a, b = _quotes(paras, i), _quotes(paras, i + 1)
        if len(a) == len(b) == 1 and a[0].get("speaker") and a[0]["speaker"] == b[0].get("speaker")                 and all(not x["text"].strip() for x in paras[i]["segs"] + paras[i + 1]["segs"] if x["kind"] == "narration"):
            for q in (a[0], b[0]):
                if str(q.get("by", "")).startswith("llm") and not q.get("flag"):
                    q["flag"], q["candidates"] = "no_alternation", []
                    stats["позначено:no_alternation"] += 1
    return stats


def review_items(chapter: int, title: str, paras: list[dict]) -> list[dict]:
    items = []
    for i, p in enumerate(paras):
        for n, s in enumerate((s for s in p["segs"] if s["kind"] == "quote"), start=1):
            if s.get("flag") or not s.get("speaker"):
                items.append({"chapter": chapter, "paragraph": i, "quote": n, "text": s["text"][:160],
                              "speaker": s.get("speaker"), "problem": s.get("flag") or s.get("by") or "не визначено",
                              "candidates": s.get("candidates") or [], "hint": s.get("hint")})
    return items


def run_validate(args) -> None:
    book_path = resolve_book(args.book)
    cast = load_notes(book_path.stem)["cast"]
    slug = slugify(book_path.stem)[:60]
    out_dir = Path(args.out_dir) if args.out_dir else SPEAKERS_DIR / slug
    files = {int(f.name[:2]): f for f in sorted(out_dir.glob("[0-9][0-9]_*.json"))}
    review = []
    for n in parse_selection(args.chapters, max(files, default=0)):
        f = files.get(n)
        if not f:
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for p in data["paragraphs"]:  # повторний запуск починає з чистого стану
            for s in p["segs"]:
                s.pop("flag", None)
                s.pop("candidates", None)
        stats = validate_chapter(data["paragraphs"], cast)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        items = review_items(n, data["title"], data["paragraphs"])
        review += items
        print(f"{n}. {data['title']}: {dict(stats) or 'змін немає'}; на розгляд {len(items)}")
    (out_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"На розгляд усього: {len(review)} → {out_dir / 'review.json'}")


# Шар 3, крок 2: сумнівні репліки ще раз розбирає модель, але з широким контекстом (усі абзаци навколо).
VERIFY_CONTEXT_BEFORE = 8
VERIFY_CONTEXT_AFTER = 3
VERIFY_MAX_CHARS = 9000

PROMPT_L3 = """You are checking a speaker annotation of a novel for an audiobook. The narrator is Araragi ("I").

Story notes:
{story}

Cast:
{cast}

Below is a passage. Quotations are marked [P.N]; a marker like [12.1=Hanekawa] is already settled and reliable. The unmarked ones are doubtful.
Think of the whole conversation: who is talking with whom, who asks and who answers, and that two people in a conversation take turns (a reply follows the question it answers). Names that are only mentioned in a quote are usually NOT the speaker.

Text:
{text}

Answer with a JSON object that gives the speaker for each of these doubtful quotations: {targets}
Choose only from: {names}. Use Unknown if it cannot be told.
"""


def _doubtful(s: dict) -> bool:
    return s["kind"] == "quote" and bool(s.get("flag") or not s.get("speaker"))


def _clusters(idx: list[int], gap: int = 6) -> list[tuple[int, int]]:
    out = []
    for i in idx:
        if out and i - out[-1][1] <= gap:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _render_masked(i: int, segs: list[dict]) -> str:
    """Як _render, але сумнівні репліки показані без імені."""
    out, q = [], 0
    for s in segs:
        if s["kind"] == "quote":
            q += 1
            known = s.get("speaker") and not s.get("flag")
            out.append(f"[{i}.{q}{'=' + s['speaker'] if known else ''}]{OPEN}{s['text']}{CLOSE}")
        else:
            out.append(s["text"])
    return "".join(out)


def verify_with_model(server, notes: dict, paras: list[dict], chapter: int) -> int:
    cast = notes["cast"]
    cast_text = "\n".join(f"- {n}: {d}" for n, d in cast.items())
    fixed = 0
    doubtful = [i for i, p in enumerate(paras) if any(_doubtful(s) for s in p["segs"])]
    for lo, hi in _clusters(doubtful):
        a, b = max(0, lo - VERIFY_CONTEXT_BEFORE), min(len(paras), hi + VERIFY_CONTEXT_AFTER + 1)
        while b - a > 1 and sum(len(_render_masked(i, paras[i]["segs"])) for i in range(a, b)) > VERIFY_MAX_CHARS:
            if lo - a > b - 1 - hi:
                a += 1
            else:
                b -= 1
        text = "\n".join(f"{i}: {_render_masked(i, paras[i]['segs'])}" for i in range(a, b))
        targets = [f"{i}.{n}" for i in range(a, b)
                   for n, s in enumerate((s for s in paras[i]["segs"] if s["kind"] == "quote"), start=1) if _doubtful(s)]
        if not targets:
            continue
        participants = [PROTAGONIST, *sorted({s["speaker"] for i in range(a, b) for s in paras[i]["segs"]
                                              if s["kind"] == "quote" and s.get("speaker") and not s.get("flag")
                                              and s["speaker"] not in (PROTAGONIST, NARRATOR)})]
        schema = {"type": "object", "additionalProperties": False, "required": targets,
                  "properties": {t: {"enum": [*participants, "Unknown"]} for t in targets}}
        prompt = PROMPT_L3.format(story=notes["story"].strip(), cast=cast_text, text=text,
                                  targets=", ".join(targets), names=", ".join(participants))
        reply = server.complete(prompt, max_tokens=60 + 25 * len(targets), temperature=0.1,
                                extra={"response_format": {"type": "json_object", "schema": schema}})
        try:
            answers = json.loads(reply)
        except ValueError:
            answers = {}
        with open(LLM_LOG, "a", encoding="utf-8") as log:
            log.write("\n".join([f"=== шар 3, розділ {chapter}, абзаци {a}-{b - 1}", prompt, "--- відповідь", reply, "", ""]))
        for key, name in answers.items():
            m = re.fullmatch(r"(\d+)\.(\d+)", key)
            if not m or name not in participants:
                continue
            quotes = _quotes(paras, int(m.group(1)))
            if not 1 <= int(m.group(2)) <= len(quotes):
                continue
            s = quotes[int(m.group(2)) - 1]
            if _doubtful(s):
                s["was"] = s.get("speaker")
                s["speaker"], s["by"] = name, "llm:3"
                s.pop("flag", None)
                s.pop("candidates", None)
                fixed += 1
    return fixed


def _clear_flags(paras: list[dict]) -> None:
    for p in paras:
        for s in p["segs"]:
            s.pop("flag", None)
            s.pop("candidates", None)


def run_verify(args) -> None:
    from .translate.lapa import LlamaServer
    book_path = resolve_book(args.book)
    notes = load_notes(book_path.stem)
    slug = slugify(book_path.stem)[:60]
    out_dir = Path(args.out_dir) if args.out_dir else SPEAKERS_DIR / slug
    files = {int(f.name[:2]): f for f in sorted(out_dir.glob("[0-9][0-9]_*.json"))}
    review = []
    with LlamaServer(ctx=args.ctx, gpu_layers=args.gpu_layers) as server:
        for n in parse_selection(args.chapters, max(files, default=0)):
            f = files.get(n)
            if not f:
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            paras = data["paragraphs"]
            _clear_flags(paras)
            validate_chapter(paras, notes["cast"])
            before = len(review_items(n, data["title"], paras))
            fixed = verify_with_model(server, notes, paras, n)
            _clear_flags(paras)  # після моделі ще раз перевіряємо узгодженість: що не пройшло — лишається на розгляд
            validate_chapter(paras, notes["cast"])
            f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            items = review_items(n, data["title"], paras)
            review += items
            print(f"{n}. {data['title']}: сумнівних було {before}, модель відповіла на {fixed}, лишилось на розгляд {len(items)}", flush=True)
    (out_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
