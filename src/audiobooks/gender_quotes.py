"""
Рід усередині лапок: чия це репліка, якщо слова автора не вказують мовця.

Етапи для кожної репліки перекладу, де є «я + дієслово минулого часу»:
  0. Програма: лапки перекладу зіставляються з лапками оригіналу (однакова кількість у абзаці), знаходяться кандидати.
  1. Модель: тип лапок (репліка / думка / вислів у лапках / написаний текст). Лишаються репліки та думки.
  2. Модель: яке питання треба поставити історії, щоб дізнатися мовця (двома формулюваннями; без наступних абзаців,
     щоб відповідь не просочилась у питання).
  3. Модель: відповідь на питання за попередніми абзацами (вікно розширюється 6 → 15 → 40) і наступними трьома
     з реченням-доказом. Програма перевіряє доказ: це не сама репліка, речення є в тексті, поруч названо мовця.
  4. Програма: рішення, лише коли обидва питання дали одного мовця з прийнятим доказом. Його рід береться з cast.
Результат: data/translations/<книга>/_gender/quotes_NN.json і quotes_NN_report.txt. Переклад не змінюється.
"""
import json
import re
from pathlib import Path

from . import gender as g
from .books import load_book, resolve_book
from .paths import TRANSLATIONS_DIR
from .speakers import SPEAKERS_DIR, load_notes, split_segments
from .text import parse_selection, slugify

CONTEXT_BEFORE = (6, 15, 40)
CONTEXT_AFTER = 3
MARK_OPEN, MARK_CLOSE = "⟦", "⟧"

EXAMPLE = ('Example from a different story (do not reuse): for the line "I finished the report." the question would be '
           '"Who had been writing a report in this story?"')
TYPE_PROMPT = """Here is a passage from a novel, with one paragraph marked >>> and one line in it marked {o}...{c}.

{text}

The marked line is inside quotation marks. Is it a line of dialogue spoken by a character, a line thought by a character, only a word/phrase/term/title quoted inside a sentence, or a written text (a note, a sign, a message)?
Answer with a JSON object."""
NEED_PROMPTS = {
    "A": """Here is a passage from a novel. The narrator is Araragi (a boy). The line marked {o}...{c} is spoken by some character.

{text}

The line will be translated into Ukrainian, where the past tense verb changes with the speaker's gender. The line alone does not tell the gender.
Think about what fact in the story you would have to find to know who says it, and write ONE short question about the story. The answer to your question must be the person who says the marked line, and the question must rely on a concrete fact (an event, an action, a request, a person), not on the line itself. {example}
Output only the question.""",
}
FIXED_QUESTION = "Who speaks the line marked with {o}...{c}?"
RETRIEVE_PROMPT = """You are reading a passage of a novel (the narrator is Araragi, a boy).

Characters:
{cast}

Passage (the paragraph with the line in question is marked >>>, the line itself {o}...{c}):
{text}

Question: {question}
Rules: answer using only the passage. The evidence must be a sentence copied from the passage that is NOT the marked line. If no other sentence settles the question, answer Unknown with an empty evidence.
Answer with a JSON object."""


def _norm(s: str) -> str:
    s = s.replace(MARK_OPEN, "").replace(MARK_CLOSE, "")
    s = re.sub(r"^\s*(?:>>>\s*)?(?:\[\d+\]\s*)?", "", s)  # модель копіює мітку абзацу «[185]»
    return re.sub(r"[\s“”\"'’]+", " ", s).strip(" .,;:!?").lower()


def render(en: list[str], i: int, target: tuple[int, int] | None = None) -> str:
    """Абзац оригіналу; якщо вказано (абзац, номер репліки), ця репліка обрамлена ⟦ ⟧."""
    if target is None or target[0] != i:
        return en[i]
    out, q = [], 0
    for s in split_segments(en[i]):
        if s["kind"] == "quote":
            q += 1
            t = f"“{s['text']}”" + ("" if not s.get("open") else "")
            out.append(f"{MARK_OPEN}{t}{MARK_CLOSE}" if q == target[1] else t)
        else:
            out.append(s["text"])
    return "".join(out)


def window(en: list[str], target: tuple[int, int], before: int, after: int) -> str:
    idx = target[0]
    a, b = max(0, idx - before), min(len(en), idx + after + 1)
    return "\n".join((">>> " if i == idx else "") + f"[{i}] {render(en, i, target)}" for i in range(a, b))


def candidates(uk: list[str], en: list[str]) -> tuple[list[dict], list[tuple[int, str]]]:
    """Репліки перекладу з «я + дієслово минулого часу»; і абзаци, де лапки оригіналу й перекладу не зіставити."""
    out, skipped = [], []
    for i, (u, e) in enumerate(zip(uk, en)):
        spans = g.quote_spans(u)
        matches = list(g.WORD.finditer(u))
        en_quotes = [s for s in split_segments(e) if s["kind"] == "quote"]
        found = []
        for t, m in enumerate(matches):
            if m.group().lower() != "я":
                continue
            ks = [k for k, (a, b) in enumerate(spans) if a < m.start() < b]
            if not ks:
                continue
            verbs = []
            for expected in ("masc", "femn"):  # chain повертає дієслова, чий рід ≠ expected
                verbs += g.chain(u, matches, t, expected, i, "probe", "?", set(), comma_stop=True)
            if verbs:
                found.append((ks[0], verbs))
        if not found:
            continue
        if len(spans) != len(en_quotes):
            skipped.append((i, f"лапок у перекладі {len(spans)}, в оригіналі {len(en_quotes)}"))
            continue
        merged = {}
        for k, verbs in found:
            merged.setdefault(k, {})
            for v in verbs:
                merged[k][v["start"]] = v
        for k, vs in merged.items():
            out.append({"par": i, "quote": k + 1, "uk": u[spans[k][0]:spans[k][1] + 1], "en": en_quotes[k]["text"],
                        "verbs": sorted(vs.values(), key=lambda v: v["start"])})
    return out, skipped


def _valid_evidence(en: list[str], target_text: str, evidence: str, answer: str, lo: int, hi: int, par: int) -> tuple[bool, str]:
    """Доказ: речення з тексту поза самою реплікою, поруч із яким названо мовця. Модель часто копіює весь абзац
    (репліка + слова автора), тому текст репліки з доказу вирізається і оцінюється залишок."""
    ev, tn = _norm(evidence), _norm(target_text)
    if not ev:
        return False, "доказ порожній"
    if ev in tn:
        return False, "доказ це сама репліка"
    rest = re.sub(r"\s+", " ", ev.replace(tn, " ")).strip() if tn in ev else ev
    if len(rest) < 4:
        return False, "після вирізання репліки нічого не лишилось"
    where = [i for i in range(lo, hi) if ev in _norm(en[i])]
    if not where:
        return False, "доказу немає в тексті"
    name = answer.lower()
    says = re.search(r"\bi (?:said|asked|answered|replied|thought|told|muttered|whispered|added|continued|called|shouted|\w+ed)\b", rest)
    if name in rest or (answer == "Araragi" and says) or any(name in _norm(en[i]).replace(tn, " ") for i in where):
        return True, f"абзац [{where[0]}]" + (" (авторські слова)" if rest != ev else "")
    return False, f"поруч із доказом не названо {answer}"


def resolve(server, notes: dict, en: list[str], cand: dict, log) -> dict:
    """Етапи 1-4 для однієї репліки. Повертає {status, speaker, type, trace}."""
    cast = notes["cast"]
    cast_text = "\n".join(f"- {n}: {d}" for n, d in cast.items())
    target = (cand["par"], cand["quote"])
    trace = []
    fmt = dict(o=MARK_OPEN, c=MARK_CLOSE, example=EXAMPLE)

    def ask(prompt: str, max_tokens: int, temperature: float = 0.0, schema: dict | None = None) -> str:
        extra = {"response_format": {"type": "json_object", "schema": schema}} if schema else None
        reply = server.complete(prompt, max_tokens=max_tokens, temperature=temperature, extra=extra)
        log.write(f"=== {target}\n{prompt}\n--- відповідь\n{reply}\n\n")
        log.flush()
        return reply

    tschema = {"type": "object", "additionalProperties": False, "required": ["type"],
               "properties": {"type": {"enum": ["dialogue", "thought", "quoted_term_or_phrase", "written_text"]}}}
    qtype = json.loads(ask(TYPE_PROMPT.format(text=window(en, target, 4, 2), **fmt), 30, schema=tschema))["type"]
    trace.append(f"тип: {qtype}")
    if qtype not in ("dialogue", "thought"):
        return {"status": "not_speech", "speaker": None, "type": qtype, "trace": trace}

    names = [*cast, "Unknown"]
    rschema = {"type": "object", "additionalProperties": False, "required": ["evidence", "answer"],
               "properties": {"evidence": {"type": "string"}, "answer": {"enum": names}}}
    answers = {}
    questions = {"A": ask(NEED_PROMPTS["A"].format(text=window(en, target, 4, 0), line=cand["en"], **fmt), 80, 0.2).strip().splitlines()[0],
                 "S": FIXED_QUESTION.format(**fmt)}
    for key, question in questions.items():
        trace.append(f"питання {key}: {question}")
        answers[key] = None
        for before in CONTEXT_BEFORE:
            lo, hi = max(0, target[0] - before), min(len(en), target[0] + CONTEXT_AFTER + 1)
            reply = ask(RETRIEVE_PROMPT.format(cast=cast_text, text=window(en, target, before, CONTEXT_AFTER), question=question, **fmt),
                        200, schema=rschema)
            try:
                r = json.loads(reply)
            except ValueError:
                continue
            if r["answer"] == "Unknown":
                trace.append(f"  {key} вікно {before}: Unknown")
                continue
            ok, why = _valid_evidence(en, cand["en"], r["evidence"], r["answer"], lo, hi, target[0])
            trace.append(f"  {key} вікно {before}: {r['answer']} | {r['evidence'][:90]} | {'ПРИЙНЯТО' if ok else 'ВІДХИЛЕНО'}: {why}")
            if ok:
                answers[key] = r["answer"]
                break
    accepted = {v for v in answers.values() if v}  # рішення: є прийнята відповідь і жодна прийнята не суперечить
    if len(accepted) == 1:
        return {"status": "decided", "speaker": accepted.pop(), "type": qtype, "answers": answers, "trace": trace}
    return {"status": "unresolved", "speaker": None, "type": qtype, "answers": answers, "trace": trace}


def _gold(slug: str, n: int) -> dict[tuple[int, int], str]:
    files = sorted((SPEAKERS_DIR / slug).glob(f"{n:02d}_*.json"))
    if not files:
        return {}
    out = {}
    for i, p in enumerate(json.loads(files[0].read_text(encoding="utf-8"))["paragraphs"]):
        q = 0
        for s in p["segs"]:
            if s["kind"] == "quote":
                q += 1
                if s.get("speaker"):
                    out[(i, q)] = "Araragi" if s["speaker"] == "narrator" else s["speaker"]
    return out


def run(args) -> None:
    from .translate.lapa import LlamaServer
    from .gender import GENDER_LOG
    book_path = resolve_book(args.book)
    notes = load_notes(book_path.stem)
    genders = g.cast_genders(notes["cast"])
    slug = slugify(book_path.stem)[:60]
    tr_dir = TRANSLATIONS_DIR / slug
    out_dir = tr_dir / "_gender"
    out_dir.mkdir(parents=True, exist_ok=True)
    chapters = load_book(args.book)
    files = {int(f.name[:2]): f for f in sorted(tr_dir.glob("[0-9][0-9]_*.txt"))}
    GENDER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LlamaServer(ctx=args.ctx, gpu_layers=args.gpu_layers) as server, open(GENDER_LOG, "a", encoding="utf-8") as log:
        for n in parse_selection(args.chapters, len(chapters)):
            f = files.get(n)
            if not f:
                continue
            uk = [ln for ln in f.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
            en = [ln for ln in chapters[n - 1][1].splitlines() if ln.strip()]
            if len(en) != len(uk):
                print(f"{n}: абзаців у оригіналі {len(en)}, у перекладі {len(uk)}: пропускаю", flush=True)
                continue
            cands, skipped = candidates(uk, en)
            if getattr(args, "paragraphs", None):
                lo, hi = (int(x) for x in args.paragraphs.split("-"))
                cands = [c for c in cands if lo <= c["par"] < hi]
            gold = _gold(slug, n)
            print(f"{n}. {chapters[n - 1][0]}: реплік із «я + дієслово» {len(cands)}, не зіставлено абзаців {len(skipped)}", flush=True)
            results, rep = [], []
            for c in cands:
                r = resolve(server, notes, en, c, log)
                key = (c["par"], c["quote"])
                r.update(par=c["par"], quote=c["quote"], gold=gold.get(key), uk=c["uk"][:120], verbs=[v["old"] for v in c["verbs"]])
                gen = genders.get(r["speaker"] or "", "")
                fixes = []
                if r["status"] == "decided" and gen:
                    for v in c["verbs"]:
                        if g.verb_gender(v["old"]) != gen:
                            new = g.inflect_verb(v["old"], gen)
                            fixes.append({"old": v["old"], "new": new})
                r["fixes"] = fixes
                results.append(r)
                mark = "" if r["gold"] is None else (" [збігається з розміткою]" if r["gold"] == r["speaker"] else
                                                       " [розмітка шарів: %s]" % r["gold"])
                rep.append(f"[{c['par']}.{c['quote']}] {r['status']} -> {r['speaker']}{mark}; дієслова {r['verbs']}; виправлення {fixes}\n"
                           f"    {r['uk']}\n    " + "\n    ".join(r["trace"]))
                print(f"  [{c['par']}.{c['quote']}] {r['status']} -> {r['speaker']}{mark}; виправлення {fixes}", flush=True)
            (out_dir / f"quotes_{n:02d}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            (out_dir / f"quotes_{n:02d}_report.txt").write_text("\n\n".join(rep) + "\n", encoding="utf-8")
            decided = [r for r in results if r["status"] == "decided"]
            agree = [r for r in decided if r["gold"] == r["speaker"]]
            known = [r for r in decided if r["gold"] is not None]
            print(f"  підсумок розділу {n}: розв'язано {len(decided)} з {len(results)}; з відомою розміткою {len(known)}, збіглося {len(agree)}; "
                  f"не мова {sum(r['status'] == 'not_speech' for r in results)}; без рішення {sum(r['status'] == 'unresolved' for r in results)}", flush=True)
