"""
Переклад англійська -> українська моделлю Lapa LLM v0.1.2 (GGUF) через llama.cpp (llama-server).

llama-server запускається автоматично як дочірній процес (OpenAI-сумісний API на localhost),
тому додаткових Python-залежностей не потрібно. Результат по розділах:
data/translations/<книга>/NN_<назва>.txt (перший рядок - назва, далі абзаци по рядку).
Прогрес кешується по шматках, тож перервану роботу можна продовжити.
"""
import hashlib
import http.client
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from ..books import load_book, resolve_book
from ..paths import CACHE_DIR, CONFIGS_DIR, MODELS_DIR, TRANSLATIONS_DIR
from ..text import parse_selection, slugify

MODEL_FILE = "lapa-v0.1.2-instruct-Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct-GGUF/resolve/main/{MODEL_FILE}"
LAPA_DIR = MODELS_DIR / "lapa"
LLAMA_DIR = MODELS_DIR / "llama.cpp"
HOST, PORT = "127.0.0.1", 8081
SERVER_LOG = CACHE_DIR / "translate" / "llama-server.log"
MAX_RESTARTS = 3
PROMPT_VERSION = "v1"  # part of the cache key: bump when the prompt changes
GROUP_CHARS = 1200      # paragraphs are translated in groups of about this size

PROMPT = """You are a professional literary translator. Translate the English text below into natural, fluent Ukrainian, preserving the style, tone, humor and narrative voice of the original.

Rules:
- Each input line is one paragraph. The output must have exactly the same number of lines, in the same order. Do not merge, split, add or omit lines.
- Keep quotation marks and dialogue punctuation natural for Ukrainian (use «» for quotes).
- Do not add notes, explanations or comments. Output only the translation.{glossary}

English text:
{text}"""


def find_llama_server() -> Path:
    exe = shutil.which("llama-server")
    if exe:
        return Path(exe)
    found = sorted(LLAMA_DIR.rglob("llama-server.exe"))
    if found:
        return found[0]
    raise SystemExit(
        f"Не знайдено llama-server. Розпакуйте збірку llama.cpp (win-vulkan-x64 або win-cpu-x64) у {LLAMA_DIR}."
    )


def load_glossary(book_name: str) -> dict[str, str]:
    """Optional configs/glossary/<book stem>.yaml: {English term: Ukrainian rendering}."""
    path = CONFIGS_DIR / "glossary" / f"{Path(book_name).stem}.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def group_paragraphs(paragraphs: list[str], max_chars: int = GROUP_CHARS) -> list[list[str]]:
    """Consecutive paragraphs grouped up to max_chars (a single long paragraph is its own group)."""
    groups, current, size = [], [], 0
    for p in paragraphs:
        if current and size + len(p) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(p)
        size += len(p)
    if current:
        groups.append(current)
    return groups


class LlamaServer:
    """Starts llama-server for the Lapa model, or reuses one that is already running."""

    def __init__(self, ctx: int = 8192, gpu_layers: int = 99):
        self.url = f"http://{HOST}:{PORT}"
        self.ctx, self.gpu_layers = ctx, gpu_layers
        self.proc = None
        self.truncated = False  # last completion hit max_tokens (usually a repetition loop)

    def _alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def __enter__(self):
        if self._alive():
            print("Використовую запущений llama-server.")
            return self
        self._start()
        return self

    def _start(self):
        model = LAPA_DIR / MODEL_FILE
        if not model.exists():
            raise SystemExit(f"Немає моделі {model}. Завантажте: {MODEL_URL}")
        exe = find_llama_server()
        cmd = [str(exe), "-m", str(model), "--host", HOST, "--port", str(PORT),
               "-c", str(self.ctx), "-ngl", str(self.gpu_layers)]
        print(f"Запускаю {exe.name} (модель {model.name}), лог: {SERVER_LOG}...")
        SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SERVER_LOG, "ab") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=log)
        for _ in range(300):  # up to ~5 min to load 7 GB
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server завершився одразу після запуску (дивіться {SERVER_LOG}).")
            if self._alive():
                print("Модель завантажено.")
                return
            time.sleep(1)
        raise SystemExit("llama-server не відповів за 5 хвилин.")

    def _stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __exit__(self, *exc):
        self._stop()

    def complete(self, prompt: str, max_tokens: int, temperature: float = 0.3, extra: dict | None = None) -> str:
        """One completion; if the server died (e.g. a GPU driver crash) restart it and retry."""
        for attempt in range(1, MAX_RESTARTS + 2):
            try:
                return self._complete(prompt, max_tokens, temperature, extra)
            except (ConnectionError, urllib.error.URLError, http.client.HTTPException) as e:
                if not self.proc or attempt > MAX_RESTARTS:
                    raise
                print(f"    llama-server впав ({type(e).__name__}); перезапуск {attempt}/{MAX_RESTARTS}...", flush=True)
                self._stop()
                self._start()

    def _complete(self, prompt: str, max_tokens: int, temperature: float, extra: dict | None = None) -> str:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            **(extra or {}),
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            choice = json.load(r)["choices"][0]
        self.truncated = choice.get("finish_reason") == "length"
        return choice["message"]["content"].strip()


_LATIN_LOOKALIKES = str.maketrans({
    "a": "а", "á": "а", "A": "А", "e": "е", "é": "е", "E": "Е", "o": "о", "ó": "о", "O": "О",
    "p": "р", "P": "Р", "c": "с", "C": "С", "y": "у", "ý": "у", "Y": "У", "x": "х", "X": "Х",
    "i": "і", "I": "І", "B": "В", "H": "Н", "K": "К", "M": "М", "T": "Т",
})


def fix_lookalikes(text: str) -> str:
    """The model sometimes puts a Latin lookalike (a, á, ý, y...) inside a Ukrainian word; TTS cannot read those.

    Only words that are mostly Cyrillic are touched, so genuine English leftovers stay as they are.
    """
    def fix(m):
        word = m.group()
        latin = re.findall(r"[A-Za-zÀ-ɏ]", word)
        if latin and len(latin) <= 2 and len(re.findall(r"[Ѐ-ӿ]", word)) >= 3:
            return word.translate(_LATIN_LOOKALIKES)
        return word
    return re.sub(r"\w+", fix, text)


def _clean_lines(output: str) -> list[str]:
    return [ln.strip() for ln in output.splitlines() if ln.strip()]


_STRETCHED = re.compile(r"\b[WwAaEeIiOoUuHh]{1,3}(?:-[WwAaEeIiOoUuHh]{1,3})+\b")  # 'Wa-a-a-ah', 'Ah-ha'


def _sound_effect(paragraph: str) -> str | None:
    """Ukrainian rendering of a paragraph that is only a stretched cry ('“Wa-a-a-ah─”' -> '“А-а-а-а─”').

    The model loops on these (it keeps extending the stretch until max_tokens), so they are rendered by rule.
    """
    if not _STRETCHED.search(paragraph) or re.search(r"[A-Za-z]{4,}", _STRETCHED.sub("", paragraph)):
        return None
    return _STRETCHED.sub(lambda m: "А" + "-а" * m.group().count("-"), paragraph)


def translate_group(server: LlamaServer, paragraphs: list[str], glossary_text: str) -> list[str]:
    cries = {i: c for i, p in enumerate(paragraphs) if len(p) <= 20 and (c := _sound_effect(p))}
    if cries:  # pure cries never reach the model: one of them can make it loop on the whole group
        rest = [p for i, p in enumerate(paragraphs) if i not in cries]
        done = iter(translate_group(server, rest, glossary_text) if rest else [])
        return [cries[i] if i in cries else next(done) for i in range(len(paragraphs))]
    prompt = PROMPT.format(glossary=glossary_text, text="\n".join(paragraphs))
    # Ukrainian output is roughly as long as the source in tokens; a generous cap stops runaway repetition early.
    budget = min(4096, sum(map(len, paragraphs)) + 300)
    lines = _clean_lines(server.complete(prompt, max_tokens=budget))
    if server.truncated:
        if len(paragraphs) > 1:
            print("  Відповідь обірвано (схоже на зациклення), перекладаю по одному абзацу.")
            return [x for p in paragraphs for x in translate_group(server, [p], glossary_text)]
        cry = _sound_effect(paragraphs[0])
        if cry:
            print(f"  Вигук «{paragraphs[0]}» замінено за правилом на «{cry}».")
            return [cry]
        print(f"  Зациклення на абзаці «{paragraphs[0][:60]}», повторюю з вищою температурою.")
        lines = _clean_lines(server.complete(prompt, max_tokens=budget, temperature=0.8))
        if server.truncated:
            print(f"  Не вдалося, лишаю без перекладу: «{paragraphs[0][:60]}».")
            return paragraphs
    if len(lines) == len(paragraphs):
        return lines
    if len(paragraphs) == 1:  # the model split a single paragraph: keep it as one
        return [" ".join(lines)]
    print(f"  Кількість абзаців не збіглася ({len(lines)} замість {len(paragraphs)}), перекладаю по одному.")
    return [x for p in paragraphs for x in translate_group(server, [p], glossary_text)]


def _glossary_block(glossary: dict[str, str], text: str = "") -> str:
    """Prompt block of fixed renderings for one group.

    Entries with non-ASCII letters (e.g. 'Mèmè Oshino') are included only when they occur in the text:
    they are usually added later, and this way adding one does not invalidate the cache of other groups.
    """
    entries = {en: uk for en, uk in glossary.items() if en.isascii() or en in text}
    if not entries:
        return ""
    lines = "\n".join(f"  {en} -> {uk}" for en, uk in entries.items())
    return f"\n- Use these fixed renderings of names and terms:\n{lines}"


def translate_chapter(server, paragraphs, glossary, cache_dir: Path) -> list[str]:
    result = []
    groups = group_paragraphs(paragraphs)
    start = time.time()
    for i, group in enumerate(groups, start=1):
        glossary_text = _glossary_block(glossary, "\n".join(group))
        key = hashlib.sha256(f"{MODEL_FILE}|{PROMPT_VERSION}|{glossary_text}|{chr(10).join(group)}".encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{key}.json"
        if cache_file.exists():
            translated = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"  Група {i}/{len(groups)} з кешу.")
        else:
            print(f"  Переклад групи {i}/{len(groups)} ({sum(map(len, group))} символів)...", flush=True)
            translated = [fix_lookalikes(x) for x in translate_group(server, group, glossary_text)]
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(translated, ensure_ascii=False), encoding="utf-8")
            print(f"    готово, {time.time() - start:.0f} с від початку розділу")
        result.extend(translated)
    return result


def limit_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """First paragraphs up to ~max_chars (for a quick test)."""
    picked, size = [], 0
    for p in paragraphs:
        if picked and size + len(p) > max_chars:
            break
        picked.append(p)
        size += len(p)
    return picked


def run(args):
    """Handler of `audiobook translate`."""
    sys.stdout.reconfigure(encoding="utf-8")
    chapters = load_book(args.book)
    book_path = resolve_book(args.book)
    slug = slugify(book_path.stem)[:60]
    glossary = load_glossary(book_path.name)
    out_dir = Path(args.out_dir) if args.out_dir else TRANSLATIONS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    with LlamaServer(ctx=args.ctx, gpu_layers=args.gpu_layers) as server:
        for n in parse_selection(args.chapters, len(chapters)):
            title, text = chapters[n - 1]
            paragraphs = text.split("\n")
            target_dir = out_dir
            if args.first_chars:  # test runs go to a subfolder so they never end up in the book
                paragraphs = limit_paragraphs(paragraphs, args.first_chars)
                target_dir = out_dir / "_samples"
                target_dir.mkdir(exist_ok=True)
            print(f"\n=== {n}. {title} ({len(paragraphs)} абзаців) ===")
            translated = translate_chapter(server, paragraphs, glossary, CACHE_DIR / "translate" / slug)
            out = target_dir / f"{n:02d}_{slugify(title)}.txt"
            out.write_text("\n".join([title, *translated]) + "\n", encoding="utf-8")
            print(f"  Збережено: {out}")
