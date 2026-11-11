r"""
Озвучує сцену розділу Higgs 3 з кількома голосами: оповідь голосом оповідача, репліки голосами мовців.
Сцена описана в configs/scenes/*.yaml (розділ, діапазон абзаців, голоси, абзаци реплік кожного мовця).
Кожен сегмент кешується (data/output/higgs_scene/cache/), тож перервану роботу можна продовжити тією ж командою.

  .venv-higgs\Scripts\python.exe tools\voice_scene_higgs.py configs\scenes\mayoi_snail_002_open.yaml
"""
import argparse
import hashlib
import re
import sys
import time

import yaml

from audiobooks.audio import save_wav, silence, unique_path
from audiobooks.paths import CONFIGS_DIR, DATA_DIR, OUTPUT_DIR
from audiobooks.text import spell_numbers, split_sentences
from audiobooks.tts.higgs_engine import VOICES_DIR, HiggsEngine

QUOTE = re.compile(r"«[^»]*»|\"[^\"]*\"")
DASHES = " \t—–-"
PAUSE_SEGMENT, PAUSE_PARAGRAPH = 0.2, 0.5


def split_paragraph(text: str, has_speaker: bool) -> list[tuple[bool, str]]:
    """[(це_репліка, текст)]. Репліка — лапки на початку абзацу, після тире авторських слів або після кінця речення;
    лапки посеред речення («дорослого одягу», «Гм») — частина оповіді. Абзац без мовця — вся оповідь."""
    if not has_speaker:
        return [(False, text)]
    segs, pos = [], 0
    for m in QUOTE.finditer(text):
        before = text[pos:m.start()]
        head = text[:m.start()].rstrip(" \t")
        if m.start() == 0 or head[-1:] in "—–-.!?…" or not head:
            if before.strip(DASHES):
                segs.append((False, before))
            segs.append((True, m.group()[1:-1]))
            pos = m.end()
    if text[pos:].strip(DASHES):
        segs.append((False, text[pos:]))
    return segs


def fingerprint(voice: str) -> str:
    """Відбиток файлів голосу: перебудований голос з тим самим іменем не має брати старі репліки з кешу."""
    h = hashlib.md5()
    for ext in (".wav", ".txt"):
        h.update((VOICES_DIR / f"{voice}{ext}").read_bytes())
    return h.hexdigest()[:8]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[«»\"]", "", text)).strip(DASHES + ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--device", default="cpu", help="cpu або xpu (потрібен .venv-higgs-xpu); кеш окремий для кожного пристрою")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"], help="eager, якщо sdpa падає на XPU")
    ap.add_argument("--paragraphs", type=int, nargs=2, metavar=("З", "ПО"), help="замість діапазону зі сцени")
    ap.add_argument("--partial-every", type=int, default=20, help="кожні N абзаців зберігати проміжний wav")
    ap.add_argument("--batch", type=int, default=8, help="скільки речень генерувати за один прохід (1 = по одному)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    scene = yaml.safe_load(open(args.scene, encoding="utf-8"))
    chapter = next((DATA_DIR / "translations").glob("Monogatari_Series_v01*/%02d_*.txt" % scene["chapter"]))
    paragraphs = [l for l in chapter.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
    first, last = args.paragraphs or scene["paragraphs"]
    speaker_of = {p: who for who, ps in scene["turns"].items() for p in ps}
    voices, narrator = scene["voices"], scene["narrator"]

    eng = HiggsEngine(device=args.device, dtype=args.dtype, attn=args.attn)
    rate = eng.sample_rate
    suffix = "" if args.device == "cpu" else f"_{args.device}"
    cache = OUTPUT_DIR / "higgs_scene" / f"cache{suffix}"
    cache.mkdir(parents=True, exist_ok=True)
    name = f"{chapter.stem}_p{first:03d}-{last:03d}{suffix}"
    out_dir = OUTPUT_DIR / "higgs_scene"

    def assemble(chunks: list[bytes], path):
        save_wav(path, b"".join(chunks), rate)

    plan = []  # сегменти сцени за порядком: (абзац, номер, мовець, чи_репліка, текст, голос, файл кешу)
    for idx in range(first, last + 1):
        who = speaker_of.get(idx)
        for k, (is_quote, raw) in enumerate(split_paragraph(paragraphs[idx], who is not None)):
            text = clean(raw)
            if not re.search(r"\w", text):  # лишилась самá пунктуація після репліки
                continue
            speaker = who if is_quote else narrator
            voice = voices[speaker]
            key = hashlib.md5(f"{voice}|{fingerprint(voice)}|{text}".encode()).hexdigest()[:10]
            plan.append((idx, k, speaker, is_quote, text, voice, cache / f"p{idx:03d}_s{k}_{key}.pcm"))

    # Пакетна генерація: речення відсутніх у кеші сегментів іде вікнами по batch*4 у порядку тексту, усередині вікна
    # пакети зібрано за довжиною (голоси в пакеті можуть бути різні). Готові сегменти одразу пишуться в кеш.
    start = time.time()
    todo = [p for p in plan if not p[6].exists()]
    window = max(args.batch * 4, 1)
    pieces_of = {id(p): split_sentences(spell_numbers(p[4]), eng.max_chars) for p in todo}
    jobs = [(p, i, s) for p in todo for i, s in enumerate(pieces_of[id(p)])]
    pending = {}  # id сегмента → {номер речення: звук}, поки не готові всі речення сегмента
    for w0 in range(0, len(jobs), window):
        chunk = jobs[w0:w0 + window]
        eng.seed = 1000 + chunk[0][0][0] * 10 + chunk[0][0][1]
        out = eng.synth_many([s for _, _, s in chunk], voices=[p[5] for p, _, _ in chunk], batch=args.batch)
        for (p, i, _), pcm_bytes in zip(chunk, out):
            pending.setdefault(id(p), {})[i] = pcm_bytes
        for pid in {id(p): p for p, _, _ in chunk}:
            p = next(q for q in todo if id(q) == pid)
            n, parts = len(pieces_of[pid]), pending[pid]
            if len(parts) == n:  # решта речень сегмента могла бути в наступному вікні: тоді допишемо його там
                p[6].write_bytes(b"".join(parts[i] + (silence(0.35, rate) if i < n - 1 else b"") for i in range(n)))
                del pending[pid]
        print(f"  згенеровано {min(w0 + window, len(jobs))}/{len(jobs)} речень (минуло {(time.time() - start) / 60:.1f} хв)",
              flush=True)

    pcm, done = [], 0
    for idx in range(first, last + 1):
        for (pidx, k, speaker, is_quote, text, voice, f) in (p for p in plan if p[0] == idx):
            seg = f.read_bytes()
            label = "репліка" if is_quote else "оповідь"
            print(f"[{idx:03d}.{k}] {speaker:12s} {label}  {len(seg) / 2 / rate:5.1f} с  | {text[:60]}  "
                  f"(минуло {(time.time() - start) / 60:.0f} хв)", flush=True)
            pcm += [seg, silence(PAUSE_SEGMENT, rate)]
        pcm.append(silence(PAUSE_PARAGRAPH - PAUSE_SEGMENT, rate))
        done += 1
        if done % args.partial_every == 0:
            assemble(pcm, out_dir / f"{name}_partial.wav")
    assemble(pcm, unique_path(out_dir, name, "v"))


if __name__ == "__main__":
    main()
