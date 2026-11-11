"""
Витягує голос закадрового дубляжу для прикладу / дотюнювання StyleTTS2.

Ідея: у дубляжній доріжці український голос лежить поверх оригінальної доріжки (музика, ефекти, японські
голоси, притишені). Оригінальна доріжка є окремим потоком, тож її віднімають з адаптивним коефіцієнтом
підсилення по частотах і часу; лишається український голос. Репліки й тайминги беруться з повних субтитрів.

Результат у data/voice_examples/<name>/: wavs/*.wav (24 кГц, моно, 16 біт), metadata.csv (файл|текст|мовець),
reference.wav (склеєні репліки ~30 с для вилучення стилю).

  .venv-styletts\Scripts\python.exe tools\extract_dub_voice.py <відео.mkv> --name araragi --start 0 --end 60
"""
import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import correlate, istft, resample_poly, stft

ROOT = Path(__file__).resolve().parents[1]
OUT_SR = 24000  # StyleTTS2 sample rate
FRAME, HOP = 2048, 512


def extract_stream(video: Path, stream: int, start: float, end: float, dest: Path):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(start), "-to", str(end), "-i", str(video), "-map", f"0:{stream}",
                    "-ac", "1", str(dest)], check=True)


def load_subs(video: Path, stream: int, style: str, start: float, end: float) -> list[tuple[float, float, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        ass = Path(tmp) / "subs.ass"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(video), "-map", f"0:{stream}", str(ass)], check=True)
        lines = ass.read_text(encoding="utf-8-sig").splitlines()

    def secs(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    items = []
    for ln in lines:
        if ln.startswith("Dialogue:"):
            p = ln.split(",", 9)
            if p[3] == style and start <= secs(p[1]) and secs(p[2]) <= end:
                items.append((secs(p[1]), secs(p[2]), re.sub(r"\{.*?\}", "", p[9]).replace(chr(92) + "N", " ").strip()))
    return sorted(items)


def merge_lines(items, max_len=14.0, max_gap=1.0):
    """Join lines that end mid-sentence (',' or '-') with the next one."""
    merged = []
    for s, e, t in items:
        if merged and merged[-1][2].rstrip()[-1:] in ",-–—" and s - merged[-1][1] <= max_gap and e - merged[-1][0] <= max_len:
            ps, _, pt = merged[-1]
            merged[-1] = (ps, e, f"{pt} {t}")
        else:
            merged.append((s, e, t))
    return merged


def remove_background(dub: np.ndarray, orig: np.ndarray, sr: int, time_win=15, freq_win=7) -> np.ndarray:
    """dub - H(t,f) * orig, where H is a local complex least-squares gain (the original is ducked under the voice)."""
    n = min(len(dub), len(orig))
    dub, orig = dub[:n], orig[:n]
    lag = np.argmax(np.abs(correlate(dub[: sr * 20], orig[: sr * 20], mode="full", method="fft"))) - (sr * 20 - 1)
    orig = np.roll(orig, lag)
    kw = dict(nperseg=FRAME, noverlap=FRAME - HOP)
    _, _, d = stft(dub, sr, **kw)
    _, _, o = stft(orig, sr, **kw)

    def smooth(x):
        return uniform_filter(x, (freq_win, time_win), mode="nearest")

    cross = d * np.conj(o)
    h = (smooth(cross.real) + 1j * smooth(cross.imag)) / (smooth(np.abs(o) ** 2) + 1e-10)
    _, clean = istft(d - h * o, sr, **kw)
    return clean[:n]


def speech_segments(clean: np.ndarray, sr: int, t0: float, merge_gap: float = 0.15) -> list[tuple[float, float]]:
    """Stretches of real speech in the cleaned track (video time). Subtitles are loose and often back to back
    (end of one = start of the next) while the voice lags behind them, so edges come from the audio itself."""
    hop = int(0.02 * sr)
    env = np.sqrt((clean[: len(clean) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    active = env > max(np.percentile(env, 95) * 0.03, 1e-4)
    gap = int(merge_gap / 0.02)
    segs, i, n = [], 0, len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and (active[j] or active[j: j + gap].any()):
            j += 1
        segs.append((t0 + i * 0.02, t0 + j * 0.02))
        i = j
    return segs


def cut_line(clean: np.ndarray, sr: int, t0: float, s: float, e: float, prev_end: float | None,
             next_start: float | None, segs: list[tuple[float, float]] | None = None,
             prev_line: tuple[float, float] | None = None, next_line: tuple[float, float] | None = None) -> np.ndarray:
    """Cut one line out of `clean` (whose first sample is at time t0 of the video).

    Subtitle times are only approximate and fixed padding either clips the end of the phrase or bleeds into the next
    line, so both edges are put at the quietest point of the margin around the subtitle time.
    """
    if segs:  # segments whose middle lies inside the subtitle window (a little wider than the subtitle)
        def owner(a: float, b: float) -> int:
            """0 = this line, -1 / 1 = the previous / next one: most overlap with its subtitle, else the closest."""
            cands = {0: (s, e)}
            if prev_line:
                cands[-1] = prev_line
            if next_line:
                cands[1] = next_line
            overlap = {k: max(0.0, min(b, y) - max(a, x)) for k, (x, y) in cands.items()}
            if max(overlap.values()) > 0:
                return max(overlap, key=lambda k: (overlap[k], k == 0))
            mid = (a + b) / 2
            return min(cands, key=lambda k: min(abs(mid - cands[k][0]), abs(mid - cands[k][1])))

        mine = [(a, b) for a, b in segs if s - 1.0 <= (a + b) / 2 <= e + 1.0 and owner(a, b) == 0]
        if mine:
            a, b = mine[0][0] - 0.08, mine[-1][1] + 0.08
            seg = clean[max(0, int((a - t0) * sr)): int((b - t0) * sr)].copy()
            fade = int(0.01 * sr)
            if len(seg) > 2 * fade:
                seg[:fade] *= np.linspace(0, 1, fade)
                seg[-fade:] *= np.linspace(1, 0, fade)
            return seg
    lo = max(s - 0.4, (prev_end + 0.02) if prev_end is not None else s - 0.4)
    hi = min(e + 0.8, (next_start - 0.02) if next_start is not None else e + 0.8)
    hop = max(1, int(0.01 * sr))
    env_start = lo

    def quietest(a: float, b: float) -> float:
        a, b = max(a, lo), min(max(b, a + 0.02), hi)
        i, j = int((a - t0) * sr), int((b - t0) * sr)
        frames = clean[i:j]
        n = len(frames) // hop
        if n < 2:
            return a
        rms = np.sqrt((frames[: n * hop].reshape(n, hop) ** 2).mean(1))
        return a + (int(np.argmin(rms)) + 0.5) * hop / sr

    a, b = quietest(lo, s + 0.05), quietest(e - 0.05, hi)
    seg = clean[int((a - t0) * sr): int((max(b, a + 0.2) - t0) * sr)].copy()
    fade = int(0.01 * sr)
    if len(seg) > 2 * fade:
        seg[:fade] *= np.linspace(0, 1, fade)
        seg[-fade:] *= np.linspace(1, 0, fade)
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--name", required=True, help="назва папки результату")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=60)
    ap.add_argument("--window", type=float, default=120, help="довжина шматка обробки в секундах (економить пам'ять)")
    ap.add_argument("--dub-stream", type=int, default=1, help="потік дубляжу (дивіться ffmpeg -i)")
    ap.add_argument("--orig-stream", type=int, default=2, help="потік оригіналу")
    ap.add_argument("--subs-stream", type=int, default=4, help="потік повних субтитрів")
    ap.add_argument("--style", default="BakeDialogue", help="стиль реплік у .ass")
    ap.add_argument("--speaker", default="")
    ap.add_argument("--min-sec", type=float, default=0.0, help="пропускати репліки коротші за стільки секунд")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    video = Path(args.video)
    out_dir = ROOT / "data" / "voice_examples" / args.name
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    speaker = args.speaker or args.name

    lines = merge_lines(load_subs(video, args.subs_stream, args.style, args.start, args.end))
    rows, pieces, i = [], [], 0
    w0 = args.start
    while w0 < args.end:
        w1 = min(w0 + args.window, args.end)
        batch = [x for x in lines if w0 <= x[0] < w1 and x[1] - x[0] >= args.min_sec]
        if batch:
            c0, c1 = max(0.0, w0 - 1), w1 + 16  # margin: a line may end after the window, lag is estimated on the margin too
            with tempfile.TemporaryDirectory() as tmp:
                d_wav, o_wav = Path(tmp) / "dub.wav", Path(tmp) / "orig.wav"
                extract_stream(video, args.dub_stream, c0, c1, d_wav)
                extract_stream(video, args.orig_stream, c0, c1, o_wav)
                dub, sr = sf.read(d_wav, dtype="float32")
                orig, _ = sf.read(o_wav, dtype="float32")
            clean = remove_background(dub, orig, sr)
            segs = speech_segments(clean, sr, c0)
            for s, e, text in batch:
                k = lines.index((s, e, text))
                prev_end = lines[k - 1][1] if k else None
                next_start = lines[k + 1][0] if k + 1 < len(lines) else None
                seg = cut_line(clean, sr, c0, s, e, prev_end, next_start, segs,
                               lines[k - 1][:2] if k else None, lines[k + 1][:2] if k + 1 < len(lines) else None)
                seg = resample_poly(seg, OUT_SR, sr)
                seg = seg / max(np.abs(seg).max(), 1e-6) * 0.9
                i += 1
                name = f"{args.name}_{i:03d}.wav"
                sf.write(out_dir / "wavs" / name, seg, OUT_SR, subtype="PCM_16")
                rows.append((f"wavs/{name}", text, speaker, round(len(seg) / OUT_SR, 2), round(s, 2)))
                pieces.append(seg)
                print(f"{name}  {s:7.1f}  {len(seg) / OUT_SR:5.1f} с  {text}", flush=True)
        w0 = w1

    with open(out_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerows(r[:3] for r in rows)
    with open(out_dir / "timings.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|").writerows((r[0], r[4], r[3]) for r in rows)  # file | start in video | duration
    gap = np.zeros(int(0.3 * OUT_SR))
    ref, total = [], 0.0
    for seg in pieces:  # reference: consecutive lines up to ~30 s
        if total > 30:
            break
        ref += [seg, gap]
        total += len(seg) / OUT_SR + 0.3
    if ref:
        sf.write(out_dir / "reference.wav", np.concatenate(ref), OUT_SR, subtype="PCM_16")
    print(f"\nУсього реплік: {len(rows)}, {sum(r[3] for r in rows):.0f} с. Папка: {out_dir}")


if __name__ == "__main__":
    main()
