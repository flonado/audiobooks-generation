"""
Створює голос StyleTTS2 (.pt зі стилем, ~2 КБ) із записів мовця: без навчання, лише вилучення стилю
енкодерами моделі. Кілька файлів усереднюються. Результат лягає в data/models/styletts2/voices.

  .venv-styletts\Scripts\python.exe tools\make_styletts2_voice.py "Койомі Арараґі" data\voice_examples\araragi\reference.wav
  .venv-styletts\Scripts\python.exe tools\make_styletts2_voice.py "Койомі Арараґі" data\voice_examples\araragi\wavs\*.wav --min-sec 2
"""
import argparse
import glob
import sys

import soundfile as sf
import torch

from audiobooks.tts.styletts2_engine import HF_MODEL, VOICES_DIR


def soften_highs(path: str, db: float, out_path: str):
    """Gentle high-shelf cut (from 3 kHz, full cut at 8 kHz). The style encoder also copies the recording's own
    brightness/noise; a dub or noisy recording then makes the cloned voice sound hoarse."""
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=None)
    spec = librosa.stft(y, n_fft=1024, hop_length=256)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    gain = np.where(freqs > 3000, 10 ** (-db * np.clip((freqs - 3000) / 5000, 0, 1) / 20), 1.0)[:, None]
    y = librosa.istft(spec * gain, hop_length=256, length=len(y))
    sf.write(out_path, y / max(np.abs(y).max(), 1e-6) * 0.9, sr, subtype="PCM_16")


def mean_real_prosody() -> torch.Tensor:
    """Mean prosody style of the real voices (copies made by this tool, named like 'Ім'я (клон)', are skipped)."""
    return torch.stack([torch.load(f).detach().reshape(-1)[128:] for f in VOICES_DIR.glob("*.pt")
                        if " (" not in f.stem and not f.stem.endswith(".dur")]).mean(0).view(1, -1)


def fit_to_real_voices(prosody: torch.Tensor, k: int) -> torch.Tensor:
    """Closest point to `prosody` in the span of the k main directions of the real voices (copies made by this tool,
    named like 'Ім'я (клон)', are not real voices)."""
    real = [torch.load(f).detach().reshape(-1)[128:] for f in VOICES_DIR.glob("*.pt") if " (" not in f.stem]
    x = torch.stack(real)
    mean = x.mean(0)
    _, _, vt = torch.linalg.svd(x - mean, full_matrices=False)
    basis = vt[:k]
    return (mean + ((prosody.reshape(-1) - mean) @ basis.T) @ basis).view(1, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="ім'я голосу (стане іменем .pt)")
    ap.add_argument("wavs", nargs="*", help="wav-файли (можна шаблон *)")
    ap.add_argument("--add-duration", action="store_true",
                    help="лише дописати до наявного голосу <name>.dur.pt (тривалості з середнього стилю справжніх голосів), "
                         "якщо клон прискорюється на довгих реченнях; сам голос не змінюється")
    ap.add_argument("--min-sec", type=float, default=1.5, help="пропускати файли коротші за стільки секунд")
    ap.add_argument("--acoustic-norm", type=float, default=2.5,
                    help="масштабувати «тембр» до цієї норми (у готових голосів ~2-3; без цього клон виходить за межі, і текст ламається)")
    ap.add_argument("--prosody-from", default="", help="голос-донор, чий темп/просодію взяти (стійкіше, ніж з запису)")
    ap.add_argument("--prosody-weight", type=float, default=0.0, help="частка просодії з запису (0 = вся від донора)")
    ap.add_argument("--fit-prosody", type=int, default=0, metavar="K",
                    help="проєктувати просодію на K головних компонент справжніх голосів: без цього запис, "
                         "що лежить поза їх простором, дає темп, який прискорюється на довгому тексті (~15 - добре)")
    ap.add_argument("--keep-offspace", type=float, default=0.0, metavar="A",
                    help="з --fit-prosody: повернути частку A (0..1) відкинутої частини просодії; "
                         "більше A = схожіше на запис, але швидше прискорюється на довгому тексті")
    ap.add_argument("--soften-highs", type=float, default=0.0, metavar="DB",
                    help="зрізати високі частоти запису перед вилученням стилю (~12 дБ), якщо клон «хрипить»")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.add_duration:
        torch.save(mean_real_prosody(), VOICES_DIR / f"{args.name}.dur.pt")
        print(f"Дописано {VOICES_DIR / (args.name + '.dur.pt')}")
        return
    if not args.wavs:
        raise SystemExit("Вкажіть wav-файли.")
    from styletts2_inference.models import StyleTTS2
    files = [f for pattern in args.wavs for f in sorted(glob.glob(pattern))]
    files = [f for f in files if sf.info(f).duration >= args.min_sec]
    if not files:
        raise SystemExit("Немає підходящих файлів.")
    model = StyleTTS2(hf_path=HF_MODEL, device="cpu")
    styles, weights = [], []
    import os
    import tempfile
    tmp = tempfile.mkdtemp()
    for f in files:
        src = f
        if args.soften_highs:
            src = os.path.join(tmp, os.path.basename(f))
            soften_highs(f, args.soften_highs, src)
        styles.append(model.extract_voice_features(src).cpu())
        weights.append(sf.info(f).duration)  # longer clips give a steadier style
        print(f"  {f}  {weights[-1]:.1f} с")
    w = torch.tensor(weights).view(-1, 1, 1)
    style = (torch.stack(styles) * w).sum(0) / w.sum()
    acoustic, prosody = style[:, :128], style[:, 128:]
    if args.acoustic_norm:
        acoustic = acoustic * (args.acoustic_norm / acoustic.norm())
    if args.fit_prosody:
        fitted = fit_to_real_voices(prosody, args.fit_prosody)
        prosody = fitted + args.keep_offspace * (prosody - fitted)
    if args.prosody_from:
        donor = torch.load(VOICES_DIR / f"{args.prosody_from}.pt")[:, 128:]
        prosody = args.prosody_weight * prosody + (1 - args.prosody_weight) * donor
    style = torch.cat([acoustic, prosody], dim=1).detach()
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    out = VOICES_DIR / f"{args.name}.pt"
    torch.save(style, out)
    print(f"Збережено {out}  форма {tuple(style.shape)} (з {len(files)} файлів)")


if __name__ == "__main__":
    main()
