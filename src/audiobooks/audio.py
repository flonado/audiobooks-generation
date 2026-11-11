"""WAV / PCM helpers (16-bit mono)."""
import wave
from pathlib import Path


def silence(seconds: float, sample_rate: int) -> bytes:
    return b"\x00" * (int(sample_rate * seconds) * 2)


def save_wav(path: Path, pcm: bytes, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    print(f"  Збережено: {path} ({len(pcm) / 2 / sample_rate / 60:.1f} хв)")


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return round(wf.getnframes() * 1000 / wf.getframerate())


def unique_path(directory: Path, base: str, tag: str) -> Path:
    """
    `<base>.wav` if it is free; otherwise `<base>_<tag>.wav`, then a counter,
    so earlier results are never overwritten.
    """
    path = directory / f"{base}.wav"
    if not path.exists():
        return path
    path = directory / f"{base}_{tag}.wav"
    counter = 2
    while path.exists():
        path = directory / f"{base}_{tag}_{counter}.wav"
        counter += 1
    return path
