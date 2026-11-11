"""Збирає .m4b (AAC, глави за розділами) з WAV-розділів за YAML-конфігом."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from .audio import wav_duration_ms
from .paths import ROOT


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg не знайдено. Встановіть його або виконайте: pip install imageio-ffmpeg")


def escape_meta(value: str) -> str:
    """Escape characters that are special in ffmpeg's ffmetadata format."""
    for ch in ("\\", "=", ";", "#", "\n"):
        value = value.replace(ch, "\\" + ch)
    return value


def load_config(config_path: Path) -> dict:
    """Paths inside the config are relative to the project root."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in ("output", "chapters"):
        if not cfg.get(key):
            raise SystemExit(f"У конфігу немає обов'язкового поля '{key}'.")

    cfg["output"] = ROOT / cfg["output"]
    if cfg.get("cover"):
        cfg["cover"] = ROOT / cfg["cover"]
        if not cfg["cover"].exists():
            raise SystemExit(f"Обкладинку не знайдено: {cfg['cover']}")

    chapters = []
    for i, item in enumerate(cfg["chapters"], start=1):
        if isinstance(item, str):
            item = {"file": item}
        path = ROOT / item["file"]
        if not path.exists():
            raise SystemExit(f"Розділ {i}: файл не знайдено: {path}")
        chapters.append({"file": path, "title": item.get("title") or path.stem})
    cfg["chapters"] = chapters
    return cfg


def build_metadata(cfg: dict, durations: list[int]) -> str:
    lines = [";FFMETADATA1"]
    for key in ("title", "author", "album", "genre", "year", "comment"):
        if cfg.get(key):
            # m4b players show 'artist' as the author and 'album' as the book title
            tag = {"author": "artist"}.get(key, key)
            lines.append(f"{tag}={escape_meta(str(cfg[key]))}")
    if cfg.get("title") and not cfg.get("album"):
        lines.append(f"album={escape_meta(str(cfg['title']))}")
    if cfg.get("narrator"):
        lines.append(f"composer={escape_meta(str(cfg['narrator']))}")

    start = 0
    for chapter, dur in zip(cfg["chapters"], durations):
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start}",
            f"END={start + dur}",
            f"title={escape_meta(chapter['title'])}",
        ]
        start += dur
    return "\n".join(lines) + "\n"


def build_m4b(cfg: dict, ffmpeg: str):
    # Chapters must share one format (same TTS engine) to be concatenated without re-encoding first.
    durations = [wav_duration_ms(c["file"]) for c in cfg["chapters"]]
    print(f"Розділів: {len(durations)}, загальна тривалість: {sum(durations) / 3600000:.2f} год.")
    cfg["output"].parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        list_file = tmp / "files.txt"
        list_file.write_text(
            "".join(f"file '{c['file'].resolve().as_posix()}'\n" for c in cfg["chapters"]), encoding="utf-8"
        )
        meta_file = tmp / "meta.txt"
        meta_file.write_text(build_metadata(cfg, durations), encoding="utf-8")

        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "warning", "-stats",
               "-f", "concat", "-safe", "0", "-i", str(list_file), "-i", str(meta_file)]
        if cfg.get("cover"):
            cmd += ["-i", str(cfg["cover"])]
        cmd += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
        if cfg.get("cover"):
            cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
        cmd += ["-c:a", "aac", "-b:a", str(cfg.get("bitrate", "64k")),
                "-ac", "1", "-movflags", "+faststart", "-f", "ipod", str(cfg["output"])]
        subprocess.run(cmd, check=True)

    print(f"Готово: {cfg['output']} ({cfg['output'].stat().st_size / 1024 / 1024:.1f} МБ)")


def run(args):
    """Handler of `audiobook m4b`."""
    sys.stdout.reconfigure(encoding="utf-8")
    build_m4b(load_config(Path(args.config)), find_ffmpeg())
