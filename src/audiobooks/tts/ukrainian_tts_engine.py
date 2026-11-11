import io
import os
import wave

from ..paths import MODELS_DIR

VOICES = ["tetiana", "mykyta", "lada", "dmytro", "oleksa"]
# Tacotron2 loses attention on long input, so it needs short fragments.
MAX_CHARS = 200


class UkrainianTtsEngine:
    name = "ukrainian-tts"
    max_chars = MAX_CHARS
    sample_rate = 22050

    def __init__(self, voice: str, speed: float = 1.0, **_):
        from ukrainian_tts.tts import TTS, Stress, Voices

        if voice not in VOICES:
            raise SystemExit(f"ukrainian-tts: невідомий голос '{voice}'. Доступні: {', '.join(VOICES)}")
        cache = MODELS_DIR / "ukrainian-tts"
        cache.mkdir(parents=True, exist_ok=True)
        # The ESPnet config refers to feats_stats.npz relatively, so load from inside the cache dir.
        cwd = os.getcwd()
        os.chdir(cache)
        try:
            self.tts = TTS(cache_folder=".", device="cpu")
        finally:
            os.chdir(cwd)
        self.voice = getattr(Voices, voice.capitalize()).value
        self.stress = Stress.Dictionary.value  # the model has no speed control

    def synth(self, text: str) -> bytes:
        buf = io.BytesIO()
        self.tts.tts(text, self.voice, self.stress, buf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            self.sample_rate = wf.getframerate()
            return wf.readframes(wf.getnframes())
