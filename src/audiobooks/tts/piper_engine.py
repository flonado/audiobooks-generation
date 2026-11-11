from ..paths import MODELS_DIR

VOICES = {
    "tetiana": "uk_UA-tetiana-high",
    "mykyta": "uk_UA-mykyta-high",
    "oleksa": "uk_UA-oleksa-high",
}
MAX_CHARS = 500


class PiperEngine:
    name = "piper"
    max_chars = MAX_CHARS

    def __init__(self, voice: str, speed: float = 1.0, noise_scale=None, noise_w=None, volume=0.85, **_):
        from piper import PiperVoice, SynthesisConfig

        if voice not in VOICES:
            raise SystemExit(f"Piper: невідомий голос '{voice}'. Доступні: {', '.join(VOICES)}")
        model = MODELS_DIR / "piper" / f"{VOICES[voice]}.onnx"
        if not model.exists():
            raise SystemExit(
                f"Немає моделі {model}. Завантажте: python -m piper.download_voices "
                f"{VOICES[voice]} --download-dir {model.parent}"
            )
        self.voice = PiperVoice.load(model)
        self.sample_rate = self.voice.config.sample_rate
        # Lower noise_scale/noise_w = steadier, less rough voice (model defaults: 0.667 / 0.8).
        # volume < 1 leaves headroom: normalized output peaks at full scale and can crackle in players.
        self.cfg = SynthesisConfig(
            length_scale=1.0 / speed, noise_scale=noise_scale, noise_w_scale=noise_w, volume=volume
        )

    def synth(self, text: str) -> bytes:
        return b"".join(c.audio_int16_bytes for c in self.voice.synthesize(text, self.cfg))
