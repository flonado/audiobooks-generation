"""
StyleTTS2, українська мультиспікерна модель patriotyk/styletts2_ukrainian_multispeaker.
Стилі голосів (.pt) лежать у data/models/styletts2/voices/.
"""
import re
from unicodedata import normalize

import numpy as np

from ..paths import CONFIGS_DIR, MODELS_DIR

VOICES_DIR = MODELS_DIR / "styletts2" / "voices"
HF_MODEL = "patriotyk/styletts2_ukrainian_multispeaker"
SAMPLE_RATE = 24000
MAX_CHARS = 400
CLAUSE_CHARS = 100  # longer sentences are split at commas (see split_clauses)
CLAUSE_PAUSE = 0.15  # seconds between clauses of one sentence


ACUTE = chr(0x301)


def keep_first_stress(text: str) -> str:
    """The stress dictionary lists accepted alternatives of some words as several marks ('ді́вчи́ни', 'за́вжди́'),
    and the model then stresses the last one ('дівчи́ни'). Keep only the first (main) mark of a word;
    hyphenated compounds keep one mark per part."""
    def fix(m):
        word = m.group()
        out = []
        for part in re.split("(-)", word):
            i = part.find(ACUTE)
            out.append(part if i < 0 else part[: i + 1] + part[i + 1:].replace(ACUTE, ""))
        return "".join(out)
    return re.sub(r"[^\s.,;:!?«»“”\"()…—–]+", fix, text)


def _accent(value: str) -> str:
    """Наголос можна писати знаком «+» перед голосною («Вогне+лап») або як U+0301 після неї."""
    return re.sub(r"\+(\w)", lambda m: m.group(1) + ACUTE, str(value)).strip()


def stress_overrides() -> dict[str, str]:
    """Усі configs/stress*.yaml: {слово: слово з наголосом}. Ключ без «+», значення з наголосом."""
    import yaml
    out: dict[str, str] = {}
    for path in sorted(CONFIGS_DIR.glob("stress*.yaml")):
        for k, v in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items():
            out[str(k).lower()] = _accent(v)
    return out


def _stem_rules(overrides: dict[str, str]) -> list[tuple[str, str]]:
    """Наголос в основі слова діє на всі його відмінкові форми: «Вогнела́п» → Вогнела́па, Вогнела́пові…
    Якщо наголос на останньому складі леми (рудьк+о́), правило лише для цієї точної форми."""
    rules = []
    for plain, val in overrides.items():
        k = val.find(ACUTE)
        stem = re.sub("[аяоеуіиь]$", "", plain)
        if 0 < k <= len(stem):
            rules.append((stem, val.replace(ACUTE, "")[:k] + ACUTE + val.replace(ACUTE, "")[k:len(stem)]))
    return sorted(rules, key=lambda r: -len(r[0]))


def apply_stress_overrides(text: str, overrides: dict[str, str]) -> str:
    if not overrides:
        return text
    rules = _stem_rules(overrides)

    def fix(m):
        word = m.group()
        plain = word.replace(ACUTE, "").lower()
        new = overrides.get(plain)
        if new is None:
            for stem, stressed in rules:
                if plain.startswith(stem) and len(plain) - len(stem) <= 5:
                    new = stressed + plain[len(stem):]
                    break
        if new is None:
            return word
        return new[0].upper() + new[1:] if word[0].isupper() else new
    return re.sub(r"[^\s.,;:!?«»“”\"()…—–-]+", fix, text)


def available_voices() -> list[str]:
    return sorted(p.stem for p in VOICES_DIR.glob("*.pt") if not p.stem.endswith(".dur"))


def split_to_parts(text: str) -> list[str]:
    """Split at .?!: followed by a space; merge fragments of <=20 chars (same rule as the official demo)."""
    parts, current = [], ""
    for i, ch in enumerate(text):
        current += ch
        if ch in ".?!:" and i < len(text) - 1 and text[i + 1] == " " and len(current) > 20:
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return parts


def split_clauses(part: str, max_len: int = CLAUSE_CHARS, min_len: int = 25) -> list[str]:
    """Split a long sentence at commas/dashes into pieces of about max_len characters.

    The model compresses time on long inputs (a cloned voice up to ~2x faster than at the start), so long
    sentences are voiced clause by clause; tiny clauses ('Звісно,') are glued to their neighbour.
    """
    if len(part) <= max_len:
        return [part]
    pieces, current = [], ""
    for clause in re.split(r"(?<=[,;:—–])" + chr(92) + "s+", part):
        if current and len(current) >= min_len and len(current) + 1 + len(clause) > max_len:
            pieces.append(current)
            current = clause
        else:
            current = f"{current} {clause}".strip()
    if current:
        pieces.append(current)
    return pieces


class StyleTTS2Engine:
    name = "styletts2"
    max_chars = MAX_CHARS
    sample_rate = SAMPLE_RATE

    def __init__(self, voice: str, speed: float = 1.0, device: str = "cpu", **_):
        import torch
        from styletts2_inference.models import StyleTTS2
        from ukrainian_word_stress import Stressifier, StressSymbol

        voice_file = VOICES_DIR / f"{voice}.pt"
        if not voice_file.exists():
            raise SystemExit(f"StyleTTS2: немає голосу '{voice}'. Доступні: {', '.join(available_voices())}")
        self.torch = torch
        self.stress_symbol = StressSymbol.CombiningAcuteAccent
        self.stressify = Stressifier()
        self.overrides = stress_overrides()
        self.model = StyleTTS2(hf_path=HF_MODEL, device=device)
        self.style = torch.load(voice_file).to(device)
        # optional <voice>.dur.pt (1x128): prosody style used ONLY to predict durations. A cloned voice whose own
        # prosody is off the real voices' range makes the duration predictor collapse to the 1-frame minimum on
        # longer sentences (words get swallowed), so its durations are taken from a normal style instead.
        dur_file = VOICES_DIR / f"{voice}.dur.pt"
        self.dur_style = torch.load(dur_file).reshape(1, -1).to(device) if dur_file.exists() else None
        self.speed = speed

    def _render(self, tokens):
        if self.dur_style is None:
            return self.model(tokens, speed=self.speed, s_prev=self.style)
        return self._forward_split_duration(tokens)

    def _forward_split_duration(self, tokens):
        """StyleTTS2.forward with durations predicted from self.dur_style and everything else from self.style."""
        torch, m = self.torch, self.model
        dtype = next(m.parameters()).dtype
        style, s_dur = self.style.to(m.device), self.dur_style.to(m.device)
        tokens = torch.cat([torch.LongTensor([0]).to(m.device), tokens.to(m.device)], axis=0).unsqueeze(0)
        with torch.no_grad():
            lengths = tokens.new_full((1,), tokens.shape[1], dtype=torch.long)
            mask = torch.arange(lengths.max()).unsqueeze(0).expand(1, -1).type_as(lengths)
            text_mask = torch.gt(mask + 1, lengths.unsqueeze(1)).to(tokens.device)
            t_en = m.text_encoder(tokens, lengths, text_mask)
            d_en = m.plbert_encoder(m.plbert(tokens, (~text_mask).int())).transpose(2, 1)
            s, ref = style[:, 128:], style[:, :128]

            x, _ = m.predictor.lstm(m.predictor.text_encoder(d_en.to(dtype=dtype), s_dur.to(dtype=dtype), lengths, text_mask))
            duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1) / self.speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_dur[-1] = pred_dur[-1].clamp(max=6)
            if m.config.model_params.decoder.type == "hifigan":
                pred_dur[0] = 30
            aln = torch.zeros(lengths, int(pred_dur.sum().data))
            frame = 0
            for i in range(aln.size(0)):
                aln[i, frame: min(frame + int(pred_dur[i]), aln.shape[1])] = 1
                frame += int(pred_dur[i])
                if frame >= aln.shape[1]:
                    break

            d = m.predictor.text_encoder(d_en.to(dtype=dtype), s.to(dtype=dtype), lengths, text_mask)
            en = d.transpose(2, 1) @ aln.unsqueeze(0).to(m.device, dtype=dtype)
            f0, noise = m.predictor.F0Ntrain(en, s.to(dtype=dtype))
            asr = (t_en @ aln.unsqueeze(0).to(m.device)).to(dtype=dtype)
            out = m.decoder(asr, f0, noise, ref.squeeze().unsqueeze(0).to(dtype=dtype))
            if m.config.model_params.decoder.type == "hifigan":
                out = out[:, :, 14500:]
            return out.squeeze()

    def _prepare(self, text: str) -> str:
        text = text.strip().replace('"', "").replace("+", self.stress_symbol)
        text = normalize("NFKC", text)
        text = re.sub(r"[᠆‐‑‒–—―⁻₋−⸺⸻]", "-", text)
        if text and text[-1] not in ".?!:-":
            text += "."
        text = re.sub(r" - ", ": ", text)
        return apply_stress_overrides(keep_first_stress(self.stressify(text)), self.overrides)

    def synth(self, text: str) -> bytes:
        from ipa_uk import ipa

        wavs = []
        pause = self.torch.zeros(int(CLAUSE_PAUSE * SAMPLE_RATE))
        for part in split_to_parts(text):
            if not part.strip():
                continue
            for i, clause in enumerate(split_clauses(part, CLAUSE_CHARS if self.dur_style is None else 10 ** 6)):
                ps = ipa(self._prepare(clause))
                if ps:
                    tokens = self.model.tokenizer.encode(ps)
                    if i:
                        wavs.append(pause.to(self.style.device))
                    wavs.append(self._render(tokens).reshape(-1).cpu())
        audio = self.torch.concatenate(wavs).cpu().numpy()
        return (np.clip(audio, -1.0, 1.0) * 32767 * 0.9).astype(np.int16).tobytes()
