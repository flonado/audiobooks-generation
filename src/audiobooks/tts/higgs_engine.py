"""
Higgs Audio v3 TTS (Boson AI, ~4B, bosonai/higgs-tts-3-4b): Qwen3-бекбон + 8 аудіокодбуків із затримкою, кодек у тому ж
safetensors. Офіційно запускається лише на CUDA (SGLang-Omni) і Mac (MLX), тут — власний інференс на transformers
(логіка семплера й упаковки промпту взята зі спільнотної реалізації Saganaki22/Higgs_v3-TTS-ComfyUI, MIT).

Голос = референсний запис + його текст (клонування в контексті, без навчання): data/models/higgs3/voices/<ім'я>.wav + .txt.
Ліцензія ваг некомерційна (Boson Higgs TTS 3 Research and Non-Commercial License); клонувати лише з дозволу мовця.
"""
import json
import re
from pathlib import Path

import numpy as np

from ..paths import MODELS_DIR

HIGGS_DIR = MODELS_DIR / "higgs3"
MODEL_DIR = HIGGS_DIR / "higgs-tts-3-4b"
VOICES_DIR = HIGGS_DIR / "voices"
CODEC_CONFIG = HIGGS_DIR / "higgs_audio_v2_tokenizer_config.json"
SAMPLE_RATE = 24000
MAX_CHARS = 260
NO_VOICE = "(без референсу)"

BOC_ID, EOC_ID, STOP_CODE = 1024, 1025, -1
AUDIO_PLACEHOLDER = -100
CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."
WEIGHT_PREFIXES = {
    "tied.embedding.text_embedding.": "backbone.model.embed_tokens.",
    "body.layers.": "backbone.model.layers.",
    "body.norm.": "backbone.model.norm.",
    "tied.head.text_head.": "backbone.lm_head.",
    "tied.embedding.modality_embeddings.0.embedding.": "modality_embedding.",
    "tied.head.modality_heads.0.": "modality_head.",
}


def available_voices() -> list[str]:
    return [NO_VOICE] + sorted(p.stem for p in VOICES_DIR.glob("*.wav") if p.with_suffix(".txt").exists())


def prepare_text(text: str, speed: float = 1.0, tags: str = "") -> str:
    """Лапки-ялинки читати не треба; темп і стиль — теги на початку речення (<|category:value|>)."""
    text = re.sub(r"[«»“”\"]", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    if speed >= 1.15:
        tags += "<|prosody:speed_fast|>"
    elif speed <= 0.85:
        tags += "<|prosody:speed_slow|>"
    return tags + text


def _torch():
    import torch
    return torch


def iter_safetensors(path: Path):
    """(ім'я, функція читання тензора) без mmap: safe_open на файлі 9 ГБ під Windows падає з access violation."""
    torch = _torch()
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64}
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
        base = 8 + size
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            start, end = meta["data_offsets"]

            def read(meta=meta, start=start, end=end):
                f.seek(base + start)
                buf = bytearray(f.read(end - start))
                return torch.frombuffer(buf, dtype=dtypes[meta["dtype"]]).reshape(meta["shape"])
            yield key, read


def _sample(logits, temperature, top_k, top_p):
    torch = _torch()
    if temperature <= 1e-5:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)
    if top_p and top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True, dim=-1)
        remove = srt.softmax(-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter_(-1, idx, remove)
        logits = torch.where(mask, float("-inf"), logits)
    return logits.softmax(-1).multinomial(1).squeeze(-1)


def apply_delay(codes):
    """[T, N] → [T+N-1, N]: кодбук c зсунутий на c кадрів, до і після — BOC/EOC."""
    torch = _torch()
    t, n = codes.shape
    out = torch.full((t + n - 1, n), EOC_ID, dtype=codes.dtype)
    for c in range(n):
        out[:c, c] = BOC_ID
        out[c:c + t, c] = codes[:, c]
    return out


def remove_delay(delayed):
    torch = _torch()
    rows, n = delayed.shape
    t = rows - (n - 1)
    if t <= 0:
        raise RuntimeError(f"Згенеровано замало кадрів ({rows}).")
    out = torch.empty((t, n), dtype=delayed.dtype)
    for c in range(n):
        out[:, c] = delayed[c:c + t, c]
    return out


class HiggsEngine:
    name = "higgs3"
    max_chars = MAX_CHARS
    sample_rate = SAMPLE_RATE

    def __init__(self, voice: str = NO_VOICE, speed: float = 1.0, device: str = "cpu", dtype: str = "bf16",
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95, seed: int = 0, tags: str = "",
                 max_ref_sec: float = 14.0, attn: str = "sdpa", **_):
        torch = _torch()
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.speed, self.tags = speed, tags
        self.temperature, self.top_k, self.top_p, self.seed = temperature, top_k, top_p, seed
        if not (MODEL_DIR / "model.safetensors").exists():
            raise SystemExit(f"Higgs: немає ваг у {MODEL_DIR} (hf download bosonai/higgs-tts-3-4b --local-dir …)")
        self.max_ref_sec, self.attn = max_ref_sec, attn
        self._voices = {}
        self._load_model()
        self.set_voice(voice)

    def _voice(self, voice: str | None):
        """(коди референсу з затримкою, текст референсу); кодування референсу кешується (воно триває ~10 с)."""
        if not voice or voice == NO_VOICE:
            return None, None
        if voice not in self._voices:
            wav = VOICES_DIR / f"{voice}.wav"
            if not wav.exists():
                raise SystemExit(f"Higgs: немає голосу '{voice}'. Доступні: {', '.join(available_voices())}")
            text = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
            self._voices[voice] = (apply_delay(self._encode_reference(wav, self.max_ref_sec)), text)
        return self._voices[voice]

    def set_voice(self, voice: str):
        """Перемкнути голос без перезавантаження моделі (ваги ~9 ГБ)."""
        self.ref_codes, self.ref_text = self._voice(voice)

    # ---------------------------------------------------------------- завантаження
    def _load_model(self):
        import torch.nn as nn
        from accelerate import init_empty_weights
        from accelerate.utils.modeling import set_module_tensor_to_device
        from tokenizers import Tokenizer
        from transformers import (HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel, PreTrainedTokenizerFast,
                                  Qwen3Config, Qwen3ForCausalLM)

        torch = self.torch
        cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        enc = cfg["audio_encoder_config"]
        self.n_cb, self.cb_vocab, hidden = enc["num_codebooks"], enc["vocab_size"], enc["out_dim"]
        text_cfg = Qwen3Config(**cfg["text_config"])
        text_cfg._attn_implementation = self.attn

        class Fused(nn.Module):  # один тензор [N*V, H] на всі кодбуки (вхід і голова пов'язані)
            def __init__(s):
                super().__init__()
                s.weight = nn.Parameter(torch.empty(self.n_cb * self.cb_vocab, hidden))

        class Model(nn.Module):
            def __init__(s):
                super().__init__()
                s.backbone = Qwen3ForCausalLM(text_cfg)
                s.modality_embedding = Fused()
                s.modality_head = Fused()

        with init_empty_weights():
            model = Model()
        model.to(dtype=self.dtype)
        names = set(dict(model.named_parameters(remove_duplicate=False)))
        codec_state, loaded = {}, set()
        for key, read in iter_safetensors(MODEL_DIR / "model.safetensors"):
            if key.startswith(CODEC_PREFIX):
                codec_state[key[len(CODEC_PREFIX):]] = read()
                continue
            target = next((t + key[len(s):] for s, t in WEIGHT_PREFIXES.items() if key.startswith(s)), key)
            if target not in names:
                continue
            tensor = read()
            if tensor.is_floating_point():
                tensor = tensor.to(self.dtype)
            # одразу на пристрій: інакше на iGPU (спільна пам'ять) модель на мить займає удвічі більше (CPU + XPU)
            set_module_tensor_to_device(model, target, device=self.device, value=tensor.contiguous())
            loaded.add(target)
        model.modality_head.weight = model.modality_embedding.weight
        try:
            model.backbone.tie_weights()
        except Exception:  # noqa: BLE001
            pass
        missing = [n for n in names if "lm_head.weight" not in n and "modality_head.weight" not in n and n not in loaded]
        if missing:
            raise RuntimeError(f"Higgs: не завантажено {len(missing)} тензорів, напр. {missing[:5]}")
        self.model = model.to(self.device).eval()

        codec_cfg = json.loads(CODEC_CONFIG.read_text(encoding="utf-8"))
        for k in ("architectures", "torch_dtype", "transformers_version"):
            codec_cfg.pop(k, None)
        codec = HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig(**codec_cfg)).eval()
        codec.load_state_dict(codec_state, strict=False)
        self.codec = codec.to(self.device)  # кодек лишаємо у fp32: у bf16 він дає артефакти
        del codec, codec_state  # хостові копії (≈0,8 ГБ) більше не потрібні: на iGPU пам'ять спільна
        import gc
        gc.collect()

        raw = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok = PreTrainedTokenizerFast(tokenizer_object=raw)
        vocab = self.tok.get_added_vocab()
        self.ids = {k: vocab[f"<|{k}|>"] for k in ("tts", "ref_audio", "text", "audio")}
        self.ids["ref_text"] = vocab.get("<|ref_text|>")

    def _encode_reference(self, path: Path, max_sec: float):
        import soundfile as sf
        import torchaudio
        torch = self.torch
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(1))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav[: int(max_sec * SAMPLE_RATE)]
        if wav.numel() < SAMPLE_RATE:
            wav = torch.nn.functional.pad(wav, (0, SAMPLE_RATE - wav.numel()))
        with torch.no_grad():
            codes = self.codec.encode(wav.view(1, 1, -1).to(self.device)).audio_codes  # [1, N, T]
        return codes.squeeze(0).transpose(0, 1).long().cpu()

    # ---------------------------------------------------------------- генерація
    def _prompt(self, text: str, ref=None) -> list[int]:
        ref_codes, ref_text = ref if ref is not None else (self.ref_codes, self.ref_text)
        ids = [self.ids["tts"]]
        n_ref = 0 if ref_codes is None else ref_codes.shape[0]
        if n_ref:
            if self.ids["ref_text"] is not None:
                ids += [self.ids["ref_text"]] + self.tok.encode(ref_text, add_special_tokens=False)
            ids += [self.ids["ref_audio"]] + [AUDIO_PLACEHOLDER] * n_ref
        return ids + [self.ids["text"]] + self.tok.encode(text, add_special_tokens=False) + [self.ids["audio"]]

    def _embed(self, codes):
        m = self.model.modality_embedding.weight
        offsets = self.torch.arange(self.n_cb, device=codes.device) * self.cb_vocab
        return self.torch.nn.functional.embedding(codes + offsets, m).sum(-2)

    def _generate(self, text: str, max_new: int):
        torch = self.torch
        ids = torch.tensor(self._prompt(text), dtype=torch.long, device=self.device)
        mask = ids == AUDIO_PLACEHOLDER
        embeds = self.model.backbone.model.embed_tokens(ids.clamp_min(0).view(1, -1))
        if self.ref_codes is not None:
            # плейсхолдери референсу — суцільний блок; присвоєння по булевій масці на XPU дає DEVICE_LOST
            start, n_ref = self._prompt(text).index(AUDIO_PLACEHOLDER), self.ref_codes.shape[0]
            ref = self._embed(self.ref_codes.to(self.device)).to(embeds.dtype).unsqueeze(0)
            embeds = torch.cat([embeds[:, :start], ref, embeds[:, start + n_ref:]], dim=1)
        out = self.model.backbone.model(inputs_embeds=embeds, use_cache=True)
        past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
        head = self.model.modality_head.weight
        n, rows, delay, countdown, last = self.n_cb, [], 0, None, None
        for step in range(max_new):
            if step % 16 == 15 and self.device.type != "cpu":
                self._free_cache()  # KV-кеш щокроку виділяє нові блоки; на iGPU вони накопичуються до OUT_OF_HOST_MEMORY
            logits = torch.nn.functional.linear(hidden, head).reshape(self.n_cb, self.cb_vocab).float()
            codes = _sample(logits, self.temperature, self.top_k, self.top_p).long()
            done = False
            if delay < n:  # старт: кодбук k вмикається на k-му кроці
                if delay + 1 < n:
                    codes[delay + 1:] = BOC_ID
                delay += 1
            elif countdown is not None:
                countdown -= 1
                done = countdown <= 0
            elif int(codes[0]) == EOC_ID:
                countdown = n - 2
                done = countdown <= 0
            if done:
                break
            rows.append(codes.cpu())
            last = codes
            out = self.model.backbone.model(inputs_embeds=self._embed(last.view(1, -1)).view(1, 1, -1).to(embeds.dtype),
                                            past_key_values=past, use_cache=True)
            past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
        else:
            print(f"  ! досягнуто max_new_tokens={max_new}: кінець фрази міг обрізатися")
        if len(rows) < n:
            raise RuntimeError(f"Higgs згенерував замало кадрів ({len(rows)}).")
        return torch.stack(rows)

    def _free_cache(self):
        """XPU/CUDA-аллокатор тримає блоки після кожного пакета; без звільнення на iGPU трапляється OUT_OF_DEVICE_MEMORY."""
        mod = getattr(self.torch, self.device.type, None)
        if hasattr(mod, "empty_cache"):
            mod.empty_cache()

    def _pcm(self, delayed) -> bytes:
        raw = remove_delay(delayed)
        audio = self.codec.decode(raw.transpose(0, 1).unsqueeze(0).to(self.device)).audio_values
        audio = audio.reshape(-1).float().cpu().numpy()
        if not np.isfinite(audio).all():
            raise RuntimeError("Higgs: у звукові дані потрапили NaN/inf (спробуйте --dtype fp32)")
        return (np.clip(audio, -1.0, 1.0) * 32767 * 0.9).astype(np.int16).tobytes()

    def synth(self, text: str) -> bytes:
        torch = self.torch
        if self.seed:
            torch.manual_seed(self.seed)
        text = prepare_text(text, self.speed, self.tags)
        plain = re.sub(r"<\|[^|]*\|>", "", text)
        max_new = int(len(plain) * 3.2) + 60
        with torch.inference_mode():
            return self._pcm(self._generate(text, max_new))

    # ---------------------------------------------------------------- пакетна генерація
    # Крок бекбона упирається в читання ваг із пам'яті, тож 8 речень за один прохід коштують ~1,5× одного (на Arc 130T:
    # 162 мс/крок для 1, 249 мс для 8). Результат статистично такий самий, але не збігається біт у біт із synth().
    def _prompt_embeds(self, text: str, ref):
        torch = self.torch
        prompt = self._prompt(text, ref)
        ids = torch.tensor(prompt, dtype=torch.long, device=self.device)
        embeds = self.model.backbone.model.embed_tokens(ids.clamp_min(0).view(1, -1))
        if ref[0] is not None:
            start, n_ref = prompt.index(AUDIO_PLACEHOLDER), ref[0].shape[0]
            block = self._embed(ref[0].to(self.device)).to(embeds.dtype).unsqueeze(0)
            embeds = torch.cat([embeds[:, :start], block, embeds[:, start + n_ref:]], dim=1)
        return embeds[0]

    def _generate_batch(self, items: list[tuple[str, tuple, int]]):
        """items: (текст з тегами, голос, ліміт кроків) → список [кадри, N] (з затримкою) для кожного."""
        torch = self.torch
        b, n = len(items), self.n_cb
        embs = [self._prompt_embeds(t, ref) for t, ref, _ in items]
        hidden_size, longest = embs[0].shape[-1], max(e.shape[0] for e in embs)
        # лівий падінг: усі промпти закінчуються в одній позиції, початок заповнюють нулі (замасковані)
        embeds = torch.stack([torch.cat([torch.zeros(longest - e.shape[0], hidden_size, dtype=e.dtype, device=e.device), e])
                              for e in embs])
        # Маски будуємо самі на CPU й передаємо готовою 4D (адитивною): вбудована побудова маски в transformers
        # використовує булеві операції, на XPU вони падають (OUT_OF_RESOURCES / DEVICE_LOST).
        lens = torch.tensor([e.shape[0] for e in embs])
        floor = torch.finfo(embeds.dtype).min

        def additive(allowed):  # bool [B,1,Q,K] на CPU → 0 / мінімум у dtype моделі на пристрої
            return torch.zeros(allowed.shape).masked_fill(~allowed, floor).to(self.device, embeds.dtype)

        valid = torch.arange(longest).view(1, -1) >= (longest - lens).view(-1, 1)  # [B, K]: не падінг
        causal = torch.tril(torch.ones(longest, longest, dtype=torch.bool))
        pos = (valid.long().cumsum(-1) - 1).clamp_min(0).to(self.device)
        out = self.model.backbone.model(inputs_embeds=embeds, position_ids=pos, use_cache=True,
                                        attention_mask=additive(causal.view(1, 1, longest, longest) & valid.view(b, 1, 1, -1)))
        past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
        nxt = lens.view(-1, 1).to(self.device)  # позиція наступного токена в кожній послідовності
        head, limits = self.model.modality_head.weight, [lim for _, _, lim in items]
        rows, delay, countdown, done = [[] for _ in range(b)], [0] * b, [None] * b, [False] * b
        runaway = [False] * b  # дійшла до ліміту кроків без кінця фрази (модель «розмазала» речення)
        for step in range(max(limits)):
            logits = torch.nn.functional.linear(hidden, head).reshape(b * n, self.cb_vocab).float()
            codes = _sample(logits, self.temperature, self.top_k, self.top_p).long().reshape(b, n).cpu()
            for i in range(b):
                if done[i]:
                    continue
                fin = False
                if delay[i] < n:  # старт: кодбук k вмикається на k-му кроці
                    if delay[i] + 1 < n:
                        codes[i, delay[i] + 1:] = BOC_ID
                    delay[i] += 1
                elif countdown[i] is not None:
                    countdown[i] -= 1
                    fin = countdown[i] <= 0
                elif int(codes[i, 0]) == EOC_ID:
                    countdown[i] = n - 2
                    fin = countdown[i] <= 0
                if fin:
                    done[i] = True
                    continue
                rows[i].append(codes[i].clone())
                if step + 1 >= limits[i]:
                    done[i], runaway[i] = True, True
            if all(done):
                break
            emb = self._embed(codes.to(self.device)).to(embeds.dtype).view(b, 1, -1)  # [B, N] → [B, 1, H]
            valid = torch.cat([valid, torch.ones(b, 1, dtype=torch.bool)], dim=1)
            out = self.model.backbone.model(inputs_embeds=emb, attention_mask=additive(valid.view(b, 1, 1, -1)),
                                            position_ids=nxt, past_key_values=past, use_cache=True)
            nxt = nxt + 1
            past, hidden = out.past_key_values, out.last_hidden_state[:, -1, :]
        for r in rows:
            if len(r) < n:
                raise RuntimeError(f"Higgs згенерував замало кадрів ({len(r)}).")
        return [torch.stack(r) for r in rows], runaway

    def _generate_safe(self, jobs):
        """_generate_batch, а якщо пам'яті пристрою не вистачило (на iGPU вона спільна із системою) — пакет навпіл."""
        try:
            return self._generate_batch(jobs)
        except RuntimeError as e:
            if len(jobs) == 1 or not any(m in str(e) for m in ("bad allocation", "OUT_OF_DEVICE_MEMORY", "out of memory")):
                raise
            print(f"  ! не вистачило пам'яті на пакет із {len(jobs)} реч., ділю навпіл")
            self._free_cache()
            half = len(jobs) // 2
            (a, ra), (b, rb) = self._generate_safe(jobs[:half]), self._generate_safe(jobs[half:])
            return a + b, ra + rb

    def synth_many(self, texts: list[str], voices: list[str] | None = None, batch: int = 8,
                   retries: int = 2) -> list[bytes]:
        """Озвучує багато текстів пакетами по `batch`; voices — голос кожного тексту (за замовчуванням поточний).
        Тексти з близькою довжиною групуються разом, щоб короткі не простоювали в очікуванні довгих.
        Речення, яке дійшло до ліміту кроків без кінця фрази (модель інколи «розмазує» його вдвічі довше), перегенеровується
        до `retries` разів із новим випадковим станом."""
        torch = self.torch
        if self.seed:
            torch.manual_seed(self.seed)
        refs = [self._voice(v) for v in voices] if voices else [(self.ref_codes, self.ref_text)] * len(texts)
        jobs = []
        for text, ref in zip(texts, refs):
            text = prepare_text(text, self.speed, self.tags)
            jobs.append((text, ref, int(len(re.sub(r"<\|[^|]*\|>", "", text)) * 3.2) + 60))
        order = sorted(range(len(jobs)), key=lambda i: jobs[i][2])
        codes_of, todo = {}, order
        with torch.inference_mode():
            for attempt in range(retries + 1):
                bad = []
                for s in range(0, len(todo), batch):
                    self._free_cache()
                    idx = todo[s:s + batch]
                    codes, runaway = self._generate_safe([jobs[i] for i in idx])
                    for i, c, r in zip(idx, codes, runaway):
                        codes_of[i] = c
                        if r:
                            bad.append(i)
                if not bad:
                    break
                if attempt < retries:
                    print(f"  ! {len(bad)} реч. без кінця фрази (ліміт кроків), перегенерація {attempt + 1}/{retries}")
                else:
                    print(f"  ! {len(bad)} реч. так і лишилися без кінця фрази: кінець міг обрізатися або звук розтягнутий")
                todo = bad
            result = [self._pcm(codes_of[i]) for i in range(len(jobs))]
        self._free_cache()
        return result
