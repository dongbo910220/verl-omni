# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whisper CER reward matching the published mlx-audio Hindi GRPO run."""

from __future__ import annotations

import re
import threading
import unicodedata
from math import gcd

import numpy as np

WHISPER_SAMPLE_RATE = 16_000
DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
_PUNCTUATION = re.compile(r"[।॥.,!?;:\"'`´’‘“”()\[\]{}<>/\\|@#%^&*_+=~–—-]")
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = _PUNCTUATION.sub(" ", normalized).lower()
    return " ".join(normalized.split())


def _edit_distance(reference, hypothesis) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def char_error_rate(hypothesis: str, reference: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(hypothesis: str, reference: str) -> float:
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _edit_distance(reference_words, hypothesis_words) / len(reference_words)


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        divisor = gcd(source_rate, target_rate)
        return resample_poly(waveform, target_rate // divisor, source_rate // divisor).astype(np.float32)
    except ImportError:
        output_length = max(1, int(waveform.size * target_rate / source_rate))
        return np.interp(
            np.linspace(0, waveform.size - 1, output_length),
            np.arange(waveform.size),
            waveform,
        ).astype(np.float32)


def _torch_dtype(name: str, device: str):
    import torch

    if str(device).startswith("cuda"):
        return getattr(torch, str(name))
    return torch.float32


def _get_whisper(model_name: str, model_revision: str, device: str, dtype: str):
    key = (model_name, model_revision, device, dtype)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            torch_dtype = _torch_dtype(dtype, device)
            processor = AutoProcessor.from_pretrained(model_name, revision=model_revision)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                revision=model_revision,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(device)
            model.eval()
            _MODEL_CACHE[key] = (model, processor, torch_dtype)
        return _MODEL_CACHE[key]


def transcribe_whisper(
    waveform: np.ndarray,
    *,
    model_name: str = DEFAULT_MODEL,
    model_revision: str = DEFAULT_MODEL_REVISION,
    device: str = "cuda:0",
    dtype: str = "float16",
    language: str = "hi",
) -> str:
    """Deterministic Whisper decoding equivalent to upstream temperature=0."""
    import torch

    model, processor, torch_dtype = _get_whisper(model_name, model_revision, device, dtype)
    features = processor(waveform, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt").input_features
    features = features.to(device=device, dtype=torch_dtype)
    with _INFERENCE_LOCK, torch.inference_mode():
        generated = model.generate(
            features,
            language=language,
            task="transcribe",
            do_sample=False,
            temperature=0.0,
            condition_on_prev_tokens=False,
        )
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _trailing_silence_fraction(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    threshold_db: float = -40.0,
) -> float:
    window = max(1, int(sample_rate * frame_ms / 1000.0))
    frame_count = waveform.size // window
    if not frame_count:
        return 0.0
    frames = waveform[: frame_count * window].reshape(frame_count, window)
    rms = np.sqrt(np.square(frames).mean(axis=1) + 1e-12)
    silent = rms < 10.0 ** (threshold_db / 20.0)
    trailing = 0
    for is_silent in silent[::-1]:
        if not is_silent:
            break
        trailing += 1
    return trailing / frame_count


def _infer_has_eos(extra_info: dict) -> bool | None:
    if "tts_has_eos" in extra_info:
        return bool(extra_info["tts_has_eos"])
    codes = extra_info.get("tts_audio_codes")
    eos_id = extra_info.get("tts_codec_eos_token_id")
    if codes is None or eos_id is None:
        return None
    codes = np.asarray(codes)
    return bool(codes.ndim == 2 and codes.shape[0] and int(codes[-1, 0]) == int(eos_id))


def compute_score(
    solution_audio,
    ground_truth: str,
    extra_info=None,
    whisper_model=DEFAULT_MODEL,
    whisper_revision=DEFAULT_MODEL_REVISION,
    whisper_device="cuda:0",
    whisper_dtype="float16",
    language="hi",
    max_asr_duration_s=30.0,
    length_weight=0.5,
    no_eos_penalty=1.0,
    silence_fraction_max=0.6,
    silence_penalty=0.5,
    silence_rms_db=-40.0,
    **kwargs,
) -> dict:
    """Return linear ``1-min(1,CER)`` plus the published degeneracy guard."""
    del kwargs
    extra_info = extra_info or {}
    target = str(extra_info.get("reward_text") or extra_info.get("text") or ground_truth or "")
    normalized_target = normalize_text(target)
    if not normalized_target:
        raise ValueError("Whisper CER reward requires a non-empty normalized target.")

    if solution_audio is None:
        waveform, sample_rate = np.empty(0, dtype=np.float32), 1
    else:
        waveform, sample_rate = solution_audio
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        sample_rate = int(sample_rate)

    finite = bool(waveform.size and sample_rate > 0 and np.isfinite(waveform).all())
    duration_s = waveform.size / sample_rate if finite else 0.0
    rms = float(np.sqrt(np.square(waveform).mean())) if finite else 0.0
    empty = not finite or duration_s < 0.1
    runaway = duration_s > float(max_asr_duration_s)
    transcript = ""
    if not empty and not runaway:
        transcript = transcribe_whisper(
            _resample(waveform, sample_rate),
            model_name=whisper_model,
            model_revision=whisper_revision,
            device=whisper_device,
            dtype=whisper_dtype,
            language=language,
        )

    cer = 1.0 if empty or runaway else char_error_rate(transcript, target)
    wer = 1.0 if empty or runaway else word_error_rate(transcript, target)
    intelligibility = 1.0 - min(1.0, max(0.0, cer))
    trailing_silence = _trailing_silence_fraction(waveform, sample_rate, threshold_db=silence_rms_db) if finite else 0.0
    has_eos = _infer_has_eos(extra_info)
    no_eos = has_eos is False
    length_reward = 0.0
    if no_eos:
        length_reward -= float(no_eos_penalty)
    if trailing_silence > float(silence_fraction_max):
        length_reward -= float(silence_penalty)
    score = intelligibility + float(length_weight) * length_reward
    sample_id = extra_info.get("id", extra_info.get("index", ""))
    generation_seed = extra_info.get("generation_seed", -1)
    if hasattr(sample_id, "item"):
        sample_id = sample_id.item()
    if hasattr(generation_seed, "item"):
        generation_seed = generation_seed.item()

    return {
        "score": float(score),
        "cer": float(cer),
        "cer_capped": float(min(1.0, max(0.0, cer))),
        "wer": float(wer),
        "intelligibility_reward": float(intelligibility),
        "length_reward": float(length_reward),
        "valid_audio": float(finite and not empty and not runaway),
        "empty": float(empty),
        "runaway": float(runaway),
        "has_eos": float(has_eos) if has_eos is not None else -1.0,
        "no_eos": float(no_eos),
        "duration_s": float(duration_s),
        "rms": rms,
        "trailing_silence": float(trailing_silence),
        "asr_text": transcript,
        "normalized_target": normalized_target,
        "normalized_hypothesis": normalize_text(transcript),
        "sample_id": str(sample_id),
        "generation_seed": int(generation_seed),
    }
