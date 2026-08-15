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
"""SenseVoice tone-pinyin reward for Chinese TTS."""

import math
import threading
from pathlib import Path

import numpy as np

_MODEL_CACHE = {}
_NORMALIZER_CACHE = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_NORMALIZER_LOCK = threading.Lock()


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, 1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def token_error_rate(reference: list[str], hypothesis: list[str]) -> float | None:
    """Return Levenshtein error rate over arbitrary token sequences."""
    return None if not reference else _edit_distance(reference, hypothesis) / len(reference)


def _get_normalizer(cache_dir: str):
    path = str(Path(cache_dir).expanduser().resolve())
    with _CACHE_LOCK:
        if path not in _NORMALIZER_CACHE:
            try:
                from tn.chinese.normalizer import Normalizer as ZhNormalizer
            except ImportError as exc:
                raise ImportError("Install verl-omni with the 'tts' extra for Chinese normalization.") from exc
            _NORMALIZER_CACHE[path] = ZhNormalizer(
                cache_dir=path,
                remove_erhua=False,
                remove_interjections=False,
                remove_puncts=True,
                overwrite_cache=False,
            )
        return _NORMALIZER_CACHE[path]


def normalize_chinese(text: str, cache_dir: str = "./cache/zh_tn") -> str:
    normalizer = _get_normalizer(cache_dir)
    with _NORMALIZER_LOCK:
        return " ".join(normalizer.normalize(str(text)).lower().split())


def to_tone_pinyin(text: str) -> list[str]:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise ImportError("Install verl-omni with the 'tts' extra for pypinyin.") from exc
    return lazy_pinyin(text, style=Style.TONE3, tone_sandhi=True, neutral_tone_with_five=True)


def _resample(waveform: np.ndarray, sample_rate: int, target_rate: int = 16_000) -> np.ndarray:
    if sample_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    output_length = max(1, round(waveform.size * target_rate / sample_rate))
    source = np.linspace(0.0, 1.0, waveform.size, endpoint=False)
    target = np.linspace(0.0, 1.0, output_length, endpoint=False)
    return np.interp(target, source, waveform).astype(np.float32)


def _get_sensevoice(model_name: str, device: str):
    key = (model_name, device)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise ImportError("Install verl-omni with the 'tts' extra for FunASR.") from exc
            _MODEL_CACHE[key] = AutoModel(
                model=model_name,
                trust_remote_code=True,
                device=device,
                disable_update=True,
            )
        return _MODEL_CACHE[key]


def transcribe_sensevoice(
    waveform,
    sample_rate,
    model_name="FunAudioLLM/SenseVoiceSmall",
    device="cpu",
    language="zh",
) -> str:
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError as exc:
        raise ImportError("Install verl-omni with the 'tts' extra for FunASR.") from exc

    audio = _resample(np.asarray(waveform, dtype=np.float32).reshape(-1), int(sample_rate))
    model = _get_sensevoice(model_name, device)
    with _INFERENCE_LOCK:
        results = model.generate(
            input=audio,
            cache={},
            language=language,
            use_itn=False,
            batch_size=1,
            disable_pbar=True,
        )
    if not results or not isinstance(results[0], dict):
        raise RuntimeError("SenseVoice returned no transcription result.")
    return rich_transcription_postprocess(str(results[0].get("text") or "")).strip()


def _has_repetition(tokens: list[str], min_span: int = 3, max_span: int = 30) -> bool:
    for span in range(min_span, min(max_span, len(tokens) // 2) + 1):
        for start in range(len(tokens) - 2 * span + 1):
            if tokens[start : start + span] == tokens[start + span : start + 2 * span]:
                return True
    return False


def compute_score(
    solution_audio,
    ground_truth: str,
    extra_info=None,
    sensevoice_model="FunAudioLLM/SenseVoiceSmall",
    sensevoice_device="cpu",
    language="zh",
    normalizer_cache_dir="./cache/zh_tn",
    reward_alpha=3.0,
    max_asr_duration_s=30.0,
    silence_rms_threshold=1e-4,
    repetition_ngram=3,
    asr_max_retries=1,
    **kwargs,
) -> dict:
    """Score waveform intelligibility as ``1 - tanh(alpha * tone-PER)``.

    Audio-health fields are diagnostics only. They intentionally do not shape
    valid samples, keeping the phase-one reward attributable to one signal.
    """
    extra_info = extra_info or {}
    target = str(extra_info.get("reward_text") or extra_info.get("text") or ground_truth or "")
    normalized_target = normalize_chinese(target, normalizer_cache_dir)
    target_pinyin = to_tone_pinyin(normalized_target)
    if not target_pinyin:
        raise ValueError("SenseVoice reward requires a non-empty normalized target.")

    if solution_audio is None:
        waveform, sample_rate = np.empty(0, dtype=np.float32), 1
    else:
        waveform, sample_rate = solution_audio
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        sample_rate = int(sample_rate)

    finite = bool(waveform.size and sample_rate > 0 and np.isfinite(waveform).all())
    duration_s = waveform.size / sample_rate if finite else 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if finite else 0.0
    empty = not finite or duration_s < 0.08 or rms < float(silence_rms_threshold)
    runaway = duration_s > float(max_asr_duration_s)

    transcript = ""
    attempts = 0
    if not empty and not runaway:
        last_error = None
        for attempts in range(1, int(asr_max_retries) + 2):
            try:
                transcript = transcribe_sensevoice(
                    waveform,
                    sample_rate,
                    sensevoice_model,
                    sensevoice_device,
                    language,
                )
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"SenseVoice failed after {attempts} attempts.") from last_error

    normalized_hypothesis = normalize_chinese(transcript, normalizer_cache_dir) if transcript else ""
    hypothesis_pinyin = to_tone_pinyin(normalized_hypothesis) if normalized_hypothesis else []
    pinyin_error_rate = 1.0 if empty or runaway else float(token_error_rate(target_pinyin, hypothesis_pinyin))
    valid_audio = finite and not empty and not runaway
    score = 1.0 - math.tanh(float(reward_alpha) * max(0.0, pinyin_error_rate))
    score = float(np.clip(score, 0.0, 1.0)) if valid_audio else 0.0

    return {
        "score": score,
        "pinyin_error_rate": pinyin_error_rate,
        "valid_audio": float(valid_audio),
        "empty": float(empty),
        "runaway": float(runaway),
        "repeated": float(_has_repetition(hypothesis_pinyin, int(repetition_ngram))),
        "duration_s": duration_s,
        "rms": rms,
        "asr_attempts": attempts,
        "asr_text": transcript,
        "normalized_target": normalized_target,
        "normalized_hypothesis": normalized_hypothesis,
        "target_pinyin": " ".join(target_pinyin),
        "hypothesis_pinyin": " ".join(hypothesis_pinyin),
    }
