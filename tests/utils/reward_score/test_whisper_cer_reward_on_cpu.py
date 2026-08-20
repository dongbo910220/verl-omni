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
"""CPU tests for the published Hindi Whisper-CER reward contract."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "verl_omni/utils/reward_score/tts/whisper_cer_reward.py"
SPEC = importlib.util.spec_from_file_location("whisper_cer_reward_test", MODULE_PATH)
reward = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reward
SPEC.loader.exec_module(reward)


def test_normalization_and_error_rates_match_upstream_devanagari_contract():
    assert reward.normalize_text(" नमस्ते।  WORLD—हाँ! ") == "नमस्ते world हाँ"
    assert reward.char_error_rate("abc", "abc") == 0.0
    assert reward.char_error_rate("ab", "abc") == pytest.approx(1 / 3)
    assert reward.word_error_rate("एक दो", "एक तीन") == pytest.approx(0.5)


def test_linear_cer_reward_uses_deterministic_transcript(monkeypatch):
    waveform = np.full(24_000, 0.1, dtype=np.float32)
    monkeypatch.setattr(reward, "transcribe_whisper", lambda *args, **kwargs: "नमस्ते दुनिया")

    matched = reward.compute_score(
        (waveform, 24_000),
        "नमस्ते दुनिया",
        extra_info={"tts_has_eos": True, "id": "hi-17", "generation_seed": np.int64(42017)},
    )
    mismatched = reward.compute_score((waveform, 24_000), "नमस्ते", extra_info={"tts_has_eos": True})

    assert matched["score"] == pytest.approx(1.0)
    assert matched["cer"] == 0.0
    assert matched["sample_id"] == "hi-17"
    assert matched["generation_seed"] == 42017
    assert mismatched["cer"] > 0.0
    assert mismatched["score"] == pytest.approx(1.0 - min(1.0, mismatched["cer"]))


def test_no_eos_and_trailing_silence_match_published_guard(monkeypatch):
    waveform = np.zeros(24_000, dtype=np.float32)
    waveform[:4_800] = 0.1
    monkeypatch.setattr(reward, "transcribe_whisper", lambda *args, **kwargs: "सही")

    result = reward.compute_score((waveform, 24_000), "सही", extra_info={"tts_has_eos": False})

    assert result["intelligibility_reward"] == pytest.approx(1.0)
    assert result["trailing_silence"] == pytest.approx(0.8)
    assert result["length_reward"] == pytest.approx(-1.5)
    assert result["score"] == pytest.approx(0.25)


def test_short_invalid_audio_skips_asr_and_receives_zero(monkeypatch):
    monkeypatch.setattr(
        reward,
        "transcribe_whisper",
        lambda *args, **kwargs: pytest.fail("Whisper should not run for an invalid short clip."),
    )

    result = reward.compute_score(
        (np.ones(100, dtype=np.float32), 24_000),
        "सही",
        extra_info={"tts_has_eos": True},
    )

    assert result["score"] == 0.0
    assert result["empty"] == 1.0
    assert result["valid_audio"] == 0.0
