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
"""CPU tests for the SenseVoice tone-pinyin reward with a stub ASR."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "verl_omni/utils/reward_score/tts/sensevoice_pinyin_reward.py"
SPEC = importlib.util.spec_from_file_location("sensevoice_pinyin_reward_test", MODULE_PATH)
reward = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reward
SPEC.loader.exec_module(reward)


def _stub_text_pipeline(monkeypatch, transcript: str):
    monkeypatch.setattr(reward, "normalize_chinese", lambda text, cache_dir: str(text).strip().lower())
    monkeypatch.setattr(reward, "to_tone_pinyin", lambda text: text.split())
    monkeypatch.setattr(reward, "transcribe_sensevoice", lambda *args, **kwargs: transcript)


def test_matching_transcript_receives_higher_single_signal_reward(monkeypatch):
    waveform = np.ones(24_000, dtype=np.float32) * 0.1
    _stub_text_pipeline(monkeypatch, "ni3 hao3")
    matched = reward.compute_score((waveform, 24_000), "ni3 hao3")

    _stub_text_pipeline(monkeypatch, "ni3")
    mismatched = reward.compute_score((waveform, 24_000), "ni3 hao3")

    assert matched["score"] == pytest.approx(1.0)
    assert matched["pinyin_error_rate"] == 0.0
    assert mismatched["pinyin_error_rate"] == pytest.approx(0.5)
    assert 0.0 < mismatched["score"] < matched["score"]


def test_silence_is_zero_reward_without_calling_asr(monkeypatch):
    monkeypatch.setattr(reward, "normalize_chinese", lambda text, cache_dir: str(text))
    monkeypatch.setattr(reward, "to_tone_pinyin", lambda text: text.split())
    monkeypatch.setattr(
        reward,
        "transcribe_sensevoice",
        lambda *args, **kwargs: pytest.fail("ASR should not run for silence"),
    )

    result = reward.compute_score((np.zeros(24_000, dtype=np.float32), 24_000), "ni3 hao3")

    assert result["score"] == 0.0
    assert result["valid_audio"] == 0.0
    assert result["empty"] == 1.0


def test_health_metrics_do_not_shape_valid_matching_audio(monkeypatch):
    waveform = np.ones(24_000, dtype=np.float32) * 0.1
    _stub_text_pipeline(monkeypatch, "a b c a b c")

    result = reward.compute_score((waveform, 24_000), "a b c a b c")

    assert result["repeated"] == 1.0
    assert result["score"] == pytest.approx(1.0)


def test_token_error_rate_handles_insertions_and_empty_reference():
    assert reward.token_error_rate(["a", "b"], ["a", "x", "b"]) == pytest.approx(0.5)
    assert reward.token_error_rate([], ["a"]) is None
