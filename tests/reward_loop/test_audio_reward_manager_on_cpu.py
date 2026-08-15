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
"""CPU tests for waveform reward routing."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.audio import AudioRewardManager


def _config(audio=None):
    return OmegaConf.create({"reward": {"audio": audio or {}}})


def _manager(compute_score, audio=None):
    return AudioRewardManager(_config(audio), MagicMock(), compute_score=compute_score)


def _data(audio=None, sample_rate=24_000, *, split="train"):
    fields = {}
    if audio is not None:
        fields = {"tts_audio": audio, "tts_audio_sample_rate": sample_rate}
    return DataProto.from_dict(
        tensors={"responses": torch.zeros(1, 4, dtype=torch.long)},
        non_tensors={
            "data_source": ["tts_reward"],
            "reward_model": [{"ground_truth": "ni3 hao3"}],
            "extra_info": [{"id": "sample-0", "split": split}],
            "tool_extra_fields": [fields],
            "global_steps": [20],
        },
    )


def test_assemble_scores_preserves_one_reward_per_sample():
    scores = AudioRewardManager.assemble_rm_scores(MagicMock(), [0.1, -1.0, 2.5])
    assert scores.shape == (3, 1)
    assert scores.dtype == torch.float32


def test_run_single_passes_waveform_and_returns_diagnostics():
    def compute_score(data_source, solution_audio, ground_truth, extra_info):
        waveform, sample_rate = solution_audio
        assert data_source == "tts_reward"
        assert waveform.dtype == np.float32
        assert waveform.shape == (24_000,)
        assert sample_rate == 24_000
        assert ground_truth == "ni3 hao3"
        assert extra_info["id"] == "sample-0"
        return {"score": 0.75, "pinyin_error_rate": 0.1}

    manager = _manager(compute_score)
    result = manager.loop.run_until_complete(manager.run_single(_data(np.ones(24_000, dtype=np.float32))))

    assert result == {
        "reward_score": 0.75,
        "reward_extra_info": {"pinyin_error_rate": 0.1},
    }


def test_missing_audio_is_forwarded_as_none():
    seen = {}

    def compute_score(solution_audio, **kwargs):
        seen["audio"] = solution_audio
        return 0.0

    manager = _manager(compute_score)
    result = manager.loop.run_until_complete(manager.run_single(_data()))

    assert seen["audio"] is None
    assert result["reward_score"] == 0.0
    assert result["reward_extra_info"] == {"acc": 0.0}


def test_validation_audio_dump_is_step_scoped(tmp_path):
    pytest.importorskip("soundfile")
    manager = _manager(
        lambda **kwargs: {"score": 1.0},
        audio={"dump_dir": str(tmp_path), "dump_validation_only": True},
    )
    waveform = np.ones(2_400, dtype=np.float32) * 0.1

    train = manager.loop.run_until_complete(manager.run_single(_data(waveform, split="train")))
    validation = manager.loop.run_until_complete(manager.run_single(_data(waveform, split="validation")))

    assert "audio_path" not in train["reward_extra_info"]
    path = validation["reward_extra_info"]["audio_path"]
    assert "step_20" in path
    assert list(tmp_path.glob("step_20/*.wav"))
