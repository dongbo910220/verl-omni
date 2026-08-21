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
"""Reward manager for waveform outputs from omni rollouts."""

import hashlib
import inspect
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

_VALIDATION_SPLITS = {"gate", "official_test", "test", "val", "validation"}


class AudioRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        if compute_score is None:
            raise ValueError("AudioRewardManager requires reward.custom_reward_function.")
        self.is_async_reward_score = inspect.iscoroutinefunction(compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        audio_config = getattr(getattr(config, "reward", None), "audio", None) or {}
        self._executor = ThreadPoolExecutor(max_workers=int(audio_config.get("score_threads", 1)))
        self._dump_dir = audio_config.get("dump_dir")
        self._dump_validation_only = bool(audio_config.get("dump_validation_only", True))

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    @staticmethod
    def _mapping(value):
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        return dict(value.items()) if hasattr(value, "items") else {}

    @classmethod
    def _extract_audio(cls, data_item, extra_info):
        batch = data_item.non_tensor_batch
        audio = extra_info.get("tts_audio", batch.get("tts_audio"))
        sample_rate = extra_info.get("tts_audio_sample_rate", batch.get("tts_audio_sample_rate", 24000))
        if audio is None:
            return None
        if isinstance(audio, list | tuple):
            chunks = [item if torch.is_tensor(item) else torch.as_tensor(item) for item in audio]
            audio = torch.cat(chunks, dim=-1) if chunks else torch.empty(0)
        if torch.is_tensor(audio):
            audio = audio.detach().float().cpu().numpy()
        waveform = np.array(audio, dtype=np.float32, copy=True).reshape(-1)
        if not waveform.size:
            return None
        if isinstance(sample_rate, list | tuple):
            sample_rate = sample_rate[-1]
        if hasattr(sample_rate, "item"):
            sample_rate = sample_rate.item()
        return waveform, int(sample_rate)

    @staticmethod
    def _scalar(value):
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[-1] if value.size else None
        return value.item() if hasattr(value, "item") else value

    def _maybe_dump(self, audio, extra_info, ground_truth):
        if self._dump_dir is None or audio is None:
            return None
        if self._dump_validation_only and extra_info.get("split") not in _VALIDATION_SPLITS:
            return None
        import soundfile as sf

        waveform, sample_rate = audio
        digest = hashlib.sha1(waveform.tobytes()).hexdigest()[:10]
        sample_id = self._scalar(extra_info.get("id", extra_info.get("index", "sample")))
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"
        step = int(self._scalar(extra_info.get("global_steps", 0)) or 0)
        output_dir = Path(os.path.expanduser(self._dump_dir)) / f"step_{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{safe_id}_{digest}.wav"
        sf.write(output_path, waveform, sample_rate)
        metadata = {
            "audio_path": str(output_path),
            "audio_sha1": digest,
            "sample_id": str(sample_id),
            "ground_truth": str(ground_truth),
            "step": step,
            "sample_rate": sample_rate,
            "num_samples": int(waveform.size),
        }
        output_path.with_suffix(".json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
        return str(output_path)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only one audio sample can be scored at a time."
        item = data[0]
        batch = item.non_tensor_batch
        extra_info = self._mapping(batch.get("extra_info", {}))
        extra_info.update(self._mapping(batch.get("tool_extra_fields")))
        extra_info["num_turns"] = batch.get("__num_turns__")
        extra_info["global_steps"] = batch.get("global_steps", 0)
        ground_truth = batch["reward_model"]["ground_truth"]
        audio = self._extract_audio(item, extra_info)
        kwargs = {
            "data_source": batch["data_source"],
            "solution_audio": audio,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
        }
        if self.is_async_reward_score:
            result = await self.compute_score(**kwargs)
        else:
            result = await self.loop.run_in_executor(self._executor, lambda: self.compute_score(**kwargs))
        if isinstance(result, dict):
            score = float(result["score"])
            reward_extra_info = {key: value for key, value in result.items() if key != "score"}
        else:
            score = float(result)
            reward_extra_info = {"acc": score}
        audio_path = self._maybe_dump(audio, extra_info, ground_truth)
        if audio_path:
            reward_extra_info["audio_path"] = audio_path
        return {"reward_score": score, "reward_extra_info": reward_extra_info}
