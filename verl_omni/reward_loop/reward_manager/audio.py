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
"""Reward manager for audio waveforms produced by omni rollouts."""

import inspect
import math

import numpy as np
import torch
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


class AudioRewardManager(RewardManagerBase):
    """Route one validated waveform per rollout to a custom reward function."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        if compute_score is None:
            raise ValueError("AudioRewardManager requires reward.custom_reward_function.")
        self.is_async_reward_score = inspect.iscoroutinefunction(compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    @staticmethod
    def _mapping(value):
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        return dict(value.items()) if hasattr(value, "items") else {}

    @classmethod
    def _extract_audio(cls, data_item, extra_info):
        batch = data_item.non_tensor_batch
        audio = extra_info.get("audio", batch.get("audio"))
        sample_rate = extra_info.get("audio_sample_rate", batch.get("audio_sample_rate"))
        if audio is None:
            raise KeyError("Audio reward requires extra_info['audio'] from the rollout.")
        if sample_rate is None:
            raise KeyError("Audio reward requires extra_info['audio_sample_rate'] from the rollout.")

        try:
            waveform = torch.as_tensor(audio).detach().float().cpu()
        except (TypeError, ValueError, RuntimeError) as exc:
            if not isinstance(audio, list | tuple):
                raise ValueError("Audio reward could not convert the waveform to numeric samples.") from exc
            try:
                chunks = [torch.as_tensor(chunk).detach().float().cpu().reshape(-1) for chunk in audio]
            except (TypeError, ValueError, RuntimeError) as chunk_exc:
                raise ValueError(
                    "Audio reward could not convert all waveform chunks to numeric samples."
                ) from chunk_exc
            waveform = torch.cat(chunks) if chunks else torch.empty(0)
        while waveform.ndim > 1 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        elif waveform.ndim != 1:
            raise ValueError(f"Expected audio shape (T,) or (C,T), got {tuple(waveform.shape)}.")
        if waveform.numel() == 0:
            raise ValueError("Audio reward received an empty waveform.")
        if not torch.isfinite(waveform).all():
            raise ValueError("Audio reward received a waveform containing NaN or infinity.")

        if isinstance(sample_rate, list | tuple):
            if len(sample_rate) != 1:
                raise ValueError("Audio reward requires exactly one sample rate per waveform.")
            sample_rate = sample_rate[0]
        if hasattr(sample_rate, "item"):
            try:
                sample_rate = sample_rate.item()
            except (RuntimeError, ValueError) as exc:
                raise ValueError("Audio reward requires one scalar sample rate per waveform.") from exc
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int | float):
            raise TypeError(f"Audio sample rate must be numeric, got {type(sample_rate).__name__}.")
        if not math.isfinite(float(sample_rate)) or float(sample_rate) <= 0 or float(sample_rate) != int(sample_rate):
            raise ValueError(f"Audio sample rate must be a positive integer, got {sample_rate!r}.")
        return waveform.numpy().astype(np.float32, copy=False), int(sample_rate)

    async def run_single(self, data: DataProto) -> dict:
        if len(data) != 1:
            raise ValueError(f"AudioRewardManager scores one sample at a time, got batch size {len(data)}.")
        item = data[0]
        batch = item.non_tensor_batch
        extra_info = self._mapping(batch.get("extra_info", {}))
        extra_info.update(self._mapping(batch.get("tool_extra_fields")))
        extra_info["num_turns"] = batch.get("__num_turns__", extra_info.get("num_turns"))
        extra_info["global_steps"] = batch.get("global_steps", extra_info.get("global_steps", 0))
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
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))
        if isinstance(result, dict):
            if "score" not in result:
                raise ValueError("Audio reward result dictionary is missing 'score'.")
            score = float(result["score"])
            reward_extra_info = {key: value for key, value in result.items() if key != "score"}
        else:
            score = float(result)
            reward_extra_info = {"acc": score}
        if not math.isfinite(score):
            raise ValueError(f"Audio reward must be finite, got {score!r}.")
        return {"reward_score": score, "reward_extra_info": reward_extra_info}
