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
"""Qwen3-TTS rollout model extensions."""

import torch
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)


def _align_prompt_embedding_dtype(model, dtype: torch.dtype) -> None:
    """Keep request embeddings compatible with the vLLM model input buffer."""
    model._embedding_dtype = dtype
    model._prompt_builder._embedding_dtype = dtype


class Qwen3TTSDtypeAlignedTalkerForConditionalGeneration(Qwen3TTSTalkerForConditionalGeneration):
    """Make Qwen3-TTS prompt embeddings follow the configured rollout dtype."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _align_prompt_embedding_dtype(self, vllm_config.model_config.dtype)
