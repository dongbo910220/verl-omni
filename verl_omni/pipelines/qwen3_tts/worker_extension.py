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
"""Qwen3-TTS vLLM worker initialization."""

import torch

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension


def _align_prompt_embedding_dtype(model, dtype: torch.dtype) -> None:
    """Keep request embeddings compatible with the vLLM model input buffer."""
    model._embedding_dtype = dtype
    model._prompt_builder._embedding_dtype = dtype


class Qwen3TTSColocateWorkerExtension(vLLMOmniColocateWorkerExtension):
    """Apply Qwen3-TTS setup after the upstream model is loaded."""

    def align_qwen3_tts_prompt_embedding_dtype(self) -> None:
        standard = self._get_standard_weight_model_and_config()
        if standard is None:
            raise RuntimeError("Qwen3-TTS rollout worker has no loaded AR model")
        model, model_config = standard
        _align_prompt_embedding_dtype(model, model_config.dtype)
