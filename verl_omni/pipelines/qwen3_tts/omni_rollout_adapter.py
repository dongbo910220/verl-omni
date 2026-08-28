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
"""Qwen3-TTS two-stage rollout adapter."""

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache

import torch
from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_tts.rollout_utils import (
    align_audio_codes,
    append_tensor_chunk,
    is_evaluation_split,
    with_rollout_generation_seed,
)
from verl_omni.pipelines.qwen3_tts.talker_forward import (
    TEXT_PROMPT_TRAILER_TOKENS,
    build_assistant_text,
    load_speaker_xvector,
    require_auto_language,
)

_PIPELINE_ID = "qwen3_tts_rl"
_SYNC_PROCESSOR = "verl_omni.pipelines.qwen3_tts.omni_rollout_adapter.talker2code2wav_token_only"
QWEN3_TTS_RL_PIPELINE = PipelineConfig(
    model_type=_PIPELINE_ID,
    model_arch=QWEN3_TTS_PIPELINE.model_arch,
    stages=(
        replace(QWEN3_TTS_PIPELINE.stages[0], final_output=True, final_output_type="latent"),
        replace(QWEN3_TTS_PIPELINE.stages[1], sync_process_input_func=_SYNC_PROCESSOR),
    ),
)


@lru_cache(maxsize=4)
def _load_speaker_vector(path: str) -> list[float]:
    return load_speaker_xvector(path).reshape(-1).tolist()


def _completion(output):
    request_output = getattr(output, "request_output", None) or output
    completions = getattr(request_output, "outputs", None)
    return completions[0] if completions else None


def _copy_plain_containers(value):
    """Copy Ray/shared-memory mapping containers without copying tensor payloads."""
    if isinstance(value, Mapping):
        return {key: _copy_plain_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_plain_containers(item) for item in value]
    return value


def talker2code2wav_token_only(source_outputs, prompt=None, _requires_multimodal_data=False):
    """Give the mutating upstream processor ordinary Python containers."""
    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_token_only as upstream_processor,
    )

    converted = []
    for source_output in source_outputs:
        source_copy = copy.copy(source_output)
        source_copy.outputs = []
        for completion in getattr(source_output, "outputs", []):
            completion_copy = copy.copy(completion)
            multimodal = getattr(completion, "multimodal_output", None)
            if isinstance(multimodal, Mapping):
                completion_copy.multimodal_output = _copy_plain_containers(multimodal)
            source_copy.outputs.append(completion_copy)
        converted.append(source_copy)
    return upstream_processor(converted, prompt, _requires_multimodal_data)


@OmniRolloutPipelineBase.register(_PIPELINE_ID)
class Qwen3TTSRolloutAdapter(OmniRolloutPipelineBase):
    @classmethod
    def _check_mode(cls, pipeline_mode):
        if pipeline_mode != "full":
            raise ValueError("Qwen3-TTS RL supports only pipeline_mode='full'.")

    @classmethod
    def build_stage_configs(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return list(QWEN3_TTS_RL_PIPELINE.stages)

    @classmethod
    def get_pipeline_id(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return _PIPELINE_ID

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        register_pipeline(QWEN3_TTS_RL_PIPELINE)

    @classmethod
    def weight_sync_stage_ids(cls, pipeline_mode="full"):
        """Sync actor weights only to stage 0; stage 1 is the frozen decoder."""
        cls._check_mode(pipeline_mode)
        return [0]

    @classmethod
    def get_stage_engine_extras(cls, stage_id, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return {"max_model_len": 65536, "max_num_batched_tokens": 65536} if stage_id == 1 else {}

    @classmethod
    def prepare_agent_sampling_params(
        cls,
        sampling_params,
        *,
        rollout_config,
        trainer_config,
        agent_inputs,
    ):
        """Seed codec-0 and residual-codebook sampling for each GRPO candidate."""
        extra_info = agent_inputs.get("extra_info")
        evaluation = is_evaluation_split(extra_info)
        candidate_count = rollout_config.val_kwargs.n if evaluation else rollout_config.n
        return with_rollout_generation_seed(
            sampling_params,
            extra_info,
            session_id=agent_inputs.get("session_id"),
            global_steps=agent_inputs.get("global_steps"),
            uid=agent_inputs.get("uid"),
            base_seed=int(trainer_config.data.get("seed", 0)),
            require_session_id=int(candidate_count) > 1,
        )

    @classmethod
    def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
        """Map the 16-codebook rollout to codec-0 policy tokens and replay fields."""
        extra = output.extra_fields
        codes, text = extra.get("tts_audio_codes"), extra.get("tts_text")
        if codes is None or text is None:
            raise RuntimeError("Qwen3-TTS rollout did not return codec codes and text.")
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[-1] != 16:
            raise ValueError("Qwen3-TTS codec codes must have shape (frames, 16).")
        codes = codes[:response_length]
        policy_ids = codes[:, 0].tolist()
        if not policy_ids:
            raise RuntimeError("Qwen3-TTS rollout returned an empty codec trajectory.")
        if output.response_logprobs is not None:
            if len(output.response_logprobs) < len(policy_ids):
                raise RuntimeError("Qwen3-TTS rollout logprobs are shorter than the policy trajectory.")
            output.response_logprobs = output.response_logprobs[: len(policy_ids)]
        text_ids = tokenizer(build_assistant_text(str(text)), return_tensors="pt", padding=False)["input_ids"]
        text_ids = torch.as_tensor(text_ids, dtype=torch.long)
        if text_ids.ndim == 1:
            text_ids = text_ids.unsqueeze(0)
        if text_ids.ndim != 2 or text_ids.shape[1] <= TEXT_PROMPT_TRAILER_TOKENS:
            raise ValueError("Qwen3-TTS assistant text tokenization returned an invalid sequence.")
        extra["tts_text_ids"] = text_ids[:, :-TEXT_PROMPT_TRAILER_TOKENS].reshape(-1).tolist()
        extra["tts_audio_codes"] = codes
        output.prompt_ids = [0]
        output.response_ids = policy_ids
        output.response_mask = [1] * len(policy_ids)
        return output

    @classmethod
    def prepare_engine_prompt(cls, prompt_ids, model_config, multi_modal_data, mm_processor_kwargs=None):
        """Build the Base-task prompt and fixed-speaker conditioning for vLLM-Omni."""
        text = model_config.tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("Qwen3-TTS received an empty text prompt.")
        speaker_path = model_config.override_config.get("tts_spk_embed_path")
        if not speaker_path:
            raise ValueError("Qwen3-TTS GRPO requires tts_spk_embed_path for the validated non-streaming replay.")
        language = require_auto_language(model_config.override_config.get("tts_language", "Auto"))
        additional_information = {
            "task_type": ["Base"],
            "text": [text],
            "language": [language],
            "non_streaming_mode": [True],
            "x_vector_only_mode": [True],
            "voice_clone_prompt": [{"ref_spk_embedding": _load_speaker_vector(speaker_path)}],
        }
        assistant_ids = model_config.tokenizer(build_assistant_text(text), padding=False)["input_ids"]
        assistant_ids = torch.as_tensor(assistant_ids).reshape(-1).tolist()
        prompt_length = len(assistant_ids) + 2
        identity = "\0".join((text, str(language), str(speaker_path))).encode()
        return {
            "prompt_token_ids": [1] * prompt_length,
            "additional_information": additional_information,
            "cache_salt": hashlib.sha256(identity).hexdigest(),
        }

    @classmethod
    def combine_engine_outputs(cls, outputs, prompt):
        """Combine stage-0 policy tokens with codec and waveform outputs."""
        policy_output = None
        policy_length = -1
        audio_codes = waveform = None
        sample_rate = None
        diagnostics = []
        for output in outputs:
            completion = _completion(output)
            if getattr(output, "stage_id", None) == 0 and completion is not None:
                length = len(getattr(completion, "token_ids", None) or [])
                if length >= policy_length:
                    policy_output, policy_length = output, length
            multimodal = getattr(output, "multimodal_output", None)
            diagnostics.append((getattr(output, "stage_id", None), type(multimodal).__name__))
            if not isinstance(multimodal, Mapping):
                continue
            codes = multimodal.get("codes")
            if isinstance(codes, Mapping):
                audio_codes = append_tensor_chunk(audio_codes, codes.get("audio"))
            if getattr(output, "stage_id", None) == 1:
                waveform = append_tensor_chunk(
                    waveform, multimodal.get("audio", multimodal.get("model_outputs")), flatten=True
                )
                sample_rate = multimodal.get("sr", multimodal.get("audio_sample_rate", sample_rate))
        if policy_output is None:
            raise RuntimeError("Qwen3-TTS rollout produced no stage-0 policy output.")
        token_ids = list(getattr(_completion(policy_output), "token_ids", None) or [])
        if audio_codes is None:
            raise RuntimeError(f"Qwen3-TTS rollout produced no codec trajectory: {diagnostics}")
        fields = {
            "tts_audio_codes": align_audio_codes(audio_codes, token_ids),
            "tts_text": prompt["additional_information"]["text"][0],
        }
        if waveform is not None:
            fields["audio"] = waveform.float().reshape(-1)
        if sample_rate is not None:
            if isinstance(sample_rate, list | tuple):
                sample_rate = sample_rate[-1]
            fields["audio_sample_rate"] = int(sample_rate.item() if hasattr(sample_rate, "item") else sample_rate)
        return policy_output, fields
