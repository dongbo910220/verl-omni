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
from verl_omni.pipelines.qwen3_tts.rollout_utils import align_audio_codes, append_tensor_chunk
from verl_omni.pipelines.qwen3_tts.talker_forward import (
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
    request_output = getattr(output, "request_output", None)
    completions = getattr(request_output, "outputs", None) if request_output is not None else None
    return completions[0] if completions else None


def _materialize(value):
    if isinstance(value, Mapping):
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    return value


def talker2code2wav_token_only(source_outputs, prompt=None, _requires_multimodal_data=False):
    """Materialize shared-memory Mapping payloads before the upstream processor mutates them."""
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
                completion_copy.multimodal_output = _materialize(multimodal)
            source_copy.outputs.append(completion_copy)
        converted.append(source_copy)
    return upstream_processor(converted, prompt, _requires_multimodal_data)


@OmniRolloutPipelineBase.register(_PIPELINE_ID)
class Qwen3TTSRolloutAdapter(OmniRolloutPipelineBase):
    _codec_eos_token_id = None

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
    def get_output_modalities(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return ["latent", "audio"]

    @classmethod
    def weight_sync_stage_ids(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return [0]

    @classmethod
    def supports_cache_engine_sleep(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return False

    @classmethod
    def get_stage_engine_extras(cls, stage_id, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return {"max_model_len": 65536, "max_num_batched_tokens": 65536} if stage_id == 1 else {}

    @classmethod
    def prepare_engine_prompt(cls, prompt_ids, model_config, multi_modal_data, mm_processor_kwargs=None):
        text = model_config.tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("Qwen3-TTS received an empty text prompt.")
        speaker_path = model_config.override_config.get("tts_spk_embed_path")
        replay_layout = str(model_config.override_config.get("tts_replay_layout", "concatenated"))
        if replay_layout not in ("concatenated", "interleaved"):
            raise ValueError("tts_replay_layout must be 'concatenated' or 'interleaved'.")
        if replay_layout == "concatenated" and not speaker_path:
            raise ValueError("Concatenated Qwen3-TTS rollout requires tts_spk_embed_path.")
        language = require_auto_language(model_config.override_config.get("tts_language", "Auto"))
        hf_config = getattr(model_config, "hf_config", None)
        talker_config = getattr(hf_config, "talker_config", hf_config)
        if talker_config is not None and getattr(talker_config, "codec_eos_token_id", None) is not None:
            cls._codec_eos_token_id = int(talker_config.codec_eos_token_id)
        additional_information = {
            # VoiceDesign uses the same no-speaker codec prefix as mlx-audio's
            # Base generate(ref_audio=None) path. It does not require a VoiceDesign checkpoint.
            "task_type": ["Base" if speaker_path else "VoiceDesign"],
            "text": [text],
            "language": [language],
            "non_streaming_mode": [replay_layout != "interleaved"],
        }
        if speaker_path:
            additional_information.update(
                {
                    "x_vector_only_mode": [True],
                    "voice_clone_prompt": [{"ref_spk_embedding": _load_speaker_vector(speaker_path)}],
                }
            )
        assistant_ids = model_config.tokenizer(build_assistant_text(text), padding=False)["input_ids"]
        assistant_ids = torch.as_tensor(assistant_ids).reshape(-1).tolist()
        if replay_layout == "interleaved":
            # Streaming prefill contains role(3), codec prefix(4 without a
            # speaker, 5 with one), and the first text token(1).
            prompt_length = 9 if speaker_path else 8
        else:
            # Legacy non-streaming x-vector layout includes the full text and
            # therefore tracks the assistant-template token count.
            prompt_length = len(assistant_ids) + 2
        identity = "\0".join((text, str(language), str(speaker_path), replay_layout)).encode()
        return {
            "prompt_token_ids": [1] * prompt_length,
            "additional_information": additional_information,
            "cache_salt": hashlib.sha256(identity).hexdigest(),
        }

    @classmethod
    def combine_engine_outputs(cls, outputs, prompt):
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
            "tts_generation_length": len(token_ids),
        }
        if cls._codec_eos_token_id is not None:
            fields["tts_codec_eos_token_id"] = cls._codec_eos_token_id
            fields["tts_has_eos"] = bool(token_ids and token_ids[-1] == cls._codec_eos_token_id)
        if waveform is not None:
            fields["tts_audio"] = waveform.float().reshape(-1)
        if sample_rate is not None:
            if isinstance(sample_rate, list | tuple):
                sample_rate = sample_rate[-1]
            fields["tts_audio_sample_rate"] = int(sample_rate.item() if hasattr(sample_rate, "item") else sample_rate)
        return policy_output, fields
