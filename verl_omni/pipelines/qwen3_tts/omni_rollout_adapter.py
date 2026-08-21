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
from verl_omni.utils.learning_trace import emit as trace_emit
from verl_omni.utils.learning_trace import enabled as learning_trace_enabled
from verl_omni.utils.learning_trace import expected_step, summarize_tensor
from verl_omni.utils.learning_trace import span as trace_span

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
    def get_worker_extension_cls(cls, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        return "verl_omni.pipelines.qwen3_tts.worker_extension.Qwen3TTSColocateWorkerExtension"

    @classmethod
    async def initialize_rollout_workers(cls, engine, pipeline_mode="full"):
        cls._check_mode(pipeline_mode)
        await engine.collective_rpc(
            method="align_qwen3_tts_prompt_embedding_dtype",
            stage_ids=[0],
        )

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
        result = {
            "prompt_token_ids": [1] * prompt_length,
            "additional_information": additional_information,
            "cache_salt": hashlib.sha256(identity).hexdigest(),
        }
        trace_emit(
            "rollout.engine_prompt",
            step=expected_step(),
            payload={
                "text": text,
                "language": language,
                "input_token_count": len(prompt_ids),
                "engine_prompt_token_count": prompt_length,
                "cache_salt": result["cache_salt"],
            },
        )
        return result

    @classmethod
    def combine_engine_outputs(cls, outputs, prompt):
        step = expected_step()
        with trace_span("rollout.stages", step=step, payload={"output_count": len(outputs)}):
            policy_output = None
            policy_length = -1
            audio_codes = waveform = None
            sample_rate = None
            diagnostics = []
            stage_rows = []
            request_ids = []
            for output in outputs:
                completion = _completion(output)
                request_output = getattr(output, "request_output", None) or output
                request_id = getattr(request_output, "request_id", None) or getattr(output, "request_id", None)
                if request_id is not None:
                    request_ids.append(str(request_id))
                stage_id = getattr(output, "stage_id", None)
                token_count = len(getattr(completion, "token_ids", None) or []) if completion is not None else 0
                if stage_id == 0 and completion is not None and token_count >= policy_length:
                    policy_output, policy_length = output, token_count
                multimodal = getattr(output, "multimodal_output", None)
                diagnostics.append((stage_id, type(multimodal).__name__))
                stage_rows.append(
                    {
                        "stage_id": stage_id,
                        "request_id": request_id,
                        "token_count": token_count,
                        "multimodal_type": type(multimodal).__name__,
                        "multimodal_keys": sorted(map(str, multimodal.keys()))
                        if isinstance(multimodal, Mapping)
                        else [],
                    }
                )
                if not isinstance(multimodal, Mapping):
                    continue
                codes = multimodal.get("codes")
                if isinstance(codes, Mapping):
                    audio_codes = append_tensor_chunk(audio_codes, codes.get("audio"))
                if stage_id == 1:
                    waveform = append_tensor_chunk(
                        waveform, multimodal.get("audio", multimodal.get("model_outputs")), flatten=True
                    )
                    sample_rate = multimodal.get("sr", multimodal.get("audio_sample_rate", sample_rate))
            if policy_output is None:
                raise RuntimeError("Qwen3-TTS rollout produced no stage-0 policy output.")
            token_ids = list(getattr(_completion(policy_output), "token_ids", None) or [])
            if audio_codes is None:
                raise RuntimeError(f"Qwen3-TTS rollout produced no codec trajectory: {diagnostics}")
            aligned_codes = align_audio_codes(audio_codes, token_ids)
            fields = {
                "tts_audio_codes": aligned_codes,
                "tts_text": prompt["additional_information"]["text"][0],
            }
            if waveform is not None:
                fields["audio"] = waveform.float().reshape(-1)
            if sample_rate is not None:
                if isinstance(sample_rate, list | tuple):
                    sample_rate = sample_rate[-1]
                fields["audio_sample_rate"] = int(sample_rate.item() if hasattr(sample_rate, "item") else sample_rate)
            if learning_trace_enabled():
                fields["_learning_trace_request_ids"] = sorted(set(request_ids))
            trace_emit(
                "rollout.stages_result",
                step=step,
                payload={
                    "request_ids": sorted(set(request_ids)),
                    "text": fields["tts_text"],
                    "stages": stage_rows,
                    "policy_token_count": len(token_ids),
                    "policy_token_ids": token_ids,
                    "audio_codes": summarize_tensor(aligned_codes, exact_limit=0),
                    "waveform": summarize_tensor(waveform, exact_limit=0) if waveform is not None else None,
                    "sample_rate": fields.get("audio_sample_rate"),
                },
                status="ok",
            )
        return policy_output, fields
