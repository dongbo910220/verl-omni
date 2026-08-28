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
"""Qwen3-TTS talker actor adapter."""

import logging
import types
from typing import Any

import torch

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.pipelines.qwen3_tts.talker_forward import (
    load_speaker_xvector,
    require_auto_language,
    tts_actor_logits,
)
from verl_omni.pipelines.qwen3_tts.transformers_compat import (
    patch_qwen3_tts_config_defaults,
    qwen3_tts_import_context,
)

logger = logging.getLogger(__name__)
_PASSTHROUGH_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
_TRAINABLE_PREFIXES = ("talker.model.", "talker.codec_head.")


def _prepare_config_for_checkpoint(config) -> None:
    speaker_config = getattr(config, "speaker_encoder_config", None)
    if speaker_config is not None:
        speaker_config.__dict__.pop("dtype", None)
        speaker_config.__dict__.pop("_dtype", None)


def _speaker_embedding(model, batch_size, device, dtype):
    cached = getattr(model, "_verl_tts_speaker_embedding", None)
    if cached is None:
        path = getattr(model.config, "tts_spk_embed_path", None)
        if not path:
            return None
        cached = load_speaker_xvector(path)
        model._verl_tts_speaker_embedding = cached
    return cached.to(device=device, dtype=dtype).expand(batch_size, -1)


def _reinitialize_rope_buffers(model):
    for submodule in model.modules():
        rope_init = getattr(submodule, "rope_init_fn", None)
        inv_freq = getattr(submodule, "inv_freq", None)
        if rope_init is None or not torch.is_tensor(inv_freq):
            continue
        new_inv_freq, scaling = rope_init(submodule.config, device=inv_freq.device)
        submodule.inv_freq.data.copy_(new_inv_freq.to(device=inv_freq.device, dtype=inv_freq.dtype))
        submodule.attention_scaling = scaling


def _qwen3_tts_forward(
    self,
    input_ids=None,
    attention_mask=None,
    tts_text_ids=None,
    tts_audio_codes=None,
    response_len=None,
    text_len=None,
    **kwargs,
):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if any(value is None for value in (tts_text_ids, tts_audio_codes, response_len, text_len)):
        raise RuntimeError("Qwen3-TTS forward is missing exact rollout codec fields.")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if not getattr(self, "_verl_tts_rope_initialized", False):
        _reinitialize_rope_buffers(self)
        self._verl_tts_rope_initialized = True
    speaker = _speaker_embedding(self, input_ids.shape[0], input_ids.device, next(self.talker.parameters()).dtype)
    return CausalLMOutputWithPast(
        logits=tts_actor_logits(
            self,
            input_ids,
            attention_mask,
            tts_text_ids,
            tts_audio_codes,
            response_len,
            text_len,
            speaker,
        )
    )


def _get_input_embeddings(self):
    return self.talker.model.codec_embedding


def _set_input_embeddings(self, value):
    self.talker.model.codec_embedding = value


@OmniModelBase.register("Qwen3TTSForConditionalGeneration", stage="talker")
class Qwen3TTSTalkerAdapter(OmniModelBase):
    @classmethod
    def load_hf_config(cls, model_path, *, trust_remote_code, attn_implementation):
        with qwen3_tts_import_context():
            from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

        patch_qwen3_tts_config_defaults(Qwen3TTSConfig)
        return Qwen3TTSConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

    @classmethod
    def get_model_class(cls):
        with qwen3_tts_import_context():
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

        return Qwen3TTSForConditionalGeneration

    @classmethod
    def get_strip_modules(cls, model_config):
        return ["speaker_encoder", "speech_tokenizer", "code2wav"]

    @classmethod
    def configure_model(cls, module, model_config):
        module = super().configure_model(module, model_config)
        _prepare_config_for_checkpoint(module.config)
        module.config.tts_spk_embed_path = model_config.override_config.get("tts_spk_embed_path")
        module.config.tts_language = require_auto_language(model_config.override_config.get("tts_language", "Auto"))
        if not module.config.tts_spk_embed_path:
            raise ValueError("Qwen3-TTS GRPO requires tts_spk_embed_path for the validated non-streaming replay.")
        module.forward = types.MethodType(_qwen3_tts_forward, module)
        module.get_input_embeddings = types.MethodType(_get_input_embeddings, module)
        module.set_input_embeddings = types.MethodType(_set_input_embeddings, module)
        module._no_split_modules = ["Qwen3TTSTalkerDecoderLayer", "Qwen3TTSDecoderLayer"]
        trainable = 0
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name.startswith(_TRAINABLE_PREFIXES))
            trainable += int(parameter.requires_grad)
        logger.info("Qwen3-TTS talker adapter enabled %d trainable parameter tensors", trainable)
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        return None

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        tokenizer.chat_template = _PASSTHROUGH_TEMPLATE
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        talker_config = getattr(model_config.hf_config, "talker_config", None)
        if talker_config is not None:
            talker_config.tie_word_embeddings = False
        return tokenizer

    @classmethod
    def prepare_model_inputs(cls, model_inputs, micro_batch, model_config):
        del model_config
        fields = micro_batch.get("extra_fields")
        if fields is None:
            raise RuntimeError(
                "Qwen3-TTS actor inputs require AgentLoopOutput.extra_fields; use the V1 agent-loop trainer path."
            )
        if hasattr(fields, "tolist"):
            fields = fields.tolist()
        if isinstance(fields, dict):
            fields = [fields]
        fields = [getattr(item, "data", item) for item in fields]
        if len(fields) != model_inputs["input_ids"].shape[0]:
            raise RuntimeError("Qwen3-TTS actor extra_fields do not match its batch size.")

        texts = [torch.as_tensor(item["tts_text_ids"], dtype=torch.long).reshape(-1) for item in fields]
        codes = [torch.as_tensor(item["tts_audio_codes"], dtype=torch.long) for item in fields]
        if any(item.ndim != 2 or item.shape[-1] != 16 for item in codes):
            raise ValueError("Qwen3-TTS codec codes must have shape (frames, 16).")
        device = model_inputs["input_ids"].device
        text_buffer = torch.zeros((len(fields), max(item.numel() for item in texts)), dtype=torch.long, device=device)
        code_buffer = torch.zeros(
            (len(fields), max(item.shape[0] for item in codes), 16), dtype=torch.long, device=device
        )
        text_lens = torch.empty(len(fields), dtype=torch.long, device=device)
        response_lens = torch.empty_like(text_lens)
        for index, (text, code) in enumerate(zip(texts, codes, strict=True)):
            text_buffer[index, : text.numel()] = text.to(device)
            code_buffer[index, : code.shape[0]] = code.to(device)
            text_lens[index], response_lens[index] = text.numel(), code.shape[0]
        model_inputs.update(
            {
                "tts_text_ids": text_buffer,
                "tts_audio_codes": code_buffer,
                "text_len": text_lens,
                "response_len": response_lens,
            }
        )
        return model_inputs
