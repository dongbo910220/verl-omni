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
"""Check the pinned upstream qwen-tts TF5 source on the repository stack."""

import importlib.util

import pytest
import torch


def test_qwen_tts_registers_and_runs_without_a_transformers_compatibility_layer():
    transformers = pytest.importorskip("transformers")
    if importlib.util.find_spec("qwen_tts") is None:
        pytest.skip("qwen-tts is an optional dependency")

    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from transformers import AutoConfig, AutoModelForTextToWaveform

    assert int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
    AutoModelForTextToWaveform.register(
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
        exist_ok=True,
    )
    assert AutoConfig.for_model("qwen3_tts").__class__ is Qwen3TTSConfig
    assert AutoModelForTextToWaveform._model_mapping[Qwen3TTSConfig] is Qwen3TTSForConditionalGeneration

    predictor = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "num_code_groups": 4,
        "layer_types": ["full_attention"],
        "pad_token_id": None,
    }
    talker = {
        "code_predictor_config": predictor,
        "vocab_size": 64,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "num_code_groups": 4,
        "text_hidden_size": 8,
        "text_vocab_size": 80,
        "spk_id": {},
        "codec_language_id": {},
        "rope_scaling": {
            "rope_type": "default",
            "type": "default",
            "mrope_section": [1, 1, 0],
            "interleaved": True,
        },
    }
    config = Qwen3TTSConfig(
        talker_config=talker,
        speaker_encoder_config={},
        tts_model_type="custom",
        tokenizer_type="12hz",
    )
    model = Qwen3TTSForConditionalGeneration(config)
    output = model.talker(
        inputs_embeds=torch.randn(2, 5, 8),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
        use_cache=False,
        output_hidden_states=False,
    )

    assert output.logits.shape == (2, 5, 64)
