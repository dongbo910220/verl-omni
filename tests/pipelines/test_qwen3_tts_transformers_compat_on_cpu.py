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
"""Optional real-package check for qwen-tts on the repository TF5 stack."""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
from packaging.version import Version

ROOT = Path(__file__).parents[2]


def _load_compat_module():
    path = ROOT / "verl_omni/pipelines/qwen3_tts/transformers_compat.py"
    spec = importlib.util.spec_from_file_location("qwen3_tts_transformers_compat_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qwen_tts_tiny_model_constructs_and_forwards_without_source_patch():
    transformers = pytest.importorskip("transformers")
    if Version(transformers.__version__).major < 5:
        pytest.skip("Transformers 5.x compatibility test")
    if importlib.util.find_spec("qwen_tts") is None:
        pytest.skip("qwen-tts is an optional dependency")

    compat = _load_compat_module()
    with compat.qwen3_tts_import_context():
        config_module = importlib.import_module("qwen_tts.core.models.configuration_qwen3_tts")
        model_module = importlib.import_module("qwen_tts.core.models.modeling_qwen3_tts")
    compat.patch_qwen3_tts_config_defaults(config_module.Qwen3TTSConfig)

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
    config = config_module.Qwen3TTSConfig(
        talker_config=talker,
        speaker_encoder_config={},
        tts_model_type="custom",
        tokenizer_type="12hz",
    )
    assert config.talker_config.pad_token_id is None
    model = model_module.Qwen3TTSForConditionalGeneration(config)
    output = model.talker(
        inputs_embeds=torch.randn(2, 5, 8),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
        use_cache=False,
        output_hidden_states=False,
    )

    assert output.logits.shape == (2, 5, 64)
