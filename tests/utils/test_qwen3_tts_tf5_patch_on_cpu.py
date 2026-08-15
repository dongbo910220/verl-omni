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
"""CPU tests for the qwen-tts Transformers 5 compatibility patch."""

import importlib.util
from pathlib import Path


def _load_patch_module():
    root = Path(__file__).parents[2]
    path = root / "examples/grpo_trainer/qwen3_tts/patches/patch_qwen_tts_tf5.py"
    spec = importlib.util.spec_from_file_location("qwen3_tts_tf5_patch_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_is_complete_and_idempotent(tmp_path):
    module = _load_patch_module()
    model_path = tmp_path / "modeling_qwen3_tts.py"
    model_path.write_text(
        "logger = logging.get_logger(__name__)\n"
        "@check_model_inputs()\n"
        "class Model:\n"
        "    rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]\n"
        "    pad = config.pad_token_id\n"
        "    kwargs = {\n"
        '                "input_embeds": inputs_embeds,\n'
        '                "attention_mask": attention_mask,\n'
        '                "cache_position": cache_position,\n'
        "    }\n"
        "            else:\n"
        "                inputs_embeds = inputs_embeds + tts_pad_embed\n"
        "        if attention_mask is not None:\n"
        "            pass\n"
    )

    module.patch_file(model_path)
    first = model_path.read_text()
    module.patch_file(model_path)

    assert model_path.read_text() == first
    assert "@check_model_inputs\n" in first
    assert "_qtts_default_rope_init" in first
    assert 'getattr(config, "pad_token_id", None)' in first
    assert '"inputs_embeds": inputs_embeds' in first
    assert '"cache_position": cache_position' not in first
    assert "past_seen_tokens = past_key_values.get_seq_length()" in first
    assert "past_seen_tokens + inputs_embeds.shape[1]" in first
