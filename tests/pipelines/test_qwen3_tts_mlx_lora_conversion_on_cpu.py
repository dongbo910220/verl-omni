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
"""CPU contracts for the published MLX-to-PEFT Qwen3-TTS adapter conversion."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "examples/grpo_trainer/qwen3_tts/data_process/convert_mlx_lora_to_peft.py"


def _load():
    spec = importlib.util.spec_from_file_location("qwen3_tts_mlx_lora_conversion_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = _load()


def _mlx_adapter():
    tensors = {}
    for module in converter._expected_modules():
        tensors[f"{module}.lora_a"] = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        tensors[f"{module}.lora_b"] = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    return tensors


def test_conversion_is_complete_and_peft_key_compatible(tmp_path):
    source = tmp_path / "adapters.safetensors"
    output = tmp_path / "peft"
    save_file(_mlx_adapter(), source)

    manifest = converter.convert_adapter(source, output, source_revision="upstream-commit")
    converted = load_file(output / "adapter_model.safetensors")
    config = json.loads((output / "adapter_config.json").read_text())

    assert manifest["source_tensor_count"] == 462
    assert manifest["converted_tensor_count"] == 462
    assert manifest["module_count"] == 231
    assert manifest["component_module_counts"] == {"talker": 196, "code_predictor": 35}
    assert len(converted) == 462
    assert all(key.startswith("base_model.model.talker.") for key in converted)
    assert all(key.endswith((".lora_A.weight", ".lora_B.weight")) for key in converted)
    assert {
        key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        for key in converted
        if key.endswith(".lora_A.weight")
    } == converter._expected_modules()
    assert config["r"] == 8
    assert config["lora_alpha"] == 16
    assert config["lora_dropout"] == pytest.approx(0.05)
    assert set(config["target_modules"]) == set(converter.TARGET_MODULES)


def test_conversion_fails_closed_on_incomplete_topology(tmp_path):
    tensors = _mlx_adapter()
    tensors.pop(next(key for key in tensors if key.endswith(".lora_b")))
    source = tmp_path / "broken.safetensors"
    save_file(tensors, source)

    with pytest.raises(ValueError, match="topology mismatch"):
        converter.convert_adapter(source, tmp_path / "output", source_revision="upstream-commit")
