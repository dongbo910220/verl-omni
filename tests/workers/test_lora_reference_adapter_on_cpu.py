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
"""CPU contracts for a frozen SFT LoRA reference policy."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
from safetensors.torch import save_file

ROOT = Path(__file__).parents[2]

verl_module = types.ModuleType("verl")
verl_utils_module = types.ModuleType("verl.utils")
verl_utils_module.__path__ = []
verl_functional_module = types.ModuleType("verl.utils.py_functional")
verl_functional_module.convert_to_regular_types = lambda value: value
verl_fs_module = types.ModuleType("verl.utils.fs")
verl_fs_module.copy_to_local = lambda path, **_: path
sys.modules.setdefault("verl", verl_module)
sys.modules.setdefault("verl.utils", verl_utils_module)
sys.modules.setdefault("verl.utils.py_functional", verl_functional_module)
sys.modules.setdefault("verl.utils.fs", verl_fs_module)


def _load():
    path = ROOT / "verl_omni/workers/engine/lora_adapter_mixin.py"
    spec = importlib.util.spec_from_file_location("lora_reference_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lora_module = _load()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 2, bias=False)

    def add_adapter(self, adapter_config, adapter_name="default"):
        if not hasattr(self, "peft_config"):
            self.peft_config = {}
        self.peft_config[adapter_name] = adapter_config
        inject_adapter_in_model(adapter_config, self, adapter_name)

    def set_adapter(self, adapter_name):
        for module in self.modules():
            if module is not self and callable(getattr(module, "set_adapter", None)):
                module.set_adapter(adapter_name)

    def disable_adapters(self):
        for module in self.modules():
            if module is not self and callable(getattr(module, "enable_adapters", None)):
                module.enable_adapters(False)

    def enable_adapters(self):
        for module in self.modules():
            if module is not self and callable(getattr(module, "enable_adapters", None)):
                module.enable_adapters(True)

    def forward(self, values):
        return self.q_proj(values)


class Engine(lora_module.LoRAAdapterMixin):
    def __init__(self, model_config):
        self.model_config = model_config


def _write_adapter(path):
    model = TinyModel()
    config = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.1, target_modules=["q_proj"], bias="none")
    model.add_adapter(config)
    state = get_peft_model_state_dict(model)
    state["q_proj.lora_A.weight"].fill_(0.25)
    state["q_proj.lora_B.weight"].fill_(0.5)
    path.mkdir()
    save_file(state, path / "adapter_model.safetensors")
    config_dict = config.to_dict()
    config_dict["target_modules"] = sorted(config.target_modules)
    config_dict["peft_type"] = "LORA"
    (path / "adapter_config.json").write_text(json.dumps(config_dict, default=str))
    return state


def test_pretrained_policy_and_reference_start_identical_and_reference_stays_frozen(tmp_path):
    checkpoint = _write_adapter(tmp_path / "adapter")
    config = SimpleNamespace(
        lora_adapter_path=str(tmp_path / "adapter"),
        reference_adapter_name="sft_reference",
        policy_state_adapters=("default",),
        use_shm=False,
        lora_dtype=None,
        lora_dropout=0.0,
    )
    engine = Engine(config)
    model = engine._build_lora_module(TinyModel())
    engine.module = model
    model.eval()

    policy = get_peft_model_state_dict(model, adapter_name="default")
    reference = get_peft_model_state_dict(model, adapter_name="sft_reference")
    for key in checkpoint:
        torch.testing.assert_close(policy[key], checkpoint[key])
        torch.testing.assert_close(reference[key], checkpoint[key])
    assert all(".default." in name for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if ".sft_reference." in name)
    assert model.peft_config["default"].lora_dropout == 0.0
    assert model.peft_config["sft_reference"].lora_dropout == 0.0

    values = torch.ones((1, 3))
    with torch.no_grad():
        reference_before = None
        with engine.use_reference_adapter():
            reference_before = model(values).clone()
        for name, parameter in model.named_parameters():
            if ".default." in name:
                parameter.add_(1.0)
        policy_after = model(values).clone()
        with engine.use_reference_adapter():
            reference_after = model(values).clone()

    torch.testing.assert_close(reference_after, reference_before)
    assert not torch.equal(policy_after, reference_after)
    assert model.q_proj.active_adapter == ["default"]


def test_reference_context_preserves_legacy_base_behavior_when_unset():
    config = SimpleNamespace(reference_adapter_name=None)
    engine = Engine(config)
    model = TinyModel()
    model.add_adapter(LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], bias="none"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                parameter.fill_(0.5)
    engine.module = model
    values = torch.ones((1, 3))

    with torch.no_grad():
        enabled = model(values).clone()
        with engine.use_reference_adapter():
            disabled = model(values).clone()

    assert not torch.equal(enabled, disabled)
    torch.testing.assert_close(model(values), enabled)


def test_pretrained_lora_without_named_reference_keeps_legacy_loader(tmp_path):
    class LegacyModel:
        def __init__(self):
            self.loaded_paths = []
            self.active_adapter = None
            self.peft_config = {"default": object()}

        def load_lora_adapter(self, path):
            self.loaded_paths.append(path)

        def set_adapter(self, name):
            self.active_adapter = name

    config = SimpleNamespace(
        lora_adapter_path=str(tmp_path / "adapter"),
        reference_adapter_name=None,
        policy_state_adapters=("default",),
        use_shm=False,
        lora_dtype=None,
    )
    engine = Engine(config)
    model = engine._build_lora_module(LegacyModel())

    assert model.loaded_paths == [str(tmp_path / "adapter")]
    assert model.active_adapter == "default"


def test_checkpoint_key_mismatch_fails_closed(tmp_path):
    _write_adapter(tmp_path / "adapter")
    weights = tmp_path / "adapter/adapter_model.safetensors"
    save_file({"wrong.lora_A.weight": torch.zeros(2, 3)}, weights)
    engine = Engine(
        SimpleNamespace(
            lora_adapter_path=str(tmp_path / "adapter"),
            reference_adapter_name="sft_reference",
            policy_state_adapters=("default",),
            use_shm=False,
            lora_dtype=None,
        )
    )

    with pytest.raises(RuntimeError, match="does not exactly match"):
        engine._build_lora_module(TinyModel())
