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
"""Reusable PEFT/LoRA adapter lifecycle helpers for training engines."""

import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from peft import LoraConfig
from verl.utils.py_functional import convert_to_regular_types

logger = logging.getLogger(__name__)


def _normalize_checkpoint_keys(state_dict, expected_keys):
    if set(state_dict) == expected_keys:
        return state_dict
    prefix = "base_model.model."
    normalized = {
        key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state_dict.items()
    }
    if len(normalized) == len(state_dict) and set(normalized) == expected_keys:
        return normalized
    missing = sorted(expected_keys - set(normalized))
    unexpected = sorted(set(normalized) - expected_keys)
    raise RuntimeError(
        "LoRA checkpoint does not exactly match the injected adapter: "
        f"missing={missing[:5]} ({len(missing)} total), "
        f"unexpected={unexpected[:5]} ({len(unexpected)} total)."
    )


def _load_lora_checkpoint(module, adapter_path: str, adapter_name: str) -> None:
    """Inject and exactly load a local PEFT adapter into any compatible model."""
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"PEFT adapter requires {config_path} and {weights_path}.")

    config = LoraConfig.from_pretrained(adapter_path)
    module.add_adapter(config, adapter_name=adapter_name)
    current = get_peft_model_state_dict(module, adapter_name=adapter_name)
    checkpoint = _normalize_checkpoint_keys(load_file(weights_path), set(current))
    set_peft_model_state_dict(module, checkpoint, adapter_name=adapter_name)

    loaded = get_peft_model_state_dict(module, adapter_name=adapter_name)
    mismatched = [
        key
        for key in current
        if not torch.equal(loaded[key].detach().cpu(), checkpoint[key].to(dtype=loaded[key].dtype).cpu())
    ]
    if mismatched:
        raise RuntimeError(f"LoRA checkpoint verification failed for {len(mismatched)} tensors: {mismatched[:5]}.")


class LoRAAdapterMixin:
    """Backend-agnostic helpers for named PEFT/LoRA policy adapters."""

    def _build_lora_module(self, module):
        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        reference_adapter_name = getattr(self.model_config, "reference_adapter_name", None)
        policy_state_adapters = tuple(getattr(self.model_config, "policy_state_adapters", ("default",)))
        extra_adapters = [adapter for adapter in policy_state_adapters if adapter not in ("default", "reference")]
        if reference_adapter_name is not None:
            if reference_adapter_name in ("default", "reference"):
                raise ValueError("reference_adapter_name must be a distinct registered PEFT adapter name.")
            if lora_adapter_path is None:
                raise ValueError("reference_adapter_name requires lora_adapter_path to snapshot a pretrained policy.")
            if reference_adapter_name not in extra_adapters:
                extra_adapters.append(reference_adapter_name)

        if lora_adapter_path is not None:
            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter from: {lora_adapter_path}")
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            _load_lora_checkpoint(module, local_adapter_path, "default")
            peft_config = getattr(module, "peft_config", {}).get("default", None)
            for adapter_name in extra_adapters:
                if adapter_name in getattr(module, "peft_config", {}):
                    continue
                if adapter_name == reference_adapter_name:
                    _load_lora_checkpoint(module, local_adapter_path, adapter_name)
                elif peft_config is not None:
                    module.add_adapter(peft_config, adapter_name=adapter_name)
        else:
            lora_config = {
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "lora_dropout": getattr(self.model_config, "lora_dropout", 0.0),
                "init_lora_weights": self.model_config.lora_init_weights,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "target_parameters": convert_to_regular_types(self.model_config.target_parameters),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module.add_adapter(LoraConfig(**lora_config), adapter_name="default")
            for adapter_name in extra_adapters:
                module.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)

        if "default" in policy_state_adapters and hasattr(module, "set_adapter"):
            module.set_adapter("default")

        lora_dtype = getattr(self.model_config, "lora_dtype", None)
        if lora_dtype is not None:
            from peft.tuners.tuners_utils import BaseTunerLayer
            from verl.utils.torch_dtypes import PrecisionType

            target_dtype = PrecisionType.to_dtype(lora_dtype)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    orig_dtype = param.dtype
                    param.data = param.data.to(target_dtype)
                    logger.debug("LoRA param %s: %s -> %s", name, orig_dtype, param.dtype)

            for submodule in module.modules():
                if isinstance(submodule, BaseTunerLayer):
                    submodule.cast_input_dtype_enabled = False

        return module

    @contextmanager
    def _adapter_state_context(self):
        """Open writable adapter parameter access (FSDP summon when applicable)."""
        from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
        from verl.utils.memory_utils import aggressive_empty_cache

        from verl_omni.utils.fsdp_utils import fsdp_summon_full_params

        is_fsdp_module = fsdp_version(self.module) in (1, 2)
        is_offload_param = getattr(self, "_is_offload_param", False)
        origin_module_device = next(self.module.parameters()).device.type
        if is_fsdp_module and (is_offload_param or origin_module_device == "cpu"):
            load_fsdp_model_to_gpu(self.module)

        ctx = fsdp_summon_full_params(self.module, writeback=True) if is_fsdp_module else nullcontext()
        try:
            with ctx:
                try:
                    yield
                finally:
                    self._set_adapter("default")
        finally:
            if is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
                aggressive_empty_cache(force_sync=True)

    def _set_adapter(self, name: str):
        module = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not hasattr(module, "set_adapter"):
            raise AttributeError(f"Module does not support set_adapter({name!r})")
        module.set_adapter(name)

    @contextmanager
    def use_adapter(self, name: str):
        """Temporarily select a named PEFT adapter.

        ``"reference"`` is a logical policy state (see ``policy_state_adapters``)
        that runs with all LoRA adapters disabled, not a registered PEFT adapter.
        """
        if name == "reference":
            with self.disable_adapter():
                yield
        else:
            self._set_adapter(name)
            try:
                yield
            finally:
                self._set_adapter("default")

    @contextmanager
    def use_reference_adapter(self):
        """Use the configured frozen reference, or the base model when unset."""
        name = getattr(self.model_config, "reference_adapter_name", None)
        if name is None:
            with self.disable_adapter():
                yield
        else:
            with self.use_adapter(name):
                yield

    def _active_adapter_trainable_params(self, adapter_name: str) -> list[torch.nn.Parameter]:
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not hasattr(peft_model, "set_adapter"):
            raise AttributeError("Module does not support PEFT adapter selection.")
        peft_model.set_adapter(adapter_name)
        return list(filter(lambda param: param.requires_grad, peft_model.parameters()))

    def copy_adapter(self, source: str = "default", target: str = "old") -> None:
        """Copy LoRA state between named policy adapters."""
        with self._adapter_state_context(), torch.no_grad():
            source_params = self._active_adapter_trainable_params(source)
            target_params = self._active_adapter_trainable_params(target)
            if len(source_params) != len(target_params) or not source_params:
                raise ValueError(
                    f"Adapter copy {source!r} -> {target!r} found mismatched params: "
                    f"{len(source_params)} vs {len(target_params)}"
                )
            for source_param, target_param in zip(source_params, target_params, strict=True):
                target_param.copy_(source_param)

    def ema_update_adapter(self, source: str = "default", target: str = "old", decay: float = 0.0) -> None:
        """EMA-update target adapter parameters from source adapter parameters."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"Adapter EMA decay must be in [0, 1], got {decay}.")
        with self._adapter_state_context(), torch.no_grad():
            source_params = self._active_adapter_trainable_params(source)
            target_params = self._active_adapter_trainable_params(target)
            if len(source_params) != len(target_params) or not source_params:
                raise ValueError(
                    f"Adapter EMA {source!r} -> {target!r} found mismatched params: "
                    f"{len(source_params)} vs {len(target_params)}"
                )
            for source_param, target_param in zip(source_params, target_params, strict=True):
                target_param.lerp_(source_param, 1.0 - decay)

    @contextmanager
    def disable_adapter(self):
        """Temporarily disable all PEFT adapters."""
        try:
            self.module.disable_adapters()
            yield
        finally:
            self.module.enable_adapters()
