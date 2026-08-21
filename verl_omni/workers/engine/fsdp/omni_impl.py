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
"""FSDP engine for omni models, registered as ``model_type="omni_model"``."""

import logging
import warnings

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoModelForMultimodalLM
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    get_init_weight_context_manager,
    load_fsdp_model_to_gpu,
    merged_lora_context,
    normalize_peft_param_name,
    offload_fsdp_model_to_cpu,
    replace_lora_wrapper,
)
from verl.utils.model import convert_weight_keys
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from verl_omni.utils.fsdp_utils import collect_lora_params
from verl_omni.workers.config import OmniModelConfig

logger = logging.getLogger(__name__)


@EngineRegistry.register(model_type="omni_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class OmniFSDPEngine(FSDPEngineWithLMHead):
    """FSDP engine for omni models"""

    @staticmethod
    def _resolve_weight_sync_dtype(weight_sync_dtype):
        if weight_sync_dtype is None or isinstance(weight_sync_dtype, torch.dtype):
            return weight_sync_dtype

        from verl.utils.torch_dtypes import PrecisionType

        return PrecisionType.to_dtype(weight_sync_dtype)

    @staticmethod
    def _cast_weight_for_sync(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
        if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
            return tensor.to(dtype=dtype, non_blocking=True)
        return tensor

    def _materialize_weight_for_sync(self, param, device, dtype: torch.dtype | None) -> torch.Tensor:
        if isinstance(param, DTensor):
            tensor = param.to(device, non_blocking=True).full_tensor()
            dtype = torch.bfloat16 if dtype is None else dtype
        else:
            tensor = param
        return self._cast_weight_for_sync(tensor, dtype)

    def _run_with_manual_ref_offload(self, call):
        adapter_cls = getattr(self, "model_adapter_cls", None)
        manual_ref_offload = (
            getattr(adapter_cls, "requires_manual_ref_offload", False)
            and getattr(self.engine_config, "forward_only", False)
            and not getattr(self.engine_config, "param_offload", True)
        )
        if not manual_ref_offload:
            return call()

        self.engine_config.forward_only = False
        try:
            return call()
        finally:
            self.engine_config.forward_only = True

    def _build_fsdp_module(self, module):
        parent_build = super()._build_fsdp_module
        return self._run_with_manual_ref_offload(lambda: parent_build(module))

    def to(self, device, model=True, optimizer=True, grad=True):
        parent_to = super().to
        return self._run_with_manual_ref_offload(lambda: parent_to(device, model=model, optimizer=optimizer, grad=grad))

    def prepare_model_inputs(self, micro_batch):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        adapter_cls = getattr(self, "model_adapter_cls", None)
        if adapter_cls is not None:
            model_inputs = adapter_cls.prepare_model_inputs(model_inputs, micro_batch, self.model_config)
        return model_inputs, output_args

    def get_per_tensor_param(
        self,
        layered_summon=False,
        base_sync_done=False,
        weight_sync_dtype=None,
        **kwargs,
    ):
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        sync_dtype = self._resolve_weight_sync_dtype(weight_sync_dtype)

        # FSDP2 CPUOffloadPolicy owns CPU<->GPU placement; calling model.to(device) here
        # leaves the module half-moved and crashes state_dict() below (#5995). The
        # per-DTensor .to(device).full_tensor() below still produces GPU tensors.
        if not self._uses_fsdp2_cpu_offload_policy:
            load_fsdp_model_to_gpu(self.module)

        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        merge_lora = self.model_config.lora.get("merge", False)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if hasattr(peft_model, "peft_config"):  # LoRA
            if not merge_lora:
                peft_config = peft_model.peft_config.get("default", None)
                # DIFF vs upstream: use verl_omni's fixed collect_lora_params
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                )
                if not base_sync_done:
                    params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
            else:  # merge lora
                return self._merged_lora_per_tensor_param(sync_dtype), None
        else:
            params = self.module.state_dict()

        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        per_tensor_param = (
            (name, self._materialize_weight_for_sync(param, device, sync_dtype)) for name, param in params.items()
        )

        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer
            from verl.utils.torch_dtypes import PrecisionType

            mixed_precision_config = self.engine_config.mixed_precision
            if mixed_precision_config is not None:
                param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            else:
                param_dtype = torch.bfloat16

            quantizer = QATQuantizer(
                mode=self._qat_config.mode,
                group_size=self._qat_config.group_size,
                ignore_patterns=list(self._qat_config.ignore_patterns),
                device=torch.device(get_device_id()),
                param_dtype=param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None

        return per_tensor_param, peft_config_dict

    def _merged_lora_per_tensor_param(self, weight_sync_dtype=None):
        """Stream materialized merged weights before restoring the actor."""
        device = get_device_id()
        sync_dtype = self._resolve_weight_sync_dtype(weight_sync_dtype)
        try:
            with merged_lora_context(self.module, backup_adapters=True):
                params = normalize_peft_param_name(self.module.state_dict())
                params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
                for name, param in params.items():
                    materialized = self._materialize_weight_for_sync(param, device, sync_dtype)
                    yield name, materialized.detach().clone()
        finally:
            log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

    def _build_module(self):
        from verl.utils.torch_dtypes import PrecisionType

        from verl_omni.pipelines.model_base import OmniModelBase

        self.model_config: OmniModelConfig
        architecture = self.model_config.architecture

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # Use the stage sub-config for the meta-tensor decision; fall back to the umbrella config.
        stage_config = getattr(
            self.model_config.hf_config, f"{self.model_config.model_stage}_config", self.model_config.hf_config
        )
        tie_word_embeddings = getattr(stage_config, "tie_word_embeddings", False)
        if not hasattr(self.model_config.hf_config, "tie_word_embeddings"):
            self.model_config.hf_config.tie_word_embeddings = tie_word_embeddings

        init_context = get_init_weight_context_manager(use_meta_tensor=not tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if getattr(self.model_config, "use_liger", False):
                logger.warning("use_liger is set but not applied for omni models; this is a no-op.")
            if getattr(self.model_config, "use_fused_kernels", False):
                logger.warning("use_fused_kernels is set but not applied for omni models; this is a no-op.")

            module = AutoModelForMultimodalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            adapter_cls = OmniModelBase.get_class_by_name(
                architecture,
                self.model_config.model_stage,
                self.model_config.get("external_lib"),
            )
            self.model_adapter_cls = adapter_cls
            module = adapter_cls.configure_model(module, self.model_config)

            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module = super()._build_lora_module(module)

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
