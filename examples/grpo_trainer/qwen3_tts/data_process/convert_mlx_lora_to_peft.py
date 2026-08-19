#!/usr/bin/env python3
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
"""Convert the published mlx-audio Qwen3-TTS LoRA checkpoint to PEFT."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXPECTED_TALKER_LAYERS = 28
EXPECTED_CODE_PREDICTOR_LAYERS = 5
EXPECTED_RANK = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_modules() -> set[str]:
    modules = set()
    for prefix, layer_count in (
        ("talker.model.layers", EXPECTED_TALKER_LAYERS),
        ("talker.code_predictor.model.layers", EXPECTED_CODE_PREDICTOR_LAYERS),
    ):
        for layer in range(layer_count):
            for target in TARGET_MODULES:
                block = "self_attn" if target in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
                modules.add(f"{prefix}.{layer}.{block}.{target}")
    return modules


def _load_and_validate(source: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if not source.is_file():
        raise FileNotFoundError(f"MLX adapter not found: {source}")

    with safe_open(source, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        tensors = {key: handle.get_tensor(key) for key in source_keys}

    expected_modules = _expected_modules()
    expected_keys = {f"{module}.lora_{side}" for module in expected_modules for side in ("a", "b")}
    actual_keys = set(tensors)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Published MLX adapter topology mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)."
        )

    converted = {}
    dtype_counts: Counter[str] = Counter()
    for module in sorted(expected_modules):
        a = tensors[f"{module}.lora_a"]
        b = tensors[f"{module}.lora_b"]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"{module}: LoRA tensors must be matrices, got A{tuple(a.shape)} B{tuple(b.shape)}.")
        if a.shape[0] != EXPECTED_RANK or b.shape[1] != EXPECTED_RANK or a.shape[0] != b.shape[1]:
            raise ValueError(f"{module}: expected rank {EXPECTED_RANK}, got A{tuple(a.shape)} B{tuple(b.shape)}.")
        if not a.is_floating_point() or not b.is_floating_point():
            raise TypeError(f"{module}: LoRA tensors must be floating point.")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"{module}: LoRA tensors contain NaN or infinity.")

        peft_module = f"base_model.model.{module}"
        converted[f"{peft_module}.lora_A.weight"] = a.contiguous()
        converted[f"{peft_module}.lora_B.weight"] = b.contiguous()
        dtype_counts[str(a.dtype)] += 1
        dtype_counts[str(b.dtype)] += 1

    audit = {
        "source_tensor_count": len(tensors),
        "converted_tensor_count": len(converted),
        "module_count": len(expected_modules),
        "rank": EXPECTED_RANK,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "component_module_counts": {
            "talker": EXPECTED_TALKER_LAYERS * len(TARGET_MODULES),
            "code_predictor": EXPECTED_CODE_PREDICTOR_LAYERS * len(TARGET_MODULES),
        },
    }
    return converted, audit


def _adapter_config(*, model_id: str, model_revision: str) -> dict[str, object]:
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": EXPECTED_RANK,
        "rank_pattern": {},
        "revision": model_revision,
        "target_modules": list(TARGET_MODULES),
        "task_type": None,
        "use_dora": False,
        "use_rslora": False,
    }


def convert_adapter(
    source: str | Path,
    output_dir: str | Path,
    *,
    source_revision: str,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    overwrite: bool = False,
) -> dict[str, object]:
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    weights_path = output_dir / "adapter_model.safetensors"
    config_path = output_dir / "adapter_config.json"
    manifest_path = output_dir / "conversion_manifest.json"
    existing = [path for path in (weights_path, config_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing conversion files: {existing}")

    converted, audit = _load_and_validate(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_weights = output_dir / ".adapter_model.safetensors.tmp"
    save_file(
        converted,
        tmp_weights,
        metadata={"format": "pt", "source_format": "mlx-audio", "source_revision": source_revision},
    )
    tmp_weights.replace(weights_path)

    config = _adapter_config(model_id=model_id, model_revision=model_revision)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "source_file": str(source),
        "source_revision": source_revision,
        "source_sha256": _sha256(source),
        "model_id": model_id,
        "model_revision": model_revision,
        "output_sha256": _sha256(weights_path),
        "mapping": {
            "module_prefix": "base_model.model.",
            ".lora_a": ".lora_A.weight",
            ".lora_b": ".lora_B.weight",
        },
        **audit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Published MLX adapters.safetensors")
    parser.add_argument("--output-dir", required=True, help="Destination PEFT adapter directory")
    parser.add_argument("--source-revision", required=True, help="Immutable upstream adapter revision")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = convert_adapter(
        args.source,
        args.output_dir,
        source_revision=args.source_revision,
        model_id=args.model_id,
        model_revision=args.model_revision,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
