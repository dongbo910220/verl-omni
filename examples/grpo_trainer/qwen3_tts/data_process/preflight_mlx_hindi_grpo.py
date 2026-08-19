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
"""Fail-closed input and environment audit for the MLX Hindi GRPO reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

BASE_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
BASE_CONFIG_SHA256 = "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011"
BASE_WEIGHTS_SHA256 = "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6"
SPEECH_TOKENIZER_SHA256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
WHISPER_WEIGHTS_SHA256 = "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"
DATA_REVISION = "5f4495c91d500742a58d1be2ab07d77f73c0acf8"
DATA_ACQUISITION_SOURCE = "SPRINGLab/IndicVoices-R_Hindi"
DATA_ACQUISITION_REVISION = "8f9913669e3505acd38b97fa38bd027f49afeeef"
ADAPTER_SOURCE_REVISION = "a8718cac15b5a40bd4926b47d0854162ff32010a"
ADAPTER_SOURCE_SHA256 = "bd06b9474b128f12863e6ad167fd055fae0ab5f48e5d18df56089707781358df"
ADAPTER_SHA256 = "0d97fd4635fa828f9c2d7fefa7321904341f64c46029a24e34611ae8863bb2e4"
VERL_REVISION = "8a694930275061f52ebd538c906ef8819af56dbd"
EXPECTED_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
EXPECTED_DATA_CONTENT_SHA256 = {
    "train": "ac42716b7ddfac3d86453e4e669ed02368274fb7ba200e811dd75ba52482abb4",
    "validation": "54c837251ed5bfe6e0a6c8bd086031448db62e1354ff92a89b93aface47e2ee1",
    "gate": "e857e548324350a5b50cba70438fd10db930c55aa8ee27e7e7f73cd5c4e5dd59",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(path: Path) -> str | None:
    marker = path / ".hf_revision"
    if marker.is_file():
        return marker.read_text().strip()
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if re.fullmatch(r"[0-9a-f]{40}", candidate.name):
            return candidate.name
    return None


def git_state(path: Path) -> dict:
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(path), "diff", "--binary"])
    status = subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True).splitlines()
    untracked_output = subprocess.check_output(
        ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"]
    )
    untracked_paths = [item.decode() for item in untracked_output.split(b"\0") if item]
    untracked = {
        relative: sha256_file(path / relative) for relative in sorted(untracked_paths) if (path / relative).is_file()
    }
    state_digest = hashlib.sha256(diff)
    for relative, digest in untracked.items():
        state_digest.update(relative.encode())
        state_digest.update(b"\0")
        state_digest.update(digest.encode())
        state_digest.update(b"\0")
    return {
        "revision": revision,
        "state_sha256": state_digest.hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "status": status,
        "untracked_sha256": untracked,
    }


def validate_adapter(path: Path) -> dict:
    from safetensors import safe_open

    config = json.loads((path / "adapter_config.json").read_text())
    manifest = json.loads((path / "conversion_manifest.json").read_text())
    weights = path / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        dtypes = {str(handle.get_tensor(key).dtype) for key in keys}

    actual_sha = sha256_file(weights)
    checks = {
        "rank": config.get("r") == 8,
        "alpha": float(config.get("lora_alpha")) == 16.0,
        "dropout": float(config.get("lora_dropout")) == 0.05,
        "targets": set(config.get("target_modules") or ()) == EXPECTED_TARGETS,
        "tensor_count": len(keys) == 462,
        "dtypes": dtypes == {"torch.float32"},
        "sha256": actual_sha == ADAPTER_SHA256,
        "source_revision": manifest.get("source_revision") == ADAPTER_SOURCE_REVISION,
        "source_sha256": manifest.get("source_sha256") == ADAPTER_SOURCE_SHA256,
        "module_count": manifest.get("module_count") == 231,
    }
    if not all(checks.values()):
        raise ValueError(f"Published SFT adapter validation failed: {checks}")
    return {"path": str(path.resolve()), "sha256": actual_sha, "checks": checks}


def validate_model_files(model: Path, whisper: Path) -> dict:
    expected = {
        model / "config.json": BASE_CONFIG_SHA256,
        model / "model.safetensors": BASE_WEIGHTS_SHA256,
        model / "speech_tokenizer/model.safetensors": SPEECH_TOKENIZER_SHA256,
        whisper / "model.safetensors": WHISPER_WEIGHTS_SHA256,
    }
    actual = {str(path.resolve()): sha256_file(path) for path in expected}
    mismatches = {
        str(path): {"expected": digest, "actual": actual[str(path.resolve())]}
        for path, digest in expected.items()
        if actual[str(path.resolve())] != digest
    }
    if mismatches:
        raise ValueError(f"Pinned model-file validation failed: {mismatches}")
    return actual


def validate_data(train: Path, validation: Path, gate: Path) -> dict:
    import pandas as pd

    expected = {train: 863, validation: 100, gate: 256}
    counts = {str(path.resolve()): len(pd.read_parquet(path)) for path in expected}
    mismatches = {
        str(path): (counts[str(path.resolve())], count)
        for path, count in expected.items()
        if counts[str(path.resolve())] != count
    }
    if mismatches:
        raise ValueError(f"IndicVoices split row-count mismatch: {mismatches}")

    manifest_path = train.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_revision") != DATA_REVISION:
        raise ValueError(f"Expected IndicVoices revision {DATA_REVISION}, got {manifest.get('source_revision')}")
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Expected IndicVoices manifest schema 2, got {manifest.get('schema_version')}")
    filter_contract = manifest.get("filter_contract") or {}
    if (
        filter_contract.get("scenario") != "Read"
        or filter_contract.get("raw_read_limit") != 1_000
        or filter_contract.get("expected_train_after_filter") != 863
    ):
        raise ValueError(f"Unexpected IndicVoices prompt-selection contract: {filter_contract}")
    acquisition = manifest.get("acquisition") or {}
    if acquisition.get("mode") == "sparse_parquet_metadata":
        if (
            acquisition.get("dataset") != DATA_ACQUISITION_SOURCE
            or acquisition.get("revision") != DATA_ACQUISITION_REVISION
            or acquisition.get("rows") != 26_318
        ):
            raise ValueError(f"Unexpected IndicVoices acquisition mirror: {acquisition}")
        if any(shard.get("x_repo_commit") != DATA_ACQUISITION_REVISION for shard in acquisition.get("shards") or []):
            raise ValueError("At least one IndicVoices metadata shard was not fetched from the pinned revision")
    elif acquisition.get("mode") != "authenticated_huggingface_stream":
        raise ValueError(f"Unknown IndicVoices acquisition mode: {acquisition.get('mode')}")
    collection = manifest.get("collection_audit") or {}
    if (
        collection.get("train") != 863
        or collection.get("validation") != 100
        or collection.get("rejected_train") != {"duration": 137}
        or collection.get("train_last_source_index") != 12_600
        or collection.get("validation_first_source_index") != 12_612
        or collection.get("validation_last_source_index") != 13_746
    ):
        raise ValueError(f"Unexpected IndicVoices collection audit: {collection}")
    manifest_counts = {name: item["count"] for name, item in manifest.get("splits", {}).items()}
    expected_manifest_counts = {"train": 863, "validation": 100, "gate": 256}
    if any(manifest_counts.get(name) != count for name, count in expected_manifest_counts.items()):
        raise ValueError(f"Unexpected split counts in {manifest_path}: {manifest_counts}")
    content_hashes = {name: item["content_sha256"] for name, item in manifest.get("splits", {}).items()}
    if any(content_hashes.get(name) != digest for name, digest in EXPECTED_DATA_CONTENT_SHA256.items()):
        raise ValueError(f"Unexpected split content hashes in {manifest_path}: {content_hashes}")
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_revision": manifest["source_revision"],
        "counts": counts,
        "file_sha256": {str(path.resolve()): sha256_file(path) for path in expected},
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--verl-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    required_paths = (
        args.repo,
        args.verl_repo,
        args.model,
        args.whisper,
        args.adapter,
        args.train,
        args.validation,
        args.gate,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    model_revision = source_revision(args.model)
    whisper_revision = source_revision(args.whisper)
    if model_revision != BASE_REVISION:
        raise ValueError(f"Expected base revision {BASE_REVISION}, got {model_revision!r} from {args.model}")
    if whisper_revision != WHISPER_REVISION:
        raise ValueError(f"Expected Whisper revision {WHISPER_REVISION}, got {whisper_revision!r} from {args.whisper}")
    verl_state = git_state(args.verl_repo)
    if verl_state["revision"] != VERL_REVISION:
        raise ValueError(f"Expected verl revision {VERL_REVISION}, got {verl_state['revision']}")
    if torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(f"Requested {args.num_gpus} GPUs, found {torch.cuda.device_count()}.")

    report = {
        "schema_version": 1,
        "contract": {
            "base_revision": BASE_REVISION,
            "whisper_revision": WHISPER_REVISION,
            "data_revision": DATA_REVISION,
            "verl_revision": VERL_REVISION,
        },
        "model": {"path": str(args.model.resolve()), "revision": model_revision},
        "whisper": {"path": str(args.whisper.resolve()), "revision": whisper_revision},
        "model_file_sha256": validate_model_files(args.model, args.whisper),
        "adapter": validate_adapter(args.adapter),
        "data": validate_data(args.train, args.validation, args.gate),
        "repo": git_state(args.repo),
        "verl": verl_state,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
