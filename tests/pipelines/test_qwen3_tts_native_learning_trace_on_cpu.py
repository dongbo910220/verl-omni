# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""CPU contracts for the optional Hindi GRPO teaching trace."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "examples/grpo_trainer/qwen3_tts/train_indicvoices_hindi_native_grpo.py"
SPEC = importlib.util.spec_from_file_location("qwen3_tts_native_learning_trace_test", SCRIPT)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def test_learning_event_is_disabled_by_default(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path)
    runtime = trainer.Runtime(rank=0, local_rank=0, world_size=2, device=torch.device("cpu"))

    trainer.learning_event(args, runtime, "disabled", step=1, value=torch.tensor([1.0]))

    assert not (tmp_path / "learning-trace").exists()


def test_learning_event_detaches_tensors_and_writes_strict_json(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, learning_trace=True)
    runtime = trainer.Runtime(rank=1, local_rank=1, world_size=2, device=torch.device("cpu"))
    value = torch.tensor([0.25, 0.75], requires_grad=True)

    trainer.learning_event(
        args,
        runtime,
        "trajectory",
        step=3,
        tensor=value,
        array=np.asarray([1, 2]),
        nonfinite=float("nan"),
    )

    path = tmp_path / "learning-trace" / "rank-1.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["event"] == "trajectory"
    assert payload["step"] == 3
    assert payload["tensor"] == {
        "dtype": "torch.float32",
        "shape": [2],
        "values": [0.25, 0.75],
    }
    assert payload["array"] == [1, 2]
    assert payload["nonfinite"] == "nan"
    assert value.grad is None
