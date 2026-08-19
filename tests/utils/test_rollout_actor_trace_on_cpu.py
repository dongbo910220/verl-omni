# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0
"""CPU tests for opt-in rollout-versus-actor traces."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rollout_actor_trace_test",
    ROOT / "verl_omni/utils/rollout_actor_trace.py",
)
trace = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = trace
SPEC.loader.exec_module(trace)

_make_probability_trace_wrapper = trace._make_probability_trace_wrapper


def test_probability_trace_saves_exact_debug_inputs_atomically(tmp_path):
    expected_metrics = {"training/rollout_probs_diff_mean": 0.01}
    calls = []

    def calculate_debug_metrics(data):
        calls.append(data)
        return expected_metrics

    data = SimpleNamespace(
        batch={
            "responses": torch.tensor([[3, 4]]),
            "response_mask": torch.tensor([[True, False]]),
            "rollout_log_probs": torch.tensor([[-0.1, -0.2]]),
            "old_log_probs": torch.tensor([[-0.3, -0.4]]),
            "ignored": torch.tensor([5]),
        }
    )
    wrapped = _make_probability_trace_wrapper(calculate_debug_metrics, tmp_path)

    assert wrapped(data) is expected_metrics
    assert calls == [data]

    trace_files = list(tmp_path.glob("probability-trace-*.pt"))
    assert [path.name for path in trace_files] == ["probability-trace-000001.pt"]
    assert not list(tmp_path.glob("*.tmp-*"))
    trace = torch.load(trace_files[0], map_location="cpu", weights_only=True)
    assert trace["metrics"] == expected_metrics
    assert set(trace["tensors"]) == {
        "responses",
        "response_mask",
        "rollout_log_probs",
        "old_log_probs",
    }
    torch.testing.assert_close(trace["tensors"]["responses"], data.batch["responses"])


def test_probability_trace_uses_monotonic_filenames(tmp_path):
    wrapped = _make_probability_trace_wrapper(lambda _: {}, tmp_path)
    data = SimpleNamespace(batch={})

    wrapped(data)
    wrapped(data)

    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "probability-trace-000001.pt",
        "probability-trace-000002.pt",
    ]
