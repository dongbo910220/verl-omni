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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from verl_omni.utils import learning_trace


@pytest.fixture(autouse=True)
def reset_trace_state(monkeypatch):
    monkeypatch.delenv(learning_trace.TRACE_DIR_ENV, raising=False)
    monkeypatch.delenv(learning_trace.TRACE_RUN_ID_ENV, raising=False)
    monkeypatch.delenv(learning_trace.TRACE_PHASE_ENV, raising=False)
    monkeypatch.delenv(learning_trace.TRACE_EXPECTED_STEP_ENV, raising=False)
    learning_trace._PROCESS_START_PID = None
    learning_trace._SEQUENCE = 0
    learning_trace._SPAN_STACK.set(())


def _records(root: Path) -> list[dict]:
    files = list(root.rglob("*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def test_disabled_trace_is_inert(tmp_path):
    assert learning_trace.emit("ignored", payload={"value": 1}) is None
    with learning_trace.span("ignored"):
        pass
    assert not list(tmp_path.rglob("*"))


def test_trace_has_process_identity_pairing_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv(learning_trace.TRACE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(learning_trace.TRACE_RUN_ID_ENV, "run-1")
    monkeypatch.setenv(learning_trace.TRACE_PHASE_ENV, "phase-a")
    learning_trace.emit(
        "example",
        step=1,
        payload={"password": "do-not-write", "token_ids": [1, 2, 3], "nested": {"api_key": "hidden"}},
    )
    with learning_trace.span("work", step=1):
        learning_trace.emit("work.detail", step=1, payload={"ok": True})

    records = _records(tmp_path)
    assert [record["sequence"] for record in records] == list(range(1, len(records) + 1))
    assert records[0]["event"] == "process.start"
    example = next(record for record in records if record["event"] == "example")
    assert example["payload"]["password"] == "<redacted>"
    assert example["payload"]["nested"]["api_key"] == "<redacted>"
    assert example["payload"]["token_ids"] == [1, 2, 3]
    begin = next(record for record in records if record["event"] == "work.begin")
    end = next(record for record in records if record["event"] == "work.end")
    detail = next(record for record in records if record["event"] == "work.detail")
    assert begin["call_id"] == end["call_id"]
    assert detail["parent_call_id"] == begin["call_id"]
    assert end["status"] == "ok"


def test_tensor_and_parameter_evidence_is_bounded_and_detects_update():
    exact = learning_trace.summarize_tensor(torch.tensor([1.0, 2.0]), exact_limit=4)
    sampled = learning_trace.summarize_tensor(torch.arange(1000), exact_limit=4, sample_limit=8)
    assert exact["values"] == [1.0, 2.0]
    assert "values" not in sampled
    assert len(sampled["sample_values"]) == 8

    module = torch.nn.Linear(4, 2)
    before = learning_trace.summarize_module_parameters(module, include_norms=True)
    module(torch.ones(1, 4)).sum().backward()
    with torch.no_grad():
        module.weight.add_(0.25)
    after = learning_trace.summarize_module_parameters(module, include_norms=True)
    assert before["parameter_fingerprint"] != after["parameter_fingerprint"]
    assert after["local_gradient_l2"] > 0


def test_tensor_sampling_uses_exact_integer_indices_for_large_parameters():
    # float32 linspace rounds 18_743_295 up to 18_743_296 and produces an invalid final index.
    tensor = torch.zeros(18_743_296, dtype=torch.uint8)
    tensor[-1] = 1
    sampled = learning_trace._sample_flat_tensor(tensor, 8)
    assert sampled.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]


def test_weight_iterator_preserves_order_and_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setenv(learning_trace.TRACE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(learning_trace.TRACE_RUN_ID_ENV, "run-1")
    monkeypatch.setenv(learning_trace.TRACE_PHASE_ENV, "phase-a")
    source = [("a", torch.arange(3)), ("b", torch.arange(4))]
    traced = list(learning_trace.traced_weight_iterator(iter(source), step=1))
    assert [name for name, _ in traced] == ["a", "b"]
    assert all(torch.equal(actual, expected) for (_, actual), (_, expected) in zip(traced, source, strict=True))
    summary = next(record for record in _records(tmp_path) if record["event"] == "weights.sender_summary")
    assert summary["payload"]["completed"] is True
    assert summary["payload"]["tensor_count"] == 2
    assert summary["payload"]["total_numel"] == 7


def test_analyzer_fails_closed_for_incomplete_trace(tmp_path, monkeypatch):
    monkeypatch.setenv(learning_trace.TRACE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(learning_trace.TRACE_RUN_ID_ENV, "run-1")
    monkeypatch.setenv(learning_trace.TRACE_PHASE_ENV, "phase-a")
    learning_trace.emit("entrypoint.config", payload={})
    script = Path(__file__).parents[2] / "scripts" / "analyze_qwen3_tts_learning_trace.py"
    env = dict(os.environ)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(tmp_path),
            "--run-id",
            "run-1",
            "--phases",
            "phase-a",
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    audit = json.loads((tmp_path / "analysis" / "required_stage_audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is False
