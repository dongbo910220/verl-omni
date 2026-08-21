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
"""Fail-closed audit for the two-process Qwen3-TTS GRPO learning trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REQUIRED_KEYS = {
    "schema_version",
    "run_id",
    "phase",
    "hostname",
    "pid",
    "sequence",
    "wall_time_unix_ns",
    "monotonic_ns",
    "event",
    "call_id",
    "payload",
}
COMMON_REQUIRED_EVENTS = {
    "entrypoint.config": 1,
    "runtime.dispatch": 1,
    "trainer.constructed": 1,
    "trainer.setup_result": 1,
    "trainer.lifecycle.begin": 1,
    "trainer.lifecycle.end": 1,
    "controller.training_step_result": 1,
    "controller.old_log_prob": 1,
    "controller.ref_log_prob": 1,
    "controller.advantage": 1,
    "controller.actor_update_result": 1,
    "engine.forward_backward_result": 2,
    "optimizer.pre_step": 2,
    "optimizer.post_step": 2,
    "weights.sync_result": 2,
    "weights.sender_summary": 2,
    "checkpoint.save_result": 1,
}
SOURCE_MAP = {
    "entrypoint.config / runtime.dispatch": "verl_omni/trainer/main_omni.py::main,run_omni",
    "trainer and controller stages": "verl_omni/trainer/omni/ray_omni_trainer.py::OmniPPOTrainerSync",
    "rollout.candidate": "verl_omni/pipelines/qwen3_tts/agent_loop.py::Qwen3TTSSingleTurnAgentLoop.run",
    "rollout.stages": "verl_omni/pipelines/qwen3_tts/omni_rollout_adapter.py::combine_engine_outputs",
    "reward.score": "verl_omni/reward_loop/reward_manager/audio.py::AudioRewardManager.run_single",
    "reward.http": "verl_omni/utils/reward_score/audio_http_scorer_client.py::compute_score",
    "model / forward / optimizer / sender weights": "verl_omni/workers/engine/fsdp/omni_impl.py::OmniFSDPEngine",
    "checkpoint controller": "verl/trainer/ppo/v1/trainer_base.py::_load_checkpoint,_save_checkpoint",
    "GRPO advantage implementation": "verl/trainer/ppo/v1/trainer_base.py::_compute_advantage",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def load_records(trace_root: Path, run_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    raw_root = trace_root / "raw" / run_id
    if not raw_root.exists() and trace_root.name == run_id:
        raw_root = trace_root
    files = sorted(raw_root.rglob("*.jsonl")) if raw_root.exists() else []
    if not files:
        return [], [f"no JSONL trace files found under {raw_root}"]

    records = []
    for file in files:
        previous_sequence = 0
        for line_number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{file}:{line_number}: malformed JSON: {exc}")
                continue
            missing = sorted(REQUIRED_KEYS - set(record))
            if missing:
                errors.append(f"{file}:{line_number}: missing keys {missing}")
            if record.get("schema_version") != SCHEMA_VERSION:
                errors.append(f"{file}:{line_number}: unsupported schema {record.get('schema_version')}")
            sequence = record.get("sequence")
            if not isinstance(sequence, int) or sequence <= previous_sequence:
                errors.append(f"{file}:{line_number}: non-monotonic sequence {sequence} after {previous_sequence}")
            if isinstance(sequence, int):
                previous_sequence = sequence
            record["_source_file"] = str(file.relative_to(trace_root))
            record["_source_line"] = line_number
            records.append(record)
    records.sort(key=lambda item: (item.get("wall_time_unix_ns", 0), item.get("pid", 0), item.get("sequence", 0)))
    return records, errors


def _event_records(records: list[dict[str, Any]], phase: str, event: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("phase") == phase and record.get("event") == event]


def _check(condition: bool, name: str, detail: Any, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": name, "passed": bool(condition), "detail": detail})


def _flatten_numeric(values: Any):
    if isinstance(values, bool):
        return
    if isinstance(values, int | float):
        yield float(values)
    elif isinstance(values, list):
        for value in values:
            yield from _flatten_numeric(value)


def _metric(metrics: dict[str, Any], fragment: str) -> float | None:
    for key, value in metrics.items():
        if fragment in key:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def audit(
    records: list[dict[str, Any]],
    phases: list[str],
    expected_candidates: int,
    load_errors: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    _check(not load_errors, "jsonl_schema_and_sequence", load_errors, checks)
    _check(bool(records), "records_present", {"record_count": len(records)}, checks)

    observed_phases = sorted({str(record.get("phase")) for record in records})
    _check(set(phases) <= set(observed_phases), "expected_phases_present", observed_phases, checks)

    error_events = [
        {
            "phase": record.get("phase"),
            "event": record.get("event"),
            "pid": record.get("pid"),
            "payload": _payload(record),
        }
        for record in records
        if record.get("status") == "error"
    ]
    _check(not error_events, "no_trace_or_runtime_error_events", error_events, checks)

    begins: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    ends: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for record in records:
        event = str(record.get("event"))
        if event.endswith(".begin"):
            begins[(str(record.get("phase")), int(record.get("pid")), event.removesuffix(".begin"))].append(
                str(record.get("call_id"))
            )
        elif event.endswith(".end"):
            ends[(str(record.get("phase")), int(record.get("pid")), event.removesuffix(".end"))].append(
                str(record.get("call_id"))
            )
    unmatched = []
    for key in sorted(set(begins) | set(ends)):
        if Counter(begins[key]) != Counter(ends[key]):
            unmatched.append({"identity": key, "begins": begins[key], "ends": ends[key]})
    _check(not unmatched, "paired_begin_end_spans", unmatched, checks)

    phase_summaries: dict[str, Any] = {}
    trainer_pids: dict[str, list[int]] = {}
    for phase in phases:
        phase_records = [record for record in records if record.get("phase") == phase]
        counts = Counter(str(record.get("event")) for record in phase_records)
        for event, minimum in COMMON_REQUIRED_EVENTS.items():
            _check(
                counts[event] >= minimum,
                f"{phase}:event:{event}",
                {"observed": counts[event], "minimum": minimum},
                checks,
            )
        for event in (
            "rollout.candidate.begin",
            "rollout.candidate.end",
            "rollout.candidate_result",
            "rollout.stages_result",
            "reward.input",
            "reward.output",
            "reward.http_response",
        ):
            _check(
                counts[event] == expected_candidates,
                f"{phase}:cardinality:{event}",
                {"observed": counts[event], "expected": expected_candidates},
                checks,
            )

        trainer_pids[phase] = sorted(
            {int(record["pid"]) for record in phase_records if record.get("event") == "trainer.constructed"}
        )
        training_steps = _event_records(records, phase, "controller.training_step_result")
        training_step_payload = _payload(training_steps[-1]) if training_steps else {}
        _check(
            training_step_payload.get("candidate_count") == expected_candidates,
            f"{phase}:controller_candidate_count",
            training_step_payload.get("candidate_count"),
            checks,
        )

        candidate_records = _event_records(records, phase, "rollout.candidate_result")
        candidate_keys = [_payload(record).get("candidate_key") for record in candidate_records]
        _check(
            len(set(candidate_keys)) == expected_candidates and None not in candidate_keys,
            f"{phase}:unique_candidate_keys",
            {"unique": len(set(candidate_keys)), "total": len(candidate_keys)},
            checks,
        )
        prompt_groups = Counter(str(key).split(":", 1)[0] for key in candidate_keys if key is not None)
        expected_group_size = 8
        _check(
            len(prompt_groups) == expected_candidates // expected_group_size
            and set(prompt_groups.values()) == {expected_group_size},
            f"{phase}:b4_g8_shape",
            dict(prompt_groups),
            checks,
        )

        reward_records = _event_records(records, phase, "reward.output")
        scores = [float(_payload(record)["score"]) for record in reward_records]
        _check(
            len(scores) == expected_candidates and all(math.isfinite(score) for score in scores),
            f"{phase}:finite_scores",
            scores,
            checks,
        )

        advantage_records = _event_records(records, phase, "controller.advantage")
        advantage_payload = _payload(advantage_records[-1]) if advantage_records else {}
        rows = advantage_payload.get("rows", [])
        _check(
            advantage_payload.get("candidate_count") == expected_candidates and len(rows) == expected_candidates,
            f"{phase}:advantage_candidate_count",
            {"payload": advantage_payload.get("candidate_count"), "rows": len(rows)},
            checks,
        )
        group_sizes = advantage_payload.get("group_sizes", {})
        _check(
            len(group_sizes) == expected_candidates // expected_group_size
            and set(group_sizes.values()) == {expected_group_size},
            f"{phase}:advantage_group_sizes",
            group_sizes,
            checks,
        )
        advantage_values = [value for row in rows for value in _flatten_numeric(row.get("advantages", []))]
        _check(
            bool(advantage_values) and all(math.isfinite(value) for value in advantage_values),
            f"{phase}:finite_advantages",
            {"count": len(advantage_values)},
            checks,
        )

        old_log_records = _event_records(records, phase, "controller.old_log_prob")
        metrics = _payload(old_log_records[-1]).get("metrics", {}) if old_log_records else {}
        diff_mean = _metric(metrics, "rollout_probs_diff_mean")
        pearson = _metric(metrics, "rollout_actor_probs_pearson_corr")
        _check(
            diff_mean is not None and diff_mean < 0.005,
            f"{phase}:rollout_actor_diff_mean",
            diff_mean,
            checks,
        )
        _check(
            pearson is not None and pearson > 0.995,
            f"{phase}:rollout_actor_pearson",
            pearson,
            checks,
        )

        optimizer_records = _event_records(records, phase, "optimizer.post_step")
        optimizer_updates = [_payload(record) for record in optimizer_records]
        _check(
            all(payload.get("parameter_fingerprint_changed") is True for payload in optimizer_updates),
            f"{phase}:optimizer_changed_parameters",
            [payload.get("parameter_fingerprint_changed") for payload in optimizer_updates],
            checks,
        )
        grad_norms = [float(payload.get("grad_norm")) for payload in optimizer_updates]
        _check(
            all(math.isfinite(value) and value > 0 for value in grad_norms),
            f"{phase}:finite_positive_grad_norms",
            grad_norms,
            checks,
        )

        save_records = _event_records(records, phase, "checkpoint.save_result")
        saved_steps = [int(record.get("step")) for record in save_records]
        expected_save_step = phases.index(phase) + 1
        _check(
            expected_save_step in saved_steps and any(_payload(record).get("exists") for record in save_records),
            f"{phase}:checkpoint_saved",
            {"saved_steps": saved_steps, "expected": expected_save_step},
            checks,
        )

        load_records = _event_records(records, phase, "checkpoint.load_result")
        loaded_steps = [int(_payload(record).get("loaded_step", -1)) for record in load_records]
        expected_load_step = phases.index(phase)
        _check(
            expected_load_step in loaded_steps,
            f"{phase}:checkpoint_loaded_step",
            {"loaded_steps": loaded_steps, "expected": expected_load_step},
            checks,
        )

        phase_summaries[phase] = {
            "record_count": len(phase_records),
            "process_count": len({(record.get("hostname"), record.get("pid")) for record in phase_records}),
            "event_counts": dict(sorted(counts.items())),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_mean": sum(scores) / len(scores) if scores else None,
            "rollout_probs_diff_mean": diff_mean,
            "rollout_probs_pearson": pearson,
            "grad_norms": grad_norms,
            "trainer_pids": trainer_pids[phase],
        }

    if len(phases) >= 2:
        _check(
            set(trainer_pids[phases[0]]).isdisjoint(trainer_pids[phases[1]]),
            "fresh_controller_process_for_resume",
            trainer_pids,
            checks,
        )
        phase_b_engine_load = _event_records(records, phases[1], "engine.checkpoint_load_result")
        _check(
            len(phase_b_engine_load) >= 2,
            "phase_b_loaded_actor_on_all_ranks",
            {"observed": len(phase_b_engine_load), "minimum": 2},
            checks,
        )
        phase_a_engine_save = _event_records(records, phases[0], "engine.checkpoint_save_result")
        saved_fingerprints = {
            int(_payload(record)["rank"]): _payload(record).get("parameter_fingerprint")
            for record in phase_a_engine_save
        }
        loaded_fingerprints = {
            int(_payload(record)["rank"]): _payload(record).get("parameter_fingerprint")
            for record in phase_b_engine_load
        }
        _check(
            len(saved_fingerprints) >= 2 and saved_fingerprints == loaded_fingerprints,
            "phase_b_parameters_match_phase_a_checkpoint",
            {"saved": saved_fingerprints, "loaded": loaded_fingerprints},
            checks,
        )
        phase_a_engine_pids = {int(record["pid"]) for record in phase_a_engine_save}
        phase_b_engine_pids = {int(record["pid"]) for record in phase_b_engine_load}
        _check(
            phase_a_engine_pids.isdisjoint(phase_b_engine_pids),
            "fresh_actor_processes_for_resume",
            {"phase_a": sorted(phase_a_engine_pids), "phase_b": sorted(phase_b_engine_pids)},
            checks,
        )

    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "check_count": len(checks),
        "failed_check_count": sum(not check["passed"] for check in checks),
        "checks": checks,
        "phases": phase_summaries,
    }


def write_outputs(records: list[dict[str, Any]], audit_result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter((str(record.get("phase")), str(record.get("event"))) for record in records)
    _write_json(
        output_dir / "event_counts.json",
        {
            phase: {event: count for (item_phase, event), count in sorted(counts.items()) if item_phase == phase}
            for phase in sorted({key[0] for key in counts})
        },
    )
    _write_json(output_dir / "required_stage_audit.json", audit_result)

    processes: dict[str, dict[str, Any]] = {}
    for record in records:
        key = f"{record.get('phase')}:{record.get('hostname')}:{record.get('pid')}"
        process = processes.setdefault(
            key,
            {
                "phase": record.get("phase"),
                "hostname": record.get("hostname"),
                "pid": record.get("pid"),
                "ranks": set(),
                "roles": set(),
                "event_count": 0,
                "first_wall_time_unix_ns": record.get("wall_time_unix_ns"),
                "last_wall_time_unix_ns": record.get("wall_time_unix_ns"),
            },
        )
        process["event_count"] += 1
        process["ranks"].add(record.get("rank"))
        process["roles"].add(record.get("role"))
        process["last_wall_time_unix_ns"] = record.get("wall_time_unix_ns")
    serializable_processes = []
    for process in processes.values():
        process["ranks"] = sorted(process["ranks"], key=lambda value: (-1 if value is None else value))
        process["roles"] = sorted(map(str, process["roles"]))
        serializable_processes.append(process)
    _write_json(output_dir / "processes.json", serializable_processes)

    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "phase",
                "wall_time_utc",
                "monotonic_ns",
                "hostname",
                "pid",
                "rank",
                "sequence",
                "step",
                "event",
                "status",
                "call_id",
                "parent_call_id",
                "source_file",
                "source_line",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "phase": record.get("phase"),
                    "wall_time_utc": record.get("wall_time_utc"),
                    "monotonic_ns": record.get("monotonic_ns"),
                    "hostname": record.get("hostname"),
                    "pid": record.get("pid"),
                    "rank": record.get("rank"),
                    "sequence": record.get("sequence"),
                    "step": record.get("step"),
                    "event": record.get("event"),
                    "status": record.get("status"),
                    "call_id": record.get("call_id"),
                    "parent_call_id": record.get("parent_call_id"),
                    "source_file": record.get("_source_file"),
                    "source_line": record.get("_source_line"),
                }
            )

    source_lines = ["# Trace Source Map", ""]
    source_lines.extend(f"- `{event}`: `{source}`" for event, source in SOURCE_MAP.items())
    (output_dir / "source_map.md").write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    summary = [
        "# Qwen3-TTS GRPO Learning Trace Audit",
        "",
        f"- Overall: **{'PASS' if audit_result['passed'] else 'FAIL'}**",
        f"- Checks: {audit_result['check_count']}",
        f"- Failed checks: {audit_result['failed_check_count']}",
        f"- Trace records: {len(records)}",
        "",
        "## Phase Summary",
        "",
        "| Phase | Records | Processes | Score mean | Diff mean | Pearson | Grad norms |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    def format_float(value: Any, digits: int) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    for phase, phase_summary in audit_result["phases"].items():
        summary.append(
            f"| {phase} | {phase_summary['record_count']} | {phase_summary['process_count']} | "
            f"{format_float(phase_summary['score_mean'], 6)} | "
            f"{format_float(phase_summary['rollout_probs_diff_mean'], 8)} | "
            f"{format_float(phase_summary['rollout_probs_pearson'], 8)} | {phase_summary['grad_norms']} |"
        )
    failed = [check for check in audit_result["checks"] if not check["passed"]]
    summary.extend(["", "## Failed Checks", ""])
    summary.extend(["None."] if not failed else [f"- `{check['name']}`: `{check['detail']}`" for check in failed])
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def write_checksums(root: Path) -> None:
    rows = []
    for file in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "manifest.sha256"):
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(f"{digest.hexdigest()}  {file.relative_to(root)}")
    (root / "manifest.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phases", default="phase-a,phase-b")
    parser.add_argument("--expected-candidates", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace_root = args.trace_root.expanduser().resolve()
    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    output_dir = args.output_dir or trace_root / "analysis"
    records, load_errors = load_records(trace_root, args.run_id)
    result = audit(records, phases, args.expected_candidates, load_errors)
    write_outputs(records, result, output_dir)
    write_checksums(trace_root)
    print(json.dumps({"passed": result["passed"], "analysis": str(output_dir)}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
