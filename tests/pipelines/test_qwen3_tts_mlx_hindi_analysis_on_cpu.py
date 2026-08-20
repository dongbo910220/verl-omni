# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0
"""CPU contracts for the staged MLX Hindi GRPO analysis gates."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "examples/grpo_trainer/qwen3_tts/data_process/analyze_mlx_hindi_grpo.py"
SPEC = importlib.util.spec_from_file_location("mlx_hindi_grpo_analysis_test", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _signal_args(source, output, **overrides):
    values = {
        "input": source,
        "output": output,
        "expected_prompts": 2,
        "candidates": 4,
        "min_valid_ratio": 0.99,
        "min_diverse_group_ratio": 0.60,
        "max_all_perfect_group_ratio": 0.30,
        "zero_variance_epsilon": 1e-8,
        "perfect_cer_epsilon": 1e-12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_signal_gate_requires_exact_groups_and_reward_diversity(tmp_path):
    source = tmp_path / "gate.jsonl"
    output = tmp_path / "gate.json"
    rows = []
    for prompt in range(2):
        for candidate in range(4):
            rows.append(
                {
                    "sample_id": f"prompt-{prompt}",
                    "score": 1.0 - 0.1 * candidate,
                    "cer": 0.1 * candidate,
                    "valid_audio": 1.0,
                }
            )
    _write_jsonl(source, rows)

    analysis.analyze_signal(_signal_args(source, output))

    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["metrics"]["diverse_group_ratio"] == 1.0


def test_signal_gate_fails_closed_on_saturated_groups(tmp_path):
    source = tmp_path / "gate.jsonl"
    output = tmp_path / "gate.json"
    rows = [
        {"sample_id": f"prompt-{prompt}", "score": 1.0, "cer": 0.0, "valid_audio": 1.0}
        for prompt in range(2)
        for _ in range(4)
    ]
    _write_jsonl(source, rows)

    with pytest.raises(SystemExit, match="reward_diversity"):
        analysis.analyze_signal(_signal_args(source, output))
    assert json.loads(output.read_text())["passed"] is False


def _training_line(step, *, lr=5e-6, diff=0.004, pearson=0.999):
    return (
        f"step:{step} - training/rollout_probs_diff_valid:1 "
        f"- training/rollout_probs_diff_mean:{diff} "
        "- training/rollout_probs_diff_max:0.01 "
        f"- training/rollout_actor_probs_pearson_corr:{pearson} "
        "- actor/loss:np.float64(0.2) - actor/grad_norm:np.float64(1.5) "
        f"- actor/lr:np.float64({lr}) - critic/score/min:0.2 - critic/score/max:0.9 "
        "- critic/score/mean:0.5 - critic/advantages/min:-1 - critic/advantages/max:1 "
        "- response_length/clip_ratio:0.0 - response/aborted_ratio:0.0\n"
    )


def test_consistency_and_health_gates_use_strict_per_step_thresholds(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(_training_line(1) + _training_line(2))
    consistency_output = tmp_path / "consistency.json"
    health_output = tmp_path / "health.json"

    analysis.analyze_consistency(
        SimpleNamespace(
            log=log,
            output=consistency_output,
            min_steps=2,
            diff_threshold=0.005,
            pearson_threshold=0.995,
            require_lr_zero=False,
        )
    )
    analysis.analyze_health(
        SimpleNamespace(
            log=log,
            output=health_output,
            min_steps=2,
            diff_threshold=0.005,
            pearson_threshold=0.995,
            max_grad_norm=10_000.0,
            min_healthy_gradient_fraction=0.8,
            reward_spread_epsilon=1e-8,
            min_reward_spread_fraction=0.6,
            max_response_clip_ratio=0.25,
            max_aborted_ratio=0.05,
        )
    )

    assert json.loads(consistency_output.read_text())["passed"] is True
    assert json.loads(health_output.read_text())["passed"] is True

    log.write_text(_training_line(1, diff=0.005))
    with pytest.raises(SystemExit, match="diff_mean_every_step"):
        analysis.analyze_consistency(
            SimpleNamespace(
                log=log,
                output=consistency_output,
                min_steps=1,
                diff_threshold=0.005,
                pearson_threshold=0.995,
                require_lr_zero=False,
            )
        )


def test_eval_pairs_the_same_complete_sample_set(tmp_path):
    base_path = tmp_path / "base.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    output = tmp_path / "eval.json"
    base = []
    candidate = []
    for index in range(2):
        common = {
            "sample_id": f"hi-{index}",
            "wer": 0.2,
            "duration_s": 3.0,
            "valid_audio": 1.0,
            "runaway": 0.0,
            "no_eos": 0.0,
            "trailing_silence": 0.0,
        }
        base.append(common | {"cer": 0.2, "score": 0.8})
        candidate.append(common | {"cer": 0.1, "score": 0.9})
    _write_jsonl(base_path, base)
    _write_jsonl(candidate_path, candidate)

    analysis.analyze_eval(
        SimpleNamespace(
            base=[base_path],
            candidate=[candidate_path],
            output=output,
            expected_rows=2,
            min_valid_ratio=0.99,
            max_runaway_ratio=0.01,
            max_no_eos_ratio=0.10,
            require_cer_improvement=True,
        )
    )

    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["paired_candidate_minus_base"]["cer"]["mean"] == pytest.approx(-0.1)
