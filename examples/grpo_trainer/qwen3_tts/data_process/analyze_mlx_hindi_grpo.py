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
"""Fail-closed analysis gates for the MLX Hindi Qwen3-TTS GRPO reproduction."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_PATTERN = re.compile(r"\bstep:(\d+)\s+-")
NUMBER = r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|nan))"
TRAINING_FIELDS = {
    "diff_valid": "training/rollout_probs_diff_valid",
    "diff_mean": "training/rollout_probs_diff_mean",
    "diff_max": "training/rollout_probs_diff_max",
    "pearson": "training/rollout_actor_probs_pearson_corr",
    "actor_lr": "actor/lr",
    "actor_loss": "actor/loss",
    "grad_norm": "actor/grad_norm",
    "kl_loss": "actor/kl_loss",
    "score_mean": "critic/score/mean",
    "score_min": "critic/score/min",
    "score_max": "critic/score/max",
    "advantage_min": "critic/advantages/min",
    "advantage_max": "critic/advantages/max",
    "response_clip_ratio": "response_length/clip_ratio",
    "aborted_ratio": "response/aborted_ratio",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    return rows


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def finish(report: dict, output: Path) -> None:
    report["passed"] = all(report["checks"].values())
    write_report(report, output)
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise SystemExit(f"Gate failed: {', '.join(failed)}")


def _as_float(row: dict, name: str, default: float = math.nan) -> float:
    try:
        return float(row.get(name, default))
    except (TypeError, ValueError):
        return default


def _prompt_group_id(row: dict) -> str:
    sample_id = str(row.get("sample_id") or "")
    if sample_id:
        return sample_id
    uid = str(row.get("uid") or "")
    parts = uid.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0]
    target = str(row.get("normalized_target") or row.get("gts") or "")
    if target:
        return target
    raise ValueError("Cannot identify a prompt group: sample_id, uid, and target are all absent.")


def analyze_signal(args) -> None:
    rows = read_jsonl(args.input)
    groups = defaultdict(list)
    for row in rows:
        groups[_prompt_group_id(row)].append(row)

    expected_rows = args.expected_prompts * args.candidates
    exact_group_sizes = all(len(group) == args.candidates for group in groups.values())
    finite_rows = [
        row for row in rows if math.isfinite(_as_float(row, "score")) and math.isfinite(_as_float(row, "cer"))
    ]
    valid_rows = [row for row in finite_rows if _as_float(row, "valid_audio", 1.0) >= 0.5]
    diverse_groups = 0
    all_perfect_groups = 0
    group_std = []
    for group in groups.values():
        scores = [_as_float(row, "score") for row in group]
        std = statistics.pstdev(scores) if scores and all(map(math.isfinite, scores)) else math.nan
        group_std.append(std)
        diverse_groups += int(math.isfinite(std) and std > args.zero_variance_epsilon)
        all_perfect_groups += int(
            len(group) == args.candidates
            and all(_as_float(row, "valid_audio", 1.0) >= 0.5 for row in group)
            and all(_as_float(row, "cer") <= args.perfect_cer_epsilon for row in group)
        )

    group_count = len(groups)
    valid_ratio = len(valid_rows) / len(rows) if rows else 0.0
    diverse_ratio = diverse_groups / group_count if group_count else 0.0
    all_perfect_ratio = all_perfect_groups / group_count if group_count else 1.0
    report = {
        "mode": "signal",
        "source": str(args.input.resolve()),
        "counts": {
            "rows": len(rows),
            "groups": group_count,
            "finite_rows": len(finite_rows),
            "valid_rows": len(valid_rows),
            "diverse_groups": diverse_groups,
            "all_perfect_groups": all_perfect_groups,
        },
        "metrics": {
            "valid_ratio": valid_ratio,
            "diverse_group_ratio": diverse_ratio,
            "zero_variance_group_ratio": 1.0 - diverse_ratio,
            "all_perfect_group_ratio": all_perfect_ratio,
            "mean_group_reward_std": statistics.fmean(value for value in group_std if math.isfinite(value))
            if any(math.isfinite(value) for value in group_std)
            else math.nan,
        },
        "thresholds": {
            "expected_prompts": args.expected_prompts,
            "candidates": args.candidates,
            "minimum_valid_ratio": args.min_valid_ratio,
            "minimum_diverse_group_ratio": args.min_diverse_group_ratio,
            "maximum_all_perfect_group_ratio": args.max_all_perfect_group_ratio,
        },
        "checks": {
            "exact_row_count": len(rows) == expected_rows,
            "exact_group_count": group_count == args.expected_prompts,
            "exact_group_sizes": exact_group_sizes,
            "valid_audio": valid_ratio >= args.min_valid_ratio,
            "reward_diversity": diverse_ratio >= args.min_diverse_group_ratio,
            "not_saturated": all_perfect_ratio <= args.max_all_perfect_group_ratio,
        },
    }
    finish(report, args.output)


def extract_training_rows(path: Path) -> list[dict[str, float | int]]:
    by_step: dict[int, dict[str, float | int]] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        step_match = STEP_PATTERN.search(line)
        if not step_match:
            continue
        row: dict[str, float | int] = {"step": int(step_match.group(1))}
        for name, label in TRAINING_FIELDS.items():
            match = re.search(rf"{re.escape(label)}:(?:np\.[A-Za-z0-9_]+\()?{NUMBER}\)?", line, re.IGNORECASE)
            if match:
                row[name] = float(match.group(1))
        step = int(row["step"])
        if len(row) > len(by_step.get(step, {})):
            by_step[step] = row
    return [by_step[step] for step in sorted(by_step)]


def _required_training_rows(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    return [row for row in rows if all(field in row for field in fields)]


def _consistency_summary(rows: list[dict], diff_threshold: float, pearson_threshold: float) -> tuple[dict, dict]:
    required = _required_training_rows(rows, ("diff_valid", "diff_mean", "pearson"))
    diff_values = [_as_float(row, "diff_mean") for row in required]
    pearson_values = [_as_float(row, "pearson") for row in required]
    finite = all(math.isfinite(value) for value in diff_values + pearson_values)
    metrics = {
        "rows_with_consistency": len(required),
        "maximum_diff_mean": max(diff_values, default=math.inf),
        "mean_diff_mean": statistics.fmean(diff_values) if diff_values else math.inf,
        "minimum_pearson": min(pearson_values, default=-math.inf),
        "mean_pearson": statistics.fmean(pearson_values) if pearson_values else -math.inf,
    }
    checks = {
        "consistency_metrics_finite": finite,
        "diff_mean_every_step": bool(required)
        and all(
            _as_float(row, "diff_valid") == 1.0 and _as_float(row, "diff_mean") < diff_threshold for row in required
        ),
        "pearson_every_step": bool(required) and all(_as_float(row, "pearson") > pearson_threshold for row in required),
    }
    return metrics, checks


def analyze_consistency(args) -> None:
    rows = extract_training_rows(args.log)
    metrics, checks = _consistency_summary(rows, args.diff_threshold, args.pearson_threshold)
    consistency_rows = _required_training_rows(rows, ("diff_valid", "diff_mean", "pearson"))
    checks["minimum_step_count"] = len(consistency_rows) >= args.min_steps
    if args.require_lr_zero:
        checks["lr_is_zero"] = len(consistency_rows) >= args.min_steps and all(
            "actor_lr" in row and abs(_as_float(row, "actor_lr")) <= 1e-15 for row in consistency_rows
        )
    report = {
        "mode": "consistency",
        "source": str(args.log.resolve()),
        "steps": [row["step"] for row in consistency_rows],
        "metrics": metrics,
        "thresholds": {
            "minimum_steps": args.min_steps,
            "diff_mean_strictly_less_than": args.diff_threshold,
            "pearson_strictly_greater_than": args.pearson_threshold,
            "require_lr_zero": args.require_lr_zero,
        },
        "checks": checks,
    }
    finish(report, args.output)


def analyze_health(args) -> None:
    rows = extract_training_rows(args.log)
    required_fields = (
        "diff_valid",
        "diff_mean",
        "pearson",
        "actor_lr",
        "actor_loss",
        "grad_norm",
        "score_min",
        "score_max",
    )
    health_rows = _required_training_rows(rows, required_fields)
    consistency_metrics, checks = _consistency_summary(health_rows, args.diff_threshold, args.pearson_threshold)
    numeric_fields = required_fields[1:]
    all_finite = all(math.isfinite(_as_float(row, field)) for row in health_rows for field in numeric_fields)
    positive_grad_fraction = (
        sum(0.0 < _as_float(row, "grad_norm") <= args.max_grad_norm for row in health_rows) / len(health_rows)
        if health_rows
        else 0.0
    )
    reward_spread_fraction = (
        sum(
            _as_float(row, "score_max") - _as_float(row, "score_min") > args.reward_spread_epsilon
            for row in health_rows
        )
        / len(health_rows)
        if health_rows
        else 0.0
    )
    clip_values = [_as_float(row, "response_clip_ratio") for row in health_rows if "response_clip_ratio" in row]
    aborted_values = [_as_float(row, "aborted_ratio") for row in health_rows if "aborted_ratio" in row]
    checks.update(
        {
            "minimum_step_count": len(health_rows) >= args.min_steps,
            "all_core_metrics_finite": all_finite,
            "positive_learning_rate": bool(health_rows)
            and all(_as_float(row, "actor_lr") > 0.0 for row in health_rows),
            "healthy_gradient_fraction": positive_grad_fraction >= args.min_healthy_gradient_fraction,
            "reward_spread_fraction": reward_spread_fraction >= args.min_reward_spread_fraction,
            "response_clip_ratio": not clip_values or max(clip_values) <= args.max_response_clip_ratio,
            "aborted_ratio": not aborted_values or max(aborted_values) <= args.max_aborted_ratio,
        }
    )
    report = {
        "mode": "health",
        "source": str(args.log.resolve()),
        "steps": [row["step"] for row in health_rows],
        "metrics": consistency_metrics
        | {
            "rows_with_all_core_metrics": len(health_rows),
            "healthy_gradient_fraction": positive_grad_fraction,
            "reward_spread_fraction": reward_spread_fraction,
            "maximum_grad_norm": max((_as_float(row, "grad_norm") for row in health_rows), default=math.inf),
            "maximum_response_clip_ratio": max(clip_values, default=0.0),
            "maximum_aborted_ratio": max(aborted_values, default=0.0),
        },
        "thresholds": {
            "minimum_steps": args.min_steps,
            "diff_mean_strictly_less_than": args.diff_threshold,
            "pearson_strictly_greater_than": args.pearson_threshold,
            "maximum_grad_norm": args.max_grad_norm,
            "minimum_healthy_gradient_fraction": args.min_healthy_gradient_fraction,
            "minimum_reward_spread_fraction": args.min_reward_spread_fraction,
            "maximum_response_clip_ratio": args.max_response_clip_ratio,
            "maximum_aborted_ratio": args.max_aborted_ratio,
        },
        "checks": checks,
    }
    finish(report, args.output)


def _evaluation_id(row: dict) -> str:
    sample_id = str(row.get("sample_id") or "")
    if sample_id:
        return sample_id
    seed = row.get("generation_seed", "")
    target = str(row.get("normalized_target") or row.get("gts") or "")
    if target:
        return f"{seed}\0{target}"
    raise ValueError("Evaluation row lacks sample_id and target.")


def _index_evaluation(rows: list[dict], source: Path) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        key = _evaluation_id(row)
        if key in indexed:
            raise ValueError(f"Duplicate evaluation id {key!r} in {source}.")
        indexed[key] = row
    return indexed


def _mean(rows: list[dict], field: str) -> float:
    values = [_evaluation_metric(row, field) for row in rows]
    return statistics.fmean(values) if values and all(map(math.isfinite, values)) else math.nan


def _evaluation_metric(row: dict, field: str) -> float:
    value = _as_float(row, field)
    if field == "cer_capped" and not math.isfinite(value):
        cer = _as_float(row, "cer")
        return min(1.0, max(0.0, cer)) if math.isfinite(cer) else math.nan
    return value


def _bootstrap_mean_ci(values: list[float], samples: int = 5_000, seed: int = 42) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]]


def _paired_outcomes(values: list[float], epsilon: float = 1e-12) -> dict[str, int]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "better": sum(value < -epsilon for value in finite),
        "equal": sum(abs(value) <= epsilon for value in finite),
        "worse": sum(value > epsilon for value in finite),
    }


def analyze_eval(args) -> None:
    if len(args.base) != len(args.candidate):
        raise ValueError("Provide the same number of --base and --candidate files.")
    paired_rows = []
    file_reports = []
    exact_counts = True
    exact_ids = True
    exact_seeds = True
    for pair_index, (base_path, candidate_path) in enumerate(zip(args.base, args.candidate, strict=True)):
        base_rows = read_jsonl(base_path)
        candidate_rows = read_jsonl(candidate_path)
        exact_counts &= len(base_rows) == args.expected_rows and len(candidate_rows) == args.expected_rows
        base = _index_evaluation(base_rows, base_path)
        candidate = _index_evaluation(candidate_rows, candidate_path)
        matching_ids = set(base) == set(candidate)
        matching_seeds = all(
            base[key].get("generation_seed") == candidate[key].get("generation_seed")
            for key in set(base) & set(candidate)
        )
        exact_ids &= matching_ids
        exact_seeds &= matching_seeds
        common = sorted(set(base) & set(candidate))
        paired_rows.extend((pair_index, key, base[key], candidate[key]) for key in common)
        file_reports.append(
            {
                "base": str(base_path.resolve()),
                "candidate": str(candidate_path.resolve()),
                "base_rows": len(base_rows),
                "candidate_rows": len(candidate_rows),
                "matching_ids": matching_ids,
                "matching_generation_seeds": matching_seeds,
                "paired_rows": len(common),
            }
        )

    base_rows = [item[2] for item in paired_rows]
    candidate_rows = [item[3] for item in paired_rows]
    metric_names = (
        "cer",
        "cer_capped",
        "wer",
        "score",
        "duration_s",
        "valid_audio",
        "runaway",
        "no_eos",
        "trailing_silence",
    )
    base_metrics = {field: _mean(base_rows, field) for field in metric_names}
    candidate_metrics = {field: _mean(candidate_rows, field) for field in metric_names}
    paired_deltas = {}
    for field in metric_names:
        values = [
            _evaluation_metric(candidate, field) - _evaluation_metric(base, field)
            for _, _, base, candidate in paired_rows
        ]
        finite_values = [value for value in values if math.isfinite(value)]
        paired_deltas[field] = {
            "mean": statistics.fmean(finite_values) if finite_values else math.nan,
            "bootstrap_95_ci": _bootstrap_mean_ci(finite_values),
        }

    capped_deltas = [
        _evaluation_metric(candidate, "cer_capped") - _evaluation_metric(base, "cer_capped")
        for _, _, base, candidate in paired_rows
    ]
    capped_pairwise = _paired_outcomes(capped_deltas)
    base_capped = base_metrics["cer_capped"]
    candidate_capped = candidate_metrics["cer_capped"]
    capped_relative_change = (
        candidate_capped / base_capped - 1.0
        if math.isfinite(base_capped) and math.isfinite(candidate_capped) and base_capped > 0.0
        else math.nan
    )

    prompt_values = defaultdict(lambda: {"base": [], "candidate": []})
    for _, key, base, candidate in paired_rows:
        prompt_values[key]["base"].append(_evaluation_metric(base, "cer_capped"))
        prompt_values[key]["candidate"].append(_evaluation_metric(candidate, "cer_capped"))
    prompt_base = [statistics.fmean(values["base"]) for values in prompt_values.values()]
    prompt_candidate = [statistics.fmean(values["candidate"]) for values in prompt_values.values()]
    prompt_deltas = [candidate - base for base, candidate in zip(prompt_base, prompt_candidate, strict=True)]
    prompt_averaged = {
        "prompt_count": len(prompt_values),
        "seeds_per_prompt": sorted({len(values["base"]) for values in prompt_values.values()}),
        "base_mean": statistics.fmean(prompt_base) if prompt_base else math.nan,
        "candidate_mean": statistics.fmean(prompt_candidate) if prompt_candidate else math.nan,
        "candidate_minus_base_mean": statistics.fmean(prompt_deltas) if prompt_deltas else math.nan,
        "candidate_minus_base_bootstrap_95_ci": _bootstrap_mean_ci(prompt_deltas),
        "paired_outcomes": _paired_outcomes(prompt_deltas),
    }

    expected_total = args.expected_rows * len(args.base)
    checks = {
        "exact_file_row_counts": exact_counts,
        "exact_paired_ids": exact_ids,
        "exact_paired_generation_seeds": exact_seeds,
        "exact_total_pair_count": len(paired_rows) == expected_total,
        "candidate_valid_audio": candidate_metrics["valid_audio"] >= args.min_valid_ratio,
        "candidate_runaway": candidate_metrics["runaway"] <= args.max_runaway_ratio,
        "candidate_no_eos": candidate_metrics["no_eos"] <= args.max_no_eos_ratio,
    }
    if args.require_cer_improvement:
        checks["cer_improved"] = paired_deltas["cer"]["mean"] < 0.0
    report = {
        "mode": "eval",
        "files": file_reports,
        "paired_count": len(paired_rows),
        "base": base_metrics,
        "candidate": candidate_metrics,
        "model_card_cer": {
            "metric": "cer_capped",
            "candidate_relative_change": capped_relative_change,
            "paired_outcomes": capped_pairwise,
            "prompt_averaged_across_seeds": prompt_averaged,
        },
        "paired_candidate_minus_base": paired_deltas,
        "thresholds": {
            "expected_rows_per_file": args.expected_rows,
            "minimum_valid_ratio": args.min_valid_ratio,
            "maximum_runaway_ratio": args.max_runaway_ratio,
            "maximum_no_eos_ratio": args.max_no_eos_ratio,
            "require_cer_improvement": args.require_cer_improvement,
        },
        "checks": checks,
    }
    finish(report, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    signal = subparsers.add_parser("signal")
    signal.add_argument("--input", type=Path, required=True)
    signal.add_argument("--output", type=Path, required=True)
    signal.add_argument("--expected-prompts", type=int, default=256)
    signal.add_argument("--candidates", type=int, default=4)
    signal.add_argument("--min-valid-ratio", type=float, default=0.99)
    signal.add_argument("--min-diverse-group-ratio", type=float, default=0.60)
    signal.add_argument("--max-all-perfect-group-ratio", type=float, default=0.30)
    signal.add_argument("--zero-variance-epsilon", type=float, default=1e-8)
    signal.add_argument("--perfect-cer-epsilon", type=float, default=1e-12)
    signal.set_defaults(func=analyze_signal)

    consistency = subparsers.add_parser("consistency")
    consistency.add_argument("--log", type=Path, required=True)
    consistency.add_argument("--output", type=Path, required=True)
    consistency.add_argument("--min-steps", type=int, default=10)
    consistency.add_argument("--diff-threshold", type=float, default=0.005)
    consistency.add_argument("--pearson-threshold", type=float, default=0.995)
    consistency.add_argument("--require-lr-zero", action="store_true")
    consistency.set_defaults(func=analyze_consistency)

    health = subparsers.add_parser("health")
    health.add_argument("--log", type=Path, required=True)
    health.add_argument("--output", type=Path, required=True)
    health.add_argument("--min-steps", type=int, default=5)
    health.add_argument("--diff-threshold", type=float, default=0.005)
    health.add_argument("--pearson-threshold", type=float, default=0.995)
    health.add_argument("--max-grad-norm", type=float, default=10_000.0)
    health.add_argument("--min-healthy-gradient-fraction", type=float, default=0.80)
    health.add_argument("--reward-spread-epsilon", type=float, default=1e-8)
    health.add_argument("--min-reward-spread-fraction", type=float, default=0.60)
    health.add_argument("--max-response-clip-ratio", type=float, default=0.25)
    health.add_argument("--max-aborted-ratio", type=float, default=0.05)
    health.set_defaults(func=analyze_health)

    evaluation = subparsers.add_parser("eval")
    evaluation.add_argument("--base", type=Path, action="append", required=True)
    evaluation.add_argument("--candidate", type=Path, action="append", required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--expected-rows", type=int, default=100)
    evaluation.add_argument("--min-valid-ratio", type=float, default=0.99)
    evaluation.add_argument("--max-runaway-ratio", type=float, default=0.01)
    evaluation.add_argument("--max-no-eos-ratio", type=float, default=0.10)
    evaluation.add_argument("--require-cer-improvement", action="store_true")
    evaluation.set_defaults(func=analyze_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
