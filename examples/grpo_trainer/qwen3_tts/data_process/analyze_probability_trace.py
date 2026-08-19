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
"""Analyze an opt-in rollout-versus-actor probability trace."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "p50": float(values.quantile(0.50).item()),
        "p90": float(values.quantile(0.90).item()),
        "p95": float(values.quantile(0.95).item()),
        "p99": float(values.quantile(0.99).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    tensors = trace["tensors"]
    actor_log = tensors["old_log_probs"].float()
    rollout_log = tensors["rollout_log_probs"].float()
    mask = tensors["response_mask"].bool()
    responses = tensors["responses"]
    if actor_log.shape != rollout_log.shape or actor_log.shape != mask.shape or mask.shape != responses.shape:
        raise ValueError(
            "trace tensors must share shape: "
            f"actor={tuple(actor_log.shape)}, rollout={tuple(rollout_log.shape)}, "
            f"mask={tuple(mask.shape)}, responses={tuple(responses.shape)}"
        )
    if not mask.any():
        raise ValueError("trace has no valid response tokens")

    actor_prob = actor_log.exp()
    rollout_prob = rollout_log.exp()
    diff = (actor_prob - rollout_prob).abs()
    valid_diff = diff[mask]
    threshold = args.threshold

    token_rows: list[dict] = []
    sequence_rows: list[dict] = []
    position_values: dict[int, list[float]] = defaultdict(list)
    token_values: dict[int, list[float]] = defaultdict(list)
    terminal_values: list[float] = []
    nonterminal_values: list[float] = []
    for sequence_index in range(diff.shape[0]):
        positions = mask[sequence_index].nonzero(as_tuple=False).flatten()
        sequence_diff = diff[sequence_index, positions]
        terminal_position = int(positions[-1].item())
        over_threshold = positions[sequence_diff >= threshold]
        first_over = int(over_threshold[0].item()) if over_threshold.numel() else None
        max_offset = int(sequence_diff.argmax().item())
        max_position = int(positions[max_offset].item())
        sequence_rows.append(
            {
                "sequence_index": sequence_index,
                "valid_tokens": int(positions.numel()),
                "diff_mean": float(sequence_diff.mean().item()),
                "diff_max": float(sequence_diff.max().item()),
                "count_ge_threshold": int((sequence_diff >= threshold).sum().item()),
                "first_ge_threshold_position": first_over,
                "max_position": max_position,
                "max_token_id": int(responses[sequence_index, max_position].item()),
                "actor_prob_at_max": float(actor_prob[sequence_index, max_position].item()),
                "rollout_prob_at_max": float(rollout_prob[sequence_index, max_position].item()),
            }
        )
        for position in positions.tolist():
            token_id = int(responses[sequence_index, position].item())
            value = float(diff[sequence_index, position].item())
            is_terminal = position == terminal_position
            token_rows.append(
                {
                    "sequence_index": sequence_index,
                    "position": position,
                    "normalized_position": position / max(terminal_position, 1),
                    "token_id": token_id,
                    "is_terminal": is_terminal,
                    "actor_log_prob": float(actor_log[sequence_index, position].item()),
                    "rollout_log_prob": float(rollout_log[sequence_index, position].item()),
                    "actor_prob": float(actor_prob[sequence_index, position].item()),
                    "rollout_prob": float(rollout_prob[sequence_index, position].item()),
                    "diff": value,
                }
            )
            position_values[position].append(value)
            token_values[token_id].append(value)
            (terminal_values if is_terminal else nonterminal_values).append(value)

    total_diff = float(valid_diff.sum().item())
    bins = []
    for name, lower, upper in (
        ("[0,0.001)", 0.0, 0.001),
        ("[0.001,0.005)", 0.001, 0.005),
        ("[0.005,0.01)", 0.005, 0.01),
        ("[0.01,0.05)", 0.01, 0.05),
        ("[0.05,0.1)", 0.05, 0.1),
        ("[0.1,inf)", 0.1, float("inf")),
    ):
        selected = valid_diff[(valid_diff >= lower) & (valid_diff < upper)]
        contribution = float(selected.sum().item())
        bins.append(
            {
                "bin": name,
                "count": int(selected.numel()),
                "token_fraction": float(selected.numel() / valid_diff.numel()),
                "diff_contribution": contribution / total_diff if total_diff else 0.0,
            }
        )

    by_position = [
        {"position": position, **_summarize(torch.tensor(values))}
        for position, values in sorted(position_values.items())
    ]
    by_token = [{"token_id": token_id, **_summarize(torch.tensor(values))} for token_id, values in token_values.items()]
    top_tokens = sorted(token_rows, key=lambda row: row["diff"], reverse=True)[:100]
    summary = {
        "source_trace": str(args.trace.resolve()),
        "source_metrics": trace.get("metrics", {}),
        "shape": list(diff.shape),
        "threshold": threshold,
        "global": _summarize(valid_diff),
        "tokens_ge_threshold": int((valid_diff >= threshold).sum().item()),
        "token_fraction_ge_threshold": float((valid_diff >= threshold).float().mean().item()),
        "bins": bins,
        "terminal": _summarize(torch.tensor(terminal_values)),
        "nonterminal": _summarize(torch.tensor(nonterminal_values)),
        "top_sequences": sorted(sequence_rows, key=lambda row: row["diff_mean"], reverse=True),
        "top_token_ids_by_mean": sorted(by_token, key=lambda row: row["mean"], reverse=True)[:50],
        "top_individual_tokens": top_tokens,
    }
    (args.output_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(args.output_dir / "per_token.csv", token_rows)
    _write_csv(args.output_dir / "per_sequence.csv", sequence_rows)
    _write_csv(args.output_dir / "by_position.csv", by_position)
    _write_csv(args.output_dir / "by_token_id.csv", by_token)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
