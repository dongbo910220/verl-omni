# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Opt-in traces for rollout-versus-actor probability debugging."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Callable

import torch

TRACE_DIR_ENV = "VERL_ROLLOUT_ACTOR_TRACE_DIR"
_TRACE_TENSOR_KEYS = (
    "input_ids",
    "prompts",
    "responses",
    "response_mask",
    "attention_mask",
    "rollout_log_probs",
    "old_log_probs",
)


def _make_probability_trace_wrapper(calculate_debug_metrics: Callable, trace_dir: Path) -> Callable:
    counter = itertools.count(1)

    def traced_calculate_debug_metrics(data):
        metrics = calculate_debug_metrics(data)
        tensors = {
            key: data.batch[key].detach().cpu()
            for key in _TRACE_TENSOR_KEYS
            if key in data.batch and isinstance(data.batch[key], torch.Tensor)
        }
        sequence = next(counter)
        trace_dir.mkdir(parents=True, exist_ok=True)
        output_path = trace_dir / f"probability-trace-{sequence:06d}.pt"
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp-{os.getpid()}")
        torch.save(
            {
                "metrics": dict(metrics),
                "tensors": tensors,
            },
            temporary_path,
        )
        os.replace(temporary_path, output_path)
        return metrics

    traced_calculate_debug_metrics._verl_omni_probability_trace = True
    return traced_calculate_debug_metrics


def install_probability_trace_from_env() -> bool:
    """Patch verl's imported metric callback only when an explicit trace dir is set."""
    trace_dir_value = os.environ.get(TRACE_DIR_ENV)
    if not trace_dir_value:
        return False

    import verl.trainer.ppo.v1.trainer_base as trainer_base

    current = trainer_base.calculate_debug_metrics
    if getattr(current, "_verl_omni_probability_trace", False):
        return False
    trainer_base.calculate_debug_metrics = _make_probability_trace_wrapper(current, Path(trace_dir_value))
    return True
