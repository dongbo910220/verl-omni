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

"""Exact GRPO math used by the published ``mlx-audio-train`` TTS run."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from verl.trainer.ppo import core_algos

MLX_ADV_ESTIMATOR = "mlx_grpo"
MLX_KL_PENALTY = "mlx_k3"
MLX_GRPO_EPSILON = 1e-4


def compute_mlx_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[Any] = None,
    norm_adv_by_std_in_grpo: bool = True,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each prompt group with population std, matching NumPy ``std``."""
    scores = token_level_rewards.sum(dim=-1)
    epsilon = MLX_GRPO_EPSILON if config is None else float(config.get("mlx_grpo_epsilon", MLX_GRPO_EPSILON))
    if epsilon < 0:
        raise ValueError("algorithm.mlx_grpo_epsilon must be non-negative.")
    if len(index) != scores.shape[0]:
        raise ValueError(f"Expected {scores.shape[0]} group ids, got {len(index)}.")

    normalized = torch.empty_like(scores)
    group_rows: dict[Any, list[int]] = {}
    for row, group_id in enumerate(index):
        group_rows.setdefault(group_id, []).append(row)

    with torch.no_grad():
        for rows in group_rows.values():
            row_index = torch.as_tensor(rows, device=scores.device)
            group_scores = scores.index_select(0, row_index)
            centered = group_scores - group_scores.mean()
            if norm_adv_by_std_in_grpo:
                # correction=0 is NumPy's default and the published MLX implementation.
                centered = centered / (group_scores.std(correction=0) + epsilon)
            normalized.index_copy_(0, row_index, centered)

        advantages = normalized.unsqueeze(-1) * response_mask
    return advantages, advantages


def mlx_k3_kl(logprob: torch.Tensor, ref_logprob: torch.Tensor) -> torch.Tensor:
    """Return the published k3 estimator with only the pre-exp clamp."""
    log_ratio = torch.clamp(ref_logprob - logprob, min=-10.0, max=10.0)
    return (torch.exp(log_ratio) - log_ratio - 1.0).contiguous()


def install_mlx_audio_grpo_compat() -> None:
    """Register the estimator and expose ``mlx_k3`` through verl's KL dispatch."""
    registered = core_algos.ADV_ESTIMATOR_REGISTRY.get(MLX_ADV_ESTIMATOR)
    if registered is None:
        core_algos.register_adv_est(MLX_ADV_ESTIMATOR)(compute_mlx_grpo_outcome_advantage)
    elif registered is not compute_mlx_grpo_outcome_advantage:
        raise RuntimeError(f"Advantage estimator {MLX_ADV_ESTIMATOR!r} is already registered by {registered!r}.")

    current = core_algos.kl_penalty
    if not getattr(current, "_verl_omni_mlx_audio_compat", False):
        original = current

        def compatible_kl_penalty(logprob, ref_logprob, kl_penalty):
            if kl_penalty == MLX_KL_PENALTY:
                return mlx_k3_kl(logprob, ref_logprob)
            return original(logprob=logprob, ref_logprob=ref_logprob, kl_penalty=kl_penalty)

        compatible_kl_penalty._verl_omni_mlx_audio_compat = True
        compatible_kl_penalty._verl_omni_original = original
        core_algos.kl_penalty = compatible_kl_penalty

    # losses.py imports kl_penalty by value, so update its local binding too.
    from verl.workers.utils import losses

    losses.kl_penalty = core_algos.kl_penalty


install_mlx_audio_grpo_compat()
