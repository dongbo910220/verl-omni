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

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from verl.trainer.ppo import core_algos
from verl.workers.utils import losses

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mlx_audio_grpo_compat_test",
    ROOT / "verl_omni/utils/mlx_audio_grpo_compat.py",
)
compat = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = compat
SPEC.loader.exec_module(compat)

MLX_ADV_ESTIMATOR = compat.MLX_ADV_ESTIMATOR
compute_mlx_grpo_outcome_advantage = compat.compute_mlx_grpo_outcome_advantage
install_mlx_audio_grpo_compat = compat.install_mlx_audio_grpo_compat
mlx_k3_kl = compat.mlx_k3_kl


def test_mlx_grpo_uses_population_standard_deviation_and_1e_4_epsilon():
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0], [10.0], [10.0]])
    mask = torch.ones_like(rewards)
    index = np.array(["a", "a", "a", "a", "b", "b"], dtype=object)

    advantages, returns = compute_mlx_grpo_outcome_advantage(rewards, mask, index)

    expected_a = (rewards[:4, 0] - 2.5) / (rewards[:4, 0].std(correction=0) + 1e-4)
    torch.testing.assert_close(advantages[:4, 0], expected_a)
    torch.testing.assert_close(advantages[4:, 0], torch.zeros(2))
    assert returns is advantages


def test_mlx_grpo_can_read_explicit_epsilon_and_mask_padding():
    rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    config = SimpleNamespace(get=lambda key, default: 0.5 if key == "mlx_grpo_epsilon" else default)

    advantages, _ = compute_mlx_grpo_outcome_advantage(
        rewards,
        mask,
        np.array(["a", "a"], dtype=object),
        config=config,
    )

    torch.testing.assert_close(advantages[:, 0], torch.tensor([-2.0 / 3.0, 2.0 / 3.0]))
    torch.testing.assert_close(advantages[:, 1], torch.zeros(2))


def test_mlx_k3_matches_published_formula_without_output_clamp():
    logprob = torch.tensor([0.0, 0.0, 0.0])
    ref_logprob = torch.tensor([-20.0, 0.5, 20.0])

    actual = mlx_k3_kl(logprob, ref_logprob)
    d = torch.tensor([-10.0, 0.5, 10.0])
    expected = torch.exp(d) - d - 1.0

    torch.testing.assert_close(actual, expected)
    assert actual[-1] > 10.0


def test_install_is_idempotent_and_updates_both_kl_bindings():
    install_mlx_audio_grpo_compat()
    first = core_algos.kl_penalty
    install_mlx_audio_grpo_compat()

    assert core_algos.ADV_ESTIMATOR_REGISTRY[MLX_ADV_ESTIMATOR] is compute_mlx_grpo_outcome_advantage
    assert core_algos.kl_penalty is first
    assert losses.kl_penalty is first
    torch.testing.assert_close(
        losses.kl_penalty(torch.tensor([0.0]), torch.tensor([0.5]), "mlx_k3"),
        mlx_k3_kl(torch.tensor([0.0]), torch.tensor([0.5])),
    )
