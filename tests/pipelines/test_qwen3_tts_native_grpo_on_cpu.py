# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""CPU contracts for the native Hindi Qwen3-TTS GRPO path."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "examples/grpo_trainer/qwen3_tts/native_grpo.py"
SPEC = importlib.util.spec_from_file_location("qwen3_tts_native_grpo_test", SCRIPT)
native = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = native
SPEC.loader.exec_module(native)


def test_parse_generation_keeps_eos_for_policy_but_not_audio():
    eos = 7
    sequences = torch.tensor([[1, 2, eos], [4, 5, 6]])
    logits = []
    for step in range(3):
        value = torch.full((2, 10), -4.0)
        value.scatter_(1, sequences[:, step : step + 1], 4.0)
        logits.append(value)
    frame0 = torch.zeros((2, 16), dtype=torch.long)
    frame1 = torch.ones((2, 16), dtype=torch.long)
    frame0[:, 0] = sequences[:, 0]
    frame1[:, 0] = sequences[:, 1]
    result = SimpleNamespace(
        sequences=sequences,
        logits=tuple(logits),
        hidden_states=((None, None), (None, frame0), (None, frame1)),
    )

    codes, audio_codes, logprobs, has_eos, lengths = native.parse_generation_result(result, eos)

    assert has_eos == [True, False]
    assert lengths == [3, 3]
    assert [value.shape for value in codes] == [torch.Size([3, 16]), torch.Size([3, 16])]
    assert [value.shape for value in audio_codes] == [torch.Size([2, 16]), torch.Size([2, 16])]
    torch.testing.assert_close(codes[0][:, 0], sequences[0])
    torch.testing.assert_close(codes[1][:, 0], sequences[1])
    assert all(value.shape == torch.Size([3]) for value in logprobs)


def test_parse_generation_fails_closed_on_misaligned_subtalker_frames():
    result = SimpleNamespace(
        sequences=torch.tensor([[1, 2]]),
        logits=(torch.zeros(1, 3), torch.zeros(1, 3)),
        hidden_states=((None, None), (None, torch.zeros((1, 16), dtype=torch.long))),
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        native.parse_generation_result(result, eos_token_id=2)


def test_group_advantages_match_population_standard_deviation():
    rewards = np.asarray([0.1, 0.2, 0.5, 0.8], dtype=np.float32)
    actual = native.group_advantages(rewards)
    expected = (rewards - rewards.mean()) / (rewards.std(ddof=0) + 1e-4)
    np.testing.assert_allclose(actual, expected)


def test_grpo_loss_matches_published_token_mean_and_k3_formula():
    actor = [torch.tensor([-1.0, -2.0], requires_grad=True), torch.tensor([-0.5], requires_grad=True)]
    reference = [torch.tensor([-1.2, -1.8]), torch.tensor([-0.4])]
    advantages = np.asarray([-1.0, 1.0], dtype=np.float32)

    loss, metrics = native.grpo_loss(actor, reference, advantages, kl_beta=0.08)

    pg = -((-1.0 * actor[0].sum()) + (1.0 * actor[1].sum())) / 3
    delta = torch.cat((reference[0] - actor[0], reference[1] - actor[1]))
    kl = (delta.exp() - delta - 1.0).mean()
    torch.testing.assert_close(loss, pg + 0.08 * kl)
    assert metrics["tokens"] == 3.0
    loss.backward()
    assert all(value.grad is not None for value in actor)


def test_probability_consistency_uses_required_strict_thresholds():
    rollout = [torch.log(torch.tensor([0.1, 0.2, 0.4, 0.8]))]
    exact = native.probability_consistency(rollout, [rollout[0].clone()])
    shifted = native.probability_consistency(
        rollout, [torch.log(torch.tensor([0.11, 0.21, 0.41, 0.81]))]
    )

    assert exact == {"tokens": 4, "diff_mean": 0.0, "pearson": 1.0, "passed": True}
    assert shifted["diff_mean"] == pytest.approx(0.01)
    assert shifted["passed"] is False


def test_public_constant_warmup_schedule_starts_at_zero():
    assert native.learning_rate_for_step(0, 5e-6, 10) == 0.0
    assert native.learning_rate_for_step(5, 5e-6, 10) == pytest.approx(2.5e-6)
    assert native.learning_rate_for_step(10, 5e-6, 10) == pytest.approx(5e-6)
    assert native.learning_rate_for_step(400, 5e-6, 10) == pytest.approx(5e-6)


def test_inference_rollout_tensors_can_be_cloned_for_actor_replay():
    with torch.inference_mode():
        rollout_codes = torch.tensor([[1] * 16, [2] * 16])

    replay_codes = rollout_codes.detach().clone()

    assert torch.is_inference(rollout_codes)
    assert not torch.is_inference(replay_codes)
