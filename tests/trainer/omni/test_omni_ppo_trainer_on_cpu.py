# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0
"""CPU contracts for the Omni v1 synchronous PPO trainer."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync


def _make_trainer(*, use_reference_policy=True, lora_rank=8, lora_adapter_path=None):
    trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "lora_rank": lora_rank,
                    "lora_adapter_path": lora_adapter_path,
                }
            }
        }
    )
    trainer.use_reference_policy = use_reference_policy
    return trainer


def test_setup_aliases_lora_reference_policy_to_actor_worker(monkeypatch):
    trainer = _make_trainer()
    actor_worker = MagicMock()
    observed_reference_flags = []

    def _parent_setup(self):
        observed_reference_flags.append(self.use_reference_policy)
        self.actor_rollout_wg = actor_worker

    monkeypatch.setattr(PPOTrainerSync, "_setup", _parent_setup)

    trainer._setup()

    assert observed_reference_flags == [False]
    assert trainer.use_reference_policy is True
    assert trainer.ref_in_actor is True
    assert trainer.ref_policy_wg is actor_worker


@pytest.mark.parametrize(
    ("use_reference_policy", "lora_rank", "lora_adapter_path"),
    [(False, 8, None), (True, 0, None)],
)
def test_setup_delegates_unchanged_without_in_actor_reference(
    monkeypatch, use_reference_policy, lora_rank, lora_adapter_path
):
    trainer = _make_trainer(
        use_reference_policy=use_reference_policy,
        lora_rank=lora_rank,
        lora_adapter_path=lora_adapter_path,
    )
    observed_reference_flags = []

    def _parent_setup(self):
        observed_reference_flags.append(self.use_reference_policy)

    monkeypatch.setattr(PPOTrainerSync, "_setup", _parent_setup)

    trainer._setup()

    assert observed_reference_flags == [use_reference_policy]
    assert trainer.use_reference_policy is use_reference_policy


def test_setup_restores_reference_flag_when_parent_setup_fails(monkeypatch):
    trainer = _make_trainer()

    def _parent_setup(self):
        assert self.use_reference_policy is False
        raise RuntimeError("setup failed")

    monkeypatch.setattr(PPOTrainerSync, "_setup", _parent_setup)

    with pytest.raises(RuntimeError, match="setup failed"):
        trainer._setup()

    assert trainer.use_reference_policy is True
    assert not hasattr(trainer, "ref_policy_wg")
