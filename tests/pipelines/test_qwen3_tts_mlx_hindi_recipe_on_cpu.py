# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0
"""Static contracts for the published MLX Hindi GRPO reproduction recipe."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "examples/grpo_trainer/qwen3_tts/config/qwen3_tts_indicvoices_hindi_grpo.yaml"
LAUNCHER = ROOT / "examples/grpo_trainer/qwen3_tts/run_qwen3_tts_indicvoices_hindi_grpo.sh"


def test_recipe_preserves_published_training_contract_and_complete_validation():
    config = yaml.safe_load(CONFIG.read_text())
    data = config["data"]
    model = config["actor_rollout_ref"]["model"]
    actor = config["actor_rollout_ref"]["actor"]
    rollout = config["actor_rollout_ref"]["rollout"]
    ref = config["actor_rollout_ref"]["ref"]
    reward = config["reward"]["custom_reward_function"]["reward_kwargs"]
    trainer = config["trainer"]

    assert data["train_batch_size"] == 4
    assert data["val_max_samples"] == 100
    assert data["max_response_length"] == 240
    assert model["lora_rank"] == 8
    assert model["lora_alpha"] == 16
    assert model["lora_dropout"] == 0.05
    assert model["lora"]["merge"] is True
    assert model["reference_adapter_name"] == "sft_reference"
    assert model["override_config"]["tts_language"] == "Auto"
    assert model["override_config"]["tts_spk_embed_path"] is None
    assert model["override_config"]["tts_replay_layout"] == "interleaved"
    assert model["override_config"]["tts_policy_logprobs_mode"] == "${actor_rollout_ref.rollout.logprobs_mode}"
    assert actor["optim"]["lr"] == 5e-6
    assert actor["optim"]["lr_warmup_steps"] == 10
    assert actor["optim"]["weight_decay"] == 0.01
    assert actor["kl_loss_coef"] == 0.08
    assert actor["kl_loss_type"] == "mlx_k3"
    assert actor["fsdp_config"]["model_dtype"] == "float32"
    assert ref["fsdp_config"]["model_dtype"] == "float32"
    assert rollout["n"] == 4
    assert rollout["temperature"] == 0.9
    assert rollout["top_p"] == 0.95
    assert rollout["top_k"] == 50
    assert rollout["logprobs_mode"] == "raw_logprobs"
    assert reward["language"] == "hi"
    assert reward["length_weight"] == 0.5
    assert trainer["log_val_generations"] == 100
    assert trainer["test_freq"] == 100
    assert trainer["save_freq"] == 50
    assert trainer["max_actor_ckpt_to_keep"] >= 8


def test_launcher_keeps_full_validation_and_fail_closed_stages():
    launcher = LAUNCHER.read_text()

    assert '[[ "$(read_rows "${VAL_FILE}")" == "100" ]]' in launcher
    assert '"trainer.log_val_generations=1024"' in launcher
    assert '"trainer.total_training_steps=10"' in launcher
    assert '"trainer.total_training_steps=5"' in launcher
    assert '"trainer.total_training_steps=200"' in launcher
    assert '"trainer.total_training_steps=400"' in launcher
    assert '"trainer.save_freq=50"' in launcher
    assert "--require-lr-zero" in launcher
    assert "--min-steps 200" in launcher
    assert "validation/0.jsonl" in launcher
    assert "validation/200.jsonl" in launcher
    assert "validation/400.jsonl" in launcher
