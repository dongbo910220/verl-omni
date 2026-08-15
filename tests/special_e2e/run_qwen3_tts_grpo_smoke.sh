#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

# Real-model two-GPU smoke for rollout -> reward -> update -> checkpoint.
# Run once with TOTAL_STEPS=1, then again with TOTAL_STEPS=2 RESUME_MODE=auto
# to exercise checkpoint restoration.

set -euo pipefail

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export TOKENIZERS_PARALLELISM=false
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_USE_FLASHINFER_SAMPLER=0
export WANDB_MODE=offline

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
SENSEVOICE_MODEL="${SENSEVOICE_MODEL:?Set SENSEVOICE_MODEL}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE}"
SPK_EMBED_PATH="${SPK_EMBED_PATH:?Set SPK_EMBED_PATH}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set a unique OUTPUT_ROOT}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
RESUME_MODE="${RESUME_MODE:-disable}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 -m verl_omni.trainer.main_omni \
    --config-path="${REPO_ROOT}/examples/grpo_trainer/qwen3_tts/config" \
    --config-name=qwen3_tts_aishell3_grpo \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    data.train_batch_size=2 \
    data.val_max_samples=2 \
    data.max_response_length=120 \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.override_config.tts_spk_embed_path=${SPK_EMBED_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    "reward.custom_reward_function.reward_kwargs.sensevoice_model=${SENSEVOICE_MODEL}" \
    "reward.custom_reward_function.reward_kwargs.normalizer_cache_dir=${OUTPUT_ROOT}/zh_tn_cache" \
    reward.custom_reward_function.reward_kwargs.max_asr_duration_s=12.0 \
    "reward.audio.dump_dir=${OUTPUT_ROOT}/audio" \
    reward.audio.dump_validation_only=false \
    trainer.val_before_train=false \
    trainer.log_val_generations=2 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    "trainer.total_training_steps=${TOTAL_STEPS}" \
    trainer.max_actor_ckpt_to_keep=2 \
    "trainer.default_local_dir=${OUTPUT_ROOT}/checkpoints" \
    "trainer.validation_data_dir=${OUTPUT_ROOT}/validation" \
    "trainer.rollout_data_dir=${OUTPUT_ROOT}/rollout" \
    "trainer.resume_mode=${RESUME_MODE}" \
    trainer.logger='[console]' \
    "$@"
