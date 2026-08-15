#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONHASHSEED="${EXPERIMENT_SEED:-42}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export TOKENIZERS_PARALLELISM=false
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_USE_FLASHINFER_SAMPLER=0
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-TTS 0.6B directory}"
SENSEVOICE_MODEL="${SENSEVOICE_MODEL:?Set SENSEVOICE_MODEL to a local SenseVoiceSmall directory}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to train_long_form.parquet}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to the fixed 100-row validation.parquet}"
SPK_EMBED_PATH="${SPK_EMBED_PATH:?Set SPK_EMBED_PATH to the fixed speaker JSON}"
RUN_STAGE="${RUN_STAGE:-train}"
NUM_GPUS="${NUM_GPUS:-2}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qwen3_tts_aishell3_grpo/${RUN_STAGE}}"
NORMALIZER_CACHE_DIR="${NORMALIZER_CACHE_DIR:-${OUTPUT_ROOT}/zh_tn_cache}"
RESUME_MODE="${RESUME_MODE:-disable}"

for path in "${MODEL_PATH}" "${SENSEVOICE_MODEL}" "${TRAIN_FILE}" "${VAL_FILE}" "${SPK_EMBED_PATH}"; do
    [[ -e "${path}" ]] || { printf 'Required path does not exist: %s\n' "${path}" >&2; exit 2; }
done

VAL_ROWS="$(python3 -c 'import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))' "${VAL_FILE}")"
[[ "${VAL_ROWS}" == "100" ]] || { printf 'VAL_FILE must contain exactly 100 rows, found %s\n' "${VAL_ROWS}" >&2; exit 2; }

EXTRA_OVERRIDES=()
case "${RUN_STAGE}" in
    gate)
        GATE_FILE="${GATE_FILE:?Set GATE_FILE to the fixed 256-row gate_long_form.parquet}"
        [[ -e "${GATE_FILE}" ]] || { printf 'Gate file does not exist: %s\n' "${GATE_FILE}" >&2; exit 2; }
        GATE_ROWS="$(python3 -c 'import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))' "${GATE_FILE}")"
        [[ "${GATE_ROWS}" == "256" ]] || { printf 'GATE_FILE must contain exactly 256 rows, found %s\n' "${GATE_ROWS}" >&2; exit 2; }
        EXTRA_OVERRIDES+=(
            "data.val_files=${GATE_FILE}"
            "data.val_max_samples=256"
            "actor_rollout_ref.rollout.val_kwargs.n=4"
            "trainer.log_val_generations=1024"
            "trainer.val_only=true"
            "trainer.total_training_steps=1"
            "trainer.save_freq=-1"
            "trainer.test_freq=-1"
        )
        ;;
    lr0)
        EXTRA_OVERRIDES+=(
            "actor_rollout_ref.actor.optim.lr=0.0"
            "trainer.total_training_steps=10"
            "trainer.save_freq=10"
            "trainer.test_freq=10"
        )
        ;;
    smoke-safe)
        EXTRA_OVERRIDES+=(
            "actor_rollout_ref.actor.optim.lr=2.0e-7"
            "trainer.total_training_steps=5"
            "trainer.save_freq=5"
            "trainer.test_freq=5"
        )
        ;;
    smoke-reference)
        EXTRA_OVERRIDES+=(
            "actor_rollout_ref.actor.optim.lr=1.0e-6"
            "trainer.total_training_steps=5"
            "trainer.save_freq=5"
            "trainer.test_freq=5"
        )
        ;;
    train)
        LEARNING_RATE="${LEARNING_RATE:-2e-7}"
        TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
        EXTRA_OVERRIDES+=(
            "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
            "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
            "trainer.save_freq=20"
            "trainer.test_freq=20"
        )
        ;;
    *)
        printf 'RUN_STAGE must be gate, lr0, smoke-safe, smoke-reference, or train; got %s\n' "${RUN_STAGE}" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m verl_omni.trainer.main_omni \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_tts_aishell3_grpo \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    "data.seed=${EXPERIMENT_SEED}" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.override_config.tts_spk_embed_path=${SPK_EMBED_PATH}" \
    "actor_rollout_ref.actor.data_loader_seed=${EXPERIMENT_SEED}" \
    "actor_rollout_ref.actor.fsdp_config.seed=${EXPERIMENT_SEED}" \
    "actor_rollout_ref.ref.fsdp_config.seed=${EXPERIMENT_SEED}" \
    "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.seed=${EXPERIMENT_SEED}" \
    "reward.custom_reward_function.reward_kwargs.sensevoice_model=${SENSEVOICE_MODEL}" \
    "reward.custom_reward_function.reward_kwargs.normalizer_cache_dir=${NORMALIZER_CACHE_DIR}" \
    "reward.audio.dump_dir=${OUTPUT_ROOT}/audio" \
    "trainer.n_gpus_per_node=${NUM_GPUS}" \
    "trainer.experiment_name=qwen3_tts_aishell3_${RUN_STAGE}_seed_${EXPERIMENT_SEED}" \
    "trainer.default_local_dir=${OUTPUT_ROOT}/checkpoints" \
    "trainer.validation_data_dir=${OUTPUT_ROOT}/validation" \
    "trainer.rollout_data_dir=${OUTPUT_ROOT}/rollout" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "${EXTRA_OVERRIDES[@]}" \
    "$@"
