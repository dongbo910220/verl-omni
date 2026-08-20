#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

RUN_STAGE="${RUN_STAGE:-gate}"
NUM_GPUS="${NUM_GPUS:-2}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Recent vLLM wheels can depend on CUDA runtime libraries installed as Python
# packages. Those directories are not always on the loader path in cloud images.
mapfile -t PYTHON_GPU_LIBRARY_DIRS < <(
    "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import site

for root in site.getsitepackages():
    site_packages = Path(root)
    candidates = [site_packages / "torch" / "lib"]
    candidates.extend(sorted((site_packages / "nvidia").glob("*/lib")))
    for candidate in candidates:
        if candidate.is_dir():
            print(candidate)
PY
)
if ((${#PYTHON_GPU_LIBRARY_DIRS[@]})); then
    PYTHON_GPU_LIBRARY_PATH="$(printf '%s:' "${PYTHON_GPU_LIBRARY_DIRS[@]}")"
    export LD_LIBRARY_PATH="${PYTHON_GPU_LIBRARY_PATH%:}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONHASHSEED="${EXPERIMENT_SEED}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export TOKENIZERS_PARALLELISM=false
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-verl_omni}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the pinned local Qwen3-TTS 0.6B snapshot}"
WHISPER_MODEL="${WHISPER_MODEL:?Set WHISPER_MODEL to the pinned local Whisper large-v3-turbo snapshot}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:?Set SFT_ADAPTER_PATH to the converted published SFT adapter}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the fixed 863-row train.parquet}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to the fixed 100-row validation.parquet}"
GATE_FILE="${GATE_FILE:?Set GATE_FILE to the fixed 256-row gate.parquet}"
VERL_REPO="${VERL_REPO:?Set VERL_REPO to verl at revision 8a694930275061f52ebd538c906ef8819af56dbd}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_PROCESS_DIR="${SCRIPT_DIR}/data_process"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/qwen3_tts_indicvoices_hindi_grpo}"
case "${RUN_STAGE}" in
    gate|lr0)
        DEFAULT_OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_STAGE}"
        ;;
    smoke|resume-smoke)
        DEFAULT_OUTPUT_ROOT="${OUTPUT_BASE}/smoke"
        ;;
    train|extend)
        DEFAULT_OUTPUT_ROOT="${OUTPUT_BASE}/formal-seed-${EXPERIMENT_SEED}"
        ;;
    *)
        printf 'RUN_STAGE must be gate, lr0, smoke, resume-smoke, train, or extend; got %s\n' "${RUN_STAGE}" >&2
        exit 2
        ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
mkdir -p "${OUTPUT_ROOT}/analysis" "${OUTPUT_ROOT}/logs"

for path in \
    "${MODEL_PATH}" \
    "${WHISPER_MODEL}" \
    "${SFT_ADAPTER_PATH}" \
    "${TRAIN_FILE}" \
    "${VAL_FILE}" \
    "${GATE_FILE}" \
    "${VERL_REPO}"; do
    [[ -e "${path}" ]] || { printf 'Required path does not exist: %s\n' "${path}" >&2; exit 2; }
done

read_rows() {
    "${PYTHON_BIN}" -c 'import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))' "$1"
}

[[ "$(read_rows "${TRAIN_FILE}")" == "863" ]] || { printf 'TRAIN_FILE must have exactly 863 rows.\n' >&2; exit 2; }
[[ "$(read_rows "${VAL_FILE}")" == "100" ]] || { printf 'VAL_FILE must have exactly 100 rows.\n' >&2; exit 2; }
[[ "$(read_rows "${GATE_FILE}")" == "256" ]] || { printf 'GATE_FILE must have exactly 256 rows.\n' >&2; exit 2; }

PREFLIGHT_REPORT="${OUTPUT_ROOT}/analysis/preflight-${RUN_STAGE}.json"
"${PYTHON_BIN}" "${DATA_PROCESS_DIR}/preflight_mlx_hindi_grpo.py" \
    --repo "${REPO_ROOT}" \
    --verl-repo "${VERL_REPO}" \
    --model "${MODEL_PATH}" \
    --whisper "${WHISPER_MODEL}" \
    --adapter "${SFT_ADAPTER_PATH}" \
    --train "${TRAIN_FILE}" \
    --validation "${VAL_FILE}" \
    --gate "${GATE_FILE}" \
    --num-gpus "${NUM_GPUS}" \
    --output "${PREFLIGHT_REPORT}"

EXTRA_OVERRIDES=()
RESUME_MODE="disable"
case "${RUN_STAGE}" in
    gate)
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
            "reward.audio.dump_validation_only=false"
            "trainer.val_before_train=false"
            "trainer.total_training_steps=10"
            "trainer.save_freq=10"
            "trainer.test_freq=-1"
        )
        ;;
    smoke)
        EXTRA_OVERRIDES+=(
            "reward.audio.dump_validation_only=false"
            "trainer.val_before_train=false"
            "trainer.total_training_steps=5"
            "trainer.save_freq=5"
            "trainer.test_freq=-1"
        )
        ;;
    resume-smoke)
        RESUME_MODE="auto"
        EXTRA_OVERRIDES+=(
            "reward.audio.dump_validation_only=false"
            "trainer.val_before_train=false"
            "trainer.total_training_steps=6"
            "trainer.save_freq=1"
            "trainer.test_freq=-1"
        )
        ;;
    train)
        TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
        [[ "${TOTAL_TRAINING_STEPS}" == "200" ]] || {
            printf 'Initial formal run is fixed at 200 steps; got %s.\n' "${TOTAL_TRAINING_STEPS}" >&2
            exit 2
        }
        EXTRA_OVERRIDES+=(
            "trainer.val_before_train=true"
            "trainer.total_training_steps=200"
            "trainer.save_freq=50"
            "trainer.test_freq=100"
        )
        ;;
    extend)
        RESUME_MODE="auto"
        EXTRA_OVERRIDES+=(
            "trainer.val_before_train=false"
            "trainer.total_training_steps=400"
            "trainer.save_freq=50"
            "trainer.test_freq=100"
        )
        ;;
esac

LOG_PATH="${OUTPUT_ROOT}/logs/${RUN_STAGE}.log"
COMMAND=(
    "${PYTHON_BIN}" -m verl_omni.trainer.main_omni
    "--config-path=${SCRIPT_DIR}/config"
    --config-name=qwen3_tts_indicvoices_hindi_grpo
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.seed=${EXPERIMENT_SEED}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.lora_adapter_path=${SFT_ADAPTER_PATH}"
    "actor_rollout_ref.actor.data_loader_seed=${EXPERIMENT_SEED}"
    "actor_rollout_ref.actor.fsdp_config.fsdp_size=${NUM_GPUS}"
    "actor_rollout_ref.actor.fsdp_config.seed=${EXPERIMENT_SEED}"
    "actor_rollout_ref.ref.fsdp_config.fsdp_size=${NUM_GPUS}"
    "actor_rollout_ref.ref.fsdp_config.seed=${EXPERIMENT_SEED}"
    "actor_rollout_ref.rollout.engine_kwargs.vllm_omni.seed=${EXPERIMENT_SEED}"
    "reward.custom_reward_function.reward_kwargs.whisper_model=${WHISPER_MODEL}"
    "reward.audio.dump_dir=${OUTPUT_ROOT}/audio"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "trainer.experiment_name=qwen3_tts_hindi_mlx_repro_${RUN_STAGE}_seed_${EXPERIMENT_SEED}"
    "trainer.default_local_dir=${OUTPUT_ROOT}/checkpoints"
    "trainer.validation_data_dir=${OUTPUT_ROOT}/validation"
    "trainer.rollout_data_dir=${OUTPUT_ROOT}/rollout"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.logger=[console]"
    "${EXTRA_OVERRIDES[@]}"
    "$@"
)

printf 'Launching stage %s with %s GPU(s); log: %s\n' "${RUN_STAGE}" "${NUM_GPUS}" "${LOG_PATH}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"

ANALYZER="${DATA_PROCESS_DIR}/analyze_mlx_hindi_grpo.py"
case "${RUN_STAGE}" in
    gate)
        "${PYTHON_BIN}" "${ANALYZER}" signal \
            --input "${OUTPUT_ROOT}/validation/0.jsonl" \
            --output "${OUTPUT_ROOT}/analysis/signal-gate.json"
        ;;
    lr0)
        "${PYTHON_BIN}" "${ANALYZER}" consistency \
            --log "${LOG_PATH}" \
            --min-steps 10 \
            --require-lr-zero \
            --output "${OUTPUT_ROOT}/analysis/lr0-consistency.json"
        ;;
    smoke)
        "${PYTHON_BIN}" "${ANALYZER}" health \
            --log "${LOG_PATH}" \
            --min-steps 5 \
            --output "${OUTPUT_ROOT}/analysis/smoke-health.json"
        ;;
    resume-smoke)
        "${PYTHON_BIN}" "${ANALYZER}" health \
            --log "${LOG_PATH}" \
            --min-steps 1 \
            --output "${OUTPUT_ROOT}/analysis/resume-smoke-health.json"
        [[ "$(tr -d '[:space:]' < "${OUTPUT_ROOT}/checkpoints/latest_checkpointed_iteration.txt")" == "6" ]] || {
            printf 'Resume smoke did not produce the step-6 checkpoint marker.\n' >&2
            exit 2
        }
        ;;
    train)
        "${PYTHON_BIN}" "${ANALYZER}" health \
            --log "${LOG_PATH}" \
            --min-steps 200 \
            --output "${OUTPUT_ROOT}/analysis/train-200-health.json"
        "${PYTHON_BIN}" "${ANALYZER}" eval \
            --base "${OUTPUT_ROOT}/validation/0.jsonl" \
            --candidate "${OUTPUT_ROOT}/validation/200.jsonl" \
            --output "${OUTPUT_ROOT}/analysis/eval-step-0-vs-200.json"
        ;;
    extend)
        "${PYTHON_BIN}" "${ANALYZER}" health \
            --log "${LOG_PATH}" \
            --min-steps 200 \
            --output "${OUTPUT_ROOT}/analysis/train-201-400-health.json"
        "${PYTHON_BIN}" "${ANALYZER}" eval \
            --base "${OUTPUT_ROOT}/validation/0.jsonl" \
            --candidate "${OUTPUT_ROOT}/validation/400.jsonl" \
            --output "${OUTPUT_ROOT}/analysis/eval-step-0-vs-400.json"
        ;;
esac
