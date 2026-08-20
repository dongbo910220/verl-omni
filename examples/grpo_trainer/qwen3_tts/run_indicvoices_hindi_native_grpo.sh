#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

RUN_STAGE="${RUN_STAGE:-lr0}"
NUM_GPUS="${NUM_GPUS:-2}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
NATIVE_PYTHON="${NATIVE_PYTHON:?Set NATIVE_PYTHON to the isolated qwen-tts 0.1.1 Python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the pinned local Qwen3-TTS 0.6B snapshot}"
WHISPER_MODEL="${WHISPER_MODEL:?Set WHISPER_MODEL to the pinned local Whisper snapshot}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:?Set SFT_ADAPTER_PATH to the verified PEFT SFT adapter}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the fixed 863-row train.parquet}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to the fixed 100-row validation.parquet}"

[[ "${NUM_GPUS}" == "2" ]] || { printf 'This reproduction requires exactly two GPUs.\n' >&2; exit 2; }
for path in "${NATIVE_PYTHON}" "${MODEL_PATH}" "${WHISPER_MODEL}" "${SFT_ADAPTER_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    [[ -e "${path}" ]] || { printf 'Required path does not exist: %s\n' "${path}" >&2; exit 2; }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/qwen3_tts_indicvoices_hindi_native_grpo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONHASHSEED="${EXPERIMENT_SEED}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMON=(
    "${NATIVE_PYTHON}" -m torch.distributed.run
    --standalone
    "--nproc_per_node=${NUM_GPUS}"
    "${SCRIPT_DIR}/train_indicvoices_hindi_native_grpo.py"
    "--model=${MODEL_PATH}"
    "--sft-adapter=${SFT_ADAPTER_PATH}"
    "--whisper-model=${WHISPER_MODEL}"
    "--train-file=${TRAIN_FILE}"
    "--validation-file=${VAL_FILE}"
    "--seed=${EXPERIMENT_SEED}"
    --group-size=4
    --rollout-microbatch-size=2
    --prompts-per-step=4
    --max-new-tokens=240
    --temperature=0.9
    --top-p=0.95
    --top-k=50
    --kl-beta=0.08
    --kl-clip=10
    --weight-decay=0.01
    --eval-every=20
    --save-every=20
)

case "${RUN_STAGE}" in
    lr0)
        OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/lr0-seed-${EXPERIMENT_SEED}}"
        ARGS=(--stage=lr0 --total-steps=10 --learning-rate=0 --warmup-steps=0)
        ;;
    smoke)
        OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/smoke-seed-${EXPERIMENT_SEED}}"
        ARGS=(--stage=smoke --total-steps=5 --learning-rate=5e-6 --warmup-steps=0)
        ;;
    resume-smoke)
        OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/smoke-seed-${EXPERIMENT_SEED}}"
        ARGS=(--stage=smoke --total-steps=6 --learning-rate=5e-6 --warmup-steps=0 --resume)
        ;;
    train)
        OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/formal-seed-${EXPERIMENT_SEED}}"
        ARGS=(
            --stage=train
            --total-steps=200
            --learning-rate=5e-6
            --warmup-steps=10
            --eval-before-train
            --save-validation-audio
        )
        ;;
    extend)
        OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/formal-seed-${EXPERIMENT_SEED}}"
        ARGS=(
            --stage=extend
            --total-steps=400
            --learning-rate=5e-6
            --warmup-steps=10
            --resume
            --save-validation-audio
        )
        ;;
    *)
        printf 'RUN_STAGE must be lr0, smoke, resume-smoke, train, or extend; got %s\n' "${RUN_STAGE}" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_DIR}/logs"
LOG_PATH="${OUTPUT_DIR}/logs/${RUN_STAGE}.log"
COMMAND=("${COMMON[@]}" "--output-dir=${OUTPUT_DIR}" "${ARGS[@]}" "$@")
printf 'Launching native stage %s; log: %s\n' "${RUN_STAGE}" "${LOG_PATH}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
cd "${REPO_ROOT}"
"${COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"
