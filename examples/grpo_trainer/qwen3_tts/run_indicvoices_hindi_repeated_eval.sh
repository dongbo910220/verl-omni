#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
NATIVE_PYTHON="${NATIVE_PYTHON:?Set NATIVE_PYTHON to the isolated qwen-tts Python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the pinned Qwen3-TTS snapshot}"
WHISPER_MODEL="${WHISPER_MODEL:?Set WHISPER_MODEL to the pinned Whisper snapshot}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:?Set SFT_ADAPTER_PATH to the verified SFT adapter}"
TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to the fixed 863-row train.parquet}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to the fixed 100-row validation.parquet}"
FORMAL_OUTPUT_DIR="${FORMAL_OUTPUT_DIR:?Set FORMAL_OUTPUT_DIR to the completed 200-step run}"
OUTPUT_BASE="${OUTPUT_BASE:-${FORMAL_OUTPUT_DIR%/formal-seed-*}}"
POLICIES="${POLICIES:-base 160 200}"
SEED_OFFSETS="${SEED_OFFSETS:-1000000 2000000 3000000}"

[[ "${NUM_GPUS}" == "2" ]] || { printf 'Repeated evaluation requires exactly two GPUs.\n' >&2; exit 2; }
[[ -f "${FORMAL_OUTPUT_DIR}/checkpoints/step-0200.pt" ]] || {
    printf 'Completed formal checkpoint is missing under %s\n' "${FORMAL_OUTPUT_DIR}" >&2
    exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONHASHSEED="${EXPERIMENT_SEED}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for policy in ${POLICIES}; do
    if [[ "${policy}" == "base" ]]; then
        checkpoint_step=0
        resume_args=()
    else
        checkpoint_step="${policy}"
        checkpoint_path="${FORMAL_OUTPUT_DIR}/checkpoints/step-$(printf '%04d' "${checkpoint_step}").pt"
        [[ -f "${checkpoint_path}" ]] || {
            printf 'Checkpoint is missing: %s\n' "${checkpoint_path}" >&2
            exit 2
        }
        resume_args=(--resume)
    fi

    for offset in ${SEED_OFFSETS}; do
        if [[ "${policy}" == "base" ]]; then
            label=base
        else
            label="step-$(printf '%04d' "${checkpoint_step}")"
        fi
        output_dir="${OUTPUT_BASE}/repeated-eval/${label}/offset-${offset}"
        summary="${output_dir}/validation/step-$(printf '%04d' "${checkpoint_step}")/summary.json"
        results="${output_dir}/validation/step-$(printf '%04d' "${checkpoint_step}")/results.jsonl"
        if [[ -f "${summary}" && "$(wc -l < "${results}")" == "100" ]]; then
            printf 'Skipping completed evaluation: %s\n' "${output_dir}"
            continue
        fi
        [[ ! -e "${output_dir}" ]] || {
            printf 'Refusing incomplete existing output: %s\n' "${output_dir}" >&2
            exit 2
        }
        mkdir -p "${output_dir}/logs"
        if [[ "${policy}" != "base" ]]; then
            mkdir -p "${output_dir}/checkpoints"
            ln "${checkpoint_path}" "${output_dir}/checkpoints/latest.pt"
        fi

        command=(
            "${NATIVE_PYTHON}" -m torch.distributed.run
            --standalone
            "--nproc_per_node=${NUM_GPUS}"
            "${SCRIPT_DIR}/train_indicvoices_hindi_native_grpo.py"
            --stage=eval
            --eval-only
            --total-steps=0
            --learning-rate=0
            "--model=${MODEL_PATH}"
            "--sft-adapter=${SFT_ADAPTER_PATH}"
            "--whisper-model=${WHISPER_MODEL}"
            "--train-file=${TRAIN_FILE}"
            "--validation-file=${VAL_FILE}"
            "--output-dir=${output_dir}"
            "--seed=${EXPERIMENT_SEED}"
            "--validation-seed-offset=${offset}"
            --save-validation-audio
            "${resume_args[@]}"
        )
        printf 'Evaluating %s with seed offset %s\n' "${label}" "${offset}"
        (cd "${REPO_ROOT}" && "${command[@]}") 2>&1 | tee "${output_dir}/logs/eval.log"
        [[ -f "${summary}" && "$(wc -l < "${results}")" == "100" ]] || {
            printf 'Evaluation did not produce a complete fixed-100 result: %s\n' "${output_dir}" >&2
            exit 2
        }
    done
done
