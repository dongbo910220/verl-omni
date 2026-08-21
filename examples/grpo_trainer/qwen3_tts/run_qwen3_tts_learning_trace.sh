#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

set -euo pipefail

TRACE_ROOT="${TRACE_ROOT:?Set TRACE_ROOT to a persistent evidence directory}"
TRACE_RUN_ID="${TRACE_RUN_ID:?Set TRACE_RUN_ID to one identifier shared by both phases}"
TRACE_PHASE="${TRACE_PHASE:?Set TRACE_PHASE to phase-a or phase-b}"
TRACE_EXPECTED_STEP="${TRACE_EXPECTED_STEP:?Set TRACE_EXPECTED_STEP to 1 or 2}"

case "${TRACE_PHASE}:${TRACE_EXPECTED_STEP}" in
    phase-a:1|phase-b:2) ;;
    *)
        printf 'Expected phase-a:1 or phase-b:2, got %s:%s\n' "${TRACE_PHASE}" "${TRACE_EXPECTED_STEP}" >&2
        exit 2
        ;;
esac

mkdir -p "${TRACE_ROOT}"
export VERL_OMNI_LEARNING_TRACE_DIR="${TRACE_ROOT}"
export VERL_OMNI_LEARNING_TRACE_RUN_ID="${TRACE_RUN_ID}"
export VERL_OMNI_LEARNING_TRACE_PHASE="${TRACE_PHASE}"
export VERL_OMNI_LEARNING_TRACE_EXPECTED_STEP="${TRACE_EXPECTED_STEP}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_qwen3_tts_grpo.sh" "$@"
