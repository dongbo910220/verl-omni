#!/usr/bin/env bash
# Qwen3-TTS full-parameter GRPO e2e smoke: real 0.6B model, two updates.
# This validates execution only; the in-process CPU reward is not a quality reward.

set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:${CPATH}}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-2}"
[[ "${NUM_GPUS}" == "2" ]] || { echo "Qwen3-TTS smoke requires exactly two GPUs" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/qwen3_tts_grpo_smoke_${USER:-user}_$$}"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/output}"
mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"

if ! "${PYTHON_BIN}" -c 'from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration; import onnxruntime, soundfile, librosa, sox'; then
    uv pip install --python "${PYTHON_BIN}" -e ".[tts]"
    uv pip install --python "${PYTHON_BIN}" --force-reinstall --no-deps \
        "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git@00969daa8064e23adc9e5f52cdf20cf247f94159"
fi

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    MODEL_PATH="$("${PYTHON_BIN}" -c \
        "from huggingface_hub import snapshot_download; print(snapshot_download('${MODEL_REPO}'))")"
fi
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Invalid MODEL_PATH: ${MODEL_PATH}" >&2; exit 2; }

"${PYTHON_BIN}" tests/special_e2e/create_dummy_qwen3_tts_grpo_data.py --output-dir "${DATA_DIR}"

MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${DATA_DIR}/train.parquet" \
VAL_FILE="${DATA_DIR}/validation.parquet" \
SPK_EMBED_PATH="${DATA_DIR}/speaker.json" \
SCORER_URL="http://unused.invalid/score" \
OUTPUT_DIR="${OUTPUT_DIR}" \
NUM_GPUS="${NUM_GPUS}" \
TOTAL_TRAINING_STEPS=2 \
TEST_FREQ=-1 \
SAVE_FREQ=-1 \
RESUME_MODE=disable \
PYTHON_BIN="${PYTHON_BIN}" \
bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    "reward.custom_reward_function.path=${REPO_ROOT}/tests/special_e2e/qwen3_tts_dummy_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.log_val_generations=0 \
    "$@"

grep -q "training/global_step.*2" "${OUTPUT_DIR}/train.log"
echo "Qwen3-TTS GRPO e2e smoke passed; artifacts: ${WORK_DIR}"
