# Qwen3-TTS GRPO

This recipe trains the codec-0 policy of `Qwen3-TTS-12Hz-0.6B-Base` with
GRPO and a single Chinese intelligibility reward. It uses public AISHELL-3
text, decodes every rollout through Qwen3-TTS code2wav, transcribes the
waveform with SenseVoiceSmall, and scores tone-pinyin error rate as:

```text
reward = clip(1 - tanh(3 * pinyin_error_rate), 0, 1)
```

Valid waveforms are scored only by tone-pinyin error. Malformed, silent, or
runaway audio receives zero; duration, silence, repetition, and termination are
also logged as health diagnostics and are not added as weighted reward terms.

## Implementation

The policy trajectory is codec codebook 0. The other 15 codebooks and the
waveform remain rollout fields used by the actor and reward:

```text
AISHELL-3 text
  -> vLLM-Omni Qwen3-TTS Talker (codec-0 policy + residual codebooks)
  -> Qwen3-TTS code2wav
  -> SenseVoice tone-pinyin reward
  -> GRPO advantage
  -> FSDP Talker update
  -> stage-0 weight sync
```

The actor trains `talker.model.*` and `talker.codec_head.*`. Speaker encoder,
speech tokenizer, and code2wav are rollout-only components. Codec alignment
fails closed unless every recoverable codec-0 row exactly matches the sampled
policy prefix.

Every prompt-group candidate receives a distinct deterministic seed for both
codec-0 and residual-codebook sampling. Training seeds include the global step;
evaluation seeds do not, so the fixed validation sample remains paired across
checkpoints while a multi-candidate Gate still produces distinct candidates.

## Install

Install the engine stack before the training dependencies:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[tts,train,dev]"
uv pip install qwen-tts==0.1.1 --no-deps
python examples/grpo_trainer/qwen3_tts/patches/patch_qwen_tts_tf5.py
```

`qwen-tts` is installed without dependencies because its Transformers pin is
incompatible with the repository's vLLM stack. The idempotent compatibility
patch adapts its modeling files to the Transformers version installed by this
repository.

## Data

Download AISHELL-3 metadata from the pinned public revision recorded by
`prepare_aishell3_text.py`, then create leakage-free train, validation, and
Gate splits:

```bash
python examples/grpo_trainer/qwen3_tts/data_process/prepare_aishell3_text.py \
  --metadata /path/to/aishell-3.jsonl \
  --output-dir /path/to/aishell3_grpo

python examples/grpo_trainer/qwen3_tts/data_process/derive_aishell3_long_form.py \
  --train /path/to/aishell3_grpo/train.parquet \
  --output-dir /path/to/aishell3_grpo
```

The fixed validation set always contains 100 rows. The long-form training set
uses only the input-side rule `text length >= 18`; its 256-row Gate is balanced
over four predefined length buckets. Neither selection uses Base-model errors.
Both scripts emit manifests and hashes.

Select one public AISHELL-3 reference clip and precompute one fixed speaker
embedding:

```bash
python examples/grpo_trainer/qwen3_tts/data_process/precompute_speaker_embedding.py \
  --model /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --reference-audio /path/to/reference.wav \
  --output /path/to/speaker.json
```

AISHELL-3 data is not redistributed by this repository. Check the dataset's
license before using it outside research.

## Run

Set the common paths once:

```bash
export MODEL_PATH=/path/to/Qwen3-TTS-12Hz-0.6B-Base
export SENSEVOICE_MODEL=/path/to/SenseVoiceSmall
export TRAIN_FILE=/path/to/aishell3_grpo/train_long_form.parquet
export VAL_FILE=/path/to/aishell3_grpo/validation.parquet
export GATE_FILE=/path/to/aishell3_grpo/gate_long_form.parquet
export SPK_EMBED_PATH=/path/to/speaker.json
```

Run the controls and training from the same Base checkpoint:

```bash
RUN_STAGE=gate bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_aishell3_grpo.sh
RUN_STAGE=lr0 bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_aishell3_grpo.sh
RUN_STAGE=smoke-safe bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_aishell3_grpo.sh
RUN_STAGE=smoke-reference bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_aishell3_grpo.sh
RUN_STAGE=train bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_aishell3_grpo.sh
```

The default formal run uses two GPUs, `B=16`, `G=4`, no KL, `lr=2e-7`, and
100 updates. It validates the same complete 100 rows at step
`0/20/40/60/80/100`; the launcher rejects any validation file whose row count
is not exactly 100. To resume through step 200 while retaining step-20
checkpoints:

```bash
RUN_STAGE=train RESUME_MODE=auto TOTAL_TRAINING_STEPS=200 \
  bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_aishell3_grpo.sh
```

## Validation checks

Before a paid run, require the 256 x 4 signal check to have at least 99% valid
waveforms, at least 60% prompt groups with reward variance, and at most 30%
all-perfect groups. Keep rollout log-prob calculation enabled and require every
comparison to be finite and valid. verl's general guidance expects mean
absolute probability difference below `0.005` and treats values above `0.01`
as an inference precision issue.

The current Qwen3-TTS path does not meet the stricter boundary: repeated H800
runs on the same 1.7B model and data measured `0.0075-0.0077`, with Pearson
correlation above `0.9988`. PR #282 reports Pearson `0.9993` but does not publish
its mean probability difference. This phase records that numerical gap but does
not use `<0.005` as a submission or execution gate; convergence is determined
from the fixed held-out validation rather than this diagnostic alone.

An earlier development run completed 200 updates and complete step-20
validation on two RTX 5090 GPUs, but it predated the per-candidate seed fix and
its held-out reward did not improve monotonically. The corrected path has
completed nonzero updates plus model/optimizer/scheduler save and restore on
the same hardware. Treat these runs as implementation evidence, not as a claim
that this single reward already proves stable TTS quality gains.
