# Qwen3-TTS GRPO with an audio reward

This example full-parameter tunes the codec-0 policy of
`Qwen/Qwen3-TTS-12Hz-0.6B-Base`. It uses verl's stock GRPO advantage,
vanilla PPO policy loss, and optional direct reference-model KL. The other
15 codec codebooks and code2wav stage remain frozen but are retained so every
candidate can be decoded and scored as audio.

The launcher follows the V1 omni-model integration guide: it calls
`verl_omni.trainer.main_omni` and expresses the recipe as CLI overrides on the
standard `omni_trainer` config, without a model-specific Trainer or config tree.

SpeechJudge-BTRM is one possible scorer. It runs behind the generic audio HTTP
client because its published Transformers environment conflicts with the
Transformers 5.x vLLM stack. The Trainer and `AudioRewardManager` do not contain
SpeechJudge-specific branches, candidate masks, ASR gates, or custom losses.

## Install

Install the engine before the training stack:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[tts,train,dev]"
uv pip install qwen-tts==0.1.1 --no-deps
```

The Qwen3-TTS adapter contains a bounded import compatibility layer for
`qwen-tts==0.1.1` on the repository's Transformers 5.x stack. It does not edit
site-packages. The system `sox` executable is also required by qwen-tts.

## Data

Training and validation parquet rows use the normal verl format:

```python
{
    "data_source": "tts",
    "prompt": [{"role": "user", "content": "Text to synthesize"}],
    "reward_model": {"style": "model", "ground_truth": "Text to synthesize"},
    "extra_info": {"id": "stable-id", "split": "train"},
}
```

Use disjoint prompts. The default recipe evaluates the same complete 100-row
validation parquet at step 0 and every 20 updates. Give validation rows fixed
`extra_info.generation_seed` values to keep candidate sampling paired across
checkpoints.

The concatenated replay layout also requires one fixed speaker embedding JSON.
Generate it once with the official Qwen3-TTS Base model's
`extract_speaker_embedding` API from a 24 kHz reference recording, then reuse
the same file for the entire run.

## Audio scorer protocol

The configured endpoint receives one JSON request per candidate:

```json
{
  "protocol_version": "1",
  "waveform_f32_base64": "...",
  "num_samples": 24000,
  "sample_rate": 24000,
  "prompt": "Text to synthesize",
  "metadata": {"id": "stable-id"}
}
```

It must return `{"score": 1.25}` and may include additional scalar metrics.
The client retries only transient network, timeout, HTTP 408/429, and 5xx
failures. Missing, malformed, or non-finite results stop the run instead of
being converted to a valid zero reward.

For SpeechJudge-BTRM, deploy the official
[`AmphionTeam/SpeechJudge`](https://github.com/AmphionTeam/SpeechJudge) code and
[`RMSnow/SpeechJudge-BTRM`](https://huggingface.co/RMSnow/SpeechJudge-BTRM)
checkpoint in a separate environment, then expose its pointwise score through
this protocol. Pin the SpeechJudge source revision and runtime versions in the
service deployment. SpeechJudge-BTRM is licensed CC-BY-NC-4.0.

## Train

```bash
MODEL_PATH=/path/to/Qwen3-TTS-12Hz-0.6B-Base \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/fixed-validation-100.parquet \
SPK_EMBED_PATH=/path/to/speaker.json \
SCORER_URL=http://scorer-host:18080/score \
OUTPUT_DIR=/path/to/output \
bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh
```

The example defaults are `B=4`, `G=8`, `lr=2e-7`, direct `low_var_kl` with
coefficient `0.12`, two GPUs, and 500 updates. These are recipe values, not
algorithm requirements. `norm_adv_by_std_in_grpo` remains at the upstream
default.

For a two-update implementation smoke test:

```bash
TOTAL_TRAINING_STEPS=2 TEST_FREQ=-1 SAVE_FREQ=1 RESUME_MODE=disable \
OUTPUT_DIR=outputs/qwen3_tts_grpo_smoke \
bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh \
  trainer.val_before_train=false trainer.log_val_generations=0
```

This smoke proves rollout, finite audio reward, optimizer update, weight sync,
and checkpoint wiring only. It is not evidence that GRPO improves held-out
speech quality; that requires the complete fixed-validation curve and paired
human listening evaluation.
