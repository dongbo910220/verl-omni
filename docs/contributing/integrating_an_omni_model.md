# How to Add a New Omni Model

Last updated: 08/21/2026.

This guide walks through adding a new omni (multimodal autoregressive) model to
the verl-omni training framework. It uses the Qwen3-Omni Thinker adapter as a
**running example**, not as the only valid pattern. Your model's architecture,
decomposition, and required adapter logic may differ. All adapter code lives
under [`verl_omni/pipelines/`](https://github.com/verl-project/verl-omni/tree/main/verl_omni/pipelines).

## 1. Understand the architecture

Decide which **training stage** you want to train and how the model decomposes:

- **Stage-split**: Multi-component omni models (thinker → talker → code2wav)
  train one selected autoregressive stage during RL post-training. Other
  components are stripped before FSDP wrapping to save memory. Qwen3-Omni
  trains the thinker; Qwen3-TTS trains the talker's codec-0 policy while its
  decoder remains rollout-only.
- **Encoder-frozen**: Vision/audio encoders are typically frozen during RL
  training (`freeze_vision_tower=True`). The training adapter's
  `get_strip_modules` excludes them from the trainable set if they are separate
  submodules.
- **Discrete-token**: Unlike diffusion models, omni models produce discrete
  text tokens. RL algorithms (GSPO, GRPO, RLOO) are selected through standard
  verl config fields (`actor.policy_loss.loss_mode`,
  `algorithm.adv_estimator`) — the adapter is algorithm-agnostic.

The stage-split decomposition described above is specific to Qwen3-Omni. Your
omni model may have a simpler (single-stage) or different multi-stage
architecture.

## 2. Create the training adapter

Subclass `OmniModelBase` (see
[`verl_omni/pipelines/model_base.py`](https://github.com/verl-project/verl-omni/tree/main/verl_omni/pipelines/model_base.py)
and implement these methods. Descriptions below use Qwen3-Omni as an example —
adapt each implementation to your model's architecture:

- **`get_strip_modules(model_config)`**: Return a list of submodule attribute
  names to delete before FSDP wrapping (e.g. `["talker", "code2wav",
  "code_predictor"]`). This removes inference-only stages from the trainable
  module and is called by the base `configure_model` implementation.

- **`configure_processor(model_path, model_config)`**: Load and configure the
  multimodal processor. For Qwen3-Omni, this swaps `processor.config` to
  `thinker_config`, binds `get_rope_index` (cast to `int64` to avoid bf16
  rounding) and `get_llm_pos_ids_for_vision` to the processor, and binds
  `dedup_pad_tokens` to collapse consecutive multimodal pad tokens that
  would otherwise be double-expanded in AR mode.

- **`configure_tokenizer(model_path, model_config)`**: Load the tokenizer.
  Qwen3-Omni loads `chat_template.json` from the model checkpoint if the
  tokenizer config does not contain one — a common pattern for models that
  ship the template separately.

- **`configure_model(module, model_config)`**: Called after the base-class
  stripping. Qwen3-Omni redirects `module.forward` →
  `module.thinker.forward`, swaps the embedding accessors, and sets
  `module._no_split_modules` to the correct decoder layer class for FSDP.
  This method runs before FSDP wrapping and LoRA injection.

- **`prepare_model_inputs(model_inputs, micro_batch, model_config)`**
  (optional): Add tensors retained by the rollout to the actor's forward call.
  This is needed when the policy sequence alone cannot reconstruct the exact
  teacher-forced inputs. Qwen3-TTS uses it to pass text tokens and all 16 codec
  codebooks while optimizing codec-0 log-probabilities.

Reference:
[`verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py`](../../verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py)

## 3. Create the rollout adapter

Subclass `OmniRolloutPipelineBase` (see
[`verl_omni/pipelines/model_base.py`](../../verl_omni/pipelines/model_base.py))
and implement:

- **`build_stage_configs(pipeline_mode)`**: Return a list of per-stage
  pipeline topology objects. Qwen3-Omni delegates to vLLM-Omni's frozen
  `QWEN3_OMNI_THINKER_ONLY_PIPELINE` for thinker-only training and supports
  `thinker_talker` / `full` modes for inference.

- **`rollout_flags(pipeline_mode)`**: Return per-stage flags dict. For
  thinker-only mode this is empty (text output). Multi-stage modes return
  `return_hidden_states` flags so intermediate hidden states flow between
  pipeline stages.

- **`get_pipeline_id(pipeline_mode)`**: Return the vLLM-Omni pipeline
  `model_type` string, used when auto-generating the deploy config YAML.

Optional overrides fall into four groups:

- Pipeline setup: `ensure_pipeline_registered`, `get_engine_hf_overrides`,
  `get_stage_engine_extras`, `get_worker_extension_cls`, and
  `initialize_rollout_workers`.
- Resource behavior: `weight_sync_stage_ids` and
  `supports_cache_engine_sleep`.
- Request construction: `prepare_engine_prompt`.
- Multi-stage output retention: `get_output_modalities` and
  `combine_engine_outputs`.

Their defaults preserve the existing single-output AR behavior. Override only
the hooks required by the model. For example, Qwen3-TTS synchronizes actor
weights only to its talker stage and retains both codec and waveform outputs;
the decoder stage never receives actor weights.

Reference:
[`verl_omni/pipelines/qwen3_omni/omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen3_omni/omni_rollout_adapter.py)

## 4. Register both adapters

Registration uses Python decorators at class-definition time:

```python
@OmniModelBase.register("YourArchitectureName", stage="thinker")
class YourThinkerAdapter(OmniModelBase):
    ...

@OmniRolloutPipelineBase.register("your_pipeline_name")
class YourRolloutAdapter(OmniRolloutPipelineBase):
    ...
```

The `architecture` key for `OmniModelBase` matches the HuggingFace config
`architectures[0]` value. The `model_type` key for
`OmniRolloutPipelineBase` matches the vLLM-Omni pipeline registry name.

To ensure registration fires before the trainer starts, import your adapter
module from [`verl_omni/pipelines/__init__.py`](../../verl_omni/pipelines/__init__.py).
The `VERL_USE_EXTERNAL_MODULES=verl_omni` environment variable triggers verl
to import `verl_omni`, which in turn imports the pipeline package and
activates all registrations. No `external_lib` CLI argument is needed.

## 5. Write the run script

The V1 trainer uses pure CLI overrides on `verl_omni.trainer.main_omni` with
no YAML config files or `--config-path/--config-name`:

```bash
export VERL_USE_EXTERNAL_MODULES=verl_omni

python3 -m verl_omni.trainer.main_omni \
    data.train_files="$HOME/data/train.parquet" \
    data.val_files="$HOME/data/test.parquet" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.strategy=fsdp2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="your_pipeline_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    "$@"
```

Key points:
- No `external_lib` — adapters are auto-registered via the Python import
  triggered by `VERL_USE_EXTERNAL_MODULES=verl_omni`.
- No `stage_configs_path` — the rollout deploy config is auto-generated
  from `pipeline_name` by `vLLMOmniHttpServer`.
- No `--config-path/--config-name` — all config comes from CLI overrides
  on `verl_omni`'s `omni_trainer.yaml` defaults.
- The `"$@"` at the end lets callers override any field without editing
  the script (e.g. `bash run.sh trainer.total_epochs=10`).

Reference:
[`examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh`](../../examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh)

For a talker-stage full-parameter example, see
[`examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh`](../../examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh).

## 6. Common pitfalls

These pitfalls are drawn from the Qwen3-Omni adapter. Some are
model-specific — verify each against your own model's architecture.

- **`_no_split_modules`**: Must be set to the correct decoder layer class
  name in `configure_model` (e.g. `Qwen3OmniMoeThinkerTextDecoderLayer`).
  FSDP uses this hint for sharding granularity — a wrong name causes the
  entire module to be treated as a single leaf, defeating parameter sharding.

- **mrope**: Qwen-style multimodal models use 3-component position IDs
  (temporal, height, width) for mrope. HuggingFace's `get_rope_index` returns
  float32 position IDs that FSDP would bf16-round. Cast to `int64` in
  `configure_processor` (see `_get_rope_index_long` in
  `thinker_training_adapter.py:98-100`).

- **`dedup_pad_tokens`**: Bind on the processor for multimodal (image/video/
  audio) training to avoid double-expansion in AR mode. Both the HF processor
  and vLLM's `_apply_prompt_updates` expand the pad token, causing a mismatch.
  The `dedup_pad_tokens` helper collapses consecutive identical multimodal
  pad tokens before sending to vLLM-Omni.

- **tokenizer `chat_template.json`**: If the model checkpoint ships
  `chat_template.json` separately (not in `tokenizer_config.json`), load it
  in `configure_tokenizer` and assign it to `tokenizer.chat_template`.
  verl's dataset loader calls `tokenizer.apply_chat_template()` and will
  fail without a template.

- **Actor/rollout probability consistency**: Autoregressive codec policies may
  combine several codebook embeddings before predicting the selected token.
  Match actor, reference, rollout, and weight-sync dtypes, then verify selected
  token log-probabilities before training. A nominal FP32 rollout fed BF16
  actor weights is not an FP32 consistency check.
