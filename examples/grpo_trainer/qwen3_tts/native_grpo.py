# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native PyTorch rollout and loss helpers for the Hindi Qwen3-TTS GRPO run.

This is deliberately not a port of the MLX training framework. It keeps the
published experiment's model path and GRPO math while using one Transformers
talker for both autoregressive rollout and teacher-forced actor replay.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

NUM_CODEBOOKS = 16
TEXT_PROMPT_TRAILER_TOKENS = 5
CONSISTENCY_DIFF_LIMIT = 0.005
CONSISTENCY_PEARSON_LIMIT = 0.995


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class TalkerTokens:
    tts_pad: int
    tts_bos: int
    tts_eos: int
    codec_pad: int
    codec_bos: int
    codec_eos: int
    codec_nothink: int
    codec_think_bos: int
    codec_think_eos: int

    @classmethod
    def from_config(cls, config):
        talker = config.talker_config
        return cls(
            int(config.tts_pad_token_id),
            int(config.tts_bos_token_id),
            int(config.tts_eos_token_id),
            int(talker.codec_pad_id),
            int(talker.codec_bos_id),
            int(talker.codec_eos_token_id),
            int(talker.codec_nothink_id),
            int(talker.codec_think_bos_id),
            int(talker.codec_think_eos_id),
        )


@dataclass
class InterleavedTalkerBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    codec_lens: list[int]
    logit_starts: list[int]


def _project_text(talker, token_ids: torch.Tensor) -> torch.Tensor:
    embeddings = talker.model.text_embedding(token_ids)
    if getattr(talker, "text_projection", None) is not None:
        embeddings = talker.text_projection(embeddings)
    return embeddings


def build_interleaved_talker_batch(
    talker,
    text_ids,
    audio_codes,
    tokens,
    *,
    speaker_embedding=None,
    sub_codebook_vocab=None,
) -> InterleavedTalkerBatch:
    """Rebuild the exact streamed text-plus-codec inputs used by generation."""
    codec_embedding = talker.model.codec_embedding
    sub_embeddings = talker.code_predictor.get_input_embeddings()
    sequences = []
    codec_lens = []
    logit_starts = []
    for index, (sample_text, sample_codes) in enumerate(zip(text_ids, audio_codes, strict=True)):
        ids = sample_text.reshape(-1).long()
        codes = sample_codes.long()
        if ids.numel() <= TEXT_PROMPT_TRAILER_TOKENS + 3:
            raise ValueError("Qwen3-TTS assistant text must contain its role and text tokens.")
        if codes.ndim != 2 or codes.shape[-1] != NUM_CODEBOOKS or not codes.shape[0]:
            raise ValueError("Qwen3-TTS codec codes must have shape (nonzero frames, 16).")
        if sub_codebook_vocab is not None:
            codes = codes.clone()
            codes[:, 1:].clamp_(0, sub_codebook_vocab - 1)

        # Generation projects the complete batched token sequence and only then
        # removes the five chat-template trailer tokens. Matching that GEMM
        # shape is required for BF16 probability consistency.
        text_embeddings = _project_text(talker, ids.unsqueeze(0))[0]
        text_embeddings = text_embeddings[:-TEXT_PROMPT_TRAILER_TOKENS]
        special_ids = ids.new_tensor([tokens.tts_bos, tokens.tts_eos, tokens.tts_pad])
        tts_bos, tts_eos, tts_pad = _project_text(talker, special_ids).split(1, dim=0)
        prefix_ids = ids.new_tensor(
            [tokens.codec_nothink, tokens.codec_think_bos, tokens.codec_think_eos]
        )
        suffix_ids = ids.new_tensor([tokens.codec_pad, tokens.codec_bos])
        codec_prefix = codec_embedding(prefix_ids)
        sample_speaker = None if speaker_embedding is None else speaker_embedding[index : index + 1]
        if sample_speaker is not None:
            sample_speaker = sample_speaker.to(device=codec_prefix.device, dtype=codec_prefix.dtype)
            codec_prefix = torch.cat((codec_prefix, sample_speaker, codec_embedding(suffix_ids)), dim=0)
        else:
            codec_prefix = torch.cat((codec_prefix, codec_embedding(suffix_ids)), dim=0)

        pad_count = codec_prefix.shape[0] - 2
        combined_text = torch.cat((tts_pad.expand(pad_count, -1), tts_bos), dim=0)
        combined = combined_text + codec_prefix[:-1]
        first_text = text_embeddings[3:4] + codec_prefix[-1:]
        prefill = torch.cat((text_embeddings[:3], combined, first_text), dim=0)
        trailing = torch.cat((text_embeddings[4:], tts_eos), dim=0)
        frame_inputs = []
        for frame in range(codes.shape[0] - 1):
            streamed_text = trailing[frame : frame + 1] if frame < trailing.shape[0] else tts_pad
            codec_parts = [codec_embedding(codes[frame : frame + 1, 0])]
            for codebook in range(1, NUM_CODEBOOKS):
                codec_parts.append(
                    sub_embeddings[codebook - 1](codes[frame : frame + 1, codebook])
                )
            # Match Qwen3TTSTalkerForConditionalGeneration.forward exactly.
            # Sequential BF16 additions move selected-token probabilities by
            # about one percent even though the mathematical sum is identical.
            streamed_codec = torch.cat(codec_parts, dim=0).sum(dim=0, keepdim=True)
            frame_inputs.append(streamed_text + streamed_codec)
        sequence = torch.cat((prefill, *frame_inputs), dim=0) if frame_inputs else prefill
        sequences.append(sequence)
        codec_lens.append(int(codes.shape[0]))
        logit_starts.append(int(prefill.shape[0] - 1))

    inputs_embeds = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    attention_mask = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    for index, sequence in enumerate(sequences):
        attention_mask[index, : sequence.shape[0]] = 1
    return InterleavedTalkerBatch(inputs_embeds, attention_mask, codec_lens, logit_starts)


def interleaved_codec0_logits(talker, batch: InterleavedTalkerBatch) -> torch.Tensor:
    return talker(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask,
        use_cache=False,
        output_hidden_states=False,
    ).logits


def cached_interleaved_codec0_logits(talker, batch: InterleavedTalkerBatch) -> torch.Tensor:
    """Replay a recorded trajectory through the rollout's KV-cache path.

    A single teacher-forced forward is mathematically equivalent but is not
    numerically equivalent on this model: its attention reductions differ from
    the cached generation path by about one percent in selected probabilities.
    This replay keeps the real rollout policy while preserving autograd.
    """
    if len(set(batch.logit_starts)) != 1:
        raise ValueError("Cached replay requires one shared no-speaker prefill length.")
    prefill_length = batch.logit_starts[0] + 1
    max_actions = max(batch.codec_lens)
    prefill_mask = batch.attention_mask[:, :prefill_length]
    output = talker(
        inputs_embeds=batch.inputs_embeds[:, :prefill_length],
        attention_mask=prefill_mask,
        use_cache=True,
        output_hidden_states=False,
    )
    logits = [output.logits[:, -1].float()]
    cache = output.past_key_values
    for action_index in range(1, max_actions):
        sequence_index = prefill_length + action_index - 1
        attention_mask = batch.attention_mask[:, : sequence_index + 1]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -1:].unsqueeze(0).expand(3, -1, -1)
        cache_position = torch.tensor(
            [sequence_index], dtype=torch.long, device=batch.inputs_embeds.device
        )
        output = talker.model(
            inputs_embeds=batch.inputs_embeds[:, sequence_index : sequence_index + 1],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            output_hidden_states=False,
            return_dict=True,
        )
        cache = output.past_key_values
        logits.append(talker.codec_head(output.last_hidden_state)[:, -1].float())
    return torch.stack(logits, dim=1)


@dataclass
class NativeRolloutGroup:
    text: str
    text_ids: torch.Tensor
    codes: list[torch.Tensor]
    audio_codes: list[torch.Tensor]
    rollout_logprobs: list[torch.Tensor]
    has_eos: list[bool]
    generation_lengths: list[int]
    rewards: np.ndarray | None = None
    reward_info: list[dict[str, Any]] = field(default_factory=list)
    prefill_captures: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def capture_first_core_forward(talker):
    """Capture the first real talker-core call without changing its inputs."""
    capture: dict[str, Any] = {}

    def pre_hook(_module, _args, kwargs):
        if capture:
            return
        capture["training"] = bool(_module.training)
        capture["grad_enabled"] = bool(torch.is_grad_enabled())
        capture["inference_mode"] = bool(torch.is_inference_mode_enabled())
        for name in ("inputs_embeds", "attention_mask", "position_ids", "cache_position"):
            value = kwargs.get(name)
            if torch.is_tensor(value):
                capture[name] = value.detach().cpu().clone()
                capture[f"{name}_stride"] = tuple(value.stride())
                capture[f"{name}_contiguous"] = bool(value.is_contiguous())
        cache = kwargs.get("past_key_values")
        capture["cache_type"] = None if cache is None else type(cache).__name__
        capture["cache_seq_length_before"] = (
            None if cache is None or not hasattr(cache, "get_seq_length") else int(cache.get_seq_length())
        )

    def post_hook(_module, _args, _kwargs, output):
        if "last_hidden_state" not in capture:
            capture["last_hidden_state"] = output.last_hidden_state.detach().cpu().clone()

    pre_handle = talker.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = talker.model.register_forward_hook(post_hook, with_kwargs=True)
    try:
        yield capture
    finally:
        pre_handle.remove()
        post_handle.remove()


def group_advantages(rewards: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1 or rewards.size < 2:
        raise ValueError("GRPO rewards must be a one-dimensional group with at least two candidates.")
    return (rewards - rewards.mean()) / (rewards.std(ddof=0) + float(epsilon))


def probability_consistency(
    rollout_logprobs: list[torch.Tensor],
    actor_logprobs: list[torch.Tensor],
) -> dict[str, float | int | bool]:
    if len(rollout_logprobs) != len(actor_logprobs):
        raise ValueError("Rollout and actor trajectory counts differ.")
    rollout = torch.cat([value.detach().float().cpu() for value in rollout_logprobs])
    actor = torch.cat([value.detach().float().cpu() for value in actor_logprobs])
    if rollout.shape != actor.shape or not rollout.numel():
        raise ValueError("Rollout and actor log-probability shapes differ or are empty.")
    rollout_probs = rollout.exp().numpy().astype(np.float64, copy=False)
    actor_probs = actor.exp().numpy().astype(np.float64, copy=False)
    diff_mean = float(np.abs(rollout_probs - actor_probs).mean())
    if float(rollout_probs.std()) == 0.0 or float(actor_probs.std()) == 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(rollout_probs, actor_probs)[0, 1])
    passed = bool(
        np.isfinite(pearson)
        and diff_mean < CONSISTENCY_DIFF_LIMIT
        and pearson > CONSISTENCY_PEARSON_LIMIT
    )
    return {
        "tokens": int(rollout.numel()),
        "diff_mean": diff_mean,
        "pearson": pearson,
        "passed": passed,
    }


def combine_consistency_payloads(payloads: list[dict[str, list[float]]]) -> dict[str, float | int | bool]:
    rollout = [torch.tensor(part["rollout"], dtype=torch.float32).log() for part in payloads if part["rollout"]]
    actor = [torch.tensor(part["actor"], dtype=torch.float32).log() for part in payloads if part["actor"]]
    return probability_consistency(rollout, actor)


def learning_rate_for_step(step: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(base_lr) * step / max(1, warmup_steps)
    return float(base_lr)


def _prepare_auto_generation_inputs(wrapper, text: str, group_size: int):
    model = wrapper.model
    talker = model.talker
    config = model.config
    full_ids = wrapper._tokenize_texts([build_assistant_text(text)])[0]
    if full_ids.shape[1] <= TEXT_PROMPT_TRAILER_TOKENS + 3:
        raise ValueError("Qwen3-TTS assistant prompt is too short for the interleaved layout.")

    projected = talker.text_projection(talker.get_text_embeddings()(full_ids))
    special_ids = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=talker.device,
        dtype=full_ids.dtype,
    )
    tts_bos, tts_eos, tts_pad = talker.text_projection(
        talker.get_text_embeddings()(special_ids)
    ).chunk(3, dim=1)
    codec_prefix_ids = torch.tensor(
        [[
            talker.config.codec_nothink_id,
            talker.config.codec_think_bos_id,
            talker.config.codec_think_eos_id,
            talker.config.codec_pad_id,
            talker.config.codec_bos_id,
        ]],
        device=talker.device,
        dtype=full_ids.dtype,
    )
    codec_prefix = talker.get_input_embeddings()(codec_prefix_ids)
    aligned_prefix = torch.cat(
        (tts_pad.expand(-1, codec_prefix.shape[1] - 2, -1), tts_bos), dim=1
    ) + codec_prefix[:, :-1]
    prefill = torch.cat(
        (projected[:, :3], aligned_prefix, projected[:, 3:4] + codec_prefix[:, -1:]), dim=1
    )
    trailing = torch.cat((projected[:, 4:-TEXT_PROMPT_TRAILER_TOKENS], tts_eos), dim=1)
    prefill = prefill.expand(group_size, -1, -1).contiguous()
    trailing = trailing.expand(group_size, -1, -1).contiguous()
    return full_ids.reshape(-1), prefill, trailing, tts_pad


def parse_generation_result(result, eos_token_id: int, num_codebooks: int = NUM_CODEBOOKS) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[bool],
    list[int],
]:
    if result.logits is None or not len(result.logits):
        raise RuntimeError("Native rollout did not return raw generation logits.")
    raw_logits = torch.stack(
        [value[:, -1] if value.ndim == 3 else value for value in result.logits], dim=1
    ).float()
    sampled = result.sequences[:, -raw_logits.shape[1] :].long()
    frame_list = [hidden[-1] for hidden in result.hidden_states if hidden[-1] is not None]
    frames = torch.stack(frame_list, dim=1).long() if frame_list else sampled.new_zeros(
        (sampled.shape[0], 0, num_codebooks)
    )
    if frames.shape[0] != sampled.shape[0] or frames.shape[2] != num_codebooks:
        raise RuntimeError(
            f"Unexpected Qwen3-TTS frame shape {tuple(frames.shape)} for sampled shape {tuple(sampled.shape)}."
        )
    if frames.shape[1] not in (sampled.shape[1] - 1, sampled.shape[1]):
        raise RuntimeError(
            f"Expected one residual-code frame per completed codec-0 input, got {frames.shape[1]} "
            f"for {sampled.shape[1]} sampled actions."
        )
    if frames.shape[1] and not torch.equal(frames[:, :, 0], sampled[:, : frames.shape[1]]):
        raise RuntimeError("Sub-talker frames are not aligned with sampled codec-0 actions.")

    selected = torch.log_softmax(raw_logits, dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    codes: list[torch.Tensor] = []
    audio_codes: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    has_eos: list[bool] = []
    lengths: list[int] = []
    for row in range(sampled.shape[0]):
        eos_rows = torch.nonzero(sampled[row] == int(eos_token_id), as_tuple=False).reshape(-1)
        ended = bool(eos_rows.numel())
        action_length = int(eos_rows[0]) + 1 if ended else sampled.shape[1]
        trajectory = sampled.new_zeros((action_length, num_codebooks))
        available = min(action_length, frames.shape[1])
        if available:
            trajectory[:available] = frames[row, :available]
        trajectory[:, 0] = sampled[row, :action_length]
        audio_length = action_length - 1 if ended else available
        codes.append(trajectory.detach())
        audio_codes.append(frames[row, :audio_length].detach())
        logprobs.append(selected[row, :action_length].detach())
        has_eos.append(ended)
        lengths.append(action_length)
    return codes, audio_codes, logprobs, has_eos, lengths


def _sample_native_rollout_batch(
    wrapper,
    text: str,
    *,
    batch_size: int,
    max_new_tokens: int = 240,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    capture_prefill: bool = False,
) -> NativeRolloutGroup:
    model = wrapper.model
    talker = model.talker
    text_ids, prefill, trailing, tts_pad = _prepare_auto_generation_inputs(wrapper, text, batch_size)
    eos_id = int(talker.config.codec_eos_token_id)
    suppress = [
        token
        for token in range(talker.config.vocab_size - 1024, talker.config.vocab_size)
        if token != eos_id
    ]
    capture_context = capture_first_core_forward(talker) if capture_prefill else nullcontext({})
    with capture_context as prefill_capture:
        result = talker.generate(
            inputs_embeds=prefill,
            attention_mask=torch.ones(prefill.shape[:2], dtype=torch.long, device=prefill.device),
            trailing_text_hidden=trailing,
            tts_pad_embed=tts_pad,
            max_new_tokens=max_new_tokens,
            min_new_tokens=0,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.0,
            subtalker_dosample=True,
            subtalker_top_k=top_k,
            subtalker_top_p=top_p,
            subtalker_temperature=temperature,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            suppress_tokens=suppress,
            output_hidden_states=True,
            output_logits=True,
            return_dict_in_generate=True,
        )
    if capture_prefill:
        active = getattr(wrapper.model, "active_adapters", None)
        prefill_capture["active_adapters"] = (
            list(active) if isinstance(active, (list, tuple)) else active
        )
    codes, audio_codes, logprobs, has_eos, lengths = parse_generation_result(result, eos_id)
    captures = [prefill_capture] if capture_prefill else []
    return NativeRolloutGroup(
        text,
        text_ids.detach(),
        codes,
        audio_codes,
        logprobs,
        has_eos,
        lengths,
        prefill_captures=captures,
    )


@torch.inference_mode()
def sample_native_rollouts(
    wrapper,
    text: str,
    *,
    group_size: int = 4,
    microbatch_size: int | None = None,
    max_new_tokens: int = 240,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int,
    capture_prefill: bool = False,
) -> NativeRolloutGroup:
    """Sample one GRPO group with a continuous RNG stream across microbatches."""
    if group_size <= 0:
        raise ValueError("Rollout group size must be positive.")
    microbatch_size = group_size if microbatch_size is None else int(microbatch_size)
    if microbatch_size <= 0 or microbatch_size > group_size:
        raise ValueError("Rollout microbatch size must be between one and group size.")
    wrapper.model.eval()
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    parts = []
    remaining = group_size
    while remaining:
        current = min(microbatch_size, remaining)
        parts.append(
            _sample_native_rollout_batch(
                wrapper,
                text,
                batch_size=current,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                capture_prefill=capture_prefill,
            )
        )
        remaining -= current
    text_ids = parts[0].text_ids
    if any(not torch.equal(text_ids, part.text_ids) for part in parts[1:]):
        raise RuntimeError("Rollout microbatches produced different text token IDs.")
    return NativeRolloutGroup(
        text=text,
        text_ids=text_ids,
        codes=[value for part in parts for value in part.codes],
        audio_codes=[value for part in parts for value in part.audio_codes],
        rollout_logprobs=[value for part in parts for value in part.rollout_logprobs],
        has_eos=[value for part in parts for value in part.has_eos],
        generation_lengths=[value for part in parts for value in part.generation_lengths],
        prefill_captures=[value for part in parts for value in part.prefill_captures],
    )


@torch.inference_mode()
def decode_rollout_audio(model, audio_codes: list[torch.Tensor]) -> tuple[list[np.ndarray], int]:
    nonempty = [(index, codes) for index, codes in enumerate(audio_codes) if codes.shape[0]]
    output = [np.empty(0, dtype=np.float32) for _ in audio_codes]
    if not nonempty:
        return output, 24_000
    decoded, sample_rate = model.speech_tokenizer.decode(
        [{"audio_codes": codes} for _, codes in nonempty]
    )
    for (index, _), waveform in zip(nonempty, decoded, strict=True):
        output[index] = np.asarray(waveform, dtype=np.float32).reshape(-1)
    return output, int(sample_rate)


def _replay_codes_logprobs(
    model,
    text_ids: torch.Tensor,
    codes: list[torch.Tensor],
) -> list[torch.Tensor]:
    talker = model.talker
    # Rollout tensors are intentionally created under inference_mode. Clone them
    # here so autograd may save codec targets and replay inputs for backward.
    text_ids = text_ids.detach().clone()
    codes = [value.detach().clone() for value in codes]
    texts = [text_ids] * len(codes)
    with torch.no_grad():
        batch = build_interleaved_talker_batch(
            talker,
            texts,
            codes,
            TalkerTokens.from_config(model.config),
            speaker_embedding=None,
            sub_codebook_vocab=int(talker.code_predictor.get_input_embeddings()[0].num_embeddings),
        )
        batch.inputs_embeds = batch.inputs_embeds.detach()
    logits = cached_interleaved_codec0_logits(talker, batch)
    selected: list[torch.Tensor] = []
    for row, trajectory in enumerate(codes):
        length = int(trajectory.shape[0])
        row_logits = logits[row, :length].float()
        row_targets = trajectory[:, 0].long()
        selected.append(torch.log_softmax(row_logits, dim=-1).gather(-1, row_targets[:, None]).squeeze(-1))
    return selected


def replay_logprobs(model, group: NativeRolloutGroup) -> list[torch.Tensor]:
    return _replay_codes_logprobs(model, group.text_ids, group.codes)


def replay_one_logprobs(model, group: NativeRolloutGroup, index: int) -> torch.Tensor:
    """Replay one candidate so long trajectories do not retain a padded G-way graph."""
    return _replay_codes_logprobs(model, group.text_ids, [group.codes[index]])[0]


def replay_subset_logprobs(
    model,
    group: NativeRolloutGroup,
    indices: list[int],
) -> list[torch.Tensor]:
    """Replay a stable candidate microbatch selected from one GRPO group."""
    if not indices:
        raise ValueError("Replay candidate indices cannot be empty.")
    return _replay_codes_logprobs(model, group.text_ids, [group.codes[index] for index in indices])


def grpo_loss(
    actor_logprobs: list[torch.Tensor],
    reference_logprobs: list[torch.Tensor],
    advantages: np.ndarray,
    *,
    kl_beta: float = 0.08,
    kl_clip: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not (len(actor_logprobs) == len(reference_logprobs) == len(advantages)):
        raise ValueError("Actor, reference, and advantage group sizes differ.")
    token_count = sum(value.numel() for value in actor_logprobs)
    if not token_count:
        raise ValueError("Cannot compute GRPO loss for an empty trajectory group.")
    pg_sum = actor_logprobs[0].new_zeros(())
    kl_sum = actor_logprobs[0].new_zeros(())
    for actor, reference, advantage in zip(
        actor_logprobs, reference_logprobs, advantages, strict=True
    ):
        if actor.shape != reference.shape:
            raise ValueError("Actor and reference trajectory lengths differ.")
        pg_sum = pg_sum - float(advantage) * actor.sum()
        delta = (reference.detach() - actor).clamp(-kl_clip, kl_clip)
        kl_sum = kl_sum + (delta.exp() - delta - 1.0).sum()
    pg = pg_sum / token_count
    kl = kl_sum / token_count
    loss = pg + float(kl_beta) * kl
    return loss, {
        "loss": float(loss.detach()),
        "pg": float(pg.detach()),
        "kl": float(kl.detach()),
        "tokens": float(token_count),
    }


def grpo_trajectory_loss(
    actor_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantage: float,
    token_count: int,
    *,
    kl_beta: float = 0.08,
    kl_clip: float = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return one candidate's additive share of the token-normalized GRPO loss."""
    if actor_logprobs.shape != reference_logprobs.shape:
        raise ValueError("Actor and reference trajectory lengths differ.")
    if token_count < actor_logprobs.numel() or token_count <= 0:
        raise ValueError("GRPO group token count is invalid.")
    pg = -float(advantage) * actor_logprobs.sum() / token_count
    delta = (reference_logprobs.detach() - actor_logprobs).clamp(-kl_clip, kl_clip)
    kl = (delta.exp() - delta - 1.0).sum() / token_count
    loss = pg + float(kl_beta) * kl
    return loss, {
        "loss": float(loss.detach()),
        "pg": float(pg.detach()),
        "kl": float(kl.detach()),
        "tokens": float(actor_logprobs.numel()),
    }
