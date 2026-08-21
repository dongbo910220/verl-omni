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
"""Helpers for collecting Qwen3-TTS rollout outputs."""

import hashlib
from collections.abc import Mapping
from typing import Any

import torch

_EVALUATION_SPLITS = {"gate", "official_test", "test", "val", "validation"}
_MAX_SAMPLING_SEED = 2**31 - 1


def _scalar(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            return None
    return value


def _extra_info_mapping(extra_info) -> dict:
    extra_info = _scalar(extra_info)
    return dict(extra_info.items()) if isinstance(extra_info, Mapping) else {}


def is_evaluation_split(extra_info) -> bool:
    info = _extra_info_mapping(extra_info)
    return str(_scalar(info.get("split")) or "").lower() in _EVALUATION_SPLITS


def validation_generation_seed(extra_info) -> int | None:
    info = _extra_info_mapping(extra_info)
    if not is_evaluation_split(info):
        return None
    seed = _scalar(info.get("generation_seed"))
    return None if seed is None else int(seed)


def rollout_generation_seed(
    extra_info,
    *,
    session_id=None,
    global_steps=None,
    uid=None,
    base_seed=0,
    require_session_id=False,
) -> int:
    """Derive a stable, group-diverse seed for both Qwen3-TTS samplers."""
    if require_session_id and session_id is None:
        raise RuntimeError("Qwen3-TTS group sampling requires a per-candidate session_id when rollout.n > 1.")
    candidate = int(_scalar(session_id) or 0)
    explicit_seed = validation_generation_seed(extra_info)
    if explicit_seed is not None:
        return (explicit_seed + candidate) % _MAX_SAMPLING_SEED

    info = _extra_info_mapping(extra_info)
    sample_id = _scalar(info.get("id", info.get("index", uid)))
    step = "evaluation" if is_evaluation_split(info) else _scalar(global_steps)
    payload = "\0".join(map(str, (int(base_seed), step, sample_id, candidate))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % _MAX_SAMPLING_SEED


def with_rollout_generation_seed(sampling_params, extra_info, **seed_kwargs):
    """Seed codec-0 and residual-codebook sampling without mutating input."""
    seed = rollout_generation_seed(extra_info, **seed_kwargs)
    seeded = dict(sampling_params)
    residual_args = dict(seeded.get("extra_args") or {})
    residual_args["tts_local_seed"] = seed
    seeded.update(seed=seed, extra_args=residual_args)
    return seeded


def append_tensor_chunk(accumulated: torch.Tensor | None, value: Any, *, flatten=False):
    if isinstance(value, list | tuple):
        value = value[-1] if value else None
    if value is None:
        return accumulated
    chunk = torch.as_tensor(value).detach().cpu()
    if flatten:
        chunk = chunk.reshape(-1)
    if not chunk.numel():
        return accumulated
    if accumulated is None:
        return chunk
    if chunk.shape[1:] != accumulated.shape[1:]:
        raise RuntimeError(f"Rollout chunks changed shape from {tuple(accumulated.shape)} to {tuple(chunk.shape)}.")
    if chunk.shape[0] >= accumulated.shape[0] and torch.equal(chunk[: accumulated.shape[0]], accumulated):
        return chunk
    if accumulated.shape[0] >= chunk.shape[0] and torch.equal(accumulated[: chunk.shape[0]], chunk):
        return accumulated
    return torch.cat((accumulated, chunk), dim=0)


def align_audio_codes(audio_codes: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Recover residual codebooks using codec-0 policy tokens as an exact invariant.

    The engine can prepend placeholder rows and need not emit residual codes
    for the last sampled token, because that token is never consumed as the
    context for another policy token. Thus ``token_ids[:-1]`` must match
    exactly; a final residual row is retained only when the engine provides it.
    """
    if audio_codes.ndim != 2 or audio_codes.shape[-1] != 16:
        raise ValueError("Qwen3-TTS codec codes must have shape (frames, 16).")
    if not token_ids:
        return audio_codes[:0].long()
    raw_codes = audio_codes.long()
    policy_ids = torch.as_tensor(token_ids, dtype=torch.long, device=raw_codes.device)
    required = len(token_ids) - 1
    candidates: list[tuple[int, int]] = []
    if required == 0:
        candidates.append((0, 0))
    else:
        for start in range(raw_codes.shape[0] - required + 1):
            if torch.equal(raw_codes[start : start + required, 0], policy_ids[:required]):
                has_final = int(
                    raw_codes.shape[0] - start >= len(token_ids)
                    and raw_codes[start + required, 0] == policy_ids[required]
                )
                candidates.append((required + has_final, start))
    if not candidates:
        raise RuntimeError(
            "Could not exactly align Qwen3-TTS residual codebooks with the sampled codec-0 policy: "
            f"response_length={len(token_ids)}, raw_codec_rows={raw_codes.shape[0]}."
        )

    copy_length, start = max(candidates)
    codes = raw_codes.new_zeros((len(token_ids), 16))
    if copy_length:
        codes[:copy_length] = raw_codes[start : start + copy_length]
    codes[:, 0] = policy_ids
    if required and not torch.equal(codes[:required, 0], policy_ids[:required]):
        raise RuntimeError("Qwen3-TTS codec alignment invariant failed after recovery.")
    return codes
