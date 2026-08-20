#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
# ruff: noqa: E402
"""Run the published Hindi Qwen3-TTS CER-GRPO recipe on native PyTorch.

The process is data parallel over two GPUs. Each rank owns two prompt groups,
then LoRA gradients are summed so one optimizer step is the published B=4,
G=4 batch. The rollout and actor use the same eager Transformers talker.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# transformers 4.57 treats the optional kernels package as unavailable when its
# import raises ImportError. The isolated native environment shares a newer
# vLLM environment that contains an incompatible kernels build.
sys.modules["kernels"] = None

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_grpo import (  # noqa: E402
    CONSISTENCY_DIFF_LIMIT,
    CONSISTENCY_PEARSON_LIMIT,
    NativeRolloutGroup,
    capture_first_core_forward,
    decode_rollout_audio,
    group_advantages,
    grpo_trajectory_loss,
    learning_rate_for_step,
    probability_consistency,
    replay_subset_logprobs,
    sample_native_rollouts,
)


def _load_reward_function():
    path = ROOT / "verl_omni/utils/reward_score/tts/whisper_cer_reward.py"
    spec = importlib.util.spec_from_file_location("qwen3_tts_native_whisper_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CER reward from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.compute_score


compute_score = _load_reward_function()

EXPECTED_TRAIN_ROWS = 863
EXPECTED_VALIDATION_ROWS = 100
EXPECTED_ADAPTER_TENSORS = 462
EXPECTED_TRAINABLE_PARAMETERS = 5_947_392
POLICY_ADAPTER = "default"
REFERENCE_ADAPTER = "sft_reference"


@dataclass
class Runtime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def rank_event(
    args: argparse.Namespace,
    runtime: Runtime,
    event: str,
    **values: Any,
) -> None:
    event_dir = args.output_dir / "rank-events"
    event_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "monotonic_s": time.monotonic(),
        "rank": runtime.rank,
        "time_s": time.time(),
        **values,
    }
    with (event_dir / f"rank-{runtime.rank}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def enable_hang_trace(args: argparse.Namespace, runtime: Runtime):
    trace_dir = args.output_dir / "hang-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    handle = (trace_dir / f"rank-{runtime.rank}.log").open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=handle, all_threads=True)
    if args.hang_trace_interval_s > 0:
        faulthandler.dump_traceback_later(args.hang_trace_interval_s, repeat=True, file=handle)
    return handle


class PromptSampler:
    def __init__(self, size: int, seed: int):
        self.size = int(size)
        self.rng = random.Random(int(seed))
        self.order = list(range(self.size))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.epoch = 0
        self.ordinal = 0
        self.seed = int(seed)

    def next(self) -> tuple[int, int]:
        if self.cursor == self.size:
            self.epoch += 1
            self.cursor = 0
            self.rng.shuffle(self.order)
        index = self.order[self.cursor]
        self.cursor += 1
        self.ordinal += 1
        generation_seed = self.seed * 1_000_003 + self.ordinal * 1_009
        return index, generation_seed

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "order": self.order,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "ordinal": self.ordinal,
            "seed": self.seed,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["size"]) != self.size or int(state["seed"]) != self.seed:
            raise ValueError("Checkpoint prompt sampler does not match this dataset or seed.")
        self.order = list(state["order"])
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        self.ordinal = int(state["ordinal"])
        self.rng.setstate(state["rng_state"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("lr0", "smoke", "train", "extend", "eval"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-beta", type=float, default=0.08)
    parser.add_argument("--kl-clip", type=float, default=10.0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--rollout-microbatch-size", type=int, default=2)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-before-train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--validation-seed-offset", type=int, default=0)
    parser.add_argument("--max-zero-variance-retries", type=int, default=64)
    parser.add_argument("--zero-variance-epsilon", type=float, default=1e-6)
    parser.add_argument("--save-validation-audio", action="store_true")
    parser.add_argument("--diagnose-prefill", action="store_true")
    parser.add_argument(
        "--hang-trace-interval-s",
        type=int,
        default=0,
        help="Periodically dump Python stacks for hang diagnosis; disabled by default.",
    )
    args = parser.parse_args()
    if args.group_size != 4 or args.prompts_per_step != 4:
        raise ValueError("The published reproduction is fixed to B=4 prompts and G=4 candidates.")
    if args.rollout_microbatch_size != 2:
        raise ValueError("The two-GPU BF16 reproduction requires rollout/replay microbatch size 2.")
    if args.max_new_tokens != 240:
        raise ValueError("The published reproduction is fixed to 240 codec frames.")
    if args.eval_every != 20:
        raise ValueError("The fixed 100-row validation set must be evaluated every 20 steps.")
    if args.stage == "lr0" and args.learning_rate != 0.0:
        raise ValueError("The lr0 stage requires --learning-rate 0.")
    if args.eval_only != (args.stage == "eval"):
        raise ValueError("The eval stage and --eval-only must be used together.")
    if args.stage == "eval" and args.learning_rate != 0.0:
        raise ValueError("The eval stage requires --learning-rate 0.")
    if args.validation_seed_offset and not args.eval_only:
        raise ValueError("--validation-seed-offset is only valid with --eval-only.")
    return args


def init_runtime() -> Runtime:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != 2:
        raise RuntimeError(f"This reproduction is fixed to two GPUs, got world_size={world_size}.")
    torch.cuda.set_device(local_rank)
    return Runtime(rank, local_rank, world_size, torch.device("cuda", local_rank))


def _prompt_text(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(f"Unexpected prompt payload: {value!r}")
    text = str(value[0].get("content") or "").strip()
    if not text:
        raise ValueError("Dataset prompt text is empty.")
    return text


def _extra_info(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if not isinstance(value, dict):
        raise ValueError(f"Unexpected extra_info payload: {value!r}")
    return value


def load_rows(path: Path, expected: int, *, validation: bool) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    if len(frame) != expected:
        raise ValueError(f"{path} must contain {expected} rows, found {len(frame)}.")
    rows = []
    for index, row in frame.iterrows():
        info = _extra_info(row["extra_info"])
        rows.append(
            {
                "index": int(index),
                "id": str(info.get("id") or f"row-{index:08d}"),
                "text": _prompt_text(row["prompt"]),
                "generation_seed": int(info.get("generation_seed", 42_000 + index)),
            }
        )
    if validation and len({row["id"] for row in rows}) != expected:
        raise ValueError("Validation sample IDs are not unique.")
    return rows


def _canonical_adapter_key(key: str) -> str:
    key = key.removeprefix("base_model.model.")
    for adapter in (POLICY_ADAPTER, REFERENCE_ADAPTER):
        key = key.replace(f".{adapter}.weight", ".weight")
    return key


def verify_adapter(model, adapter_path: Path) -> None:
    source = load_file(adapter_path / "adapter_model.safetensors")
    loaded = get_peft_model_state_dict(model, adapter_name=POLICY_ADAPTER)
    source_by_key = {_canonical_adapter_key(key): value for key, value in source.items()}
    loaded_by_key = {_canonical_adapter_key(key): value for key, value in loaded.items()}
    if len(source_by_key) != EXPECTED_ADAPTER_TENSORS or set(source_by_key) != set(loaded_by_key):
        raise RuntimeError(
            f"SFT adapter key mismatch: source={len(source_by_key)}, loaded={len(loaded_by_key)}."
        )
    mismatched = [
        key
        for key in source_by_key
        if not torch.equal(
            source_by_key[key].float().cpu(), loaded_by_key[key].detach().float().cpu()
        )
    ]
    if mismatched:
        raise RuntimeError(f"SFT adapter value verification failed for {mismatched[:5]}.")


def activate_adapter(model, name: str) -> None:
    model.set_adapter(name)
    for parameter_name, parameter in model.named_parameters():
        if "lora_" in parameter_name:
            parameter.requires_grad_(f".{POLICY_ADAPTER}." in parameter_name)


def load_model(args: argparse.Namespace, runtime: Runtime):
    import qwen_tts
    import transformers
    from qwen_tts import Qwen3TTSModel

    if transformers.__version__ != "4.57.3":
        raise RuntimeError(
            f"qwen-tts 0.1.1 requires transformers 4.57.3, got {transformers.__version__}."
        )
    wrapper = Qwen3TTSModel.from_pretrained(
        str(args.model),
        device_map=str(runtime.device),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    wrapper.model = PeftModel.from_pretrained(
        wrapper.model,
        str(args.sft_adapter),
        adapter_name=POLICY_ADAPTER,
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    wrapper.model.load_adapter(
        str(args.sft_adapter), adapter_name=REFERENCE_ADAPTER, is_trainable=False
    )
    wrapper.model.eval()
    verify_adapter(wrapper.model, args.sft_adapter)
    default = get_peft_model_state_dict(wrapper.model, adapter_name=POLICY_ADAPTER)
    set_peft_model_state_dict(
        wrapper.model,
        {key: value.detach().clone() for key, value in default.items()},
        adapter_name=REFERENCE_ADAPTER,
    )
    reference = get_peft_model_state_dict(wrapper.model, adapter_name=REFERENCE_ADAPTER)
    default_by_key = {_canonical_adapter_key(key): value for key, value in default.items()}
    reference_by_key = {_canonical_adapter_key(key): value for key, value in reference.items()}
    if set(default_by_key) != set(reference_by_key):
        raise RuntimeError("Policy and reference adapter key spaces differ.")
    mismatched_reference = [
        key for key in default_by_key if not torch.equal(default_by_key[key], reference_by_key[key])
    ]
    if mismatched_reference:
        raise RuntimeError(
            f"Frozen SFT reference differs from the initial policy: {mismatched_reference[:5]}."
        )
    activate_adapter(wrapper.model, POLICY_ADAPTER)
    parameters = [
        parameter
        for name, parameter in wrapper.model.named_parameters()
        if "lora_" in name and f".{POLICY_ADAPTER}." in name
    ]
    trainable = sum(parameter.numel() for parameter in parameters)
    if len(parameters) != EXPECTED_ADAPTER_TENSORS or trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"Unexpected policy adapter: tensors={len(parameters)}, parameters={trainable}."
        )
    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "qwen_tts": str(Path(qwen_tts.__file__).resolve()),
    }
    return wrapper, parameters, versions


def score_group(
    wrapper,
    group: NativeRolloutGroup,
    args: argparse.Namespace,
    runtime: Runtime,
) -> NativeRolloutGroup:
    waveforms, sample_rate = decode_rollout_audio(wrapper.model, group.audio_codes)
    info = []
    for waveform, has_eos in zip(waveforms, group.has_eos, strict=True):
        info.append(
            compute_score(
                (waveform, sample_rate),
                group.text,
                extra_info={"reward_text": group.text, "tts_has_eos": has_eos},
                whisper_model=str(args.whisper_model),
                whisper_device=str(runtime.device),
                whisper_dtype="float16",
                language="hi",
                max_asr_duration_s=30.0,
                length_weight=0.5,
                no_eos_penalty=1.0,
                silence_fraction_max=0.6,
                silence_penalty=0.5,
                silence_rms_db=-40.0,
            )
        )
    group.rewards = np.asarray([item["score"] for item in info], dtype=np.float32)
    group.reward_info = info
    return group


def _broadcast(runtime: Runtime, payload: Any) -> Any:
    values = [payload if runtime.primary else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def build_step_groups(
    wrapper,
    train_rows: list[dict[str, Any]],
    sampler: PromptSampler,
    args: argparse.Namespace,
    runtime: Runtime,
) -> tuple[dict[int, NativeRolloutGroup], dict[str, float]]:
    local_groups: dict[int, NativeRolloutGroup] = {}
    pending = list(range(args.prompts_per_step)) if runtime.primary else None
    attempts = 0
    skipped = 0
    while True:
        assignments = None
        if runtime.primary:
            if not pending:
                assignments = []
            else:
                assignments = []
                for slot in pending:
                    prompt_index, generation_seed = sampler.next()
                    assignments.append(
                        {"slot": slot, "prompt_index": prompt_index, "seed": generation_seed}
                    )
        assignments = _broadcast(runtime, assignments)
        if not assignments:
            break
        local_status: dict[int, bool] = {}
        for assignment in assignments:
            slot = int(assignment["slot"])
            if slot % runtime.world_size != runtime.rank:
                continue
            row = train_rows[int(assignment["prompt_index"])]
            rollout_started = time.monotonic()
            rank_event(
                args,
                runtime,
                "rollout_start",
                prompt_index=int(assignment["prompt_index"]),
                seed=int(assignment["seed"]),
                slot=slot,
                text_chars=len(row["text"]),
            )
            group = sample_native_rollouts(
                wrapper,
                row["text"],
                group_size=args.group_size,
                microbatch_size=args.rollout_microbatch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=int(assignment["seed"]),
                capture_prefill=args.diagnose_prefill,
            )
            rank_event(
                args,
                runtime,
                "rollout_done",
                elapsed_s=time.monotonic() - rollout_started,
                generation_lengths=group.generation_lengths,
                has_eos=group.has_eos,
                slot=slot,
            )
            reward_started = time.monotonic()
            group = score_group(wrapper, group, args, runtime)
            active = float(group.rewards.std(ddof=0)) >= args.zero_variance_epsilon
            rank_event(
                args,
                runtime,
                "reward_done",
                active=active,
                elapsed_s=time.monotonic() - reward_started,
                rewards=group.rewards.tolist(),
                reward_std=float(group.rewards.std(ddof=0)),
                slot=slot,
            )
            local_status[slot] = active
            if active:
                local_groups[slot] = group
            else:
                skipped += 1
        gathered: list[dict[int, bool] | None] = [None] * runtime.world_size
        rank_event(args, runtime, "group_status_gather_start", local_status=local_status)
        dist.all_gather_object(gathered, local_status)
        rank_event(args, runtime, "group_status_gather_done")
        if runtime.primary:
            statuses = {slot: active for part in gathered for slot, active in (part or {}).items()}
            pending = [int(item["slot"]) for item in assignments if not statuses[int(item["slot"])]]
            attempts += len(pending)
            if attempts > args.max_zero_variance_retries:
                raise RuntimeError("Too many zero-variance reward groups while filling one GRPO step.")
    if len(local_groups) != args.prompts_per_step // runtime.world_size:
        raise RuntimeError(
            f"Rank {runtime.rank} built {len(local_groups)} active groups, expected "
            f"{args.prompts_per_step // runtime.world_size}."
        )
    skipped_parts: list[int | None] = [None] * runtime.world_size
    rank_event(args, runtime, "skipped_gather_start", local_group_count=len(local_groups))
    dist.all_gather_object(skipped_parts, skipped)
    rank_event(args, runtime, "skipped_gather_done", skipped=skipped)
    return local_groups, {"skipped_groups": float(sum(int(value or 0) for value in skipped_parts))}


def _allreduce_gradients(parameters: list[torch.nn.Parameter], runtime: Runtime) -> int:
    gradients = 0
    for parameter in parameters:
        present = torch.tensor(int(parameter.grad is not None), device=runtime.device)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        if int(present) == 0:
            continue
        if int(present) != runtime.world_size or parameter.grad is None:
            raise RuntimeError("LoRA gradient presence differs across data-parallel ranks.")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("Non-finite LoRA gradient encountered.")
        # The published MLX trainer sums four accumulated prompt-group gradients.
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        gradients += 1
    return gradients


def _consistency_across_ranks(
    rollout_logprobs: list[torch.Tensor],
    actor_logprobs: list[torch.Tensor],
    runtime: Runtime,
) -> dict[str, Any]:
    payload = {
        "rollout": torch.cat([value.detach().float().exp().cpu() for value in rollout_logprobs]).tolist(),
        "actor": torch.cat([value.detach().float().exp().cpu() for value in actor_logprobs]).tolist(),
    }
    gathered: list[dict[str, list[float]] | None] = [None] * runtime.world_size
    dist.all_gather_object(gathered, payload)
    rollout = [torch.tensor(part["rollout"], dtype=torch.float64).log() for part in gathered if part]
    actor = [torch.tensor(part["actor"], dtype=torch.float64).log() for part in gathered if part]
    return probability_consistency(rollout, actor)


def _mean_group_metrics(parts: list[dict[str, float]]) -> dict[str, float]:
    keys = set().union(*(part.keys() for part in parts))
    return {key: float(np.mean([part[key] for part in parts if key in part])) for key in keys}


def _write_consistency_failure_trace(
    records: list[dict[str, Any]],
    *,
    step: int,
    args: argparse.Namespace,
    runtime: Runtime,
) -> None:
    trace_dir = args.output_dir / "consistency-failures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"step-{step:04d}-rank-{runtime.rank}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            rollout = record["rollout_logprobs"].detach().float().cpu()
            actor = record["actor_logprobs"].detach().float().cpu()
            token_ids = record["token_ids"].detach().long().cpu()
            if rollout.shape != actor.shape or rollout.shape != token_ids.shape:
                raise RuntimeError("Consistency failure trace tensors are not aligned.")
            rollout_probs = rollout.exp()
            actor_probs = actor.exp()
            for position in range(rollout.numel()):
                payload = {
                    "abs_prob_diff": abs(
                        float(rollout_probs[position]) - float(actor_probs[position])
                    ),
                    "actor_logprob": float(actor[position]),
                    "actor_prob": float(actor_probs[position]),
                    "candidate": int(record["candidate"]),
                    "position": position,
                    "rank": runtime.rank,
                    "rollout_logprob": float(rollout[position]),
                    "rollout_prob": float(rollout_probs[position]),
                    "slot": int(record["slot"]),
                    "step": step,
                    "token_id": int(token_ids[position]),
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _compare_prefill_captures(
    rollout: dict[str, Any],
    actor: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    tensor_names = (
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "cache_position",
        "last_hidden_state",
    )
    for name in tensor_names:
        left = rollout.get(name)
        right = actor.get(name)
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            summary[name] = {"present": [torch.is_tensor(left), torch.is_tensor(right)]}
            continue
        item: dict[str, Any] = {
            "dtype": [str(left.dtype), str(right.dtype)],
            "exact": bool(left.shape == right.shape and torch.equal(left, right)),
            "shape": [list(left.shape), list(right.shape)],
        }
        if left.shape == right.shape:
            difference = (left.float() - right.float()).abs()
            item["max_abs_diff"] = float(difference.max()) if difference.numel() else 0.0
            item["mean_abs_diff"] = float(difference.mean()) if difference.numel() else 0.0
        summary[name] = item
    for name in (
        "active_adapters",
        "cache_seq_length_before",
        "cache_type",
        "grad_enabled",
        "inference_mode",
        "training",
    ):
        summary[name] = {"rollout": rollout.get(name), "actor": actor.get(name)}
    for name in ("inputs_embeds", "attention_mask", "position_ids", "cache_position"):
        summary[f"{name}_layout"] = {
            "rollout_contiguous": rollout.get(f"{name}_contiguous"),
            "rollout_stride": rollout.get(f"{name}_stride"),
            "actor_contiguous": actor.get(f"{name}_contiguous"),
            "actor_stride": actor.get(f"{name}_stride"),
        }
    return summary


def _write_prefill_diagnosis(
    wrapper,
    groups: dict[int, NativeRolloutGroup],
    args: argparse.Namespace,
    runtime: Runtime,
    *,
    step: int,
) -> None:
    if not args.diagnose_prefill:
        return
    activate_adapter(wrapper.model, POLICY_ADAPTER)
    output_dir = args.output_dir / "prefill-diagnosis" / f"step-{step:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for slot in sorted(groups):
        group = groups[slot]
        expected = (len(group.codes) + args.rollout_microbatch_size - 1) // args.rollout_microbatch_size
        if len(group.prefill_captures) != expected:
            raise RuntimeError(
                f"Expected {expected} rollout prefill captures for slot {slot}, "
                f"got {len(group.prefill_captures)}."
            )
        for microbatch, rollout_capture in enumerate(group.prefill_captures):
            first = microbatch * args.rollout_microbatch_size
            candidates = list(range(first, min(first + args.rollout_microbatch_size, len(group.codes))))
            with torch.inference_mode(), capture_first_core_forward(
                wrapper.model.talker
            ) as actor_capture:
                replay_subset_logprobs(wrapper.model, group, candidates)
            active = getattr(wrapper.model, "active_adapters", None)
            actor_capture["active_adapters"] = (
                list(active) if isinstance(active, (list, tuple)) else active
            )
            payload = {
                "actor": actor_capture,
                "candidates": candidates,
                "rollout": rollout_capture,
                "slot": slot,
            }
            artifact = output_dir / (
                f"rank-{runtime.rank}-slot-{slot}-microbatch-{microbatch}.pt"
            )
            torch.save(payload, artifact)
            summaries.append(
                {
                    "artifact": str(artifact),
                    "candidates": candidates,
                    "comparison": _compare_prefill_captures(
                        rollout_capture, actor_capture
                    ),
                    "slot": slot,
                }
            )
    summary_path = output_dir / f"rank-{runtime.rank}-summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rank_event(
        args,
        runtime,
        "prefill_diagnosis_done",
        artifact=str(summary_path),
        summaries=summaries,
    )


def _replay_groups_for_diagnosis(
    wrapper,
    groups: dict[int, NativeRolloutGroup],
    args: argparse.Namespace,
    context,
) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    activate_adapter(wrapper.model, POLICY_ADAPTER)
    with context:
        for slot in sorted(groups):
            group = groups[slot]
            for first_candidate in range(0, len(group.codes), args.rollout_microbatch_size):
                candidates = list(
                    range(
                        first_candidate,
                        min(
                            first_candidate + args.rollout_microbatch_size,
                            len(group.codes),
                        ),
                    )
                )
                values.extend(
                    value.detach()
                    for value in replay_subset_logprobs(
                        wrapper.model,
                        group,
                        candidates,
                    )
                )
    return values


def train_step(
    wrapper,
    groups: dict[int, NativeRolloutGroup],
    parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    runtime: Runtime,
    step: int,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    local_metrics: list[dict[str, float]] = []
    rollout_trace: list[torch.Tensor] = []
    actor_trace: list[torch.Tensor] = []
    trace_records: list[dict[str, Any]] = []
    for slot in sorted(groups):
        group = groups[slot]
        advantages = group_advantages(group.rewards)
        token_count = sum(int(value.shape[0]) for value in group.codes)
        pg_value = 0.0
        kl_value = 0.0
        actor_values: list[torch.Tensor] = []
        for first_candidate in range(0, len(advantages), args.rollout_microbatch_size):
            candidates = list(
                range(
                    first_candidate,
                    min(first_candidate + args.rollout_microbatch_size, len(advantages)),
                )
            )
            rank_event(
                args,
                runtime,
                "reference_replay_start",
                candidates=candidates,
                slot=slot,
                trajectory_lengths=[int(group.codes[index].shape[0]) for index in candidates],
            )
            activate_adapter(wrapper.model, REFERENCE_ADAPTER)
            with torch.no_grad():
                references = [
                    value.detach()
                    for value in replay_subset_logprobs(wrapper.model, group, candidates)
                ]
            rank_event(
                args, runtime, "reference_replay_done", candidates=candidates, slot=slot
            )
            activate_adapter(wrapper.model, POLICY_ADAPTER)
            rank_event(args, runtime, "actor_replay_start", candidates=candidates, slot=slot)
            actors = replay_subset_logprobs(wrapper.model, group, candidates)
            rank_event(args, runtime, "actor_replay_done", candidates=candidates, slot=slot)
            losses = []
            for candidate, actor, reference in zip(
                candidates, actors, references, strict=True
            ):
                loss, trajectory_metrics = grpo_trajectory_loss(
                    actor,
                    reference,
                    float(advantages[candidate]),
                    token_count,
                    kl_beta=args.kl_beta,
                    kl_clip=args.kl_clip,
                )
                losses.append(loss)
                pg_value += trajectory_metrics["pg"]
                kl_value += trajectory_metrics["kl"]
                actor_values.append(actor.detach())
                trace_records.append(
                    {
                        "actor_logprobs": actor.detach(),
                        "candidate": candidate,
                        "rollout_logprobs": group.rollout_logprobs[candidate],
                        "slot": slot,
                        "token_ids": group.codes[candidate][:, 0],
                    }
                )
            torch.stack(losses).sum().backward()
            rank_event(args, runtime, "backward_done", candidates=candidates, slot=slot)

        metrics = {
            "loss": pg_value + args.kl_beta * kl_value,
            "pg": pg_value,
            "kl": kl_value,
            "tokens": float(token_count),
        }
        rollout_trace.extend(group.rollout_logprobs)
        actor_trace.extend(actor_values)
        reward_info = group.reward_info
        metrics.update(
            {
                "reward": float(group.rewards.mean()),
                "cer": float(np.mean([item["cer"] for item in reward_info])),
                "cer_capped": float(np.mean([item["cer_capped"] for item in reward_info])),
                "wer": float(np.mean([item["wer"] for item in reward_info])),
                "no_eos": float(np.mean([item["no_eos"] for item in reward_info])),
                "duration_s": float(np.mean([item["duration_s"] for item in reward_info])),
                "reward_std": float(group.rewards.std(ddof=0)),
            }
        )
        local_metrics.append(metrics)
    rank_event(args, runtime, "consistency_gather_start")
    local_consistency = probability_consistency(rollout_trace, actor_trace)
    rank_event(args, runtime, "local_consistency_done", **local_consistency)
    consistency = _consistency_across_ranks(rollout_trace, actor_trace, runtime)
    rank_event(args, runtime, "consistency_gather_done", **consistency)
    if not consistency["passed"]:
        _write_consistency_failure_trace(
            trace_records,
            step=step,
            args=args,
            runtime=runtime,
        )
        _write_prefill_diagnosis(
            wrapper,
            groups,
            args,
            runtime,
            step=step,
        )
        no_grad_trace = _replay_groups_for_diagnosis(
            wrapper,
            groups,
            args,
            torch.no_grad(),
        )
        inference_trace = _replay_groups_for_diagnosis(
            wrapper,
            groups,
            args,
            torch.inference_mode(),
        )
        rank_event(
            args,
            runtime,
            "consistency_mode_diagnosis",
            actor_vs_inference=probability_consistency(actor_trace, inference_trace),
            actor_vs_no_grad=probability_consistency(actor_trace, no_grad_trace),
            rollout_vs_inference=probability_consistency(rollout_trace, inference_trace),
            rollout_vs_no_grad=probability_consistency(rollout_trace, no_grad_trace),
        )
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(
            "Rollout/actor consistency gate failed: "
            f"diff_mean={consistency['diff_mean']:.8f} (must be < {CONSISTENCY_DIFF_LIMIT}), "
            f"pearson={consistency['pearson']:.8f} (must be > {CONSISTENCY_PEARSON_LIMIT})."
        )
    rank_event(args, runtime, "gradient_allreduce_start")
    gradient_tensors = _allreduce_gradients(parameters, runtime)
    rank_event(args, runtime, "gradient_allreduce_done", gradient_tensors=gradient_tensors)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0))
    if not np.isfinite(grad_norm):
        raise RuntimeError("Gradient norm is not finite.")
    optimizer.step()
    rank_event(args, runtime, "optimizer_step_done", grad_norm=grad_norm)

    local_summary = _mean_group_metrics(local_metrics)
    gathered: list[dict[str, float] | None] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local_summary)
    summary = _mean_group_metrics([part for part in gathered if part])
    summary.update(consistency)
    summary.update({"grad_norm": grad_norm, "gradient_tensors": gradient_tensors})
    return summary


def _adapter_sha256(model) -> str:
    digest = hashlib.sha256()
    state = get_peft_model_state_dict(model, adapter_name=POLICY_ADAPTER)
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verified_rank_adapter_sha256(
    model,
    args: argparse.Namespace,
    runtime: Runtime,
    label: str,
) -> str:
    digest = _adapter_sha256(model)
    gathered: list[str | None] = [None] * runtime.world_size
    rank_event(args, runtime, "policy_hash_gather_start", label=label, policy_sha256=digest)
    dist.all_gather_object(gathered, digest)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"Policy adapter hashes differ across ranks at {label}: {gathered}.")
    rank_event(args, runtime, "policy_hash_gather_done", label=label, policy_sha256=digest)
    return digest


def _rng_state(runtime: Runtime) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(runtime.device),
    }


def _restore_rng_state(state: dict[str, Any], runtime: Runtime) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], runtime.device)


def _optimizer_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    if torch.is_tensor(state):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {key: _optimizer_to_cpu(value) for key, value in state.items()}
    if isinstance(state, list):
        return [_optimizer_to_cpu(value) for value in state]
    if isinstance(state, tuple):
        return tuple(_optimizer_to_cpu(value) for value in state)
    return state


def save_checkpoint(
    wrapper,
    optimizer: torch.optim.Optimizer,
    sampler: PromptSampler,
    step: int,
    args: argparse.Namespace,
    runtime: Runtime,
    *,
    archive: bool,
) -> None:
    rng_by_rank: list[dict[str, Any] | None] = [None] * runtime.world_size
    dist.all_gather_object(rng_by_rank, _rng_state(runtime))
    if runtime.primary:
        checkpoint_dir = args.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        policy = {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(
                wrapper.model, adapter_name=POLICY_ADAPTER
            ).items()
        }
        payload = {
            "schema_version": 1,
            "step": int(step),
            "policy": policy,
            "optimizer": _optimizer_to_cpu(optimizer.state_dict()),
            "policy_sha256": _adapter_sha256(wrapper.model),
            "rng_by_rank": rng_by_rank,
            "sampler": sampler.state_dict(),
            "args": vars(args),
        }
        latest = checkpoint_dir / "latest.pt"
        temporary = checkpoint_dir / ".latest.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(latest)
        (checkpoint_dir / "latest_step.txt").write_text(f"{step}\n", encoding="ascii")
        if archive:
            snapshot = checkpoint_dir / f"step-{step:04d}.pt"
            snapshot_tmp = checkpoint_dir / f".step-{step:04d}.pt.tmp"
            if snapshot_tmp.exists():
                snapshot_tmp.unlink()
            os.link(latest, snapshot_tmp)
            snapshot_tmp.replace(snapshot)
            adapter_dir = checkpoint_dir / f"adapter-step-{step:04d}"
            adapter_tmp = checkpoint_dir / f".adapter-step-{step:04d}.tmp"
            if adapter_tmp.exists():
                shutil.rmtree(adapter_tmp)
            wrapper.model.save_pretrained(
                adapter_tmp,
                safe_serialization=True,
                selected_adapters=[POLICY_ADAPTER],
            )
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            adapter_tmp.replace(adapter_dir)
    dist.barrier()


def load_checkpoint(
    wrapper,
    optimizer: torch.optim.Optimizer,
    sampler: PromptSampler,
    args: argparse.Namespace,
    runtime: Runtime,
) -> int:
    path = args.output_dir / "checkpoints/latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Resume requested but checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(wrapper.model, payload["policy"], adapter_name=POLICY_ADAPTER)
    restored_hash = _adapter_sha256(wrapper.model)
    if restored_hash != payload.get("policy_sha256", restored_hash):
        raise RuntimeError("Restored policy adapter does not match the checkpoint digest.")
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(runtime.device)
    if runtime.primary:
        sampler.load_state_dict(payload["sampler"])
    if "rng_by_rank" in payload:
        _restore_rng_state(payload["rng_by_rank"][runtime.rank], runtime)
    step = int(payload["step"])
    shared_step = _broadcast(runtime, step if runtime.primary else None)
    if step != shared_step:
        raise RuntimeError("Checkpoint step differs across ranks.")
    return step


def evaluate(
    wrapper,
    rows: list[dict[str, Any]],
    step: int,
    args: argparse.Namespace,
    runtime: Runtime,
) -> None:
    import soundfile as sf

    activate_adapter(wrapper.model, POLICY_ADAPTER)
    output_dir = args.output_dir / "validation" / f"step-{step:04d}"
    audio_dir = output_dir / "audio"
    if args.save_validation_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
    local_results = []
    for row in rows[runtime.rank :: runtime.world_size]:
        group = sample_native_rollouts(
            wrapper,
            row["text"],
            group_size=1,
            microbatch_size=1,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=row["generation_seed"],
        )
        waveforms, sample_rate = decode_rollout_audio(wrapper.model, group.audio_codes)
        reward = compute_score(
            (waveforms[0], sample_rate),
            row["text"],
            extra_info={
                "reward_text": row["text"],
                "tts_has_eos": group.has_eos[0],
                "id": row["id"],
                "generation_seed": row["generation_seed"],
            },
            whisper_model=str(args.whisper_model),
            whisper_device=str(runtime.device),
            whisper_dtype="float16",
            language="hi",
            max_asr_duration_s=30.0,
            length_weight=0.5,
            no_eos_penalty=1.0,
            silence_fraction_max=0.6,
            silence_penalty=0.5,
            silence_rms_db=-40.0,
        )
        if args.save_validation_audio:
            sf.write(audio_dir / f"{row['id']}.wav", waveforms[0], sample_rate)
        local_results.append(
            {
                "index": row["index"],
                "id": row["id"],
                "text": row["text"],
                "generation_seed": row["generation_seed"],
                "step": step,
                "frames": group.generation_lengths[0],
                "has_eos": group.has_eos[0],
                **reward,
            }
        )
    gathered: list[list[dict[str, Any]] | None] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local_results)
    if runtime.primary:
        results = sorted(
            [item for part in gathered for item in (part or [])], key=lambda item: item["index"]
        )
        if len(results) != EXPECTED_VALIDATION_ROWS or len({item["id"] for item in results}) != len(results):
            raise RuntimeError("Complete fixed-100 validation did not produce exactly 100 unique rows.")
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "step": step,
            "count": len(results),
            "cer": float(np.mean([item["cer"] for item in results])),
            "cer_capped": float(np.mean([item["cer_capped"] for item in results])),
            "wer": float(np.mean([item["wer"] for item in results])),
            "score": float(np.mean([item["score"] for item in results])),
            "no_eos_ratio": float(np.mean([item["no_eos"] for item in results])),
            "duration_s": float(np.mean([item["duration_s"] for item in results])),
            "trailing_silence": float(np.mean([item["trailing_silence"] for item in results])),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("VALIDATION " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    dist.barrier()


def write_manifest(
    args: argparse.Namespace,
    runtime: Runtime,
    versions: dict[str, str],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    if not runtime.primary:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "stage": args.stage,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": versions,
        "hardware": [torch.cuda.get_device_name(index) for index in range(runtime.world_size)],
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "policy": {
            "base_precision": "bfloat16",
            "adapter_precision": "float32",
            "attention": "eager",
            "lora_dropout_config": 0.05,
            "lora_dropout_runtime": 0.0,
            "rollout_actor_backend": "same Transformers talker",
            "rollout_replay_microbatch_size": args.rollout_microbatch_size,
        },
        "consistency_gate": {
            "diff_mean_lt": CONSISTENCY_DIFF_LIMIT,
            "pearson_gt": CONSISTENCY_PEARSON_LIMIT,
            "enforced_every_step": True,
        },
        "validation_contract": "same fixed 100 rows, one fixed seed each, step 0 and every 20 steps",
    }
    (args.output_dir / "runtime-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    runtime = init_runtime()
    hang_trace = enable_hang_trace(args, runtime)
    try:
        train_rows = load_rows(args.train_file, EXPECTED_TRAIN_ROWS, validation=False)
        validation_rows = load_rows(
            args.validation_file, EXPECTED_VALIDATION_ROWS, validation=True
        )
        if args.validation_seed_offset:
            validation_rows = [
                {
                    **row,
                    "generation_seed": int(row["generation_seed"])
                    + args.validation_seed_offset,
                }
                for row in validation_rows
            ]
        wrapper, parameters, versions = load_model(args, runtime)
        rank_event(args, runtime, "model_loaded")
        sampler = PromptSampler(len(train_rows), args.seed)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        step = load_checkpoint(wrapper, optimizer, sampler, args, runtime) if args.resume else 0
        if not args.eval_only and step >= args.total_steps:
            raise ValueError(f"Checkpoint step {step} is already at or beyond {args.total_steps}.")
        write_manifest(args, runtime, versions, train_rows, validation_rows)
        initial_hash = verified_rank_adapter_sha256(
            wrapper.model, args, runtime, "initial"
        )
        if args.eval_only:
            evaluate(wrapper, validation_rows, step, args, runtime)
            if runtime.primary:
                print(f"Completed eval-only stage at step={step}.", flush=True)
            return
        if args.eval_before_train and step == 0:
            evaluate(wrapper, validation_rows, step, args, runtime)
            save_checkpoint(wrapper, optimizer, sampler, step, args, runtime, archive=True)

        metrics_path = args.output_dir / "training-metrics.jsonl"
        while step < args.total_steps:
            started = time.monotonic()
            pre_update_hash = verified_rank_adapter_sha256(
                wrapper.model, args, runtime, f"step-{step + 1}-pre"
            )
            rank_event(args, runtime, "step_start", step=step + 1)
            groups, group_metrics = build_step_groups(
                wrapper, train_rows, sampler, args, runtime
            )
            lr = learning_rate_for_step(step, args.learning_rate, args.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            metrics = train_step(
                wrapper, groups, parameters, optimizer, args, runtime, step + 1
            )
            post_update_hash = verified_rank_adapter_sha256(
                wrapper.model, args, runtime, f"step-{step + 1}-post"
            )
            step += 1
            metrics.update(group_metrics)
            metrics.update(
                {
                    "step": step,
                    "lr": lr,
                    "elapsed_s": time.monotonic() - started,
                    "stage": args.stage,
                    "policy_changed": pre_update_hash != post_update_hash,
                    "policy_sha256_post": post_update_hash,
                    "policy_sha256_pre": pre_update_hash,
                }
            )
            if args.stage == "lr0":
                if post_update_hash != initial_hash or pre_update_hash != post_update_hash:
                    raise RuntimeError("lr=0 changed the policy adapter weights.")
                metrics["policy_sha256"] = post_update_hash
            elif lr > 0.0 and pre_update_hash == post_update_hash:
                raise RuntimeError("A nonzero optimizer step did not change the policy adapter.")
            if runtime.primary:
                append_metrics(metrics_path, metrics)
                print("TRAIN " + json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
            archive = step % args.save_every == 0 or step == args.total_steps
            save_checkpoint(wrapper, optimizer, sampler, step, args, runtime, archive=archive)
            if args.stage in ("train", "extend") and step % args.eval_every == 0:
                evaluate(wrapper, validation_rows, step, args, runtime)
        if runtime.primary:
            print(f"Completed stage={args.stage} through step={step}.", flush=True)
    finally:
        if args.hang_trace_interval_s > 0:
            faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():
            dist.destroy_process_group()
        hang_trace.close()


if __name__ == "__main__":
    main()
