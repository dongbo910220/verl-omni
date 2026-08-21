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
"""Opt-in structured traces for learning the real Qwen3-TTS GRPO path.

The tracer is deliberately inert unless ``VERL_OMNI_LEARNING_TRACE_DIR`` is
set. Each process writes its own JSONL file so Ray workers never contend for a
shared append handle.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import os
import re
import socket
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_SCHEMA_VERSION = 1
TRACE_DIR_ENV = "VERL_OMNI_LEARNING_TRACE_DIR"
TRACE_RUN_ID_ENV = "VERL_OMNI_LEARNING_TRACE_RUN_ID"
TRACE_PHASE_ENV = "VERL_OMNI_LEARNING_TRACE_PHASE"
TRACE_ROLE_ENV = "VERL_OMNI_LEARNING_TRACE_ROLE"
TRACE_EXPECTED_STEP_ENV = "VERL_OMNI_LEARNING_TRACE_EXPECTED_STEP"

_LOCK = threading.Lock()
_SEQUENCE = 0
_PROCESS_START_PID: int | None = None
_SPAN_STACK: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "verl_omni_learning_trace_span_stack", default=()
)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "credentials",
    "password",
    "refresh_token",
    "access_token",
    "secret",
}


def enabled() -> bool:
    """Return whether structured tracing is enabled for this process."""

    return bool(os.environ.get(TRACE_DIR_ENV, "").strip())


def expected_step(default: int | None = None) -> int | None:
    """Return the phase's expected global step, if the launcher supplied one."""

    raw = os.environ.get(TRACE_EXPECTED_STEP_ENV)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _safe_component(value: str, fallback: str) -> str:
    value = _SAFE_COMPONENT.sub("_", value.strip()).strip("._")
    return value or fallback


def _run_id() -> str:
    return _safe_component(os.environ.get(TRACE_RUN_ID_ENV, "learning-trace"), "learning-trace")


def _phase() -> str:
    return _safe_component(os.environ.get(TRACE_PHASE_ENV, "unspecified"), "unspecified")


def _rank_info() -> dict[str, int | None]:
    def parse(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    rank = parse("RANK")
    local_rank = parse("LOCAL_RANK")
    world_size = parse("WORLD_SIZE")
    torch = sys.modules.get("torch")
    try:
        dist = torch.distributed if torch is not None else None
        if dist is not None and dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
    except Exception:
        pass
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def _trace_path(identity: Mapping[str, Any]) -> Path:
    root = Path(os.environ[TRACE_DIR_ENV]).expanduser().resolve()
    rank = identity.get("rank")
    rank_part = "na" if rank is None else str(rank)
    filename = f"{identity['hostname']}-pid{identity['pid']}-rank{rank_part}.jsonl"
    return root / "raw" / _run_id() / _phase() / filename


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith("_password") or normalized.endswith("_secret")


def _local_tensor(value):
    try:
        if hasattr(value, "to_local"):
            value = value.to_local()
    except Exception:
        pass
    return value


def _sample_flat_tensor(tensor, limit: int):
    import torch

    tensor = _local_tensor(tensor).detach().reshape(-1)
    if tensor.numel() <= limit:
        return tensor
    indices = torch.linspace(0, tensor.numel() - 1, steps=limit, device=tensor.device).long()
    return tensor.index_select(0, indices)


def summarize_tensor(value: Any, *, exact_limit: int = 1024, sample_limit: int = 256) -> dict[str, Any]:
    """Return bounded tensor evidence with explicit exact-vs-sampled semantics."""

    import torch

    tensor = _local_tensor(value)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach()
    numel = int(tensor.numel())
    if tensor.device.type == "meta":
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": "meta",
            "numel": numel,
            "materialized": False,
        }
    sampled = _sample_flat_tensor(tensor, max(1, sample_limit)).cpu() if numel else tensor.reshape(-1).cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    try:
        digest.update(sampled.contiguous().numpy().tobytes())
    except Exception:
        digest.update(repr(sampled.tolist()).encode())

    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": numel,
        "sha256": digest.hexdigest(),
        "sha256_scope": "full" if numel <= sample_limit else f"evenly_sampled_{sample_limit}",
    }
    if numel and (tensor.is_floating_point() or tensor.is_complex()):
        stats = sampled.float()
        finite = torch.isfinite(stats)
        result["stats_scope"] = "full" if numel <= sample_limit else f"evenly_sampled_{sample_limit}"
        result["finite"] = int(finite.sum().item())
        if bool(finite.any()):
            finite_stats = stats[finite]
            result.update(
                min=float(finite_stats.min().item()),
                max=float(finite_stats.max().item()),
                mean=float(finite_stats.mean().item()),
                l2=float(torch.linalg.vector_norm(finite_stats).item()),
            )
    if numel <= exact_limit:
        result["values"] = tensor.cpu().tolist()
        result["values_scope"] = "full"
    elif numel:
        result["sample_values"] = sampled.tolist()
        result["values_scope"] = result["sha256_scope"]
    return result


def _sanitize(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    if key is not None and _is_sensitive_key(key):
        return "<redacted>"
    if depth > 12:
        return "<max-depth>"
    if value is None or isinstance(value, bool | int | str):
        return value if not isinstance(value, str) or len(value) <= 8192 else value[:8192] + "<truncated>"
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return summarize_tensor(value)
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return summarize_tensor(value)
        if isinstance(value, np.generic):
            return _sanitize(value.item(), key=key, depth=depth + 1)
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item_value, key=str(item_key), depth=depth + 1)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list | tuple | set):
        items = list(value)
        limit = 20000
        output = [_sanitize(item, depth=depth + 1) for item in items[:limit]]
        if len(items) > limit:
            output.append({"truncated_items": len(items) - limit})
        return output
    if hasattr(value, "item"):
        try:
            return _sanitize(value.item(), key=key, depth=depth + 1)
        except Exception:
            pass
    return repr(value)[:8192]


def _next_sequence() -> int:
    global _SEQUENCE
    _SEQUENCE += 1
    return _SEQUENCE


def _identity() -> dict[str, Any]:
    rank_info = _rank_info()
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": _run_id(),
        "phase": _phase(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "role": os.environ.get(TRACE_ROLE_ENV, os.environ.get("ROLE", "unknown")),
        **rank_info,
    }


def _record(
    event: str,
    *,
    step: int | None,
    payload: Any,
    call_id: str,
    parent_call_id: str | None,
    status: str | None,
) -> dict[str, Any]:
    identity = _identity()
    record = {
        **identity,
        "sequence": _next_sequence(),
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_unix_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "event": event,
        "step": step,
        "call_id": call_id,
        "parent_call_id": parent_call_id,
        "status": status,
        "payload": _sanitize(payload),
    }
    return record


def _write_record(record: Mapping[str, Any]) -> None:
    path = _trace_path(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()


def emit(
    event: str,
    *,
    step: int | None = None,
    payload: Any = None,
    call_id: str | None = None,
    parent_call_id: str | None = None,
    status: str | None = None,
) -> str | None:
    """Append one structured event and return its call identifier."""

    if not enabled():
        return None
    if not event or not isinstance(event, str):
        raise ValueError("trace event must be a non-empty string")
    stack = _SPAN_STACK.get()
    parent_call_id = parent_call_id if parent_call_id is not None else (stack[-1] if stack else None)
    call_id = call_id or f"{os.getpid()}-{uuid.uuid4().hex[:12]}"

    global _PROCESS_START_PID, _SEQUENCE
    with _LOCK:
        if _PROCESS_START_PID != os.getpid():
            _SEQUENCE = 0
            start_id = f"{os.getpid()}-process-start"
            _write_record(
                _record(
                    "process.start",
                    step=None,
                    payload={"argv": sys.argv, "cwd": os.getcwd(), "python": sys.executable},
                    call_id=start_id,
                    parent_call_id=None,
                    status="ok",
                )
            )
            _PROCESS_START_PID = os.getpid()
        _write_record(
            _record(
                event,
                step=step,
                payload=payload,
                call_id=call_id,
                parent_call_id=parent_call_id,
                status=status,
            )
        )
    return call_id


@contextmanager
def span(event: str, *, step: int | None = None, payload: Any = None) -> Iterator[str | None]:
    """Emit paired ``.begin``/``.end`` records around a real runtime call."""

    if not enabled():
        yield None
        return
    parent = _SPAN_STACK.get()[-1] if _SPAN_STACK.get() else None
    call_id = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    emit(f"{event}.begin", step=step, payload=payload, call_id=call_id, parent_call_id=parent, status="begin")
    token = _SPAN_STACK.set((*_SPAN_STACK.get(), call_id))
    start = time.monotonic_ns()
    try:
        yield call_id
    except BaseException as exc:
        _SPAN_STACK.reset(token)
        emit(
            f"{event}.end",
            step=step,
            payload={
                "duration_ns": time.monotonic_ns() - start,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
            call_id=call_id,
            parent_call_id=parent,
            status="error",
        )
        raise
    else:
        _SPAN_STACK.reset(token)
        emit(
            f"{event}.end",
            step=step,
            payload={"duration_ns": time.monotonic_ns() - start},
            call_id=call_id,
            parent_call_id=parent,
            status="ok",
        )


def extract_step(data: Any = None, fallback: int | None = None) -> int | None:
    """Best-effort extraction of a propagated global step from TensorDict metadata."""

    for key in ("learning_trace_global_step", "global_steps", "global_step"):
        for getter in (
            lambda key=key: data[key],
            lambda key=key: data.get(key),
        ):
            try:
                value = getter()
            except Exception:
                continue
            for attr in ("data", "value"):
                try:
                    value = getattr(value, attr)
                except Exception:
                    pass
            try:
                if hasattr(value, "reshape"):
                    value = value.reshape(-1)[0]
                if hasattr(value, "item"):
                    value = value.item()
                return int(value)
            except Exception:
                continue
    return expected_step(fallback)


def _parameter_group(name: str) -> str:
    cleaned = name.replace("_fsdp_wrapped_module.", "").lstrip(".")
    parts = [part for part in cleaned.split(".") if part]
    return ".".join(parts[:3]) if parts else "<root>"


def summarize_module_parameters(module: Any, *, include_norms: bool) -> dict[str, Any]:
    """Summarize every local parameter shard and fingerprint bounded samples."""

    import torch

    dtype_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    total_numel = trainable_numel = grad_numel = 0
    parameter_count = trainable_count = gradients_present = 0
    parameter_sq = gradient_sq = 0.0
    parameter_max_abs = gradient_max_abs = 0.0
    fingerprint = hashlib.sha256()
    representative: list[dict[str, Any]] = []

    for name, parameter in module.named_parameters():
        parameter_count += 1
        local = _local_tensor(parameter.detach())
        local_numel = int(local.numel())
        total_numel += local_numel
        dtype_counts[str(local.dtype)] += local_numel
        group_counts[_parameter_group(name)] += local_numel
        if parameter.requires_grad:
            trainable_count += 1
            trainable_numel += local_numel

        fingerprint.update(name.encode())
        fingerprint.update(str(tuple(parameter.shape)).encode())
        fingerprint.update(str(local.dtype).encode())
        if local.device.type == "meta":
            if len(representative) < 16 and parameter.requires_grad:
                representative.append(
                    {
                        "name": name,
                        "global_shape": list(parameter.shape),
                        "local_numel": local_numel,
                        "dtype": str(local.dtype),
                        "materialized": False,
                    }
                )
            continue
        if local_numel:
            sampled = _sample_flat_tensor(local, 8).cpu()
            try:
                fingerprint.update(sampled.contiguous().numpy().tobytes())
            except Exception:
                fingerprint.update(repr(sampled.tolist()).encode())
            if len(representative) < 16 and parameter.requires_grad:
                representative.append(
                    {
                        "name": name,
                        "global_shape": list(parameter.shape),
                        "local_numel": local_numel,
                        "dtype": str(local.dtype),
                        "sample_values": sampled.tolist(),
                    }
                )
            if include_norms:
                norm = float(torch.linalg.vector_norm(local.float()).item())
                parameter_sq += norm * norm
                parameter_max_abs = max(parameter_max_abs, float(local.abs().max().item()))

        gradient = parameter.grad
        if gradient is not None:
            gradient = _local_tensor(gradient.detach())
            gradients_present += 1
            grad_numel += int(gradient.numel())
            if include_norms and gradient.numel():
                grad_norm = float(torch.linalg.vector_norm(gradient.float()).item())
                gradient_sq += grad_norm * grad_norm
                gradient_max_abs = max(gradient_max_abs, float(gradient.abs().max().item()))

    result: dict[str, Any] = {
        "parameter_count": parameter_count,
        "local_parameter_numel": total_numel,
        "trainable_parameter_count": trainable_count,
        "local_trainable_numel": trainable_numel,
        "gradients_present": gradients_present,
        "local_gradient_numel": grad_numel,
        "dtype_numel": dict(sorted(dtype_counts.items())),
        "largest_groups_by_local_numel": [
            {"group": group, "local_numel": count} for group, count in group_counts.most_common(24)
        ],
        "parameter_fingerprint": fingerprint.hexdigest(),
        "representative_trainable_parameters": representative,
    }
    if include_norms:
        result.update(
            local_parameter_l2=math.sqrt(parameter_sq),
            local_parameter_max_abs=parameter_max_abs,
            local_gradient_l2=math.sqrt(gradient_sq),
            local_gradient_max_abs=gradient_max_abs,
        )
    return result


def traced_weight_iterator(parameters, *, step: int | None, payload: Mapping[str, Any] | None = None):
    """Wrap a weight iterator without materializing it or changing iteration order."""

    if not enabled():
        return parameters

    def iterator():
        digest = hashlib.sha256()
        dtype_counts: Counter[str] = Counter()
        tensor_count = total_numel = 0
        completed = False
        with span("weights.sender_iteration", step=step, payload=payload):
            try:
                for name, tensor in parameters:
                    local = _local_tensor(tensor.detach())
                    tensor_count += 1
                    total_numel += int(local.numel())
                    dtype_counts[str(local.dtype)] += int(local.numel())
                    digest.update(str(name).encode())
                    digest.update(str(tuple(local.shape)).encode())
                    digest.update(str(local.dtype).encode())
                    if local.numel():
                        sampled = _sample_flat_tensor(local, 8).cpu()
                        try:
                            digest.update(sampled.contiguous().numpy().tobytes())
                        except Exception:
                            digest.update(repr(sampled.tolist()).encode())
                    yield name, tensor
                completed = True
            finally:
                emit(
                    "weights.sender_summary",
                    step=step,
                    payload={
                        **dict(payload or {}),
                        "completed": completed,
                        "tensor_count": tensor_count,
                        "total_numel": total_numel,
                        "dtype_numel": dict(sorted(dtype_counts.items())),
                        "fingerprint": digest.hexdigest(),
                    },
                    status="ok" if completed else "partial",
                )

    return iterator()
