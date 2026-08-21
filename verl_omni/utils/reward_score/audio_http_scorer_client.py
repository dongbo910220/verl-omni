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
"""JSON HTTP client for audio reward services in a separate runtime."""

import asyncio
import base64
import math
from typing import Any

import aiohttp
import numpy as np

from verl_omni.utils.learning_trace import emit as trace_emit
from verl_omni.utils.learning_trace import span as trace_span

PROTOCOL_VERSION = "1"


class _RetryableHTTPError(RuntimeError):
    pass


def _scalar_metadata(extra_info: dict | None) -> dict[str, str | int | float | bool | None]:
    metadata = {}
    for key, value in (extra_info or {}).items():
        if key in {"audio", "audio_sample_rate"}:
            continue
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        elif hasattr(value, "item"):
            try:
                value = value.item()
            except (RuntimeError, ValueError):
                continue
        if value is None or isinstance(value, str | int | bool):
            metadata[str(key)] = value
        elif isinstance(value, float) and math.isfinite(value):
            metadata[str(key)] = value
    return metadata


def _serialize_request(solution_audio, ground_truth: str, extra_info: dict | None) -> dict[str, Any]:
    if not isinstance(solution_audio, tuple) or len(solution_audio) != 2:
        raise TypeError("Audio HTTP scorer expects solution_audio=(waveform, sample_rate).")
    waveform, sample_rate = solution_audio
    waveform = np.asarray(waveform, dtype="<f4").reshape(-1)
    if waveform.size == 0:
        raise ValueError("Audio HTTP scorer received an empty waveform.")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio HTTP scorer received a non-finite waveform.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int | float):
        raise TypeError("Audio HTTP scorer sample rate must be numeric.")
    if not math.isfinite(float(sample_rate)) or float(sample_rate) <= 0 or int(sample_rate) != sample_rate:
        raise ValueError(f"Audio HTTP scorer sample rate must be a positive integer, got {sample_rate!r}.")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "waveform_f32_base64": base64.b64encode(waveform.tobytes()).decode("ascii"),
        "num_samples": int(waveform.size),
        "sample_rate": int(sample_rate),
        "prompt": str(ground_truth or ""),
        "metadata": _scalar_metadata(extra_info),
    }


def _validate_response(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("Audio scorer response must be a JSON object.")
    if "error" in payload:
        raise RuntimeError(f"Audio scorer error: {payload['error']}")
    if "score" not in payload:
        raise RuntimeError("Audio scorer response is missing 'score'.")
    try:
        score = float(payload["score"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Audio scorer returned an invalid score: {payload['score']!r}.") from exc
    if not math.isfinite(score):
        raise RuntimeError(f"Audio scorer returned a non-finite score: {score!r}.")

    diagnostics = {}
    for key, value in payload.items():
        if key == "score":
            continue
        if value is None or isinstance(value, str | int | bool):
            diagnostics[str(key)] = value
        elif isinstance(value, float) and math.isfinite(value):
            diagnostics[str(key)] = value
        else:
            raise RuntimeError(f"Audio scorer diagnostic {key!r} must be a finite JSON scalar, got {value!r}.")
    return {"score": score, **diagnostics}


async def _session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    session = getattr(compute_score, "_session", None)
    session_loop = getattr(compute_score, "_session_loop", None)
    if session is None or session.closed or session_loop is not loop:
        if session is not None and not session.closed:
            await session.close()
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        compute_score._session = session
        compute_score._session_loop = loop
    return session


async def _request_score(server_url: str, payload: dict, timeout_s: float) -> dict:
    session = await _session()
    try:
        async with session.post(
            server_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as response:
            if response.status != 200:
                detail = await response.text()
                error = f"Audio scorer returned HTTP {response.status}: {detail}"
                if response.status in {408, 429} or 500 <= response.status < 600:
                    raise _RetryableHTTPError(error)
                raise RuntimeError(error)
            try:
                result = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, ValueError) as exc:
                raise RuntimeError("Audio scorer returned malformed JSON.") from exc
    except asyncio.TimeoutError as exc:
        raise _RetryableHTTPError(f"Audio scorer timed out after {timeout_s} seconds.") from exc
    return _validate_response(result)


async def compute_score(
    solution_audio,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    server_url: str,
    timeout_s: float = 120.0,
    max_retries: int = 2,
    retry_backoff_s: float = 0.5,
    **kwargs,
) -> dict:
    """Send one waveform to an external scorer and return its finite score."""
    del kwargs
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float) or not math.isfinite(float(timeout_s)):
        raise ValueError("timeout_s must be a finite number.")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive.")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise ValueError("max_retries must be an integer.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if (
        isinstance(retry_backoff_s, bool)
        or not isinstance(retry_backoff_s, int | float)
        or not math.isfinite(float(retry_backoff_s))
    ):
        raise ValueError("retry_backoff_s must be a finite number.")
    if retry_backoff_s < 0:
        raise ValueError("retry_backoff_s must be non-negative.")
    payload = _serialize_request(solution_audio, ground_truth, extra_info)
    step = int(payload["metadata"].get("global_steps") or 0)
    candidate_key = payload["metadata"].get("_learning_trace_candidate_key")

    last_error = None
    for attempt in range(max_retries + 1):
        with trace_span(
            "reward.http",
            step=step,
            payload={"candidate_key": candidate_key, "attempt": attempt + 1, "timeout_s": timeout_s},
        ):
            trace_emit(
                "reward.http_request",
                step=step,
                payload={
                    "candidate_key": candidate_key,
                    "attempt": attempt + 1,
                    "num_samples": payload["num_samples"],
                    "sample_rate": payload["sample_rate"],
                    "prompt": payload["prompt"],
                },
            )
            try:
                result = await _request_score(server_url, payload, timeout_s)
            except (_RetryableHTTPError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
                last_error = exc
                trace_emit(
                    "reward.http_retryable_error",
                    step=step,
                    payload={"candidate_key": candidate_key, "attempt": attempt + 1, "error": str(exc)},
                    status="error",
                )
            else:
                trace_emit(
                    "reward.http_response",
                    step=step,
                    payload={"candidate_key": candidate_key, "attempt": attempt + 1, "result": result},
                    status="ok",
                )
                return result
        if attempt < max_retries:
            await asyncio.sleep(retry_backoff_s * (2**attempt))
    raise RuntimeError(f"Audio scoring failed after {max_retries + 1} attempts: {last_error}") from last_error
