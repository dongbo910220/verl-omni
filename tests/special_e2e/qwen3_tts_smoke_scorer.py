#!/usr/bin/env python3
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
"""Deterministic duration reward service for the Qwen3-TTS execution smoke."""

import argparse
import base64
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _Handler(BaseHTTPRequestHandler):
    server_version = "Qwen3TTSSmokeScorer/1.0"

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(404, {"error": "not found"})
            return
        self._write_json(200, {"status": "ready", "reward_mode": "duration_smoke_only"})

    def do_POST(self) -> None:
        if self.path != "/score":
            self._write_json(404, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            if payload.get("protocol_version") != "1":
                raise ValueError("protocol_version must be '1'")
            num_samples = int(payload["num_samples"])
            sample_rate = int(payload["sample_rate"])
            waveform = base64.b64decode(payload["waveform_f32_base64"], validate=True)
            if num_samples <= 0 or sample_rate <= 0 or len(waveform) != num_samples * 4:
                raise ValueError("invalid float32 waveform shape")
            duration_s = num_samples / sample_rate
            if not math.isfinite(duration_s):
                raise ValueError("duration must be finite")
            result = {"score": duration_s, "duration_s": duration_s, "reward_mode": "duration_smoke_only"}
            with self.server.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, allow_nan=False) + "\n")
            self._write_json(200, result)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._write_json(400, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.log_path = args.log
    server.serve_forever()


if __name__ == "__main__":
    main()
