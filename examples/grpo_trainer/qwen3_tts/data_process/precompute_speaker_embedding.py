# Copyright 2026 Gulp AI Inc and/or its affiliates
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
"""Precompute one fixed Qwen3-TTS clone speaker embedding as JSON."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SPEAKER_SAMPLE_RATE = 24_000


def load_reference_audio(path: Path) -> tuple[np.ndarray, int]:
    import librosa

    waveform, _ = librosa.load(path, sr=SPEAKER_SAMPLE_RATE, mono=True)
    return np.asarray(waveform, dtype=np.float32), SPEAKER_SAMPLE_RATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device, dtype=torch.bfloat16)
    waveform, sample_rate = load_reference_audio(args.reference_audio)
    with torch.no_grad():
        embedding = model.model.extract_speaker_embedding(
            waveform,
            int(sample_rate),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(embedding.detach().reshape(-1).float().cpu().tolist()))
    print(f"Wrote {embedding.numel()}-dimensional speaker embedding to {args.output}")


if __name__ == "__main__":
    main()
