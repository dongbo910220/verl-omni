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
"""Create deterministic Qwen3-TTS GRPO smoke data and speaker conditioning."""

import argparse
import json
from pathlib import Path

import pandas as pd

TRAIN_TEXTS = (
    "Please read this sentence at a calm and steady pace.",
    "A short speech sample checks the complete training path.",
    "Clear pronunciation makes this audio easy to inspect.",
    "The weather is pleasant and the morning train is on time.",
    "Four simple prompts are enough for one smoke-test batch.",
    "This second batch verifies another optimizer update.",
    "Generated speech is decoded before the reward is computed.",
    "The final checkpoint confirms that training completed.",
)
VALIDATION_TEXTS = (
    "This is fixed validation sample one.",
    "This is fixed validation sample two.",
    "This is fixed validation sample three.",
    "This is fixed validation sample four.",
)


def _row(text: str, sample_id: str, split: str) -> dict:
    extra_info = {"id": sample_id, "split": split}
    return {
        "data_source": "tts",
        "prompt": [{"role": "user", "content": text}],
        "reward_model": {"style": "model", "ground_truth": text},
        "extra_info": extra_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [_row(text, f"train-{index}", "train") for index, text in enumerate(TRAIN_TEXTS)]
    validation_rows = [_row(text, f"validation-{index}", "validation") for index, text in enumerate(VALIDATION_TEXTS)]
    pd.DataFrame(train_rows).to_parquet(args.output_dir / "train.parquet", index=False)
    pd.DataFrame(validation_rows).to_parquet(args.output_dir / "validation.parquet", index=False)

    # Qwen3-TTS Base expects a 1024-dimensional speaker x-vector. A unit-norm
    # deterministic fixture is sufficient for execution testing.
    speaker = [1.0 / 32.0] * 1024
    (args.output_dir / "speaker.json").write_text(json.dumps(speaker), encoding="utf-8")


if __name__ == "__main__":
    main()
