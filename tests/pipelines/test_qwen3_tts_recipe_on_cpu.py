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
"""CPU tests for the public AISHELL-3 Qwen3-TTS recipe."""

import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples/grpo_trainer/qwen3_tts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = _load("qwen3_tts_prepare_aishell3_test", EXAMPLE / "data_process/prepare_aishell3_text.py")
long_form = _load("qwen3_tts_long_form_test", EXAMPLE / "data_process/derive_aishell3_long_form.py")


def _record(index: int, text: str | None = None):
    return {
        "index": f"utt-{index}",
        "text": text or f"unique text {index}",
        "wav_path": f"audio/{index}.wav",
        "language": "zh",
    }


def test_aishell3_split_is_deterministic_disjoint_and_marks_guards():
    records = [_record(index) for index in range(30)]
    first, first_audit = prepare.build_splits(
        records,
        lambda text: text,
        train_size=10,
        validation_size=4,
        gate_size=4,
        guard_size=2,
        reference_index=12,
    )
    second, second_audit = prepare.build_splits(
        records,
        lambda text: text,
        train_size=10,
        validation_size=4,
        gate_size=4,
        guard_size=2,
        reference_index=12,
    )

    assert first == second
    assert first_audit == second_audit
    assert len(first["train"]) == 10
    assert len(first["validation"]) == 4
    assert len(first["gate"]) == 4
    assert sum(row["extra_info"]["naturalness_guard"] for row in first["validation"]) == 2
    train_text = {row["extra_info"]["normalized_text"] for row in first["train"]}
    validation_text = {row["extra_info"]["normalized_text"] for row in first["validation"]}
    assert train_text.isdisjoint(validation_text)
    assert first_audit["raw_cross_split_overlap"] == 0
    assert first_audit["normalized_cross_split_overlap"] == 0


def test_aishell3_split_skips_validation_leakage_and_backfills_train():
    records = [_record(index) for index in range(20)]
    records[2]["text"] = records[-2]["text"]

    splits, audit = prepare.build_splits(
        records,
        lambda text: text.lower(),
        train_size=8,
        validation_size=2,
        gate_size=2,
        guard_size=1,
        reference_index=10,
    )

    assert len(splits["train"]) == 8
    assert [item["metadata_position"] for item in audit["skipped_validation_overlap"]] == [2]
    assert audit["training_backfill_count"] == 1


def test_long_form_gate_uses_only_predeclared_length_buckets():
    rows = []
    index = 0
    for length in (18, 19, 20, 21, 22, 23, 24, 25):
        for variant in range(2):
            text = chr(97 + index) * length
            rows.append(
                {
                    "prompt": [{"role": "user", "content": text}],
                    "extra_info": {
                        "id": f"row-{index}",
                        "text": text,
                        "normalized_text": f"{text}-{variant}",
                    },
                }
            )
            index += 1

    splits, audit = long_form.derive_long_form(rows, gate_size=8, gate_seed=43)

    assert len(splits["train_long_form"]) == len(rows)
    assert len(splits["gate_long_form"]) == 8
    assert {row["extra_info"]["difficulty_bucket"] for row in splits["gate_long_form"]} == {
        "18-19",
        "20-21",
        "22-23",
        "24+",
    }
    assert all(bucket["gate_samples"] == 2 for bucket in audit["buckets"].values())


def test_recipe_keeps_complete_validation_and_one_reward_signal():
    config_path = EXAMPLE / "config/qwen3_tts_aishell3_grpo.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert config["data"]["val_max_samples"] == 100
    assert config["data"]["validation_shuffle"] is False
    assert config["actor_rollout_ref"]["rollout"]["n"] == 4
    assert config["actor_rollout_ref"]["rollout"]["val_kwargs"]["n"] == 1
    assert config["actor_rollout_ref"]["actor"]["use_kl_loss"] is False
    assert config["algorithm"]["use_kl_in_reward"] is False
    assert config["reward"]["custom_reward_function"]["name"] == "compute_score"
    assert config["trainer"]["log_val_generations"] == 100
    assert config["trainer"]["test_freq"] == 20
    assert config["trainer"]["total_training_steps"] == 100


def test_launcher_enforces_validation_size_and_step20_schedule():
    launcher = (EXAMPLE / "run_qwen3_tts_aishell3_grpo.sh").read_text()

    assert '[[ "${VAL_ROWS}" == "100" ]]' in launcher
    assert '[[ "${GATE_ROWS}" == "256" ]]' in launcher
    assert '"actor_rollout_ref.rollout.val_kwargs.n=4"' in launcher
    assert '"actor_rollout_ref.actor.optim.lr=0.0"' in launcher
    assert '"actor_rollout_ref.actor.optim.lr=2.0e-7"' in launcher
    assert '"actor_rollout_ref.actor.optim.lr=1.0e-6"' in launcher
    assert '"trainer.save_freq=20"' in launcher
    assert '"trainer.test_freq=20"' in launcher
    assert 'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"' in launcher
