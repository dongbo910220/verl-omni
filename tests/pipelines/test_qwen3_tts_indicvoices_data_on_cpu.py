# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU contracts for the IndicVoices-R Hindi prompt reconstruction."""

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "examples/grpo_trainer/qwen3_tts/data_process/prepare_indicvoices_hindi_grpo.py"
SPEC = importlib.util.spec_from_file_location("indicvoices_hindi_grpo_data_test", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def _sample(index, *, scenario="Read", duration=2.0, snr=25.0, normalized=None, audio=b"wav"):
    return {
        "text": f"raw {index}",
        "normalized": normalized if normalized is not None else f"normalized {index}",
        "verbatim": f"verbatim {index}",
        "speaker_id": f"speaker-{index % 3}",
        "scenario": scenario,
        "task_name": "tts",
        "gender": "Female",
        "age_group": "18-30",
        "duration": duration,
        "snr": snr,
        "audio": {"bytes": audio, "path": f"{index}.wav"},
    }


def test_first_raw_read_window_is_filtered_before_heldout_collection():
    samples = [
        _sample(0, scenario="Extempore"),
        _sample(1, duration=0.9),
        _sample(2, snr=19.9),
        _sample(3, normalized="preferred text"),
        _sample(4, normalized="", audio=b""),
        _sample(5, normalized=""),
    ]

    train, validation, audit = prepare.collect_read_prompt_splits(
        samples,
        raw_read_limit=3,
        validation_count=1,
    )

    assert [record["source_index"] for record in train] == [3]
    assert train[0]["text"] == "preferred text"
    assert train[0]["audio_sha256"] == hashlib.sha256(b"wav").hexdigest()
    assert [record["source_index"] for record in validation] == [5]
    assert validation[0]["text"] == "raw 5"
    assert audit["raw_read_seen_including_validation_scan"] == 5
    assert audit["rejected_train"] == {"duration": 1, "snr": 1}
    assert audit["rejected_validation"] == {"audio": 1}


def test_split_preserves_source_order_and_builds_fixed_seed_pairs():
    samples = [_sample(index) for index in range(25)]
    train, validation, _ = prepare.collect_read_prompt_splits(
        samples,
        raw_read_limit=20,
        validation_count=5,
    )
    splits, audit = prepare.build_splits(
        train,
        validation,
        expected_read_prompts=None,
        expected_validation_prompts=5,
        gate_size=3,
    )

    assert [row["extra_info"]["source_index"] for row in splits["train"]] == list(range(20))
    assert [row["extra_info"]["source_index"] for row in splits["validation"]] == list(range(20, 25))
    assert len(splits["validation_seed_2"]) == 5
    assert splits["validation"][0]["extra_info"]["id"] == splits["validation_seed_2"][0]["extra_info"]["id"]
    assert (
        splits["validation"][0]["extra_info"]["generation_seed"]
        != splits["validation_seed_2"][0]["extra_info"]["generation_seed"]
    )
    assert len(splits["gate"]) == 3
    assert audit["train_read"] == 20
    assert not audit["raw_text_cross_split_overlap"]


def test_sparse_metadata_audio_evidence_replaces_embedded_bytes():
    sample = _sample(0, audio=b"")
    sample["_audio_verified"] = True
    train, validation, audit = prepare.collect_read_prompt_splits(
        [sample, _sample(1)],
        raw_read_limit=1,
        validation_count=1,
    )

    assert len(train) == 1
    assert train[0]["audio_sha256"] is None
    assert train[0]["audio_verified_from_parquet_footer"] is True
    assert len(validation) == 1
    assert not audit["rejected_train"]


def test_published_read_count_is_fail_closed():
    samples = [_sample(index) for index in range(25)]
    train, validation, _ = prepare.collect_read_prompt_splits(
        samples,
        raw_read_limit=20,
        validation_count=5,
    )

    try:
        prepare.build_splits(
            train,
            validation,
            expected_read_prompts=863,
            expected_validation_prompts=5,
            gate_size=3,
        )
    except ValueError as error:
        assert "Expected 863 published Read prompts" in str(error)
    else:
        raise AssertionError("A mismatched published Read-prompt count must fail.")
