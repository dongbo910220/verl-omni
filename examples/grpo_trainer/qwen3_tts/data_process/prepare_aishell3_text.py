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
"""Prepare the public AISHELL-3 text recipe for Qwen3-TTS GRPO."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DATA_SOURCE = "SparkAudio/voxbox/aishell-3"
SOURCE_REVISION = "31221f1dfd7da5bde031eee7db31ac54966f7431"
SOURCE_URL = f"https://huggingface.co/datasets/SparkAudio/voxbox/resolve/{SOURCE_REVISION}/metadata/aishell-3.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_normalizer(cache_dir: str):
    try:
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
    except ImportError as exc:
        message = "Install WeTextProcessing before preparing AISHELL-3 data."
        raise ImportError(message) from exc
    normalizer = ZhNormalizer(
        cache_dir=str(Path(cache_dir).expanduser().resolve()),
        remove_erhua=False,
        remove_interjections=False,
        remove_puncts=True,
        overwrite_cache=False,
    )

    def normalize(text: str) -> str:
        return " ".join(normalizer.normalize(str(text)).lower().split())

    return normalize


def load_metadata(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"index", "text", "wav_path"}.difference(record)
            if missing:
                raise ValueError(f"Metadata line {line_number} is missing fields: {sorted(missing)}")
            records.append(record)
    if not records:
        raise ValueError("AISHELL-3 metadata is empty.")
    return records


def _make_row(
    record: dict,
    normalized_text: str,
    split: str,
    split_index: int,
    generation_seed: int | None = None,
    naturalness_guard: bool = False,
) -> dict:
    text = str(record["text"]).strip()
    extra_info = {
        "split": split,
        "id": str(record["index"]),
        "index": split_index,
        "source_index": str(record["index"]),
        "text": text,
        "raw_text": text,
        "reward_text": text,
        "normalized_text": normalized_text,
        "source_wav_path": str(record["wav_path"]),
        "naturalness_guard": naturalness_guard,
    }
    if generation_seed is not None:
        extra_info["generation_seed"] = generation_seed
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": text}],
        "ability": "text_to_speech",
        "reward_model": {"style": "rule", "ground_truth": text},
        "extra_info": extra_info,
    }


def build_splits(
    records: list[dict],
    normalize,
    train_size: int = 80_000,
    validation_size: int = 100,
    gate_size: int = 256,
    gate_seed: int = 42,
    guard_size: int = 20,
    validation_seed_base: int = 42_000,
    reference_index: int = 80_000,
    max_text_chars: int = 256,
) -> tuple[dict[str, list[dict]], dict]:
    if train_size <= 0 or validation_size <= 0 or gate_size <= 0:
        raise ValueError("train_size, validation_size, and gate_size must be positive.")
    if len(records) < train_size + validation_size:
        raise ValueError(f"Need at least {train_size + validation_size} records, found {len(records)}.")
    if not 0 <= guard_size <= validation_size:
        raise ValueError("guard_size must be between zero and validation_size.")

    normalized_cache = {}

    def normalized(record):
        text = str(record["text"]).strip()
        if not text:
            raise ValueError(f"Empty text for {record['index']}")
        if len(text) > max_text_chars:
            raise ValueError(f"Text exceeds {max_text_chars} characters for {record['index']}")
        if text not in normalized_cache:
            normalized_cache[text] = str(normalize(text)).strip()
            if len(normalized_cache) % 5_000 == 0:
                print(f"Normalized {len(normalized_cache):,} unique AISHELL-3 texts", flush=True)
        if not normalized_cache[text]:
            raise ValueError(f"Normalization produced empty text for {record['index']}")
        return normalized_cache[text]

    validation_records = records[-validation_size:]
    validation_norm = [normalized(record) for record in validation_records]
    validation_raw = [str(record["text"]).strip() for record in validation_records]
    validation_norm_set = set(validation_norm)
    validation_start = len(records) - validation_size
    train_records = []
    train_norm = []
    train_source_positions = []
    skipped_validation_overlap = []
    for source_position, record in enumerate(records[:validation_start]):
        norm = normalized(record)
        if norm in validation_norm_set:
            skipped_validation_overlap.append(
                {
                    "metadata_position": source_position,
                    "source_index": str(record["index"]),
                    "text": str(record["text"]).strip(),
                    "normalized_text": norm,
                }
            )
            continue
        train_records.append(record)
        train_norm.append(norm)
        train_source_positions.append(source_position)
        if len(train_records) == train_size:
            break
    if len(train_records) != train_size:
        raise ValueError(f"Could only build {len(train_records)} leakage-free training rows.")

    train_raw = [str(record["text"]).strip() for record in train_records]
    raw_overlap = set(train_raw).intersection(validation_raw)
    normalized_overlap = set(train_norm).intersection(validation_norm)
    if raw_overlap or normalized_overlap:
        raise ValueError(
            "AISHELL-3 train/validation overlap after fixed split: "
            f"raw={len(raw_overlap)}, normalized={len(normalized_overlap)}"
        )
    validation_norm_groups = {}
    for record, norm in zip(validation_records, validation_norm, strict=True):
        validation_norm_groups.setdefault(norm, []).append(str(record["index"]))
    validation_duplicate_groups = {
        norm: source_ids for norm, source_ids in validation_norm_groups.items() if len(source_ids) > 1
    }

    guard_rng = random.Random(f"aishell3-guard:{gate_seed}")
    guard_indices = set(guard_rng.sample(range(validation_size), guard_size))
    train_rows = [
        _make_row(record, norm, "train", index)
        for index, (record, norm) in enumerate(zip(train_records, train_norm, strict=True))
    ]
    validation_rows = [
        _make_row(
            record,
            norm,
            "validation",
            index,
            generation_seed=validation_seed_base + index,
            naturalness_guard=index in guard_indices,
        )
        for index, (record, norm) in enumerate(zip(validation_records, validation_norm, strict=True))
    ]

    unique_train_indices = []
    seen_normalized = set()
    for index, norm in enumerate(train_norm):
        if norm not in seen_normalized:
            unique_train_indices.append(index)
            seen_normalized.add(norm)
    if len(unique_train_indices) < gate_size:
        raise ValueError(f"Only {len(unique_train_indices)} unique train prompts are available for the Gate.")
    gate_rng = random.Random(f"aishell3-gate:{gate_seed}")
    selected_gate_indices = sorted(gate_rng.sample(unique_train_indices, gate_size))
    gate_rows = [
        _make_row(train_records[source_index], train_norm[source_index], "gate", gate_index)
        for gate_index, source_index in enumerate(selected_gate_indices)
    ]

    occupied_positions = set(train_source_positions) | set(range(len(records) - validation_size, len(records)))
    occupied_norm = set(train_norm) | set(validation_norm)
    reference_record = None
    for offset in range(len(records)):
        candidate_index = (reference_index + offset) % len(records)
        if candidate_index in occupied_positions:
            continue
        candidate = records[candidate_index]
        candidate_norm = normalized(candidate)
        if candidate_norm not in occupied_norm:
            reference_record = {
                "metadata_position": candidate_index,
                "source_index": str(candidate["index"]),
                "text": str(candidate["text"]).strip(),
                "normalized_text": candidate_norm,
                "wav_path": str(candidate["wav_path"]),
            }
            break
    if reference_record is None:
        raise ValueError("Could not select a disjoint AISHELL-3 reference-audio candidate.")

    audit = {
        "source_records": len(records),
        "train_raw_unique": len(set(train_raw)),
        "train_normalized_unique": len(set(train_norm)),
        "validation_raw_unique": len(set(validation_raw)),
        "validation_normalized_unique": len(set(validation_norm)),
        "validation_normalized_duplicate_groups": validation_duplicate_groups,
        "raw_cross_split_overlap": len(raw_overlap),
        "normalized_cross_split_overlap": len(normalized_overlap),
        "training_source_position_min": min(train_source_positions),
        "training_source_position_max": max(train_source_positions),
        "training_backfill_count": sum(position >= train_size for position in train_source_positions),
        "skipped_validation_overlap": skipped_validation_overlap,
        "gate_source_positions": [train_source_positions[index] for index in selected_gate_indices],
        "naturalness_guard_ids": [validation_rows[index]["extra_info"]["id"] for index in sorted(guard_indices)],
        "reference_candidate": reference_record,
        "text_length": {
            "train_min": min(map(len, train_raw)),
            "train_max": max(map(len, train_raw)),
            "validation_min": min(map(len, validation_raw)),
            "validation_max": max(map(len, validation_raw)),
        },
        "source_language_counts": dict(sorted(Counter(str(record.get("language", "")) for record in records).items())),
    }
    return {"train": train_rows, "validation": validation_rows, "gate": gate_rows}, audit


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalizer-cache-dir", default="./cache/zh_tn")
    parser.add_argument("--train-size", type=int, default=80_000)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--gate-size", type=int, default=256)
    parser.add_argument("--gate-seed", type=int, default=42)
    parser.add_argument("--guard-size", type=int, default=20)
    parser.add_argument("--validation-seed-base", type=int, default=42_000)
    parser.add_argument("--reference-index", type=int, default=80_000)
    parser.add_argument("--max-text-chars", type=int, default=256)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    args = parser.parse_args()

    records = load_metadata(args.metadata)
    normalize = build_normalizer(args.normalizer_cache_dir)
    splits, audit = build_splits(
        records,
        normalize,
        train_size=args.train_size,
        validation_size=args.validation_size,
        gate_size=args.gate_size,
        gate_seed=args.gate_seed,
        guard_size=args.guard_size,
        validation_seed_base=args.validation_seed_base,
        reference_index=args.reference_index,
        max_text_chars=args.max_text_chars,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_source": DATA_SOURCE,
        "source_revision": args.source_revision,
        "source_url": SOURCE_URL.replace(SOURCE_REVISION, args.source_revision),
        "metadata_path": str(args.metadata.resolve()),
        "metadata_sha256": _sha256(args.metadata),
        "gate_seed": args.gate_seed,
        "validation_seed_base": args.validation_seed_base,
        "audit": audit,
        "splits": {},
    }
    for split, rows in splits.items():
        path = args.output_dir / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        manifest["splits"][split] = {
            "path": str(path.resolve()),
            "count": len(rows),
            "content_sha256": _content_hash(rows),
            "file_sha256": _sha256(path),
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
