#!/usr/bin/env python3
"""Rebuild the published 863-prompt IndicVoices-R Hindi GRPO split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DATA_SOURCE = "ai4bharat/indicvoices_r"
SOURCE_REVISION = "5f4495c91d500742a58d1be2ab07d77f73c0acf8"
CONFIG_NAME = "Hindi"
ACQUISITION_SOURCE = "SPRINGLab/IndicVoices-R_Hindi"
ACQUISITION_REVISION = "8f9913669e3505acd38b97fa38bd027f49afeeef"
MIN_DURATION = 1.0
MAX_DURATION = 14.0
MIN_SNR = 20.0
RAW_READ_LIMIT = 1_000
EXPECTED_READ_PROMPTS = 863
VALIDATION_COUNT = 100
EXPERIMENT_SEED = 42
METADATA_COLUMNS = (
    "text",
    "verbatim",
    "normalized",
    "speaker_id",
    "scenario",
    "task_name",
    "gender",
    "age_group",
    "snr",
    "duration",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _class_names(parquet_file, field: str) -> list[str] | None:
    metadata = parquet_file.schema_arrow.metadata or {}
    huggingface = metadata.get(b"huggingface")
    if not huggingface:
        return None
    features = json.loads(huggingface)["info"]["features"]
    return features.get(field, {}).get("names")


def load_metadata_samples(metadata_dir: Path) -> tuple[list[dict], dict]:
    """Load a pinned public re-shard without materializing its embedded audio bytes."""
    import pyarrow.parquet as pq

    manifest_path = metadata_dir / "metadata-manifest-00-10.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset") != ACQUISITION_SOURCE:
        raise ValueError(f"Unexpected acquisition dataset in {manifest_path}: {manifest.get('dataset')}")
    if manifest.get("revision") != ACQUISITION_REVISION:
        raise ValueError(f"Unexpected acquisition revision in {manifest_path}: {manifest.get('revision')}")

    reports = manifest.get("shards") or []
    expected_shards = [f"data/train-{index:05d}-of-00010.parquet" for index in range(10)]
    if [report.get("shard") for report in reports] != expected_shards:
        raise ValueError(f"Acquisition manifest does not contain the ten ordered train shards: {manifest_path}")

    samples = []
    shard_audit = []
    for report in reports:
        if report.get("x_repo_commit") != ACQUISITION_REVISION:
            raise ValueError(f"Unpinned acquisition response for {report.get('shard')}: {report.get('x_repo_commit')}")
        if report.get("audio_non_null_rows") != report.get("rows"):
            raise ValueError(f"Incomplete audio column reported for {report.get('shard')}")

        path = metadata_dir / f"{Path(report['shard']).stem}.metadata.parquet"
        parquet_file = pq.ParquetFile(path)
        scenario_names = _class_names(parquet_file, "scenario")
        gender_names = _class_names(parquet_file, "gender")
        age_names = _class_names(parquet_file, "age_group")
        if scenario_names != ["Extempore", "Read"]:
            raise ValueError(f"Unexpected scenario labels in {path}: {scenario_names}")
        table = parquet_file.read(columns=list(METADATA_COLUMNS))
        if table.num_rows != report.get("rows"):
            raise ValueError(f"Row-count mismatch for {path}: {table.num_rows} != {report.get('rows')}")
        for row in table.to_pylist():
            row["scenario"] = scenario_names[row["scenario"]]
            if gender_names:
                row["gender"] = gender_names[row["gender"]]
            if age_names:
                row["age_group"] = age_names[row["age_group"]]
            row["_audio_verified"] = True
            samples.append(row)
        shard_audit.append(
            {
                "shard": report["shard"],
                "rows": report["rows"],
                "remote_size": report["remote_size"],
                "etag": report.get("etag"),
                "x_repo_commit": report.get("x_repo_commit"),
                "fetched_bytes": report["fetched_bytes"],
            }
        )

    if len(samples) != 26_318:
        raise ValueError(f"Expected 26,318 source rows from the acquisition mirror, found {len(samples)}")
    return samples, {
        "mode": "sparse_parquet_metadata",
        "dataset": ACQUISITION_SOURCE,
        "revision": ACQUISITION_REVISION,
        "logical_source": DATA_SOURCE,
        "logical_source_revision": SOURCE_REVISION,
        "row_order_equivalence": "inferred from ordered re-shard; original gated bytes were unavailable",
        "audio_evidence": "Parquet footer statistics prove audio.bytes has zero nulls in every row group",
        "metadata_manifest": str(manifest_path.resolve()),
        "metadata_manifest_sha256": _sha256_file(manifest_path),
        "rows": len(samples),
        "shards": shard_audit,
    }


def _record(sample: dict, source_index: int) -> tuple[dict | None, str | None]:
    duration = sample.get("duration", 0.0)
    if not (MIN_DURATION <= duration <= MAX_DURATION):
        return None, "duration"
    snr = sample.get("snr", 0.0)
    if snr is not None and snr < MIN_SNR:
        return None, "snr"
    text = str(sample.get("normalized") or sample.get("text") or "").strip()
    if not text:
        return None, "text"

    audio = sample.get("audio") or {}
    audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
    audio_verified = bool(sample.get("_audio_verified"))
    if not audio_bytes and not audio_verified:
        return None, "audio"

    return {
        "source_index": source_index,
        "text": text,
        "raw_text": str(sample.get("text") or "").strip(),
        "verbatim": str(sample.get("verbatim") or "").strip(),
        "normalized": str(sample.get("normalized") or "").strip(),
        "speaker_id": str(sample.get("speaker_id") or ""),
        "scenario": str(sample.get("scenario") or ""),
        "task_name": str(sample.get("task_name") or ""),
        "gender": str(sample.get("gender") or ""),
        "age_group": str(sample.get("age_group") or ""),
        "duration": float(duration),
        "snr": None if snr is None else float(snr),
        "audio_path": str(audio.get("path") or "") if isinstance(audio, dict) else "",
        "audio_bytes": len(audio_bytes) if audio_bytes else None,
        "audio_sha256": _sha256_bytes(audio_bytes) if audio_bytes else None,
        "audio_verified_from_parquet_footer": audio_verified,
    }, None


def collect_read_prompt_splits(
    samples,
    *,
    raw_read_limit: int = RAW_READ_LIMIT,
    validation_count: int = VALIDATION_COUNT,
) -> tuple[list[dict], list[dict], dict]:
    """Filter the first 1,000 raw Read rows, then hold out the next 100 valid Read rows."""
    train = []
    validation = []
    rejected_train = Counter()
    rejected_validation = Counter()
    raw_read_seen = 0
    scanned = 0
    train_last_source_index = None

    for source_index, sample in enumerate(samples):
        scanned += 1
        if str(sample.get("scenario") or "").strip().casefold() != "read":
            continue
        raw_read_seen += 1
        in_train_window = raw_read_seen <= raw_read_limit
        record, reason = _record(sample, source_index)
        if in_train_window:
            train_last_source_index = source_index
            if reason:
                rejected_train[reason] += 1
            else:
                train.append(record)
            continue

        if reason:
            rejected_validation[reason] += 1
            continue
        validation.append(record)
        if len(validation) >= validation_count:
            break

    if raw_read_seen < raw_read_limit:
        raise ValueError(f"Found only {raw_read_seen}/{raw_read_limit} raw Read rows")
    if len(validation) != validation_count:
        raise ValueError(f"Collected only {len(validation)}/{validation_count} held-out Read prompts")
    audit = {
        "scanned": scanned,
        "raw_read_limit": raw_read_limit,
        "raw_read_seen_including_validation_scan": raw_read_seen,
        "train": len(train),
        "validation": len(validation),
        "train_last_source_index": train_last_source_index,
        "validation_first_source_index": validation[0]["source_index"],
        "validation_last_source_index": validation[-1]["source_index"],
        "rejected_train": dict(sorted(rejected_train.items())),
        "rejected_validation": dict(sorted(rejected_validation.items())),
        "task_counts_train": dict(sorted(Counter(row["task_name"] for row in train).items())),
        "task_counts_validation": dict(sorted(Counter(row["task_name"] for row in validation).items())),
    }
    return train, validation, audit


def _row(record: dict, split: str, index: int, *, generation_seed: int | None = None) -> dict:
    text = record["text"]
    info = {
        "split": split,
        "id": f"indicvoices-hi-{record['source_index']:08d}",
        "index": index,
        "source_index": record["source_index"],
        "text": text,
        "raw_text": record["raw_text"],
        "reward_text": text,
        "normalized_text": record["normalized"],
        "scenario": record["scenario"],
        "task_name": record["task_name"],
        "speaker_id": record["speaker_id"],
        "duration": record["duration"],
        "snr": record["snr"],
        "audio_sha256": record["audio_sha256"],
    }
    if generation_seed is not None:
        info["generation_seed"] = generation_seed
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": text}],
        "ability": "text_to_speech",
        "reward_model": {"style": "rule", "ground_truth": text},
        "extra_info": info,
    }


def build_splits(
    train: list[dict],
    validation: list[dict],
    *,
    seed: int = EXPERIMENT_SEED,
    expected_read_prompts: int | None = EXPECTED_READ_PROMPTS,
    expected_validation_prompts: int = VALIDATION_COUNT,
    gate_size: int = 256,
    validation_seed_base: int = 42_000,
    validation_seed_2_base: int = 142_000,
) -> tuple[dict[str, list[dict]], dict]:
    if expected_read_prompts is not None and len(train) != expected_read_prompts:
        raise ValueError(f"Expected {expected_read_prompts} published Read prompts, found {len(train)}")
    if len(validation) != expected_validation_prompts:
        raise ValueError(f"Expected {expected_validation_prompts} held-out Read prompts, found {len(validation)}")
    if len(train) < gate_size:
        raise ValueError(f"Need {gate_size} Read prompts for the gate, found {len(train)}")

    gate_rng = random.Random(f"indicvoices-hindi-gate:{seed}")
    gate_indices = sorted(gate_rng.sample(range(len(train)), gate_size))
    splits = {
        "train": [_row(record, "train", index) for index, record in enumerate(train)],
        "validation": [
            _row(record, "validation", index, generation_seed=validation_seed_base + index)
            for index, record in enumerate(validation)
        ],
        "validation_seed_2": [
            _row(record, "validation", index, generation_seed=validation_seed_2_base + index)
            for index, record in enumerate(validation)
        ],
        "gate": [_row(train[source_index], "gate", index) for index, source_index in enumerate(gate_indices)],
    }
    train_text = {row["text"] for row in train}
    validation_text = {row["text"] for row in validation}
    train_normalized = {row["normalized"] for row in train if row["normalized"]}
    validation_normalized = {row["normalized"] for row in validation if row["normalized"]}
    audit = {
        "train_read": len(train),
        "validation": len(validation),
        "validation_seed_2": len(validation),
        "gate": len(gate_indices),
        "train_read_source_indices": [row["source_index"] for row in train],
        "validation_source_indices": [row["source_index"] for row in validation],
        "gate_train_positions": gate_indices,
        "raw_text_cross_split_overlap": sorted(train_text & validation_text),
        "normalized_text_cross_split_overlap": sorted(train_normalized & validation_normalized),
    }
    return splits, audit


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--raw-read-limit", type=int, default=RAW_READ_LIMIT)
    parser.add_argument("--seed", type=int, default=EXPERIMENT_SEED)
    parser.add_argument("--expected-read-prompts", type=int, default=EXPECTED_READ_PROMPTS)
    parser.add_argument("--validation-count", type=int, default=VALIDATION_COUNT)
    parser.add_argument("--gate-size", type=int, default=256)
    args = parser.parse_args()

    if args.metadata_dir:
        samples, acquisition = load_metadata_samples(args.metadata_dir)
    else:
        from datasets import Audio, load_dataset

        samples = load_dataset(
            DATA_SOURCE,
            CONFIG_NAME,
            split="train",
            streaming=True,
            revision=args.source_revision,
            token=True,
        ).cast_column("audio", Audio(decode=False))
        acquisition = {
            "mode": "authenticated_huggingface_stream",
            "dataset": DATA_SOURCE,
            "revision": args.source_revision,
        }

    train, validation, collection_audit = collect_read_prompt_splits(
        samples,
        raw_read_limit=args.raw_read_limit,
        validation_count=args.validation_count,
    )
    splits, split_audit = build_splits(
        train,
        validation,
        seed=args.seed,
        expected_read_prompts=args.expected_read_prompts,
        expected_validation_prompts=args.validation_count,
        gate_size=args.gate_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "source_candidates.jsonl"
    with source_path.open("w", encoding="utf-8") as handle:
        for split, records in (("train", train), ("validation", validation)):
            for record in records:
                handle.write(json.dumps(record | {"selection_split": split}, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 2,
        "data_source": DATA_SOURCE,
        "config_name": CONFIG_NAME,
        "source_revision": args.source_revision,
        "acquisition": acquisition,
        "filter_contract": {
            "source_order": "streaming train order",
            "scenario": "Read",
            "scenario_selection_precedes_quality_filters": True,
            "raw_read_limit": args.raw_read_limit,
            "duration": [MIN_DURATION, MAX_DURATION],
            "minimum_snr": MIN_SNR,
            "text_preference": ["normalized", "text"],
            "requires_audio": True,
            "expected_train_after_filter": args.expected_read_prompts,
            "validation": f"next {args.validation_count} quality-passing Read rows after raw training window",
            "training_shuffle": "performed by the trainer with the fixed experiment seed",
        },
        "source_candidates_path": str(source_path.resolve()),
        "source_candidates_sha256": _sha256_file(source_path),
        "collection_audit": collection_audit,
        "split_audit": split_audit,
        "splits": {},
    }
    for split, rows in splits.items():
        path = args.output_dir / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        manifest["splits"][split] = {
            "path": str(path.resolve()),
            "count": len(rows),
            "content_sha256": _content_hash(rows),
            "file_sha256": _sha256_file(path),
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
