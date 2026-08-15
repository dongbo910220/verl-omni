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
"""Derive an input-only AISHELL-3 long-form training split and signal Gate."""

import argparse
import hashlib
import json
import random
from pathlib import Path

DEFAULT_BUCKETS = ((18, 20), (20, 22), (22, 24), (24, None))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bucket_label(lower: int, upper: int | None) -> str:
    return f"{lower}+" if upper is None else f"{lower}-{upper - 1}"


def derive_long_form(
    rows: list[dict],
    min_chars: int = 18,
    gate_size: int = 256,
    gate_seed: int = 43,
    buckets: tuple[tuple[int, int | None], ...] = DEFAULT_BUCKETS,
) -> tuple[dict[str, list[dict]], dict]:
    if min_chars <= 0 or gate_size <= 0:
        raise ValueError("min_chars and gate_size must be positive.")
    if gate_size % len(buckets):
        raise ValueError("gate_size must be divisible by the number of length buckets.")

    hard_rows = []
    bucket_candidates = {_bucket_label(*bucket): [] for bucket in buckets}
    bucket_seen = {label: set() for label in bucket_candidates}
    for row in rows:
        extra_info = dict(row.get("extra_info") or {})
        text = str(extra_info.get("text") or extra_info.get("raw_text") or "").strip()
        normalized_text = str(extra_info.get("normalized_text") or text).strip()
        if len(text) < min_chars:
            continue

        train_row = dict(row)
        train_extra = dict(extra_info)
        train_extra.update(split="train-long-form", index=len(hard_rows), text_length_chars=len(text))
        train_row["extra_info"] = train_extra
        hard_rows.append(train_row)

        for lower, upper in buckets:
            if len(text) >= lower and (upper is None or len(text) < upper):
                label = _bucket_label(lower, upper)
                if normalized_text not in bucket_seen[label]:
                    bucket_seen[label].add(normalized_text)
                    bucket_candidates[label].append(train_row)
                break

    if not hard_rows:
        raise ValueError("No rows satisfy the long-form filter.")

    per_bucket = gate_size // len(buckets)
    selected = []
    bucket_audit = {}
    for lower, upper in buckets:
        label = _bucket_label(lower, upper)
        candidates = bucket_candidates[label]
        if len(candidates) < per_bucket:
            raise ValueError(f"Length bucket {label} has only {len(candidates)} unique prompts; need {per_bucket}.")
        rng = random.Random(f"aishell3-long-form:{gate_seed}:{label}")
        chosen = rng.sample(candidates, per_bucket)
        selected.extend((label, row) for row in chosen)
        bucket_audit[label] = {
            "unique_candidates": len(candidates),
            "gate_samples": per_bucket,
        }

    gate_rows = []
    for gate_index, (label, row) in enumerate(selected):
        gate_row = dict(row)
        gate_extra = dict(row["extra_info"])
        gate_extra.update(split="gate", index=gate_index, difficulty_bucket=label)
        gate_row["extra_info"] = gate_extra
        gate_rows.append(gate_row)

    audit = {
        "source_rows": len(rows),
        "train_rows": len(hard_rows),
        "train_unique_normalized": len({row["extra_info"]["normalized_text"] for row in hard_rows}),
        "min_chars": min_chars,
        "gate_size": gate_size,
        "gate_seed": gate_seed,
        "buckets": bucket_audit,
        "gate_source_ids": [row["extra_info"]["id"] for row in gate_rows],
    }
    return {"train_long_form": hard_rows, "gate_long_form": gate_rows}, audit


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-chars", type=int, default=18)
    parser.add_argument("--gate-size", type=int, default=256)
    parser.add_argument("--gate-seed", type=int, default=43)
    args = parser.parse_args()

    source_rows = pd.read_parquet(args.train).to_dict("records")
    splits, audit = derive_long_form(
        source_rows,
        min_chars=args.min_chars,
        gate_size=args.gate_size,
        gate_seed=args.gate_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_train_path": str(args.train.resolve()),
        "source_train_sha256": _sha256(args.train),
        "audit": audit,
        "splits": {},
    }
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        manifest["splits"][name] = {
            "path": str(path.resolve()),
            "count": len(rows),
            "file_sha256": _sha256(path),
        }
    (args.output_dir / "manifest_long_form.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
