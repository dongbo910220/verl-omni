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
"""Fetch only the metadata columns needed from large Hugging Face Parquet shards."""

from __future__ import annotations

import argparse
import json
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET = "SPRINGLab/IndicVoices-R_Hindi"
REVISION = "8f9913669e3505acd38b97fa38bd027f49afeeef"
SHARDS = tuple(f"data/train-{index:05d}-of-00010.parquet" for index in range(10))
LAST_METADATA_COLUMN = "duration"


def _range_get(url: str, start: int, end: int, timeout: int) -> tuple[bytes, dict]:
    import requests

    response = requests.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=(30, timeout),
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Server ignored byte range {start}-{end} for {url}: HTTP {response.status_code}")
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(f"Expected {expected} bytes for {url}, received {len(response.content)}")
    headers = {key.casefold(): value for key, value in response.headers.items()}
    for redirect in response.history:
        for key, value in redirect.headers.items():
            headers.setdefault(key.casefold(), value)
    return response.content, headers


def _remote_size(url: str, timeout: int) -> tuple[int, dict]:
    payload, headers = _range_get(url, 0, 0, timeout)
    if payload != b"P":
        raise RuntimeError(f"Unexpected Parquet magic prefix for {url}: {payload!r}")
    content_range = headers.get("content-range", "")
    try:
        size = int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"Missing file size in Content-Range for {url}: {content_range!r}") from error
    return size, headers


def _column_start(column) -> int:
    offsets = [column.data_page_offset]
    if column.dictionary_page_offset is not None and column.dictionary_page_offset >= 0:
        offsets.append(column.dictionary_page_offset)
    return min(offsets)


def _metadata_ranges(parquet_file) -> tuple[list[tuple[int, int]], dict]:
    metadata = parquet_file.metadata
    ranges = []
    audio_rows = 0
    for group_index in range(metadata.num_row_groups):
        group = metadata.row_group(group_index)
        last_index = None
        audio_column = None
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            if column.path_in_schema == LAST_METADATA_COLUMN:
                last_index = column_index
            if column.path_in_schema == "audio.bytes":
                audio_column = column
        if last_index is None or audio_column is None:
            raise RuntimeError(f"Required columns are missing from row group {group_index}")
        audio_stats = audio_column.statistics
        if audio_stats is None or audio_stats.null_count != 0 or audio_column.num_values != group.num_rows:
            raise RuntimeError(f"audio.bytes is not complete in row group {group_index}")
        audio_rows += group.num_rows

        columns = [group.column(index) for index in range(last_index + 1)]
        start = min(_column_start(column) for column in columns)
        end = max(_column_start(column) + column.total_compressed_size - 1 for column in columns)
        ranges.append((start, end))
    return ranges, {
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "audio_non_null_rows": audio_rows,
    }


def fetch_shard(
    shard: str,
    output_dir: Path,
    *,
    endpoint: str,
    workers: int,
    timeout: int,
) -> dict:
    import pyarrow.parquet as pq

    url = f"{endpoint.rstrip('/')}/datasets/{DATASET}/resolve/{REVISION}/{shard}"
    output = output_dir / f"{Path(shard).stem}.metadata.parquet"
    size, response_headers = _remote_size(url, timeout)
    trailer, _ = _range_get(url, size - 8, size - 1, timeout)
    footer_size = struct.unpack("<I", trailer[:4])[0]
    if trailer[4:] != b"PAR1":
        raise RuntimeError(f"Unexpected Parquet footer magic for {url}: {trailer[4:]!r}")
    footer_start = size - footer_size - 8
    footer, _ = _range_get(url, footer_start, size - 1, timeout)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.truncate(size)
        handle.seek(0)
        handle.write(b"PAR1")
        handle.seek(footer_start)
        handle.write(footer)

    parquet_file = pq.ParquetFile(output)
    ranges, audit = _metadata_ranges(parquet_file)

    def fetch(item: tuple[int, int]) -> tuple[int, bytes]:
        start, end = item
        payload, _ = _range_get(url, start, end, timeout)
        return start, payload

    fetched = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item for item in ranges}
        for future in as_completed(futures):
            start, payload = future.result()
            fetched.append((start, payload))
    with output.open("r+b") as handle:
        for start, payload in sorted(fetched):
            handle.seek(start)
            handle.write(payload)

    columns = [
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
    ]
    table = pq.read_table(output, columns=columns)
    if table.num_rows != audit["rows"]:
        raise RuntimeError(f"Sparse metadata verification failed for {output}")

    return {
        "shard": shard,
        "url": url,
        "output": str(output.resolve()),
        "remote_size": size,
        "etag": response_headers.get("etag"),
        "x_repo_commit": response_headers.get("x-repo-commit"),
        "footer_start": footer_start,
        "fetched_ranges": [[start, end] for start, end in ranges],
        "fetched_bytes": sum(end - start + 1 for start, end in ranges) + len(footer) + 1,
        **audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=len(SHARDS))
    args = parser.parse_args()
    if not (0 <= args.start_shard < args.end_shard <= len(SHARDS)):
        raise ValueError("Require 0 <= start-shard < end-shard <= 10")

    shard_reports = []
    for shard in SHARDS[args.start_shard : args.end_shard]:
        report = fetch_shard(
            shard,
            args.output_dir,
            endpoint=args.endpoint,
            workers=args.workers,
            timeout=args.timeout,
        )
        shard_reports.append(report)
        print(json.dumps(report, ensure_ascii=True, sort_keys=True), flush=True)

    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "revision": REVISION,
        "endpoint": args.endpoint,
        "shards": shard_reports,
    }
    manifest_path = args.output_dir / f"metadata-manifest-{args.start_shard:02d}-{args.end_shard:02d}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
