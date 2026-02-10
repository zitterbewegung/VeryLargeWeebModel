#!/usr/bin/env python3
"""Placeholder dedupe stage keyed by canonical_key."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

from _stage_utils import (
    descriptor_or_default,
    now_utc_iso,
    read_jsonl,
    record_history,
    stable_hash,
    write_checkpoint,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate records by canonical key.")
    parser.add_argument("--crossing", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_records = read_jsonl(args.input)

    seen: Set[str] = set()
    output: List[Dict[str, object]] = []
    for record in source_records:
        key = record.get("canonical_key")
        if not isinstance(key, str) or not key:
            descriptor = descriptor_or_default(record, "")
            key = stable_hash({"descriptor": descriptor})

        if key in seen:
            continue
        seen.add(key)

        out = dict(record)
        out.update(
            {
                "stage": "dedupe_keys",
                "crossing": args.crossing,
                "shard": args.shard,
                "canonical_key": key,
                "history": record_history(out, "dedupe"),
                "generated_at_utc": now_utc_iso(),
            }
        )
        output.append(out)

    written = write_jsonl(args.output, output)
    write_checkpoint(
        args.checkpoint,
        processed=len(source_records),
        written=written,
        notes={
            "stage": "dedupe_keys",
            "crossing": args.crossing,
            "shard": args.shard,
        },
    )
    print(
        "dedupe_keys complete: "
        f"crossing={args.crossing} shard={args.shard} "
        f"processed={len(source_records)} written={written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
