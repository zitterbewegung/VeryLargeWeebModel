#!/usr/bin/env python3
"""Placeholder canonicalization for nonalternating records."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from _stage_utils import (
    canonicalize_descriptor_text,
    descriptor_or_default,
    now_utc_iso,
    read_jsonl,
    record_history,
    stable_hash,
    write_checkpoint,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonicalize nonalternating records."
    )
    parser.add_argument("--crossing", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_records = read_jsonl(args.input)
    output: List[Dict[str, object]] = []

    for record in source_records:
        descriptor = descriptor_or_default(record, "")
        canonical_descriptor = canonicalize_descriptor_text(descriptor)
        out = dict(record)
        out.update(
            {
                "stage": "nonalt_canonicalize",
                "crossing": args.crossing,
                "shard": args.shard,
                "descriptor": descriptor,
                "nonalt_canonical_descriptor": canonical_descriptor,
                "history": record_history(out, "nonalt-canonicalize"),
                "generated_at_utc": now_utc_iso(),
            }
        )
        out["canonical_key"] = stable_hash(
            {
                "stage": "nonalt_canonicalize",
                "canonical_descriptor": canonical_descriptor,
            }
        )
        output.append(out)

    written = write_jsonl(args.output, output)
    write_checkpoint(
        args.checkpoint,
        processed=len(source_records),
        written=written,
        notes={
            "stage": "nonalt_canonicalize",
            "crossing": args.crossing,
            "shard": args.shard,
        },
    )
    print(
        "nonalt_canonicalize complete: "
        f"crossing={args.crossing} shard={args.shard} "
        f"processed={len(source_records)} written={written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
