#!/usr/bin/env python3
"""Placeholder alternating closure stage (T/OTS closure pass)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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
    parser = argparse.ArgumentParser(
        description="Apply placeholder T/OTS closure to alternating candidates."
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
        out = dict(record)
        out.update(
            {
                "stage": "alt_close_t_ots",
                "crossing": args.crossing,
                "shard": args.shard,
                "descriptor": descriptor,
                "history": record_history(out, "T/OTS-close"),
                "closure_applied": True,
                "generated_at_utc": now_utc_iso(),
            }
        )
        out["canonical_key"] = stable_hash(
            {
                "stage": "alt_close_t_ots",
                "descriptor": descriptor,
            }
        )
        output.append(out)

    written = write_jsonl(args.output, output)
    write_checkpoint(
        args.checkpoint,
        processed=len(source_records),
        written=written,
        notes={
            "stage": "alt_close_t_ots",
            "crossing": args.crossing,
            "shard": args.shard,
        },
    )
    print(
        "alt_close_t_ots complete: "
        f"crossing={args.crossing} shard={args.shard} "
        f"processed={len(source_records)} written={written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
