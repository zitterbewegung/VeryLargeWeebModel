#!/usr/bin/env python3
"""Placeholder alternating generator stage (D/ROTS style seed expansion)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from _stage_utils import (
    descriptor_or_default,
    generate_dt_descriptor,
    now_utc_iso,
    read_jsonl,
    record_history,
    stable_hash,
    write_checkpoint,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate placeholder alternating records for D/ROTS stage."
    )
    parser.add_argument("--crossing", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_records = read_jsonl(args.input)

    output: List[Dict[str, object]] = []
    if source_records:
        for idx, record in enumerate(source_records):
            descriptor = descriptor_or_default(
                record,
                generate_dt_descriptor(
                    crossing=args.crossing,
                    shard=args.shard,
                    local_index=idx,
                    nonalternating=False,
                ),
            )
            out = dict(record)
            out.update(
                {
                    "stage": "alt_generate_d_rots",
                    "crossing": args.crossing,
                    "shard": args.shard,
                    "descriptor": descriptor,
                    "history": record_history(out, "D/ROTS-seed"),
                    "generated_at_utc": now_utc_iso(),
                }
            )
            out["canonical_key"] = stable_hash(
                {
                    "stage": "alt_generate_d_rots",
                    "descriptor": descriptor,
                }
            )
            output.append(out)
    else:
        for idx in range(args.count):
            descriptor = generate_dt_descriptor(
                crossing=args.crossing,
                shard=args.shard,
                local_index=idx,
                nonalternating=False,
            )
            output.append(
                {
                    "stage": "alt_generate_d_rots",
                    "crossing": args.crossing,
                    "shard": args.shard,
                    "local_index": idx,
                    "descriptor": descriptor,
                    "history": ["D/ROTS-seed"],
                    "generated_at_utc": now_utc_iso(),
                    "canonical_key": stable_hash(
                        {
                            "stage": "alt_generate_d_rots",
                            "descriptor": descriptor,
                        }
                    ),
                }
            )

    written = write_jsonl(args.output, output)
    write_checkpoint(
        args.checkpoint,
        processed=len(source_records),
        written=written,
        notes={
            "stage": "alt_generate_d_rots",
            "crossing": args.crossing,
            "shard": args.shard,
        },
    )
    print(
        "alt_generate_d_rots complete: "
        f"crossing={args.crossing} shard={args.shard} "
        f"processed={len(source_records)} written={written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
