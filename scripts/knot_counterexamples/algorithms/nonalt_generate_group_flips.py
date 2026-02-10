#!/usr/bin/env python3
"""Placeholder nonalternating generation stage via grouped sign flips."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from _stage_utils import (
    descriptor_or_default,
    generate_dt_descriptor,
    now_utc_iso,
    parse_dt_descriptor,
    read_jsonl,
    record_history,
    stable_hash,
    write_checkpoint,
    write_jsonl,
)


def flip_groups(descriptor: str, seed: int, max_flips: int) -> List[str]:
    parsed = parse_dt_descriptor(descriptor)
    if parsed is None:
        return [descriptor]

    outputs = [descriptor]
    for group in range(max_flips):
        values = list(parsed)
        step = 2 + ((seed + group) % 3)
        for idx in range(group, len(values), step):
            values[idx] = -values[idx]
        outputs.append("DT:" + str(values))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate placeholder nonalternating variants by grouped flips."
    )
    parser.add_argument("--crossing", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--max-flips", type=int, default=2)
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
                    nonalternating=True,
                ),
            )
            variants = flip_groups(descriptor, seed=(args.shard + idx), max_flips=args.max_flips)
            for variant_id, variant in enumerate(variants):
                out = dict(record)
                out.update(
                    {
                        "stage": "nonalt_generate_group_flips",
                        "crossing": args.crossing,
                        "shard": args.shard,
                        "descriptor": variant,
                        "variant_id": variant_id,
                        "history": record_history(out, "group-flip"),
                        "generated_at_utc": now_utc_iso(),
                    }
                )
                out["canonical_key"] = stable_hash(
                    {
                        "stage": "nonalt_generate_group_flips",
                        "descriptor": variant,
                    }
                )
                output.append(out)
    else:
        for idx in range(args.count):
            descriptor = generate_dt_descriptor(
                crossing=args.crossing,
                shard=args.shard,
                local_index=idx,
                nonalternating=True,
            )
            output.append(
                {
                    "stage": "nonalt_generate_group_flips",
                    "crossing": args.crossing,
                    "shard": args.shard,
                    "local_index": idx,
                    "descriptor": descriptor,
                    "variant_id": 0,
                    "history": ["group-flip-seed"],
                    "generated_at_utc": now_utc_iso(),
                    "canonical_key": stable_hash(
                        {
                            "stage": "nonalt_generate_group_flips",
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
            "stage": "nonalt_generate_group_flips",
            "crossing": args.crossing,
            "shard": args.shard,
            "max_flips": args.max_flips,
        },
    )
    print(
        "nonalt_generate_group_flips complete: "
        f"crossing={args.crossing} shard={args.shard} "
        f"processed={len(source_records)} written={written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
