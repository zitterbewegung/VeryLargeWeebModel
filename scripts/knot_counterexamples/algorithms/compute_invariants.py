#!/usr/bin/env python3
"""Placeholder invariant computation stage."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

from _stage_utils import (
    descriptor_or_default,
    now_utc_iso,
    read_jsonl,
    record_history,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute lightweight pseudo-invariants.")
    parser.add_argument("--crossing", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pseudo_invariants(crossing: int, descriptor: str) -> Dict[str, int]:
    digest = hashlib.sha1(descriptor.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    signature = int(value % 21) - 10
    determinant = int(value % 997) + 1
    alexander_eval = int(value % 257) - 128
    return {
        "crossing_level": crossing,
        "pseudo_signature": signature,
        "pseudo_determinant": determinant,
        "pseudo_alexander_at_minus_one": alexander_eval,
        "hash_bucket": int(value % 8192),
    }


def main() -> int:
    args = parse_args()
    source_records = read_jsonl(args.input)
    output: List[Dict[str, object]] = []

    for record in source_records:
        descriptor = descriptor_or_default(record, "")
        out = dict(record)
        out.update(
            {
                "stage": "compute_invariants",
                "crossing": args.crossing,
                "shard": args.shard,
                "descriptor": descriptor,
                "invariants": pseudo_invariants(args.crossing, descriptor),
                "history": record_history(out, "invariants"),
                "generated_at_utc": now_utc_iso(),
            }
        )
        output.append(out)

    written = write_jsonl(args.output, output)
    print(
        "compute_invariants complete: "
        f"crossing={args.crossing} shard={args.shard} "
        f"processed={len(source_records)} written={written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
