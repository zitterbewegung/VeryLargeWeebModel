#!/usr/bin/env python3
"""Candidate search for Kauffman's strong conjecture (specialized case).

Conjecture form used here:
    K is slice  <=>  untwisted Whitehead double Wh_0(K) is slice

Input requirement:
- Each line must contain a pair: "<K descriptor> | <Wh_0(K) descriptor>"
  where each side is a normal descriptor accepted by SnapPy helpers
  (DT:[...], BRAID:[...], or knot/link name).

This script does not construct Whitehead doubles automatically; it evaluates
provided pairs and flags mismatched slice-likely status as candidates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from common import (
    CandidateRecord,
    DependencyError,
    append_jsonl,
    as_int,
    compute_basic_invariants,
    is_perfect_square,
    iter_descriptor_lines,
    parse_link_descriptor,
    print_progress,
    read_checkpoint,
    require_snappy,
    simplify_link,
    write_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for K vs untwisted Whitehead-double slice-status mismatches."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input file with lines in '<K> | <Wh_0(K)>' format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/whitehead_double_candidates.jsonl"),
        help="JSONL output path for candidate records.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out/whitehead_double_candidates.ckpt"),
        help="Checkpoint file storing last processed physical line number.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of non-comment input lines to process this run.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Stop after writing this many candidate records.",
    )
    return parser.parse_args()


def parse_pair(line: str) -> Tuple[str, str]:
    if "|" not in line:
        raise ValueError("Expected line format '<K descriptor> | <Wh_0(K) descriptor>'")
    left, right = line.split("|", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError("Both descriptors must be non-empty")
    return left, right


def slice_likely(invariants: dict) -> Tuple[Optional[bool], dict]:
    signature = as_int(invariants.get("signature"))
    determinant = as_int(invariants.get("determinant"))

    signature_zero = signature == 0 if signature is not None else None
    det_square = is_perfect_square(determinant)

    if signature_zero is None or det_square is None:
        return None, {
            "signature_zero": signature_zero,
            "determinant_square": det_square,
        }
    return bool(signature_zero and det_square), {
        "signature_zero": signature_zero,
        "determinant_square": det_square,
    }


def main() -> int:
    args = parse_args()
    try:
        snappy = require_snappy()
    except DependencyError as exc:
        print(str(exc), flush=True)
        return 2

    start_line = read_checkpoint(args.checkpoint) if args.resume else 0
    processed = 0
    written = 0
    print_progress(
        f"Kauffman-strong scan started at line {start_line + 1} from {args.input}."
    )

    for line_number, raw in iter_descriptor_lines(
        args.input, start_line=start_line, limit=args.limit
    ):
        processed += 1
        try:
            base_desc, double_desc = parse_pair(raw)
            base_link = parse_link_descriptor(snappy, base_desc)
            double_link = parse_link_descriptor(snappy, double_desc)
            simplify_link(base_link)
            simplify_link(double_link)
            base_inv = compute_basic_invariants(base_link)
            double_inv = compute_basic_invariants(double_link)
        except Exception as exc:
            write_checkpoint(args.checkpoint, line_number)
            print_progress(
                f"[line {line_number}] parse/eval failed; skipped: {exc}"
            )
            continue

        base_slice, base_flags = slice_likely(base_inv)
        double_slice, double_flags = slice_likely(double_inv)
        if base_slice is None or double_slice is None:
            write_checkpoint(args.checkpoint, line_number)
            continue

        if base_slice == double_slice:
            write_checkpoint(args.checkpoint, line_number)
            continue

        reasons = [
            "Slice-likely filter disagrees between K and Wh_0(K).",
            "Mismatch is a candidate only; sliceness is not decided by this script.",
        ]
        metadata = {
            "base_descriptor": base_desc,
            "double_descriptor": double_desc,
            "base_slice_likely": base_slice,
            "double_slice_likely": double_slice,
            "base_flags": base_flags,
            "double_flags": double_flags,
            "proof_status": "heuristic_slice_filters_only",
        }
        record = CandidateRecord(
            conjecture="Kauffman Strong (specialized)",
            status="candidate_not_certified",
            line_number=line_number,
            descriptor=raw,
            reasons=reasons,
            invariants={
                "base": base_inv,
                "whitehead_double": double_inv,
            },
            metadata=metadata,
        )
        append_jsonl(args.output, record.to_json_dict())
        written += 1
        write_checkpoint(args.checkpoint, line_number)

        if args.max_candidates is not None and written >= args.max_candidates:
            break

    print_progress(
        f"Kauffman-strong scan complete. processed={processed}, "
        f"candidates={written}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
