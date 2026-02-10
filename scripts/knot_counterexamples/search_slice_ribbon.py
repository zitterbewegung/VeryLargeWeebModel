#!/usr/bin/env python3
"""Candidate search for the Slice-Ribbon conjecture.

This script is a filter, not a full proof engine:
- It flags knots that look slice-like via computable invariants.
- Ribbon detection is marked as unknown unless external certification is added.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    CandidateRecord,
    DependencyError,
    append_jsonl,
    as_int,
    compute_basic_invariants,
    is_perfect_square,
    iter_descriptor_lines,
    optional_regina,
    parse_link_descriptor,
    print_progress,
    read_checkpoint,
    regina_h1_rank_from_isosig,
    require_snappy,
    simplify_link,
    write_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for slice-ribbon counterexample candidates."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Descriptor file (one knot per line: DT:[...], BRAID:[...], or SnapPy name).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/slice_ribbon_candidates.jsonl"),
        help="JSONL output path for candidate records.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out/slice_ribbon_candidates.ckpt"),
        help="Checkpoint file storing last processed physical line number.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if it exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of non-comment input lines to process this run.",
    )
    parser.add_argument(
        "--min-crossings",
        type=int,
        default=12,
        help="Skip knots with crossing number below this threshold (if known).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Stop after writing this many candidate records.",
    )
    parser.add_argument(
        "--use-regina",
        action="store_true",
        help="Add optional Regina cross-checks when available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        snappy = require_snappy()
    except DependencyError as exc:
        print(str(exc), flush=True)
        return 2

    regina = optional_regina() if args.use_regina else None
    if args.use_regina and regina is None:
        print_progress(
            "Regina not found; proceeding without Regina cross-checks."
        )

    start_line = read_checkpoint(args.checkpoint) if args.resume else 0
    processed = 0
    written = 0

    print_progress(
        f"Slice-Ribbon scan started at line {start_line + 1} from {args.input}."
    )

    for line_number, descriptor in iter_descriptor_lines(
        args.input, start_line=start_line, limit=args.limit
    ):
        processed += 1
        try:
            link = parse_link_descriptor(snappy, descriptor)
            simplify_link(link)
            inv = compute_basic_invariants(link)
        except Exception as exc:
            write_checkpoint(args.checkpoint, line_number)
            print_progress(
                f"[line {line_number}] parse/eval failed; skipped: {exc}"
            )
            continue

        crossing = as_int(inv.get("crossing_number"))
        if crossing is not None and crossing < args.min_crossings:
            write_checkpoint(args.checkpoint, line_number)
            continue

        signature = as_int(inv.get("signature"))
        determinant = as_int(inv.get("determinant"))
        signature_zero = signature == 0 if signature is not None else None
        determinant_square = is_perfect_square(determinant)

        # Conservative slice-likely filter:
        # zero signature + square determinant are necessary in many families,
        # but not sufficient for smooth sliceness.
        slice_likely = bool(signature_zero) and bool(determinant_square)
        if not slice_likely:
            write_checkpoint(args.checkpoint, line_number)
            continue

        reasons = [
            "Signature is 0.",
            "Determinant is a perfect square.",
            "Ribbon status unresolved by this script.",
        ]
        metadata = {
            "signature_zero": signature_zero,
            "determinant_square": determinant_square,
            "slice_likely_filter": True,
            "ribbon_status": "unknown_requires_extra_obstruction_or_construction",
        }
        if regina is not None:
            metadata["regina_h1_rank"] = regina_h1_rank_from_isosig(
                inv.get("isosig"), regina
            )

        record = CandidateRecord(
            conjecture="Slice-Ribbon",
            status="candidate_not_certified",
            line_number=line_number,
            descriptor=descriptor,
            reasons=reasons,
            invariants=inv,
            metadata=metadata,
        )
        append_jsonl(args.output, record.to_json_dict())
        written += 1
        write_checkpoint(args.checkpoint, line_number)

        if args.max_candidates is not None and written >= args.max_candidates:
            break

    print_progress(
        f"Slice-Ribbon scan complete. processed={processed}, candidates={written}, "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
