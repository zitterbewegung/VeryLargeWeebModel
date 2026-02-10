#!/usr/bin/env python3
"""Candidate search for the Generalized Property R conjecture.

Workflow:
1) Load links from descriptors.
2) Perform 0-surgery on each component.
3) Keep links whose filled manifold has expected first-homology free rank.

This yields candidates; it does not decide handle-slide equivalence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    CandidateRecord,
    DependencyError,
    append_jsonl,
    as_int,
    call_first_method,
    compute_basic_invariants,
    homology_free_rank,
    iter_descriptor_lines,
    manifold_copy,
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
        description="Search for Generalized Property R counterexample candidates."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Descriptor file with one link per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/generalized_property_r_candidates.jsonl"),
        help="JSONL output path for candidate records.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out/generalized_property_r_candidates.ckpt"),
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
        "--components",
        type=int,
        default=2,
        help="Required number of components/cusps in the input link.",
    )
    parser.add_argument(
        "--min-crossings",
        type=int,
        default=10,
        help="Skip links with crossing number below this threshold (if known).",
    )
    parser.add_argument(
        "--use-regina",
        action="store_true",
        help="Add optional Regina cross-checks when available.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Stop after writing this many candidate records.",
    )
    return parser.parse_args()


def zero_surgery_h1_rank(snappy: any, link: any, expected_cusps: int) -> tuple:
    """Return (rank, identify, isosig, homology_string) for full 0-surgery."""
    exterior = call_first_method(link, ["exterior"])
    if exterior is None:
        return None, None, None, None

    filled = manifold_copy(snappy, exterior)
    for cusp in range(expected_cusps):
        call_first_method(filled, ["dehn_fill"], (0, 1), cusp)

    homology_obj = call_first_method(filled, ["homology"])
    rank = homology_free_rank(homology_obj)
    identify = call_first_method(filled, ["identify"])
    isosig = call_first_method(filled, ["triangulation_isosig", "triangulationIsoSig"])
    return rank, identify, isosig, str(homology_obj)


def main() -> int:
    args = parse_args()
    try:
        snappy = require_snappy()
    except DependencyError as exc:
        print(str(exc), flush=True)
        return 2

    regina = optional_regina() if args.use_regina else None
    if args.use_regina and regina is None:
        print_progress("Regina not found; proceeding without Regina cross-checks.")

    start_line = read_checkpoint(args.checkpoint) if args.resume else 0
    processed = 0
    written = 0
    print_progress(
        f"Generalized Property R scan started at line {start_line + 1} from {args.input}."
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

        num_cusps = as_int(inv.get("num_cusps"))
        if num_cusps is None or num_cusps != args.components:
            write_checkpoint(args.checkpoint, line_number)
            continue

        crossing = as_int(inv.get("crossing_number"))
        if crossing is not None and crossing < args.min_crossings:
            write_checkpoint(args.checkpoint, line_number)
            continue

        rank, filled_identify, filled_isosig, filled_homology = zero_surgery_h1_rank(
            snappy, link, args.components
        )
        if rank is None or rank != args.components:
            write_checkpoint(args.checkpoint, line_number)
            continue

        reasons = [
            f"{args.components}-component link under full 0-surgery has H1 free rank {rank}.",
            "Link is nontrivial by crossing threshold.",
            "Handle-slide equivalence to unlink is unresolved by this script.",
        ]
        metadata = {
            "filled_h1_rank": rank,
            "filled_identify": str(filled_identify),
            "filled_homology": filled_homology,
            "filled_isosig": str(filled_isosig),
            "handle_slide_status": "unknown_requires_extra_certificate",
        }
        if regina is not None:
            metadata["regina_h1_rank"] = regina_h1_rank_from_isosig(
                filled_isosig, regina
            )

        record = CandidateRecord(
            conjecture="Generalized Property R",
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
        f"Generalized Property R scan complete. processed={processed}, "
        f"candidates={written}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
