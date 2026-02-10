#!/usr/bin/env python3
"""Bounded n-move search on braid words (n = 3 or 4).

This script is intended for generating hard instances and candidate
counterexamples for move-conjectures. Failure to unknot within bounds is
not a proof; it is a search result.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from common import (
    CandidateRecord,
    DependencyError,
    append_jsonl,
    as_int,
    compute_basic_invariants,
    identify_looks_like_unknot,
    iter_descriptor_lines,
    parse_braid_word,
    parse_link_descriptor,
    print_progress,
    read_checkpoint,
    require_snappy,
    simplify_link,
    write_checkpoint,
)

BraidWord = Tuple[int, ...]


def normalize_braid_word(word: Sequence[int]) -> BraidWord:
    """Cancel adjacent inverse generators until stable."""
    current = list(word)
    changed = True
    while changed:
        changed = False
        out: List[int] = []
        idx = 0
        while idx < len(current):
            if idx + 1 < len(current) and current[idx] == -current[idx + 1]:
                changed = True
                idx += 2
                continue
            out.append(current[idx])
            idx += 1
        current = out
    return tuple(current)


def uniform_twist_block(block: Sequence[int]) -> bool:
    """Check whether all entries are same generator with same sign."""
    if not block:
        return False
    first = block[0]
    if first == 0:
        return False
    same_sign = all((value > 0) == (first > 0) for value in block)
    same_generator = all(abs(value) == abs(first) for value in block)
    return same_sign and same_generator


def generate_n_move_neighbors(word: BraidWord, move_size: int) -> Iterable[Tuple[BraidWord, int]]:
    """Apply n-move replacement to each uniform n-twist block."""
    if len(word) < move_size:
        return

    mutable = list(word)
    for start in range(0, len(mutable) - move_size + 1):
        block = mutable[start : start + move_size]
        if not uniform_twist_block(block):
            continue
        replaced = mutable[:start] + [-value for value in block] + mutable[start + move_size :]
        yield normalize_braid_word(replaced), start


def word_to_link(snappy: Any, word: BraidWord) -> Any:
    return snappy.Link(braid_closure=list(word))


def looks_unknotted(link: Any) -> bool:
    """Heuristic unknot recognition based on identification output."""
    inv = compute_basic_invariants(link)
    if identify_looks_like_unknot(inv.get("identify")):
        return True
    crossing = as_int(inv.get("crossing_number"))
    if crossing == 0:
        return True
    return False


def bounded_n_move_search(
    snappy: Any,
    initial_word: Sequence[int],
    move_size: int,
    max_depth: int,
    max_nodes: int,
) -> Dict[str, Any]:
    """Breadth-first bounded search for unknot via local n-moves."""
    start = normalize_braid_word(initial_word)
    queue: Deque[Tuple[BraidWord, int, List[Dict[str, Any]]]] = deque()
    queue.append((start, 0, []))
    seen: set[BraidWord] = {start}
    explored = 0

    while queue and explored < max_nodes:
        word, depth, path = queue.popleft()
        explored += 1

        try:
            link = word_to_link(snappy, word)
            simplify_link(link)
            if looks_unknotted(link):
                return {
                    "found_unknot": True,
                    "depth": depth,
                    "path": path,
                    "explored_nodes": explored,
                    "visited_states": len(seen),
                }
        except Exception:
            # Ignore malformed intermediate states and continue search.
            pass

        if depth >= max_depth:
            continue

        for neighbor, position in generate_n_move_neighbors(word, move_size):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(
                (
                    neighbor,
                    depth + 1,
                    path + [{"position": position, "move_size": move_size}],
                )
            )

    return {
        "found_unknot": False,
        "depth": None,
        "path": None,
        "explored_nodes": explored,
        "visited_states": len(seen),
    }


def parse_args(default_move_size: Optional[int] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded n-move search on braid words."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input file with one braid descriptor per line (BRAID:[...]).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/n_move_hard_instances.jsonl"),
        help="JSONL output path for hard-instance records.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out/n_move_hard_instances.ckpt"),
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
        "--max-depth",
        type=int,
        default=5,
        help="BFS depth bound for move search.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=4000,
        help="Node expansion bound for move search.",
    )
    parser.add_argument(
        "--min-crossings",
        type=int,
        default=10,
        help="Skip inputs below this crossing number (if known).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Stop after writing this many hard-instance records.",
    )
    if default_move_size is None:
        parser.add_argument(
            "--move-size",
            type=int,
            choices=(3, 4),
            required=True,
            help="Local move size (3 or 4).",
        )
    else:
        parser.set_defaults(move_size=default_move_size)
    return parser.parse_args()


def move_conjecture_name(move_size: int) -> str:
    if move_size == 3:
        return "Montesinos-Nakanishi (3-move)"
    if move_size == 4:
        return "Nakanishi (4-move)"
    return f"{move_size}-move conjecture"


def main(
    argv: Optional[Sequence[str]] = None,
    default_move_size: Optional[int] = None,
    conjecture_name: Optional[str] = None,
) -> int:
    del argv  # argparse uses sys.argv directly in this script.
    args = parse_args(default_move_size=default_move_size)
    move_size = int(args.move_size)
    conj_name = conjecture_name or move_conjecture_name(move_size)

    try:
        snappy = require_snappy()
    except DependencyError as exc:
        print(str(exc), flush=True)
        return 2

    start_line = read_checkpoint(args.checkpoint) if args.resume else 0
    processed = 0
    written = 0
    print_progress(
        f"{conj_name} scan started at line {start_line + 1} from {args.input}."
    )

    for line_number, descriptor in iter_descriptor_lines(
        args.input, start_line=start_line, limit=args.limit
    ):
        processed += 1

        try:
            word = tuple(parse_braid_word(descriptor))
            link = word_to_link(snappy, normalize_braid_word(word))
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

        result = bounded_n_move_search(
            snappy=snappy,
            initial_word=word,
            move_size=move_size,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
        )

        if result["found_unknot"]:
            write_checkpoint(args.checkpoint, line_number)
            continue

        reasons = [
            f"No unknot found under bounded {move_size}-move search.",
            "This is a hard instance, not a proof of non-equivalence.",
        ]
        metadata = {
            "move_size": move_size,
            "search_max_depth": args.max_depth,
            "search_max_nodes": args.max_nodes,
            "explored_nodes": result["explored_nodes"],
            "visited_states": result["visited_states"],
            "proof_status": "bounded_search_only",
        }
        record = CandidateRecord(
            conjecture=conj_name,
            status="hard_instance_not_certified",
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
        f"{conj_name} scan complete. processed={processed}, "
        f"hard_instances={written}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
