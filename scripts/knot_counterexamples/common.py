#!/usr/bin/env python3
"""Shared helpers for counterexample search scripts.

These tools are intentionally conservative:
- They produce candidate counterexamples for hard conjectures.
- They do not claim proofs unless an invariant-based obstruction is explicit.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


class DependencyError(RuntimeError):
    """Raised when an optional math package is required but unavailable."""


@dataclass
class CandidateRecord:
    """Structured output record written to JSONL."""

    conjecture: str
    status: str
    line_number: int
    descriptor: str
    reasons: List[str]
    invariants: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def require_snappy() -> Any:
    """Import SnapPy lazily, with a clear install hint."""
    try:
        import snappy  # type: ignore
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "SnapPy is required. Install with: python3 -m pip install snappy"
        ) from exc
    return snappy


def optional_regina() -> Optional[Any]:
    """Import Regina if available, else return None."""
    try:
        import regina  # type: ignore
    except ModuleNotFoundError:
        return None
    return regina


def iter_descriptor_lines(
    input_path: Path,
    start_line: int = 0,
    limit: Optional[int] = None,
) -> Iterator[Tuple[int, str]]:
    """Yield non-empty descriptor lines with 1-based physical line numbers."""
    emitted = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if line_number <= start_line:
                continue
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            yield line_number, value
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def read_checkpoint(checkpoint_path: Path) -> int:
    """Return last processed line number from checkpoint (0 if absent)."""
    if not checkpoint_path.exists():
        return 0
    raw = checkpoint_path.read_text(encoding="utf-8").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def write_checkpoint(checkpoint_path: Path, line_number: int) -> None:
    """Persist last processed physical line number."""
    checkpoint_path.write_text(f"{line_number}\n", encoding="utf-8")


def append_jsonl(output_path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON object line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _call_method(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(obj, name, None)
    if not callable(method):
        return None
    try:
        return method(*args, **kwargs)
    except Exception:
        return None


def call_first_method(
    obj: Any,
    method_names: Sequence[str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Try several method names and return the first non-None value."""
    for name in method_names:
        value = _call_method(obj, name, *args, **kwargs)
        if value is not None:
            return value
    return None


def normalize_json_value(value: Any) -> Any:
    """Convert library objects to JSON-friendly representations."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_json_value(val) for key, val in value.items()}
    return str(value)


def parse_braid_word(descriptor: str) -> List[int]:
    """Parse BRAID descriptors.

    Accepted formats:
    - BRAID:[1,-2,3]
    - [1,-2,3]
    """
    text = descriptor.strip()
    if text.startswith("BRAID:"):
        text = text[len("BRAID:") :].strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("Expected braid descriptor like BRAID:[1,-2,3]")
    word = ast.literal_eval(text)
    if not isinstance(word, list):
        raise ValueError("Braid descriptor must decode to a list of integers")
    output: List[int] = []
    for value in word:
        if not isinstance(value, int):
            raise ValueError("Braid word entries must be integers")
        if value == 0:
            raise ValueError("Braid generator index cannot be 0")
        output.append(value)
    return output


def parse_link_descriptor(snappy: Any, descriptor: str) -> Any:
    """Create a SnapPy Link from a text descriptor.

    Supported formats:
    - DT:[...]
    - BRAID:[...]
    - [...]  (interpreted as DT code)
    - Any SnapPy knot/link name, e.g. 8_20
    """
    value = descriptor.strip()
    if value.startswith("BRAID:"):
        return snappy.Link(braid_closure=parse_braid_word(value))
    if value.startswith("DT:"):
        return snappy.Link(value)
    if value.startswith("[") and value.endswith("]"):
        return snappy.Link("DT:" + value)
    return snappy.Link(value)


def simplify_link(link: Any) -> None:
    """Run global simplification if available."""
    _call_method(link, "simplify", "global")


def as_int(value: Any) -> Optional[int]:
    """Best-effort integer extraction from numeric/string values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
    return None


def try_alexander_determinant(alexander: Any) -> Optional[int]:
    """Try computing |Delta(-1)| from a polynomial-like object."""
    if alexander is None:
        return None
    try:
        evaluated = alexander(-1)  # type: ignore[misc]
        numeric = as_int(evaluated)
        if numeric is not None:
            return abs(numeric)
    except Exception:
        return None
    return None


def compute_basic_invariants(link: Any, include_group: bool = False) -> Dict[str, Any]:
    """Compute common SnapPy invariants with graceful fallbacks."""
    invariants: Dict[str, Any] = {}

    crossing = call_first_method(link, ["crossing_number"])
    signature = call_first_method(link, ["signature"])
    determinant = call_first_method(link, ["determinant"])
    alexander = call_first_method(link, ["alexander_polynomial"])
    jones = call_first_method(link, ["jones_polynomial"])
    dt_code = call_first_method(link, ["DT_code"])

    if determinant is None:
        determinant = try_alexander_determinant(alexander)

    invariants["crossing_number"] = normalize_json_value(crossing)
    invariants["signature"] = normalize_json_value(signature)
    invariants["determinant"] = normalize_json_value(determinant)
    invariants["alexander_polynomial"] = normalize_json_value(alexander)
    invariants["jones_polynomial"] = normalize_json_value(jones)
    invariants["dt_code"] = normalize_json_value(dt_code)

    exterior = call_first_method(link, ["exterior"])
    if exterior is not None:
        invariants["num_cusps"] = normalize_json_value(
            call_first_method(exterior, ["num_cusps", "numCusps"])
        )
        invariants["volume"] = normalize_json_value(
            call_first_method(exterior, ["volume"])
        )
        homology = call_first_method(exterior, ["homology"])
        invariants["homology"] = normalize_json_value(homology)
        invariants["identify"] = normalize_json_value(
            call_first_method(exterior, ["identify"])
        )
        invariants["isosig"] = normalize_json_value(
            call_first_method(exterior, ["triangulation_isosig", "triangulationIsoSig"])
        )
        if include_group:
            invariants["fundamental_group"] = normalize_json_value(
                call_first_method(exterior, ["fundamental_group"])
            )

    return invariants


def identify_looks_like_unknot(identify_value: Any) -> bool:
    """Heuristic unknot detector from manifold identification text."""
    if identify_value is None:
        return False
    text = str(identify_value).lower()
    tokens = ("unknot", "0_1", "k0_1", "u_1")
    return any(token in text for token in tokens)


def is_perfect_square(value: Optional[int]) -> Optional[bool]:
    """Return whether value is a perfect square, or None if unknown."""
    if value is None:
        return None
    if value < 0:
        return False
    root = math.isqrt(value)
    return root * root == value


def _parse_rank_from_homology_text(text: str) -> Optional[int]:
    """Parse free rank from strings like 'Z + Z/2 + Z^2'."""
    cleaned = text.replace(" ", "")
    if cleaned in {"0", ""}:
        return 0

    rank = 0
    for match in re.finditer(r"Z(?:\^(\d+))?(?!/)", cleaned):
        exponent = match.group(1)
        rank += int(exponent) if exponent is not None else 1
    if rank == 0:
        return None
    return rank


def homology_free_rank(homology_value: Any) -> Optional[int]:
    """Best-effort free-rank extraction from homology objects."""
    if homology_value is None:
        return None

    rank = call_first_method(
        homology_value, ["rank", "freeRank", "free_rank", "betti_number", "b1"]
    )
    parsed = as_int(rank)
    if parsed is not None:
        return parsed

    return _parse_rank_from_homology_text(str(homology_value))


def manifold_copy(snappy: Any, manifold: Any) -> Any:
    """Best-effort manifold copy to avoid in-place mutation."""
    if manifold is None:
        raise ValueError("manifold cannot be None")

    copied = call_first_method(snappy, ["Manifold"], manifold)
    if copied is not None:
        return copied

    copied = call_first_method(manifold, ["copy"])
    if copied is not None:
        return copied

    raise RuntimeError("Could not copy manifold")


def regina_h1_rank_from_isosig(isosig: Any, regina: Optional[Any] = None) -> Optional[int]:
    """Compute H1 free rank in Regina from a triangulation isosig."""
    if isosig is None:
        return None
    if regina is None:
        regina = optional_regina()
    if regina is None:
        return None

    isosig_text = str(isosig).strip()
    if not isosig_text:
        return None

    tri = None
    tri_cls = getattr(regina, "Triangulation3", None)
    if tri_cls is None:
        return None

    if hasattr(tri_cls, "fromIsoSig"):
        try:
            tri = tri_cls.fromIsoSig(isosig_text)
        except Exception:
            tri = None
    if tri is None:
        return None

    homology = call_first_method(tri, ["homology"])
    return homology_free_rank(homology)


def print_progress(message: str) -> None:
    """Consistent progress logging for long batch runs."""
    print(message, flush=True)
