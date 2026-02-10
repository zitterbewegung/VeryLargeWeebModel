#!/usr/bin/env python3
"""Placeholder milestone validator against CSV expected counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate crossing milestone counts.")
    parser.add_argument("--crossing", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser.parse_args()


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def aggregate_stage_count(workspace: Path, stage: str, crossing: int) -> int:
    pattern = workspace / "generated" / stage / f"c{crossing:02d}"
    if not pattern.exists():
        return 0
    total = 0
    for output in pattern.glob("s*/output.jsonl"):
        total += count_jsonl_records(output)
    return total


def parse_optional_int(value: str) -> Optional[int]:
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def milestone_expected(csv_path: Path, crossing: int) -> Dict[str, Optional[int]]:
    expected_alt: Optional[int] = None
    expected_nonalt: Optional[int] = None
    if not csv_path.exists():
        return {"alternating": None, "nonalternating": None}

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = parse_optional_int(str(row.get("crossing", "")).strip())
            if value != crossing:
                continue
            expected_alt = parse_optional_int(str(row.get("expected_alternating", "")))
            expected_nonalt = parse_optional_int(str(row.get("expected_nonalternating", "")))
            break
    return {"alternating": expected_alt, "nonalternating": expected_nonalt}


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    crossing = args.crossing

    observed_alt = aggregate_stage_count(workspace, "alt_dedupe", crossing)
    observed_nonalt = aggregate_stage_count(workspace, "nonalt_dedupe", crossing)

    expected = milestone_expected(
        workspace / "manifests" / "milestone_counts.csv",
        crossing,
    )

    status = "ok"
    errors = []
    if expected["alternating"] is not None and expected["alternating"] != observed_alt:
        status = "mismatch"
        errors.append(
            f"alternating expected={expected['alternating']} observed={observed_alt}"
        )
    if expected["nonalternating"] is not None and expected["nonalternating"] != observed_nonalt:
        status = "mismatch"
        errors.append(
            f"nonalternating expected={expected['nonalternating']} observed={observed_nonalt}"
        )

    result = {
        "crossing": crossing,
        "status": status,
        "expected": expected,
        "observed": {
            "alternating": observed_alt,
            "nonalternating": observed_nonalt,
        },
        "errors": errors,
    }

    metrics_path = workspace / "metrics" / f"milestone_validation_c{crossing:02d}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, sort_keys=True))
    if status != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
