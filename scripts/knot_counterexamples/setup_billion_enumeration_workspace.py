#!/usr/bin/env python3
"""Initialize a workspace for billion-scale knot enumeration runs.

This script creates:
- Standard directory layout under a chosen root.
- A SQLite state DB for runs/tasks/milestones.
- A manifest template and milestone CSV template.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


DEFAULT_ROOT = Path("/Volumes/Holomorphic/BigKnotTables")

DIRECTORIES = [
    "tables",
    "seeds/alternating",
    "seeds/nonalternating",
    "generated/alternating",
    "generated/nonalternating",
    "canonical/alternating",
    "canonical/nonalternating",
    "invariants",
    "checkpoints",
    "logs",
    "logs/tasks",
    "manifests",
    "metrics",
    "results",
    "shards",
    "tmp",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_directories(root: Path) -> List[Path]:
    created: List[Path] = []
    for relative in DIRECTORIES:
        path = root / relative
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def init_state_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              created_at_utc TEXT NOT NULL,
              manifest_path TEXT NOT NULL,
              notes TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
              task_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              stage TEXT NOT NULL,
              crossing INTEGER NOT NULL,
              shard INTEGER NOT NULL,
              status TEXT NOT NULL,
              command TEXT NOT NULL,
              input_path TEXT,
              output_path TEXT,
              checkpoint_path TEXT,
              log_path TEXT,
              started_at_utc TEXT,
              finished_at_utc TEXT,
              exit_code INTEGER,
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_status
            ON tasks(status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_run_stage_crossing
            ON tasks(run_id, stage, crossing)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS milestones (
              crossing INTEGER PRIMARY KEY,
              expected_alternating INTEGER,
              expected_nonalternating INTEGER,
              observed_alternating INTEGER,
              observed_nonalternating INTEGER,
              verified INTEGER NOT NULL DEFAULT 0,
              checked_at_utc TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def default_manifest(root: Path) -> Dict[str, object]:
    return {
        "version": 1,
        "created_at_utc": now_utc_iso(),
        "workspace_root": str(root),
        "crossings": {
            "start": 16,
            "end": 24,
            "stride": 1,
        },
        "sharding": {
            "num_shards": 256,
            "key_hash": "sha1",
            "prefix_bytes": 2,
        },
        "stage_order": [
            "alt_generate_d_rots",
            "alt_close_t_ots",
            "alt_canonicalize_flype",
            "alt_dedupe",
            "nonalt_generate_group_flips",
            "nonalt_canonicalize",
            "nonalt_dedupe",
            "compute_invariants",
            "validate_milestones",
        ],
        "stages": {
            "alt_generate_d_rots": {
                "enabled": True,
                "command_template": (
                    "python algorithms/alt_generate_d_rots.py "
                    "--crossing {n} --shard {shard} "
                    "--input {input} --output {output} --checkpoint {checkpoint}"
                ),
            },
            "alt_close_t_ots": {
                "enabled": True,
                "command_template": (
                    "python algorithms/alt_close_t_ots.py "
                    "--crossing {n} --shard {shard} "
                    "--input {input} --output {output} --checkpoint {checkpoint}"
                ),
            },
            "alt_canonicalize_flype": {
                "enabled": True,
                "command_template": (
                    "python algorithms/alt_canonicalize_flype.py "
                    "--crossing {n} --shard {shard} "
                    "--input {input} --output {output} --checkpoint {checkpoint}"
                ),
            },
            "alt_dedupe": {
                "enabled": True,
                "command_template": (
                    "python algorithms/dedupe_keys.py "
                    "--crossing {n} --shard {shard} "
                    "--input {input} --output {output} --checkpoint {checkpoint}"
                ),
            },
            "nonalt_generate_group_flips": {
                "enabled": True,
                "command_template": (
                    "python algorithms/nonalt_generate_group_flips.py "
                    "--crossing {n} --shard {shard} "
                    "--input {input} --output {output} --checkpoint {checkpoint}"
                ),
            },
            "nonalt_canonicalize": {
                "enabled": True,
                "command_template": (
                    "python algorithms/nonalt_canonicalize.py "
                    "--crossing {n} --shard {shard} "
                    "--input {input} --output {output} --checkpoint {checkpoint}"
                ),
            },
            "nonalt_dedupe": {
                "enabled": True,
                "command_template": (
                    "python algorithms/dedupe_keys.py "
                    "--crossing {n} --shard {shard} "
                    "--input {input} --output {output} --checkpoint {checkpoint}"
                ),
            },
            "compute_invariants": {
                "enabled": True,
                "command_template": (
                    "python algorithms/compute_invariants.py "
                    "--crossing {n} --shard {shard} "
                    "--input {input} --output {output}"
                ),
            },
            "validate_milestones": {
                "enabled": True,
                "command_template": (
                    "python algorithms/validate_milestones.py "
                    "--crossing {n} --workspace {workspace_root}"
                ),
            },
        },
    }


def write_manifest(path: Path, manifest: Dict[str, object], force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return True


def write_milestone_csv(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "crossing",
                "expected_alternating",
                "expected_nonalternating",
                "source",
                "notes",
            ]
        )
        for crossing in range(3, 26):
            writer.writerow([crossing, "", "", "literature", "fill before full run"])
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize workspace layout for billion-scale knot enumeration."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Workspace root directory (default: /Volumes/Holomorphic/BigKnotTables).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite manifest and milestone template files if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    created_dirs = create_directories(root)
    db_path = root / "checkpoints" / "enumeration_state.sqlite"
    init_state_db(db_path)

    manifest_path = root / "manifests" / "billion_enumeration_manifest.json"
    milestone_path = root / "manifests" / "milestone_counts.csv"

    manifest_written = write_manifest(
        manifest_path, default_manifest(root), force=args.force
    )
    milestones_written = write_milestone_csv(milestone_path, force=args.force)

    print(f"Workspace root: {root}")
    print(f"State DB: {db_path}")
    print(f"Created {len(created_dirs)} new directories.")
    if created_dirs:
        for path in created_dirs:
            print(f"  - {path}")

    if manifest_written:
        print(f"Wrote manifest template: {manifest_path}")
    else:
        print(f"Manifest exists (unchanged): {manifest_path}")

    if milestones_written:
        print(f"Wrote milestone template: {milestone_path}")
    else:
        print(f"Milestone template exists (unchanged): {milestone_path}")

    print("Next:")
    print("  1. Fill milestone_counts.csv with known literature counts.")
    print("  2. Replace algorithm command templates with real generators.")
    print("  3. Generate task shards and start dry runs at low crossing numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
