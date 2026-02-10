#!/usr/bin/env python3
"""Generate sharded task JSONL from an enumeration manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate task queue files from billion_enumeration_manifest.json."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest JSON path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL task file.",
    )
    parser.add_argument(
        "--crossing-start",
        type=int,
        default=None,
        help="Override crossing start (inclusive).",
    )
    parser.add_argument(
        "--crossing-end",
        type=int,
        default=None,
        help="Override crossing end (inclusive).",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Optional subset of stage names.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Manifest must decode to a JSON object")
    return raw


def iter_crossings(start: int, end: int, stride: int) -> Iterable[int]:
    if stride <= 0:
        raise ValueError("crossing stride must be positive")
    current = start
    while current <= end:
        yield current
        current += stride


def task_paths(root: Path, stage: str, crossing: int, shard: int) -> Tuple[str, str, str]:
    stage_dir = root / "generated" / stage / f"c{crossing:02d}" / f"s{shard:04d}"
    input_path = stage_dir / "input.jsonl"
    output_path = stage_dir / "output.jsonl"
    checkpoint_path = root / "checkpoints" / stage / f"c{crossing:02d}_s{shard:04d}.ckpt"
    return str(input_path), str(output_path), str(checkpoint_path)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    workspace_root = Path(str(manifest["workspace_root"]))
    crossings = dict(manifest["crossings"])
    sharding = dict(manifest["sharding"])
    stages = dict(manifest["stages"])
    stage_order = list(manifest["stage_order"])

    start = int(crossings["start"]) if args.crossing_start is None else args.crossing_start
    end = int(crossings["end"]) if args.crossing_end is None else args.crossing_end
    stride = int(crossings.get("stride", 1))
    num_shards = int(sharding["num_shards"])

    selected_stage_names: List[str]
    if args.stages:
        selected_stage_names = args.stages
    else:
        selected_stage_names = [name for name in stage_order if stages.get(name, {}).get("enabled", True)]

    for stage_name in selected_stage_names:
        if stage_name not in stages:
            raise ValueError(f"Unknown stage: {stage_name}")

    tasks: List[Dict[str, object]] = []
    for crossing in iter_crossings(start, end, stride):
        for stage_name in selected_stage_names:
            stage_cfg = dict(stages[stage_name])
            template = str(stage_cfg["command_template"])

            # Validate stage order dependencies by requiring predecessor completion.
            stage_index = stage_order.index(stage_name)
            dependency_stage = stage_order[stage_index - 1] if stage_index > 0 else None

            for shard in range(num_shards):
                input_path, output_path, checkpoint_path = task_paths(
                    workspace_root, stage_name, crossing, shard
                )
                command = template.format(
                    n=crossing,
                    shard=shard,
                    input=input_path,
                    output=output_path,
                    checkpoint=checkpoint_path,
                    workspace_root=str(workspace_root),
                )
                task_id = f"{stage_name}:c{crossing:02d}:s{shard:04d}"
                depends_on = None
                if dependency_stage is not None:
                    depends_on = f"{dependency_stage}:c{crossing:02d}:s{shard:04d}"

                tasks.append(
                    {
                        "task_id": task_id,
                        "stage": stage_name,
                        "crossing": crossing,
                        "shard": shard,
                        "depends_on": depends_on,
                        "command": command,
                        "input_path": input_path,
                        "output_path": output_path,
                        "checkpoint_path": checkpoint_path,
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, sort_keys=True))
            handle.write("\n")

    print(f"Generated {len(tasks)} tasks at {args.out}")
    print(f"Stages: {selected_stage_names}")
    print(f"Crossing range: {start}..{end}, shards: {num_shards}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
