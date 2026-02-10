#!/usr/bin/env python3
"""Shared helpers for placeholder enumeration stage backends."""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj = {"descriptor": text, "raw": text}
            if isinstance(obj, dict):
                records.append(obj)
            else:
                records.append({"value": obj})
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_checkpoint(
    path: Path,
    *,
    processed: int,
    written: int,
    notes: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "updated_at_utc": now_utc_iso(),
        "processed": processed,
        "written": written,
    }
    if notes:
        payload["notes"] = notes
    ensure_parent(path)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def parse_dt_descriptor(descriptor: str) -> Optional[List[int]]:
    text = descriptor.strip()
    if not text.startswith("DT:"):
        return None
    dt_payload = text[len("DT:") :].strip()
    try:
        obj = ast.literal_eval(dt_payload)
    except Exception:
        return None
    if not isinstance(obj, list):
        return None
    output: List[int] = []
    for value in obj:
        if not isinstance(value, int):
            return None
        output.append(value)
    return output


def descriptor_or_default(record: Dict[str, Any], fallback: str) -> str:
    for key in ("descriptor", "dt_code", "input_descriptor", "value"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def generate_dt_descriptor(
    *,
    crossing: int,
    shard: int,
    local_index: int,
    nonalternating: bool = False,
) -> str:
    values: List[int] = []
    for i in range(crossing):
        label = 2 * (i + 1)
        sign = -1 if ((i + shard + local_index) % 3 == 0) else 1
        if nonalternating and ((i + local_index) % 5 == 0):
            sign *= -1
        values.append(sign * label)
    if nonalternating and crossing > 2:
        values[crossing // 2] = -values[crossing // 2]
    return "DT:" + str(values)


def canonicalize_descriptor_text(descriptor: str) -> str:
    parsed = parse_dt_descriptor(descriptor)
    if parsed is None:
        stripped = "".join(descriptor.split())
        reverse = stripped[::-1]
        return min(stripped, reverse)

    plain = tuple(parsed)
    flipped = tuple(-x for x in reversed(parsed))
    canonical = min(plain, flipped)
    return "DT:" + str(list(canonical))


def record_history(record: Dict[str, Any], step: str) -> List[str]:
    history = record.get("history")
    if not isinstance(history, list):
        history = []
    out = [str(item) for item in history]
    out.append(step)
    return out

