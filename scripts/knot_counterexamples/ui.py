#!/usr/bin/env python3
"""Streamlit UI for knot counterexample search scripts."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_WORK_ROOT = (
    Path("/Volumes/Holomorphic/BigKnotTables")
    if Path("/Volumes/Holomorphic/BigKnotTables").exists()
    else PROJECT_ROOT
)
DEFAULT_RESULTS_DIR = DEFAULT_WORK_ROOT / "results"
DEFAULT_CHECKPOINT_DIR = DEFAULT_WORK_ROOT / "checkpoints"
DEFAULT_LOG_DIR = DEFAULT_WORK_ROOT / "logs"
RUN_REGISTRY_PATH = DEFAULT_RESULTS_DIR / "ui_runs.json"


SEARCH_CONFIG: Dict[str, Dict[str, Any]] = {
    "Slice-Ribbon": {
        "script": "search_slice_ribbon.py",
        "input_default": DEFAULT_WORK_ROOT / "tables" / "knots_dt.txt",
        "output_default": DEFAULT_WORK_ROOT / "results" / "slice_ribbon_candidates.jsonl",
        "checkpoint_default": DEFAULT_WORK_ROOT / "checkpoints" / "slice_ribbon.ckpt",
    },
    "Generalized Property R": {
        "script": "search_generalized_property_r.py",
        "input_default": DEFAULT_WORK_ROOT / "tables" / "links_dt.txt",
        "output_default": DEFAULT_WORK_ROOT / "results" / "property_r_candidates.jsonl",
        "checkpoint_default": DEFAULT_WORK_ROOT / "checkpoints" / "property_r.ckpt",
    },
    "Montesinos-Nakanishi (3-move)": {
        "script": "search_three_move.py",
        "input_default": DEFAULT_WORK_ROOT / "tables" / "braids.txt",
        "output_default": DEFAULT_WORK_ROOT / "results" / "three_move_hard_instances.jsonl",
        "checkpoint_default": DEFAULT_WORK_ROOT / "checkpoints" / "three_move.ckpt",
    },
    "Nakanishi (4-move)": {
        "script": "search_four_move.py",
        "input_default": DEFAULT_WORK_ROOT / "tables" / "braids.txt",
        "output_default": DEFAULT_WORK_ROOT / "results" / "four_move_hard_instances.jsonl",
        "checkpoint_default": DEFAULT_WORK_ROOT / "checkpoints" / "four_move.ckpt",
    },
    "Kauffman Strong (specialized)": {
        "script": "search_whitehead_double.py",
        "input_default": DEFAULT_WORK_ROOT / "tables" / "whitehead_pairs.txt",
        "output_default": DEFAULT_WORK_ROOT / "results" / "whitehead_candidates.jsonl",
        "checkpoint_default": DEFAULT_WORK_ROOT / "checkpoints" / "whitehead.ckpt",
    },
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_registry() -> List[Dict[str, Any]]:
    if not RUN_REGISTRY_PATH.exists():
        return []
    try:
        data = json.loads(RUN_REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def save_registry(records: List[Dict[str, Any]]) -> None:
    ensure_parent(RUN_REGISTRY_PATH)
    RUN_REGISTRY_PATH.write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def get_exit_code(status_path: Path) -> int | None:
    if not status_path.exists():
        return None
    try:
        text = status_path.read_text(encoding="utf-8").strip()
        return int(text)
    except Exception:
        return None


def run_status(record: Dict[str, Any]) -> str:
    pid = int(record.get("pid", 0))
    status_path = Path(record.get("status_path", ""))
    if pid > 0 and process_is_running(pid):
        return "running"
    exit_code = get_exit_code(status_path)
    if exit_code is None:
        return "stopped_or_unknown"
    if exit_code == 0:
        return "completed_ok"
    return f"completed_error({exit_code})"


def stop_run(record: Dict[str, Any]) -> Tuple[bool, str]:
    pid = int(record.get("pid", 0))
    if pid <= 0:
        return False, "invalid pid"
    try:
        os.killpg(pid, signal.SIGTERM)
        return True, "sent SIGTERM"
    except ProcessLookupError:
        return False, "process not found"
    except Exception as exc:
        return False, f"stop failed: {exc}"


def tail_lines(path: Path, max_lines: int = 300, max_bytes: int = 2_000_000) -> List[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    offset = max(0, size - max_bytes)
    with path.open("rb") as handle:
        if offset > 0:
            handle.seek(offset)
            handle.readline()
        data = handle.read().decode("utf-8", errors="replace")
    lines = data.splitlines()
    return lines[-max_lines:]


def parse_jsonl_tail(path: Path, max_records: int = 300) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for line in tail_lines(path, max_lines=max_records * 2):
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    return parsed[-max_records:]


def build_command(
    search_name: str,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    resume: bool,
    limit: int | None,
    max_candidates: int | None,
    min_crossings: int | None,
    use_regina: bool,
    components: int | None,
    max_depth: int | None,
    max_nodes: int | None,
) -> List[str]:
    script_path = SCRIPT_DIR / SEARCH_CONFIG[search_name]["script"]
    cmd = [
        os.environ.get("PYTHON_EXECUTABLE", os.sys.executable),
        str(script_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--checkpoint",
        str(checkpoint_path),
    ]
    if resume:
        cmd.append("--resume")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if max_candidates is not None:
        cmd.extend(["--max-candidates", str(max_candidates)])

    if search_name == "Slice-Ribbon":
        if min_crossings is not None:
            cmd.extend(["--min-crossings", str(min_crossings)])
        if use_regina:
            cmd.append("--use-regina")
    elif search_name == "Generalized Property R":
        if components is not None:
            cmd.extend(["--components", str(components)])
        if min_crossings is not None:
            cmd.extend(["--min-crossings", str(min_crossings)])
        if use_regina:
            cmd.append("--use-regina")
    elif search_name in ("Montesinos-Nakanishi (3-move)", "Nakanishi (4-move)"):
        if min_crossings is not None:
            cmd.extend(["--min-crossings", str(min_crossings)])
        if max_depth is not None:
            cmd.extend(["--max-depth", str(max_depth)])
        if max_nodes is not None:
            cmd.extend(["--max-nodes", str(max_nodes)])
    elif search_name == "Kauffman Strong (specialized)":
        pass

    return cmd


def launch_run(command: List[str], cwd: Path, log_path: Path, status_path: Path) -> int:
    ensure_parent(log_path)
    ensure_parent(status_path)
    if status_path.exists():
        status_path.unlink()

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== Run started at {now_utc_iso()} UTC ===\n")
        handle.write("Command: " + shlex.join(command) + "\n\n")

    quoted_cmd = shlex.join(command)
    shell_cmd = (
        f"{quoted_cmd} >> {shlex.quote(str(log_path))} 2>&1; "
        f"echo $? > {shlex.quote(str(status_path))}"
    )

    proc = subprocess.Popen(
        ["/bin/bash", "-lc", shell_cmd],
        cwd=str(cwd),
        start_new_session=True,
    )
    return proc.pid


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    status_counts: Counter[str] = Counter()
    conjecture_counts: Counter[str] = Counter()
    for record in records:
        status_counts[str(record.get("status", "unknown"))] += 1
        conjecture_counts[str(record.get("conjecture", "unknown"))] += 1
    return {
        "by_status": dict(status_counts),
        "by_conjecture": dict(conjecture_counts),
    }


def format_run_label(record: Dict[str, Any]) -> str:
    status = run_status(record)
    run_id = record.get("id", "unknown")
    search_name = record.get("search_name", "unknown")
    return f"{run_id} | {search_name} | {status}"


def main() -> None:
    st.set_page_config(page_title="Knot Counterexample Search", layout="wide")
    st.title("Knot Counterexample Search UI")
    st.caption(
        "Launch SnapPy/Regina searches, inspect live logs, and explore JSONL results."
    )

    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.subheader("Environment")
        st.code(f"UI script dir: {SCRIPT_DIR}")
        st.code(f"Project root: {PROJECT_ROOT}")
        st.code(f"Work root: {DEFAULT_WORK_ROOT}")
        st.code(f"Run registry: {RUN_REGISTRY_PATH}")

    st.header("Launch Search")
    search_name = st.selectbox("Search", list(SEARCH_CONFIG.keys()))
    defaults = SEARCH_CONFIG[search_name]

    input_path = Path(
        st.text_input("Input file", str(defaults["input_default"]))
    )
    output_path = Path(
        st.text_input("Output JSONL", str(defaults["output_default"]))
    )
    checkpoint_path = Path(
        st.text_input("Checkpoint file", str(defaults["checkpoint_default"]))
    )
    resume = st.checkbox("Resume from checkpoint", value=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        limit_value = st.number_input(
            "Limit lines (0 = no limit)",
            min_value=0,
            value=0,
            step=100,
        )
        limit = int(limit_value) if int(limit_value) > 0 else None
    with col_b:
        max_candidates_value = st.number_input(
            "Max candidates (0 = unlimited)",
            min_value=0,
            value=0,
            step=100,
        )
        max_candidates = int(max_candidates_value) if int(max_candidates_value) > 0 else None
    with col_c:
        min_crossings_default = 12 if search_name == "Slice-Ribbon" else 10
        min_crossings_value = st.number_input(
            "Min crossings (0 = unset)",
            min_value=0,
            value=min_crossings_default if search_name != "Kauffman Strong (specialized)" else 0,
            step=1,
        )
        min_crossings = int(min_crossings_value) if int(min_crossings_value) > 0 else None

    use_regina = False
    components = None
    max_depth = None
    max_nodes = None

    if search_name in ("Slice-Ribbon", "Generalized Property R"):
        use_regina = st.checkbox("Use Regina cross-checks", value=True)

    if search_name == "Generalized Property R":
        components = int(
            st.number_input("Required components", min_value=1, value=2, step=1)
        )

    if search_name in ("Montesinos-Nakanishi (3-move)", "Nakanishi (4-move)"):
        col_d, col_e = st.columns(2)
        with col_d:
            max_depth = int(st.number_input("Max move depth", min_value=1, value=6, step=1))
        with col_e:
            max_nodes = int(st.number_input("Max BFS nodes", min_value=100, value=10000, step=100))

    command = build_command(
        search_name=search_name,
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        resume=resume,
        limit=limit,
        max_candidates=max_candidates,
        min_crossings=min_crossings,
        use_regina=use_regina,
        components=components,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )

    st.subheader("Command Preview")
    st.code(shlex.join(command), language="bash")

    launch_col, _ = st.columns([1, 5])
    with launch_col:
        if st.button("Launch Run", type="primary"):
            if not input_path.exists():
                st.error(f"Input path does not exist: {input_path}")
            else:
                run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
                log_path = DEFAULT_LOG_DIR / f"{run_id}.log"
                status_path = DEFAULT_LOG_DIR / f"{run_id}.status"
                pid = launch_run(
                    command=command,
                    cwd=PROJECT_ROOT,
                    log_path=log_path,
                    status_path=status_path,
                )
                records = load_registry()
                records.append(
                    {
                        "id": run_id,
                        "search_name": search_name,
                        "pid": pid,
                        "command": command,
                        "command_shell": shlex.join(command),
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "checkpoint_path": str(checkpoint_path),
                        "log_path": str(log_path),
                        "status_path": str(status_path),
                        "cwd": str(PROJECT_ROOT),
                        "started_at_utc": now_utc_iso(),
                    }
                )
                save_registry(records)
                st.success(f"Started {run_id} (pid={pid})")

    st.header("Run Monitor")
    records = load_registry()
    records = sorted(records, key=lambda r: str(r.get("started_at_utc", "")), reverse=True)
    if not records:
        st.info("No runs recorded yet.")
        return

    run_options = [record["id"] for record in records]
    selected_id = st.selectbox("Select run", run_options)
    selected = next(record for record in records if record["id"] == selected_id)

    status = run_status(selected)
    st.write(
        {
            "run_id": selected["id"],
            "search": selected["search_name"],
            "pid": selected["pid"],
            "status": status,
            "started_at_utc": selected.get("started_at_utc"),
            "output_path": selected.get("output_path"),
            "checkpoint_path": selected.get("checkpoint_path"),
            "log_path": selected.get("log_path"),
        }
    )
    st.code(str(selected.get("command_shell", "")), language="bash")

    col_x, col_y, col_z = st.columns(3)
    with col_x:
        if st.button("Stop selected run"):
            ok, msg = stop_run(selected)
            if ok:
                st.warning(msg)
            else:
                st.error(msg)
    with col_y:
        if st.button("Reload run registry"):
            st.rerun()
    with col_z:
        auto_refresh = st.checkbox("Auto-refresh while running", value=True)

    log_path = Path(str(selected.get("log_path", "")))
    st.subheader("Log Tail")
    st.text_area(
        "latest log lines",
        "\n".join(tail_lines(log_path, max_lines=350)),
        height=320,
    )

    st.header("Result Explorer")
    output_path_value = st.text_input(
        "Output JSONL to inspect",
        str(selected.get("output_path", "")),
    )
    output_records = parse_jsonl_tail(Path(output_path_value), max_records=300)
    if not output_records:
        st.info("No JSONL records found yet for the selected output file.")
    else:
        summary = summarize_records(output_records)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("Counts by Status")
            st.json(summary["by_status"])
        with col_s2:
            st.subheader("Counts by Conjecture")
            st.json(summary["by_conjecture"])

        table_rows = []
        for record in output_records:
            reasons = record.get("reasons", [])
            if isinstance(reasons, list):
                reason_preview = "; ".join(str(item) for item in reasons)
            else:
                reason_preview = str(reasons)
            table_rows.append(
                {
                    "conjecture": record.get("conjecture"),
                    "status": record.get("status"),
                    "line_number": record.get("line_number"),
                    "descriptor": record.get("descriptor"),
                    "reasons": reason_preview[:220],
                }
            )
        st.subheader("Recent Records")
        st.dataframe(table_rows, use_container_width=True)
        st.subheader("Most Recent Raw Record")
        st.json(output_records[-1])

    st.subheader("All Runs")
    run_rows = []
    for record in records:
        run_rows.append(
            {
                "run_id": record.get("id"),
                "search": record.get("search_name"),
                "status": run_status(record),
                "pid": record.get("pid"),
                "started_at_utc": record.get("started_at_utc"),
                "output_path": record.get("output_path"),
            }
        )
    st.dataframe(run_rows, use_container_width=True)

    if auto_refresh and status == "running":
        time.sleep(2.0)
        st.rerun()


if __name__ == "__main__":
    main()
