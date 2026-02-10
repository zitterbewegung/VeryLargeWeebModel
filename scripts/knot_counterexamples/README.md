# Knot Counterexample Search Scripts

This folder contains practical search scripts for conjectures where a single
counterexample is enough.

Design pattern:
1. Fast SnapPy filtering over large candidate pools.
2. Optional Regina cross-check where available.
3. Output candidate or hard-instance JSONL records with checkpoints.

Important: several target conjectures are deep 4D problems. These scripts are
candidate generators unless the output explicitly includes a full obstruction.

## Dependencies

- SnapPy: required for all scripts.
- Regina: optional, used for cross-check fields when present.
- Streamlit: optional, for the web UI.

Install:

```bash
python3 -m pip install snappy regina streamlit
```

## Input Formats

One descriptor per line unless otherwise noted.

- `DT:[4,-16,24,...]`
- `BRAID:[1,-4,2,3,...]`
- SnapPy names such as `8_20`

For the Whitehead-double script only:

- `<base descriptor> | <untwisted Whitehead double descriptor>`

Lines beginning with `#` and empty lines are ignored.

## Scripts

- `search_slice_ribbon.py`
  - Flags slice-ribbon candidates using computable slice-like filters.
  - Ribbon status remains unresolved without extra certification.

- `search_generalized_property_r.py`
  - Scans links, runs full 0-surgery, keeps cases with expected homology rank.
  - Handle-slide equivalence remains unresolved.

- `search_three_move.py`
  - Bounded 3-move BFS on braid words.
  - Reports hard instances when unknot path is not found within bounds.

- `search_four_move.py`
  - Bounded 4-move BFS on braid words.
  - Reports hard instances when unknot path is not found within bounds.

- `search_whitehead_double.py`
  - Evaluates `(K, Wh_0(K))` pairs and flags slice-like mismatches.

- `ui.py`
  - Streamlit dashboard for launching searches, monitoring logs, and inspecting JSONL outputs.

- `setup_billion_enumeration_workspace.py`
  - Creates billion-scale workspace layout, SQLite state DB, manifest template, and milestone template.

- `generate_enumeration_tasks.py`
  - Generates sharded task JSONL from the enumeration manifest.

- `run_enumeration_tasks.py`
  - Executes task JSONL with dependency checks and per-task logging.

- `BILLION_ENUMERATION_PLAYBOOK.md`
  - End-to-end execution plan for large crossing-level enumeration.

## Example Commands

Slice-Ribbon candidates from a large DT list:

```bash
python3 scripts/knot_counterexamples/search_slice_ribbon.py \
  --input /Volumes/Holomorphic/BigKnotTables/tables/knots_dt.txt \
  --output out/slice_ribbon_candidates.jsonl \
  --resume
```

Generalized Property R candidates on 2-component links:

```bash
python3 scripts/knot_counterexamples/search_generalized_property_r.py \
  --input /Volumes/Holomorphic/BigKnotTables/tables/links_dt.txt \
  --components 2 \
  --output out/property_r_candidates.jsonl \
  --resume
```

3-move bounded hard-instance scan:

```bash
python3 scripts/knot_counterexamples/search_three_move.py \
  --input /Volumes/Holomorphic/BigKnotTables/tables/5braids.txt \
  --max-depth 6 \
  --max-nodes 10000 \
  --output out/three_move_hard_instances.jsonl \
  --resume
```

4-move bounded hard-instance scan:

```bash
python3 scripts/knot_counterexamples/search_four_move.py \
  --input /Volumes/Holomorphic/BigKnotTables/tables/candidate_braids.txt \
  --max-depth 6 \
  --max-nodes 10000 \
  --output out/four_move_hard_instances.jsonl \
  --resume
```

Kauffman strong conjecture specialized-pair scan:

```bash
python3 scripts/knot_counterexamples/search_whitehead_double.py \
  --input /Volumes/Holomorphic/BigKnotTables/tables/whitehead_pairs.txt \
  --output out/whitehead_double_candidates.jsonl \
  --resume
```

## Suggested Next Upgrade

For rigorous 3-move and 4-move obstructions, add a Burnside-group or quandle
invariant backend and mark those outputs as certified when invariants differ
from the unknot.

## UI Usage

Launch the UI:

```bash
streamlit run scripts/knot_counterexamples/ui.py
```

Or use the helper:

```bash
scripts/knot_counterexamples/run_ui.sh
```

With `uv`:

```bash
uv pip install streamlit snappy regina
uv run streamlit run scripts/knot_counterexamples/ui.py
```

## Billion Enumeration Bootstrap

```bash
python3 scripts/knot_counterexamples/setup_billion_enumeration_workspace.py \
  --root /Volumes/Holomorphic/BigKnotTables

python3 scripts/knot_counterexamples/generate_enumeration_tasks.py \
  --manifest /Volumes/Holomorphic/BigKnotTables/manifests/billion_enumeration_manifest.json \
  --out /Volumes/Holomorphic/BigKnotTables/manifests/tasks.jsonl

python3 scripts/knot_counterexamples/run_enumeration_tasks.py \
  --tasks /Volumes/Holomorphic/BigKnotTables/manifests/tasks.jsonl \
  --db /Volumes/Holomorphic/BigKnotTables/checkpoints/enumeration_state.sqlite \
  --workspace-root /Volumes/Holomorphic/BigKnotTables \
  --dry-run
```
