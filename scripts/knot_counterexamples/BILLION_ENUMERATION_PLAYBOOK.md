# Billion-Scale Knot Enumeration Playbook

This playbook turns the Hoste survey workflow into an operational pipeline for:

- `/Volumes/Holomorphic/BigKnotTables` (data + checkpoints + logs)
- `/Volumes/Holomorphic/BigKnotTables/scripts/knot_counterexamples` (control scripts)

The core algorithmic strategy is:

1. Build alternating level `n+1` from level `n` using local operators (`D/ROTS`), then close under `T/OTS`.
2. Collapse duplicates with flype-aware canonical forms.
3. Generate nonalternating candidates by grouped crossing flips (avoid naive `2^n` explosion).
4. Re-canonicalize, dedupe, compute invariants, and gate each crossing level against known counts.

Important: this repo currently provides orchestration and checkpoint tooling.
You still need to plug in algorithm backends for:

- `algorithms/alt_generate_d_rots.py`
- `algorithms/alt_close_t_ots.py`
- `algorithms/alt_canonicalize_flype.py`
- `algorithms/nonalt_generate_group_flips.py`
- `algorithms/nonalt_canonicalize.py`
- `algorithms/dedupe_keys.py`
- `algorithms/compute_invariants.py`
- `algorithms/validate_milestones.py`

## 1) Bootstrap environment

```bash
cd /Volumes/Holomorphic/BigKnotTables
uv venv .venv-knotsearch --python 3.11
source /Volumes/Holomorphic/BigKnotTables/.venv-knotsearch/bin/activate
uv pip install snappy regina streamlit
```

## 2) Initialize workspace layout + DB + manifest templates

```bash
python /Volumes/Holomorphic/BigKnotTables/scripts/knot_counterexamples/setup_billion_enumeration_workspace.py \
  --root /Volumes/Holomorphic/BigKnotTables
```

Outputs:

- `checkpoints/enumeration_state.sqlite`
- `manifests/billion_enumeration_manifest.json`
- `manifests/milestone_counts.csv`
- full directory scaffold (`generated/`, `canonical/`, `logs/`, etc.)

## 3) Fill milestone gates before large runs

Edit:

- `/Volumes/Holomorphic/BigKnotTables/manifests/milestone_counts.csv`

Add literature counts per crossing level. Treat mismatches as stop conditions.

## 4) Configure manifest for your cluster size

Edit:

- `/Volumes/Holomorphic/BigKnotTables/manifests/billion_enumeration_manifest.json`

Tune:

- `crossings.start`, `crossings.end`
- `sharding.num_shards` (start 64 or 128; scale to 256/512 when stable)
- `stages.*.command_template` (point to real backend scripts/binaries)

## 5) Generate task queue

```bash
python /Volumes/Holomorphic/BigKnotTables/scripts/knot_counterexamples/generate_enumeration_tasks.py \
  --manifest /Volumes/Holomorphic/BigKnotTables/manifests/billion_enumeration_manifest.json \
  --out /Volumes/Holomorphic/BigKnotTables/manifests/tasks_c16_c18.jsonl \
  --crossing-start 16 \
  --crossing-end 18
```

For first validation, restrict to one stage:

```bash
python /Volumes/Holomorphic/BigKnotTables/scripts/knot_counterexamples/generate_enumeration_tasks.py \
  --manifest /Volumes/Holomorphic/BigKnotTables/manifests/billion_enumeration_manifest.json \
  --out /Volumes/Holomorphic/BigKnotTables/manifests/tasks_alt_generate_c16.jsonl \
  --crossing-start 16 \
  --crossing-end 16 \
  --stages alt_generate_d_rots
```

## 6) Dry-run tasks (dependency check, no execution)

```bash
python /Volumes/Holomorphic/BigKnotTables/scripts/knot_counterexamples/run_enumeration_tasks.py \
  --tasks /Volumes/Holomorphic/BigKnotTables/manifests/tasks_alt_generate_c16.jsonl \
  --db /Volumes/Holomorphic/BigKnotTables/checkpoints/enumeration_state.sqlite \
  --workspace-root /Volumes/Holomorphic/BigKnotTables \
  --dry-run
```

## 7) Execute tasks

```bash
python /Volumes/Holomorphic/BigKnotTables/scripts/knot_counterexamples/run_enumeration_tasks.py \
  --tasks /Volumes/Holomorphic/BigKnotTables/manifests/tasks_c16_c18.jsonl \
  --db /Volumes/Holomorphic/BigKnotTables/checkpoints/enumeration_state.sqlite \
  --workspace-root /Volumes/Holomorphic/BigKnotTables
```

To isolate one stage for debugging:

```bash
python /Volumes/Holomorphic/BigKnotTables/scripts/knot_counterexamples/run_enumeration_tasks.py \
  --tasks /Volumes/Holomorphic/BigKnotTables/manifests/tasks_c16_c18.jsonl \
  --db /Volumes/Holomorphic/BigKnotTables/checkpoints/enumeration_state.sqlite \
  --workspace-root /Volumes/Holomorphic/BigKnotTables \
  --stage alt_canonicalize_flype \
  --crossing 16
```

## 8) Operational guardrails

Use these for billion-scale safety:

1. Keep outputs append-only per `(stage, crossing, shard)`.
2. Write canonical keys before expensive invariants.
3. Enforce per-stage checkpoint files (`checkpoints/<stage>/...`).
4. Stop on first milestone mismatch.
5. Run low crossings (`<=18`) end-to-end before expanding range.
6. Keep `logs/tasks/<task_id>.log` for every task.
7. Do not mix manifest versions in one run.

## 9) Suggested shard sizing

1. Local workstation: `num_shards = 32..64`.
2. Single high-memory server: `num_shards = 128`.
3. Multi-node cluster/object store: `num_shards = 256..1024`.

Rule of thumb: each shard should fit in memory for canonicalization + dedupe.

## 10) Known limitations in this scaffold

1. No algorithm backend is bundled yet for D/ROTS/T/OTS/master-array canonicalization.
2. Task runner is dependency-aware but sequential (good for deterministic debugging).
3. For cluster execution, wrap the generated tasks into your scheduler (Slurm/Ray/K8s).

## 11) UI integration

Use Streamlit UI for counterexample-search scripts while you debug backend primitives:

```bash
streamlit run /Volumes/Holomorphic/BigKnotTables/scripts/knot_counterexamples/ui.py
```

Once backends are stable, add an "Enumeration" tab to `ui.py` that calls these same manifest/task scripts.
