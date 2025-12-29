# Stage 2: Groundable Constraint Seed Generation (Multi-Persona)

Stage2 takes **Stage1 multi-persona outputs** and generates **DB-groundable, structured constraints**
for each persona **inside the TravelPlanner sandbox database**.

- Model: `gpt-4.1`
- Output: OpenAI Structured Outputs (Pydantic schema)

---

## What Stage2 does

Input:
- Stage1 output JSON (per scenario): `{source_id, initial_info, group_personas...}`
- TravelPlanner official DB folder (CSV tables)
- TravelPlanner dataset row `reference_information` (used to build a *per-instance* DB snapshot)

Output (per scenario):
- `group_personas[i].structured_requirement` filled with:
  - `hard_constraints` (non-negotiable)
  - `soft_constraints` (negotiable)
- Optional meta JSON per example (validation attempts, db snapshot, etc.)

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Set env

Create `.env`:

```bash
OPENAI_API_KEY=...
```

### 3) Run Stage2

```bash
python3 run_stage2.py \
  --split test \
  --db_root "/Users/ky/AgenticAI/Challenge/benchmarks/travelplanner/official/database" \
  --stage1_dir "../stage1_persona_aug/outputs_stage1/test" \
  --schema_mode schema_alias_snapshot \
  --max_records 10 \
  --workers 2 \
  --write_meta \
  --strict_values
```

Outputs:

```
outputs_stage2/
  test/
    schema_alias_snapshot/
      <source_id>.json
      meta/
        <source_id>.json
```

---

## Prompt context modes (schema ablation knob)

`--schema_mode` controls how much DB schema context is injected into the **same fixed prompt**:

1) `none`
- No explicit schema text in the prompt.
- The model still must conform to the structured output schema.

2) `schema_only`
- Tables + columns only.

3) `schema_alias`
- Tables + columns + **alias map** (allowed `field` strings).

4) `schema_alias_snapshot` (recommended default)
- `schema_alias` + per-instance snapshot (allowed categorical values + numeric ranges).

This supports the paper ablation:
(i) no schema, (ii) schema only, (iii) schema+alias, (iv) schema+alias+snapshot.

---

## Validator strictness

`--strict_values`:
- If enabled, categorical values (e.g., cuisine / room_type / house_rule) must appear in the **instance snapshot**.
- If disabled, categorical values are checked against a global enum list (more permissive).

For paper-facing results, enabling `--strict_values` usually makes the "regen cost" story cleaner.

---

## Why Stage2 builds its own DB snapshot (and Stage1 can stay minimal)

Stage2 uses the official TravelPlanner dataset's `reference_information` field to build a per-instance snapshot.
This is more robust than passing a snapshot from Stage1 because:

- Stage1 should be purely persona-focused (no DB coupling).
- Stage2 is the first place that *needs* DB grounding.
- Snapshot logic can evolve without re-running Stage1.

**Implication:** Stage1 does **not** need to store any DB snapshot/db_summary fields.

---

## Paper experiments (A/B/C)

See `paper_experiments/README.md`:

- **A**: creativity/diversity stats
- **B**: validator-loop efficiency (regen cost)
- **C**: schema block ablation runner

