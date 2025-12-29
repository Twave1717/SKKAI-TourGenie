# Paper experiments (A/B/C)

This folder contains **paper-facing** evaluation utilities for Stage 2.

Stage 2 claim: *"we generate creative + diverse constraint seeds while staying groundable in a closed DB sandbox"*.
Reviewers will ask (1) how creative/diverse, (2) how much regeneration it costs, (3) whether schema injection matters.

We provide three low-ROI-high-impact experiments:

## A. Constraint diversity / creativity stats

Compute simple but convincing statistics:

- #constraints per persona distribution
- field diversity (restaurant/accommodation/attraction/flight/ground coverage)
- operator diversity
- keyword-proxy ratio (park/trail/museum style constraints via keyword fields)

Run:

```bash
python3 paper_experiments/A_diversity/compute_diversity_stats.py \
  --in_dir outputs_stage2/test/schema_alias_snapshot \
  --out artifacts/A_diversity.json \
  --md  artifacts/A_diversity.md
```

## B. Validator loop efficiency (regen cost)

Compute:

- 1-pass valid rate
- avg attempts / avg retries
- top validation error types

This requires `run_stage2.py --write_meta` (meta is written to `<out_dir>/.../meta/`).

Run:

```bash
python3 paper_experiments/B_validator/compute_validator_efficiency.py \
  --meta_dir outputs_stage2/test/schema_alias_snapshot/meta \
  --out artifacts/B_validator.json \
  --md  artifacts/B_validator.md
```

## C. Schema block ablation

Run Stage 2 under four prompt-context modes, then summarize A/B metrics:

(i) `none` (no extra schema text)  
(ii) `schema_only` (tables+columns)  
(iii) `schema_alias` (tables+columns + alias map)  
(iv) `schema_alias_snapshot` (iii + per-instance snapshot)

Run:

```bash
python3 paper_experiments/C_schema_ablation/run_schema_ablation.py \
  --db_root "/path/to/travelplanner/official/database" \
  --stage1_dir "../stage1_persona_aug/outputs_stage1/test" \
  --max_records 50 \
  --workers 4
```

The script will create `outputs_stage2/test/<schema_mode>/...` and write an aggregated summary to `artifacts/C_schema_ablation_summary.json`.
