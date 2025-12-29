# Stage 1 Persona Augmentation (TravelPlanner → Multi-Persona)

This folder contains **Stage 1** of a multi-person travel planning benchmark pipeline.

Stage 1 input: one TravelPlanner record  
Stage 1 output: **group personas** (N = people_number)

✅ Refactor highlights
- **Removed TripCraft residual enums** (traveler_type / purpose_of_travel / spending_preference / location_preference).
- Persona profile dimensions are now **Stravl-style survey dimensions** (age / budget tier / experiences / scenery / etc.).
- Supports **grounded persona reference injection** (rule-based retrieval + diversity reranking) from the public Stravl dataset.
- Writes optional **meta/debug info** (inputs, mapping, candidate references) into each output JSON.

---

## Quick start

From `stage1_persona_aug/`:

```bash
# 1) install deps
pip install -r requirements.txt

# 2) set API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3) run Stage 1 on TravelPlanner test (10 examples)
python3 run_stage1.py --max_records 10 --workers 2 --store_meta
```

Outputs:
- `outputs_stage1/test/<source_id>.json`

Caches:
- HF cache: `data/hf_cache/`
- Minimal TravelPlanner cache: `data/travelplanner_min/test.jsonl`
- Compact DB summaries: `data/travelplanner_db_summary/test.json`
- Stravl CSV: `data/stravl/Stravl_Travel_Preference_Data.csv`  
  (or set `STRAVL_CSV_PATH=/path/to/Stravl_Travel_Preference_Data.csv`)

---

## Stage 1 Input

We read `osunlp/TravelPlanner` (HF) and normalize each row into `initial_info`:

```json
{
  "org": "...",
  "dest": "...",
  "days": 3,
  "visiting_city_number": 1,
  "date": ["YYYY-MM-DD", "..."],
  "people_number": 2,
  "query": "(rewritten to be self-consistent)",
  "budget_anchor": 2800,
  "local_constraint_anchor": {
    "house rule": null,
    "cuisine": null,
    "room type": null,
    "transportation": null
  },
  "level": "easy"
}
```

Normalization rules (see `travelplanner_loader.py`):
- If `people_number < 2`, we upsample using `--people_choices` (default: `2,4`).
- If we upsample people, we **scale total budget** proportionally to keep pppn stable.
- If `budget` is missing, we try to parse the first `$...` amount from `query`.
- We **rewrite** `query` deterministically to avoid contradictions.

---

## Stage 1 Output (schema)

The model output is validated with OpenAI Structured Outputs using `structured_output.py`.

Top-level:
- `source_id`: stable id (uses TravelPlanner `id` if available)
- `initial_info`: the normalized scenario
- `group_personas`: list of length `people_number`

Each persona contains:
- `name`, `role`, `archetype`
- `profile` (Stravl-style dimensions)
- `budget_profile.max_budget_multiplier` (0.7~1.4)
- `grounding_anchors` (2~4, overwritten to scenario anchors)
- `seed_preferences` (3~5 short strings)

### Budget note (important)
`profile.budget_tier` uses Stravl **per-person-per-night** bucket labels (e.g., `$100-$249`), but in this project it is treated as a **relative spending tier** mapped from TravelPlanner’s `pppn = budget_anchor / (people_number * days)` using quantiles (if available).  
See `persona_reference_retrieval.py` for mapping logic.

---

## Persona reference injection (people × 10)

By default, Stage 1 loads Stravl CSV and retrieves `K = people_number * ref_multiplier` reference cards.
- retrieval: rule-score + MMR-style diversity reranking
- injected into the prompt as a JSON array

You can disable it with:
```bash
python3 run_stage1.py --no_persona_refs
```

---

## Meta/debug logging

Enable `--store_meta` to append a `meta` block to each output JSON containing:
- minimal raw input row
- normalization details (upsample, budget parsing)
- derived trip features (pppn, mapped budget tier, etc.)
- DB summary (compact, extracted from TravelPlanner `reference_information`)
- selected Stravl persona reference cards (capped to 50 for size)

---

## Files

- `run_stage1.py`: main runner (HF load + caching + LLM calls)
- `structured_output.py`: Pydantic schema for structured output
- `travelplanner_loader.py`: normalization + anchor generation
- `hf_dataset_cache.py`: HF dataset loading + jsonl caching
- `persona_reference_retrieval.py`: rule-score + MMR retrieval from Stravl
- `stravl_codec.py`: robust Stravl FORM_* decoding
- `db_summary.py`: compact summary extractor for TravelPlanner reference_information
- `tp_quantiles.py`: load precomputed TravelPlanner quantiles (optional)
- `prompts/`: system/user prompts
