#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from db_schema import build_db_schema_snapshot, load_or_build_global_schema
from structured_output import Stage2LLMOutput
from validator import validate_stage2


DATASET_NAME = "osunlp/TravelPlanner"

# Must match Stage1's stable-id payload keys (DO NOT include reference_information)
STABLE_ID_KEYS = [
    "org",
    "dest",
    "days",
    "visiting_city_number",
    "date",
    "people_number",
    "local_constraint",
    "budget",
    "query",
    "level",
    "id",
]




# ---------------------------
# Prompt helpers
# ---------------------------

UNIT_HINTS = {
    # Restaurants
    "restaurant.avg_cost": "USD per person per meal",
    "restaurant.rating": "0..5 rating",
    "restaurant.count": "number of matching rows in scenario",

    # Accommodations
    "accommodation.price": "USD per night per listing (NOT total trip budget)",
    "accommodation.review": "0..5 review score",
    "accommodation.minimum_nights": "nights",
    "accommodation.maximum_occupancy": "people per listing",
    "accommodation.count": "number of matching rows in scenario",

    # Attractions
    "attraction.count": "number of matching rows in scenario",

    # Flights
    "flight.price": "USD one-way per person",
    "flight.duration_minutes": "minutes",
    "flight.distance": "miles",
    "flight.count": "number of matching rows in scenario",

    # Ground
    "ground.cost": "USD (per trip)",
    "ground.duration_minutes": "minutes",
    "ground.distance": "km",
}

BUDGET_TIER_ENUM = ["Frugal", "Budget", "Comfort", "Luxury"]
LEGACY_BUDGET_TIER_MAP = {
    "$0-$49": "Frugal",
    "$50-$99": "Budget",
    "$100-$249": "Comfort",
    "$300+": "Luxury",
}


def _normalize_budget_tier(v: Any) -> Any:
    if not isinstance(v, str) or not v.strip():
        return v
    s = v.strip()
    if s in BUDGET_TIER_ENUM:
        return s
    if s in LEGACY_BUDGET_TIER_MAP:
        return LEGACY_BUDGET_TIER_MAP[s]
    low = s.lower()
    if "frugal" in low:
        return "Frugal"
    if "budget" in low:
        return "Budget"
    if "comfort" in low:
        return "Comfort"
    if "lux" in low:
        return "Luxury"
    return s


def _budget_guidance(initial_info: Dict[str, Any], *, budget_multiplier: float, budget_tier: str) -> Dict[str, Any]:
    """Lightweight numeric hints to prevent unit mistakes."""
    ba = initial_info.get("budget_anchor")
    days = int(initial_info.get("days") or max(1, len(initial_info.get("date") or [])) or 1)
    people = int(initial_info.get("people_number") or 1)
    if not isinstance(ba, (int, float)) or ba <= 0:
        return {}

    total_target = float(ba) * float(budget_multiplier)
    per_day_total = total_target / max(1, days)

    # coarse shares by tier (not a solver)
    share_hotel = {"Frugal": 0.25, "Budget": 0.33, "Comfort": 0.40, "Luxury": 0.50}.get(budget_tier, 0.33)
    share_flight = {"Frugal": 0.40, "Budget": 0.35, "Comfort": 0.30, "Luxury": 0.25}.get(budget_tier, 0.35)
    share_food = {"Frugal": 0.18, "Budget": 0.20, "Comfort": 0.22, "Luxury": 0.25}.get(budget_tier, 0.20)

    # derived maxima (recommendations, not strict)
    hotel_max_per_night = per_day_total * share_hotel
    flight_max_oneway_pp = (total_target * share_flight) / max(1, (2 * people))  # ~2 one-way legs per person
    meal_max_pp = (total_target * share_food) / max(1, (days * 2 * people))  # ~2 restaurant meals per person per day

    return {
        "budget_anchor_total": int(round(float(ba))),
        "budget_target_total": int(round(total_target)),
        "units_note": "budget_anchor=total trip budget; accommodation.price=per-night per listing; flight.price=one-way per person; restaurant.avg_cost=per-person per meal.",
        "suggested_max": {
            "accommodation.price": int(round(hotel_max_per_night)),
            "flight.price": int(round(flight_max_oneway_pp)),
            "restaurant.avg_cost": int(round(meal_max_pp)),
        },
    }
def stable_source_id(row_min: Dict[str, Any]) -> str:
    """Stable id compatible with Stage1 (id or sha1 over row_min JSON)."""
    rid = row_min.get("id")
    if isinstance(rid, str) and rid.strip():
        return rid.strip()
    payload = json.dumps(row_min, ensure_ascii=False, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:16]
    return f"row_{h}"


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _load_split_full(split: str, cache_path: Path, hf_cache_dir: Path) -> List[Dict[str, Any]]:
    """Load TravelPlanner split with `reference_information` (cached as jsonl)."""
    if cache_path.exists():
        return [json.loads(l) for l in cache_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    try:
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    except Exception as e:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets`.\n"
            f"Import error: {e}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _try_load():
        try:
            cfgs = get_dataset_config_names(DATASET_NAME)
            cfg = split if split in cfgs else cfgs[0]
            try:
                split_names = get_dataset_split_names(DATASET_NAME, cfg)
            except Exception:
                split_names = [cfg]
            split_to_use = split if split in split_names else split_names[0]
            return load_dataset(DATASET_NAME, cfg, split=split_to_use, cache_dir=str(hf_cache_dir))
        except Exception:
            # last resort
            return load_dataset(DATASET_NAME, name=split, split=split, cache_dir=str(hf_cache_dir))

    ds = _try_load()
    keep = STABLE_ID_KEYS + ["reference_information"]

    rows: List[Dict[str, Any]] = []
    with cache_path.open("w", encoding="utf-8") as f:
        for r in ds:
            obj = {k: r.get(k) for k in keep}
            rows.append(obj)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return rows


def _schema_tables_text(schema: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Tables (CSV) and columns:")
    for t in schema.get("tables", []):
        cols = t.get("columns") or []
        if cols:
            lines.append(f"- {t['table']}: " + ", ".join(cols))
        else:
            lines.append(f"- {t['table']}: (missing)")
    return "\n".join(lines)


def _schema_alias_text(schema: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Allowed constraint fields (use these exact `field` strings):")
    for a in schema.get("field_aliases", []):
        f = str(a.get("field") or "")
        unit = UNIT_HINTS.get(f, "")
        unit_txt = f" (unit: {unit})" if unit else ""
        lines.append(f"- {a['field']} -> {a['table']}.{a['column']}{unit_txt}")
    return "\n".join(lines)

def _build_prompt_blocks(schema_mode: str, global_schema: Dict[str, Any], db_snapshot: Dict[str, Any]) -> Tuple[str, str]:
    """Return (schema_block, snapshot_block) for the user prompt."""
    if schema_mode == "none":
        return (
            "No explicit DB schema context is provided in the prompt.\n"
            "You must still only use `field` values permitted by the output JSON schema.",
            "(not provided)",
        )

    schema_block = _schema_tables_text(global_schema)

    if schema_mode in ("schema_alias", "schema_alias_snapshot"):
        schema_block = schema_block + "\n\n" + _schema_alias_text(global_schema)

    snapshot_block = "(not provided)"
    if schema_mode == "schema_alias_snapshot":
        snapshot_block = json.dumps(db_snapshot, ensure_ascii=False, indent=2)

    return schema_block, snapshot_block


def _compact_personas(stage1_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compact persona view for prompting (keeps only Stage2-relevant fields).

    Adds `budget_guidance` to prevent common unit mistakes (e.g., using total trip
    budget as a per-night accommodation price).
    """
    initial_info = stage1_obj.get("initial_info") or {}
    out: List[Dict[str, Any]] = []
    for p in (stage1_obj.get("group_personas") or []):
        prof = p.get("profile") or {}
        budget_tier = _normalize_budget_tier(prof.get("budget_tier"))

        bp = p.get("budget_profile") or {}
        mult = bp.get("max_budget_multiplier")
        try:
            mult_f = float(mult) if mult is not None else 1.0
        except Exception:
            mult_f = 1.0

        out.append(
            {
                "name": p.get("name"),
                "role": p.get("role"),
                "archetype": p.get("archetype"),
                "profile": {
                    "profile_text": prof.get("profile_text", ""),
                    "age_range": prof.get("age_range"),
                    "budget_tier": budget_tier,
                    "season": prof.get("season"),
                    "experiences": prof.get("experiences") or [],
                    "scenery": prof.get("scenery") or [],
                    "activity_level": prof.get("activity_level"),
                    "safety_conscious": prof.get("safety_conscious"),
                    "destination_popularity": prof.get("destination_popularity"),
                },
                "budget_profile": bp,
                "budget_guidance": _budget_guidance(initial_info, budget_multiplier=mult_f, budget_tier=str(budget_tier or "Budget")),
                "seed_preferences": p.get("seed_preferences") or [],
                "grounding_anchors": p.get("grounding_anchors") or [],
            }
        )
    return out

def _render_user_prompt(user_tmpl: str, *, initial_info: Dict[str, Any], personas: List[Dict[str, Any]], schema_block: str, snapshot_block: str) -> str:
    # Use .replace() (not .format()) to avoid accidental brace collisions.
    s = user_tmpl
    s = s.replace("{initial_info_json}", json.dumps(initial_info, ensure_ascii=False, indent=2))
    s = s.replace("{personas_json}", json.dumps(personas, ensure_ascii=False, indent=2))
    s = s.replace("{schema_block}", schema_block)
    s = s.replace("{snapshot_block}", snapshot_block)
    return s


def _generate_stage2(
    client: OpenAI,
    sys_prompt: str,
    user_tmpl: str,
    *,
    initial_info: Dict[str, Any],
    personas_compact: List[Dict[str, Any]],
    schema_block: str,
    snapshot_block: str,
    db_snapshot: Dict[str, Any],
    temperature: float,
    max_retries: int,
    strict_values: bool,
    reject_noop: bool,
    schema_mode: str,
    drop_bad: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (llm_output_dict, validation_meta)."""
    user_prompt = _render_user_prompt(
        user_tmpl,
        initial_info=initial_info,
        personas=personas_compact,
        schema_block=schema_block,
        snapshot_block=snapshot_block,
    )

    attempt_logs: List[Dict[str, Any]] = []
    last_err = ""

    for attempt in range(1, max_retries + 2):
        user_prompt_retry = user_prompt if not last_err else (user_prompt + "\n\n[VALIDATION ERROR]\n" + last_err)

        try:
            resp = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt_retry},
                ],
                temperature=temperature,
                text_format=Stage2LLMOutput,
            )
            out: Stage2LLMOutput = resp.output_parsed
            obj = out.model_dump()
        except Exception as e:
            msg = f"parse_error: {type(e).__name__}: {e}"
            attempt_logs.append({"attempt": attempt, "valid": False, "errors": [msg]})
            last_err = msg
            continue

        errs, sanitized, dropped = validate_stage2(
            persona_requirements=obj.get("persona_requirements") or [],
            n_personas=len(personas_compact),
            db_snapshot=db_snapshot,
            strict_values=strict_values,
            reject_noop=reject_noop,
            drop_bad=drop_bad,
        )
        if errs:
            attempt_logs.append({"attempt": attempt, "valid": False, "errors": errs[:25]})
            last_err = "\n".join(f"- {e}" for e in errs[:18])
            continue
        # Replace with sanitized requirements when dropping bad ones
        if drop_bad:
            obj["persona_requirements"] = sanitized

        attempt_logs.append({"attempt": attempt, "valid": True, "errors": []})
        vmeta = {
            "attempts": attempt,
            "first_pass_valid": attempt == 1,
            "attempt_logs": attempt_logs,
            "final_valid": True,
            "dropped_constraints": dropped if drop_bad else [],
        }
        return obj, vmeta

    vmeta = {"attempts": len(attempt_logs), "first_pass_valid": False, "attempt_logs": attempt_logs, "final_valid": False}
    raise RuntimeError(f"Stage2 generation failed. Last error:\n{last_err}\nMETA={json.dumps(vmeta, ensure_ascii=False)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="test|validation|train (default: test)")

    # Prompt-context ablation knob
    ap.add_argument(
        "--schema_mode",
        default="schema_alias_snapshot",
        choices=["none", "schema_only", "schema_alias", "schema_alias_snapshot"],
        help="how much DB schema context to include in the prompt",
    )

    # Validator strictness (independent)
    ap.add_argument("--strict_values", action="store_true", help="strict categorical checks against instance snapshot")

    ap.add_argument("--db_root", required=True, help="TravelPlanner DB root directory (official layout root)")
    ap.add_argument("--schema_cache", default="data/schema_cache/global_schema.json", help="cache path for global schema json")

    ap.add_argument("--stage1_dir", required=True, help="where Stage1 JSONs are stored (test split recommended)")
    ap.add_argument("--max_records", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_retries", type=int, default=2)

    ap.add_argument("--data_dir", default="data", help="where to cache downloaded TravelPlanner inputs")
    ap.add_argument("--out_dir", default="outputs_stage2", help="base output directory")
    ap.add_argument("--no_resume", action="store_true", help="do not skip existing outputs")
    ap.add_argument("--write_meta", action="store_true", help="write a meta JSON per example to <out>/meta/")
    ap.add_argument("--reject_noop", action="store_true", help="treat overly-loose numeric thresholds as errors (stricter)")
    ap.add_argument("--drop_bad_constraints", action="store_true", help="drop invalid/no-op constraints instead of regenerating")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Put it in .env or env vars.")

    here = Path(__file__).resolve().parent
    data_dir = (Path(args.data_dir) if Path(args.data_dir).is_absolute() else (here / args.data_dir)).resolve()
    hf_cache_dir = data_dir / "hf_cache"
    split_cache = data_dir / "travelplanner_full" / f"{args.split}.jsonl"

    out_base = (Path(args.out_dir) if Path(args.out_dir).is_absolute() else (here / args.out_dir)).resolve() / args.split / args.schema_mode
    out_base.mkdir(parents=True, exist_ok=True)
    out_meta = out_base / "meta"
    if args.write_meta:
        out_meta.mkdir(parents=True, exist_ok=True)

    # Load prompts
    sys_prompt = _read_text(here / "prompts" / "system_prompt.txt")
    user_tmpl = _read_text(here / "prompts" / "user_prompt.txt")

    # Load TravelPlanner rows and build stable_id -> row map
    rows = _load_split_full(args.split, split_cache, hf_cache_dir)
    row_map: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        row_min = {k: r.get(k) for k in STABLE_ID_KEYS}
        sid = stable_source_id(row_min)
        row_map[sid] = r

    # Load global schema once
    db_root = Path(args.db_root).expanduser()
    schema_cache = Path(args.schema_cache)
    if not schema_cache.is_absolute():
        schema_cache = (here / schema_cache).resolve()
    global_schema = load_or_build_global_schema(db_root, schema_cache)

    # Stage1 files (support stage1_dir/data and stage1_dir/meta layout)
    stage1_dir = Path(args.stage1_dir).expanduser().resolve()
    data_dir = stage1_dir / "data" if (stage1_dir / "data").exists() else stage1_dir
    stage1_files = sorted([p for p in data_dir.glob("*.json") if p.is_file()])
    if args.max_records and args.max_records > 0:
        stage1_files = stage1_files[: args.max_records]
    if not stage1_files:
        raise SystemExit(f"No Stage1 json files found in {data_dir}")

    client = OpenAI()

    def _job(p: Path) -> str:
        stage1_obj = json.loads(p.read_text(encoding="utf-8"))
        source_id = stage1_obj.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise RuntimeError(f"Missing source_id in {p}")
        out_path = out_base / f"{source_id}.json"
        if (not args.no_resume) and out_path.exists():
            return "skipped"

        row = row_map.get(source_id)
        if row is None:
            raise RuntimeError(f"TravelPlanner row not found for source_id={source_id} (split={args.split})")

        initial_info = stage1_obj.get("initial_info") or {}
        personas_compact = _compact_personas(stage1_obj)

        db_snapshot = build_db_schema_snapshot(row.get("reference_information"))
        schema_block, snapshot_block = _build_prompt_blocks(args.schema_mode, global_schema, db_snapshot)

        llm_obj, vmeta = _generate_stage2(
            client,
            sys_prompt,
            user_tmpl,
            initial_info=initial_info,
            personas_compact=personas_compact,
            schema_block=schema_block,
            snapshot_block=snapshot_block,
            db_snapshot=db_snapshot,
            temperature=args.temperature,
            max_retries=args.max_retries,
            strict_values=args.strict_values,
            reject_noop=args.reject_noop,
            schema_mode=args.schema_mode,
            drop_bad=args.drop_bad_constraints,
        )

        # Merge requirements back into full personas
        reqs = llm_obj.get("persona_requirements") or []
        req_by_idx = {
            int(x.get("persona_index")): x.get("structured_requirement")
            for x in reqs
            if isinstance(x, dict) and isinstance(x.get("persona_index"), int)
        }

        merged_personas = []
        for i, full_p in enumerate(stage1_obj.get("group_personas") or []):
            cp = dict(full_p)
            # Normalize legacy budget tier strings (keeps Stage2 outputs consistent)
            prof = cp.get("profile") or {}
            if isinstance(prof, dict):
                prof["budget_tier"] = _normalize_budget_tier(prof.get("budget_tier"))
                cp["profile"] = prof
            cp["structured_requirement"] = req_by_idx.get(i)
            merged_personas.append(cp)

        out_obj = {"source_id": source_id, "initial_info": initial_info, "group_personas": merged_personas}
        out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.write_meta:
            meta_obj = {
                "source_id": source_id,
                "stage2_model": "gpt-4.1",
                "schema_mode": args.schema_mode,
                "strict_values": args.strict_values,
                "drop_bad_constraints": args.drop_bad_constraints,
                "db_root": str(db_root),
                "schema_cache": str(schema_cache),
                "db_schema_snapshot": db_snapshot,
                "input_personas_compact": personas_compact,
                "validation": vmeta,
            }
            (out_meta / f"{source_id}.json").write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        return "ok"

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_job, p) for p in stage1_files]
        for f in tqdm(cf.as_completed(futures), total=len(futures), desc=f"Stage2 ({args.split}/{args.schema_mode})"):
            _ = f.result()


if __name__ == "__main__":
    main()
