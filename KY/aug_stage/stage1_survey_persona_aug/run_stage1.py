#!/usr/bin/env python3
from __future__ import annotations

"""Stage 1 Survey: Conflict-aware Persona Retrieval (No LLM Augmentation).

This refactored Stage 1 pipeline:
1. Loads TravelPlanner test split
2. Uses improved MMR to retrieve conflict-aware persona groups from Stravl
3. Outputs raw Stravl personas (NO LLM re-generation)

Key differences from baseline:
- NO GPT-4.1 calls (academic rigor: direct survey data usage)
- Conflict-maximizing retrieval (compatible enemies strategy)
- Stravl personas are used as-is (preserving original survey responses)
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from core import (
    TripContext,
    create_trip_context,
    retrieve_conflicting_group,
    stable_retrieval_seed,
    VectorDimensions,
    decode_form_fields,
    get_ci,
    parse_int_list,
    load_tp_pppn_quantiles,
    stable_source_id,
    normalize_initial_info,
    load_travelplanner_split,
)

import csv
import random


# ---------------------------
# Stravl library loading
# ---------------------------

def _ensure_stravl_csv(data_dir: Path) -> Optional[Path]:
    """Return csv_path if exists; else try to download."""
    override = os.getenv("STRAVL_CSV_PATH")
    if override:
        p = Path(override).expanduser().resolve()
        return p if p.exists() else None

    p = (data_dir / "stravl" / "Stravl_Travel_Preference_Data.csv").resolve()
    if p.exists():
        return p

    # Auto-download from GitHub
    try:
        from urllib.request import urlopen
        import shutil

        url = "https://raw.githubusercontent.com/Stravl/Stravl-Data/main/Stravl_Travel_Preference_Data.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with urlopen(url, timeout=60) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(p)
        return p
    except Exception:
        return None


def load_stravl_library(csv_path: Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    """Load and decode Stravl preferences into a compact in-memory library."""
    lib: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            decoded = decode_form_fields(row)

            # Keep both decoded + coded buckets for scoring
            b_list = parse_int_list(get_ci(row, "FORM_B"))
            c_list = parse_int_list(get_ci(row, "FORM_C"))
            b_code = b_list[0] if b_list else None
            c_code = c_list[0] if c_list else None

            lib.append(
                {
                    "ref_id": f"stravl_{i}",
                    "source": "stravl",
                    "budget_code": b_code,
                    "season_code": c_code,
                    **decoded,
                }
            )
    return lib


# ---------------------------
# TravelPlanner -> Stravl mapping
# ---------------------------

def infer_season_code(dates: List[str]) -> Optional[int]:
    """Map month -> Stravl season code (0=Winter, 1=Spring, 2=Summer, 3=Fall)."""
    from datetime import datetime

    if not dates:
        return None

    def parse_ymd(s: str) -> Optional[datetime]:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

    dt = parse_ymd(dates[0]) or parse_ymd(dates[-1])
    if not dt:
        return None

    m = dt.month
    if m in (12, 1, 2):
        return 0  # Winter
    if m in (3, 4, 5):
        return 1  # Spring
    if m in (6, 7, 8):
        return 2  # Summer
    return 3  # Fall


def compute_pppn(budget_anchor: Optional[int], people: int, days: int) -> Optional[float]:
    """Compute per-person-per-night budget."""
    if budget_anchor is None or people <= 0 or days <= 0:
        return None
    try:
        return float(budget_anchor) / float(people * days)
    except Exception:
        return None


def auto_select_conflict_strategy(trip_context: TripContext) -> str:
    """Automatically select the most appropriate conflict strategy based on trip characteristics.

    Uses a scoring system to determine the best strategy for each trip.

    Strategy selection logic:
    - Budget-constrained trips → budget_war (focus on spending conflicts)
    - Short trips + small groups → pace_war (focus on activity speed conflicts)
    - Large groups → taste_war (focus on interest/experience conflicts)
    - Balanced trips → adaptive (dynamic weighting based on context)

    Args:
        trip_context: Trip context object

    Returns:
        Strategy name: 'budget_war', 'pace_war', 'taste_war', or 'adaptive'
    """
    scores = {
        "budget_war": 0,
        "pace_war": 0,
        "taste_war": 0,
        "adaptive": 1,  # Base score for adaptive
    }

    # Budget-constrained trips strongly prefer budget_war
    if trip_context.is_budget_constrained:
        scores["budget_war"] += 10

    # Short trips prefer pace_war, but not if large group
    if trip_context.is_short_trip:
        scores["pace_war"] += 5
        if not trip_context.is_large_group:
            scores["pace_war"] += 3

    # Large groups strongly prefer taste_war
    if trip_context.is_large_group:
        scores["taste_war"] += 8
        scores["pace_war"] -= 2  # Reduce pace_war for large groups

    # Multi-city trips benefit from adaptive
    if trip_context.is_multi_city:
        scores["adaptive"] += 4

    # Medium-length trips (4-7 days) prefer adaptive
    if not trip_context.is_short_trip and trip_context.days <= 7:
        scores["adaptive"] += 3

    # Select strategy with highest score
    return max(scores, key=scores.get)


def map_pppn_to_budget_code(
    pppn: Optional[float], tp_quantiles: Optional[Dict[str, float]]
) -> Optional[int]:
    """Map TravelPlanner pppn -> Stravl budget code (0..3) using quantiles."""
    if pppn is None:
        return None

    if tp_quantiles and all(k in tp_quantiles for k in ("q25", "q50", "q75")):
        q25, q50, q75 = (
            float(tp_quantiles["q25"]),
            float(tp_quantiles["q50"]),
            float(tp_quantiles["q75"]),
        )
        if pppn <= q25:
            return 0
        if pppn <= q50:
            return 1
        if pppn <= q75:
            return 2
        return 3

    # Fallback: literal Stravl bins
    if pppn < 50:
        return 0
    if pppn < 100:
        return 1
    if pppn < 250:
        return 2
    return 3


# ---------------------------
# Single record processing
# ---------------------------

def process_single_record(
    r: Dict[str, Any],
    *,
    stravl_lib: List[Dict[str, Any]],
    people_choices: List[int],
    tp_quantiles: Optional[Dict[str, float]],
    data_out_dir: Path,
    meta_out_dir: Optional[Path],
    args: Any,
    rng: random.Random,
) -> Optional[str]:
    """Process a single TravelPlanner record.

    Returns:
        source_id if processed, None if skipped
    """
    sid = stable_source_id(r)
    data_path = data_out_dir / f"{sid}.json"
    meta_path = meta_out_dir / f"{sid}.json" if meta_out_dir else None

    if (not args.no_resume) and data_path.exists():
        if not args.write_meta or (meta_path and meta_path.exists()):
            return None  # Skip existing

    # --- Normalize initial_info (with people upsampling) ---
    initial_info, norm_meta = normalize_initial_info(r, people_choices=people_choices, rng=rng)

    people = initial_info["people_number"]
    days = initial_info["days"]
    budget_anchor = initial_info.get("budget_anchor")

    dates = initial_info.get("date") or []
    if isinstance(dates, str):
        dates = [dates]
    season_code = infer_season_code(dates)
    pppn = compute_pppn(budget_anchor, people, days)
    mapped_budget_code = map_pppn_to_budget_code(pppn, tp_quantiles)

    trip_context = create_trip_context(
        initial_info,
        pppn=pppn,
        season_code=season_code,
        mapped_budget_code=mapped_budget_code,
    )

    # --- Auto-select conflict strategy based on trip characteristics ---
    # If user specified a strategy via CLI, use it; otherwise auto-select
    if args.conflict_strategy == "auto":
        selected_strategy = auto_select_conflict_strategy(trip_context)
    else:
        selected_strategy = args.conflict_strategy

    # --- Retrieve personas ---
    k_base = args.k_personas if args.k_personas > 0 else people
    k_expanded = k_base * args.k_multiplier  # For 3-stage pipeline: retrieve k×10 personas
    retrieval_seed = stable_retrieval_seed(trip_context, base_seed=args.seed)

    selected_personas = retrieve_conflicting_group(
        stravl_lib,
        trip_context,
        k=k_expanded,  # Retrieve expanded set for Stage 1.5
        lambda_param=args.lambda_param,
        conflict_strategy=selected_strategy,
        prefilter_size=args.prefilter,
        seed=retrieval_seed,
    )

    # --- Output format ---
    # Keep it simple: just initial_info + raw Stravl personas
    output = {
        "source_id": sid,
        "initial_info": initial_info,
        "personas": selected_personas,  # Raw Stravl data (no LLM augmentation)
        "target_final_count": k_base,  # Final target count for Stage 1.7 (e.g., 2 or 4)
    }

    data_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Meta (optional) ---
    if args.write_meta and meta_path:
        # Calculate actual prefilter size used
        if args.prefilter == 0:
            actual_prefilter = k_expanded * 10
        elif args.prefilter == -1:
            actual_prefilter = 0
        else:
            actual_prefilter = args.prefilter

        meta = {
            "raw_row_keys": list(r.keys()),
            "normalization": norm_meta,
            "trip_context": {
                "pppn": pppn,
                "season_code": season_code,
                "mapped_budget_code": mapped_budget_code,
                "is_short_trip": trip_context.is_short_trip,
                "is_budget_constrained": trip_context.is_budget_constrained,
                "is_large_group": trip_context.is_large_group,
                "is_multi_city": trip_context.is_multi_city,
            },
            "retrieval": {
                "k_base": k_base,  # Target final count (e.g., 2 or 4)
                "k_multiplier": args.k_multiplier,
                "k_expanded": k_expanded,  # Actual retrieved count (e.g., 20 or 40)
                "lambda": args.lambda_param,
                "conflict_strategy_requested": args.conflict_strategy,
                "conflict_strategy_selected": selected_strategy,
                "prefilter_size_requested": args.prefilter,
                "prefilter_size_actual": actual_prefilter,
                "seed": retrieval_seed,
                "stravl_lib_size": len(stravl_lib),
                "selected_ref_ids": [p.get("ref_id") for p in selected_personas],
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return sid


# ---------------------------
# Main pipeline
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1 Survey: Conflict-aware persona retrieval (no LLM)"
    )
    ap.add_argument("--split", default="test", help="TravelPlanner split (test recommended)")
    ap.add_argument("--max_records", type=int, default=0, help="0 = all")
    ap.add_argument("--people_choices", default="2,3,4,5,6,7,8", help="Upsample choices when people_number < 2")
    ap.add_argument("--seed", type=int, default=42)

    # Retrieval parameters
    ap.add_argument("--k_personas", type=int, default=0, help="0 = use people_number from data")
    ap.add_argument("--k_multiplier", type=int, default=10, help="Retrieve k * multiplier personas for Stage 1.5")
    ap.add_argument("--lambda_param", type=float, default=0.6, help="MMR lambda (relevance vs conflict)")
    ap.add_argument(
        "--conflict_strategy",
        default="auto",
        choices=["auto", "adaptive", "budget_war", "pace_war", "taste_war"],
        help="Conflict weight strategy ('auto' = auto-select based on trip characteristics)",
    )
    ap.add_argument("--prefilter", type=int, default=0, help="Pre-filter pool size (0 = auto: k_expanded*10, -1 = no filter)")

    # Data paths
    ap.add_argument("--data_dir", default="data", help="Cache directory")
    ap.add_argument("--out_dir", default="outputs/stage1", help="Output directory")
    ap.add_argument("--no_resume", action="store_true", help="Do not skip existing outputs")
    ap.add_argument("--stravl_max_rows", type=int, default=0, help="0 = all rows")

    # Meta logging
    ap.add_argument("--write_meta", action="store_true", help="Write meta/debug info")

    # Performance
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")

    args = ap.parse_args()

    if args.split != "test":
        raise SystemExit("This refactor is for TravelPlanner `test` only. Use --split test.")

    load_dotenv()

    here = Path(__file__).resolve().parent
    data_dir = (here / args.data_dir).resolve()
    out_dir = (here / args.out_dir / args.split).resolve()
    data_out_dir = out_dir / "data"
    meta_out_dir = out_dir / "meta"
    data_out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_meta:
        meta_out_dir.mkdir(parents=True, exist_ok=True)

    # Load TravelPlanner
    hf_cache_dir = data_dir / "hf_cache"
    split_cache = data_dir / "travelplanner_min" / f"{args.split}.jsonl"

    rows, _ = load_travelplanner_split(
        args.split,
        cache_min_path=split_cache,
        cache_db_summary_path=None,
        hf_cache_dir=hf_cache_dir,
        keep_reference_information=False,
    )

    if args.max_records and args.max_records > 0:
        rows = rows[: args.max_records]

    # Parse people_choices for upsampling
    people_choices = [int(x) for x in args.people_choices.split(",") if x.strip()]
    rng = random.Random(args.seed)

    # Load TravelPlanner pppn quantiles
    tp_quantiles = load_tp_pppn_quantiles(here / "artifacts")

    # Load Stravl library
    stravl_csv = _ensure_stravl_csv(data_dir)
    if stravl_csv is None:
        raise SystemExit("Stravl CSV not found. Set STRAVL_CSV_PATH or download manually.")

    stravl_lib = load_stravl_library(stravl_csv, max_rows=args.stravl_max_rows)
    print(f"Loaded {len(stravl_lib)} Stravl personas from {stravl_csv}")

    # Process each TravelPlanner record (with optional threading)
    if args.workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_single_record,
                    r,
                    stravl_lib=stravl_lib,
                    people_choices=people_choices,
                    tp_quantiles=tp_quantiles,
                    data_out_dir=data_out_dir,
                    meta_out_dir=meta_out_dir if args.write_meta else None,
                    args=args,
                    rng=rng,
                ): r
                for r in rows
            }

            # Show progress bar
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Stage1 Survey ({args.split})"
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing record: {e}")
    else:
        # Sequential processing (single worker)
        for r in tqdm(rows, desc=f"Stage1 Survey ({args.split})"):
            try:
                process_single_record(
                    r,
                    stravl_lib=stravl_lib,
                    people_choices=people_choices,
                    tp_quantiles=tp_quantiles,
                    data_out_dir=data_out_dir,
                    meta_out_dir=meta_out_dir if args.write_meta else None,
                    args=args,
                    rng=rng,
                )
            except Exception as e:
                print(f"Error processing record: {e}")

    print(f"✅ Stage 1 Survey complete. Outputs in {data_out_dir}")


if __name__ == "__main__":
    main()
