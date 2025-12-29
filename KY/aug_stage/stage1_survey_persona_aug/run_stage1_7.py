#!/usr/bin/env python3
from __future__ import annotations

"""Stage 1.7: Final Persona Selection via Solvable Conflict.

This pipeline:
1. Loads Stage 1.5 outputs (k×10 personas with alpha values)
2. Selects the best N-persona combination that forms a "solvable conflict"
3. Saves final N personas for downstream planning

Solvable Conflict Definition:
- Hard constraints (α≥9) do NOT conflict
- Soft constraints (4≤α<9) DO conflict
- At least 2 dimensions have conflicts
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from stage1_7 import select_best_combination


def process_single_record(
    stage1_5_data: Dict[str, Any],
    stage1_7_out_dir: Path,
    meta_out_dir: Path,
    args: Any,
) -> str:
    """Process a single Stage 1.5 record to select final personas.

    Args:
        stage1_5_data: Stage 1.5 output JSON
        stage1_7_out_dir: Stage 1.7 output directory
        meta_out_dir: Metadata output directory
        args: CLI arguments

    Returns:
        Result status message
    """
    source_id = stage1_5_data.get("source_id")
    if not source_id:
        return "ERROR: Missing source_id"

    data_path = stage1_7_out_dir / f"{source_id}.json"
    meta_path = meta_out_dir / f"{source_id}.json" if args.write_meta else None

    # Skip if already exists
    if (not args.no_resume) and data_path.exists():
        if not args.write_meta or (meta_path and meta_path.exists()):
            return "SKIPPED"

    # Extract data
    initial_info = stage1_5_data.get("initial_info", {})
    personas = stage1_5_data.get("personas", [])
    target_final_count = stage1_5_data.get("target_final_count", 2)

    if not personas:
        return "ERROR: No personas found"

    if len(personas) < target_final_count:
        return f"ERROR: Not enough personas ({len(personas)} < {target_final_count})"

    # Select best combination
    result = select_best_combination(
        personas,
        target_count=target_final_count,
        max_combinations=args.max_combinations,
    )

    if result is None:
        return "ERROR: No solvable combination found"

    best_personas, analysis = result

    # Save output
    output = {
        "source_id": source_id,
        "initial_info": initial_info,
        "personas": best_personas,
    }

    data_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save metadata
    if args.write_meta and meta_path:
        meta = {
            "source_id": source_id,
            "stage1_5_persona_count": len(personas),
            "target_final_count": target_final_count,
            "final_persona_count": len(best_personas),
            "conflict_analysis": analysis,
            "selected_ref_ids": [p.get("ref_id") for p in best_personas],
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return f"OK: {len(best_personas)} personas, {analysis['conflict_count']} conflicts, score={analysis['score']:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1.7: Final persona selection via solvable conflict"
    )
    ap.add_argument(
        "--stage1_5_dir",
        default="outputs/stage1_5/test/data",
        help="Stage 1.5 output directory",
    )
    ap.add_argument(
        "--out_dir", default="outputs/stage1_7/test", help="Stage 1.7 output directory"
    )
    ap.add_argument("--max_records", type=int, default=0, help="0 = all")
    ap.add_argument("--no_resume", action="store_true", help="Re-process existing outputs")

    # Selection parameters
    ap.add_argument(
        "--max_combinations",
        type=int,
        default=1000,
        help="Max combinations to evaluate per record",
    )

    # Meta logging
    ap.add_argument("--write_meta", action="store_true", help="Write meta/debug info")

    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    stage1_5_dir = (here / args.stage1_5_dir).resolve()
    out_dir = (here / args.out_dir).resolve()
    data_out_dir = out_dir / "data"
    meta_out_dir = out_dir / "meta"

    data_out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_meta:
        meta_out_dir.mkdir(parents=True, exist_ok=True)

    # Load all Stage 1.5 outputs
    stage1_5_files = sorted(stage1_5_dir.glob("*.json"))

    if not stage1_5_files:
        raise SystemExit(f"No Stage 1.5 outputs found in {stage1_5_dir}")

    if args.max_records and args.max_records > 0:
        stage1_5_files = stage1_5_files[: args.max_records]

    print(f"Found {len(stage1_5_files)} Stage 1.5 outputs")

    # Process each record
    results = {"OK": 0, "SKIPPED": 0, "ERROR": 0}

    for stage1_5_file in tqdm(stage1_5_files, desc="Stage 1.7 Final Selection"):
        try:
            stage1_5_data = json.loads(stage1_5_file.read_text(encoding="utf-8"))
            status = process_single_record(
                stage1_5_data, data_out_dir, meta_out_dir, args
            )

            # Categorize result
            if status.startswith("OK"):
                results["OK"] += 1
            elif status.startswith("SKIPPED"):
                results["SKIPPED"] += 1
            else:
                results["ERROR"] += 1
                print(f"\n{stage1_5_file.name}: {status}")

        except Exception as e:
            results["ERROR"] += 1
            print(f"\nError processing {stage1_5_file.name}: {e}")

    print(f"\n✅ Stage 1.7 complete.")
    print(f"   Processed: {results['OK']}")
    print(f"   Skipped: {results['SKIPPED']}")
    print(f"   Errors: {results['ERROR']}")
    print(f"   Outputs in {data_out_dir}")


if __name__ == "__main__":
    main()
