#!/usr/bin/env python3
from __future__ import annotations

"""Stage 1.5 + 1.7 Iterative Pipeline.

This pipeline iteratively processes personas until a solvable combination is found:
1. Load Stage 1 output (k×10 personas)
2. Round 1: Process K personas → Alpha survey → Check solvability
3. If not solvable: Round 2: Process K more personas → Check again
4. Repeat up to 10 rounds or until solvable combination found

This approach saves LLM costs by only processing personas as needed.

Cost savings:
- If solvable in round 1: ~90% cost reduction (process K instead of k×10)
- Average case: ~60-80% cost reduction
- Worst case (10 rounds): Same cost as batch approach

Usage:
    # Basic usage
    python run_stage1_5_1_7_iterative.py

    # Custom settings
    python run_stage1_5_1_7_iterative.py --personas_per_round 3 --max_rounds 10
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from stage1_5 import (
    build_alpha_survey_prompt,
    run_async_pipeline,
)
from stage1_7 import select_best_combination


def process_personas_round(
    personas: List[Dict[str, Any]],
    trip_context: Dict[str, Any],
    model: str,
    temperature: float,
    max_retries: int,
    max_concurrent: int,
) -> List[Dict[str, Any]]:
    """Process a batch of personas with alpha survey (async).

    Args:
        personas: List of personas to process
        trip_context: Trip context dict
        model: LLM model to use
        temperature: Sampling temperature
        max_retries: Max retry attempts
        max_concurrent: Max concurrent API calls

    Returns:
        List of enriched personas
    """
    # Build prompts
    prompt_data_list = []
    for persona in personas:
        persona_id = persona.get("ref_id", "unknown")
        prompt = build_alpha_survey_prompt(persona, trip_context)
        prompt_data_list.append({
            "persona_id": persona_id,
            "prompt": prompt,
        })

    # Run async pipeline
    results = run_async_pipeline(
        prompt_data_list,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        max_concurrent=max_concurrent,
    )

    # Enrich personas
    enriched_personas = []
    for persona, result in zip(personas, results):
        if result is not None:
            enriched = {
                **persona,
                "alpha_survey": result,
            }
            enriched_personas.append(enriched)

    return enriched_personas


def process_single_record_iterative(
    stage1_data: Dict[str, Any],
    out_dir_1_5: Path,
    out_dir_1_7: Path,
    meta_dir_1_7: Path,
    args: Any,
) -> Dict[str, Any]:
    """Process a single Stage 1 record with iterative refinement.

    Args:
        stage1_data: Stage 1 output JSON
        out_dir_1_5: Stage 1.5 output directory
        out_dir_1_7: Stage 1.7 output directory
        meta_dir_1_7: Stage 1.7 metadata directory
        args: CLI arguments

    Returns:
        Processing result dict
    """
    source_id = stage1_data.get("source_id")
    if not source_id:
        return {"status": "ERROR", "message": "Missing source_id"}

    # Check if already processed
    output_path_1_5 = out_dir_1_5 / f"{source_id}.json"
    output_path_1_7 = out_dir_1_7 / f"{source_id}.json"
    meta_path_1_7 = meta_dir_1_7 / f"{source_id}.json" if args.write_meta else None

    if (not args.no_resume) and output_path_1_7.exists():
        if not args.write_meta or (meta_path_1_7 and meta_path_1_7.exists()):
            return {"status": "SKIPPED", "message": "Already processed"}

    # Extract data
    initial_info = stage1_data.get("initial_info", {})
    all_personas = stage1_data.get("personas", [])
    target_final_count = stage1_data.get("target_final_count", 2)

    if not all_personas:
        return {"status": "ERROR", "message": "No personas found"}

    # Build trip context
    trip_context = {
        "people_number": initial_info.get("people_number", 2),
        "days": initial_info.get("days", 3),
        "budget_anchor": initial_info.get("budget_anchor"),
        "org": initial_info.get("org", "Unknown"),
        "dest": initial_info.get("dest", "Unknown"),
    }

    # Iterative processing
    enriched_personas = []
    personas_per_round = args.personas_per_round
    max_rounds = min(args.max_rounds, (len(all_personas) + personas_per_round - 1) // personas_per_round)

    print(f"\n{source_id}: Starting iterative processing ({len(all_personas)} personas, max {max_rounds} rounds)")

    for round_idx in range(max_rounds):
        # Get next batch of personas
        start_idx = round_idx * personas_per_round
        end_idx = min(start_idx + personas_per_round, len(all_personas))
        current_batch = all_personas[start_idx:end_idx]

        if not current_batch:
            break

        print(f"  Round {round_idx + 1}/{max_rounds}: Processing {len(current_batch)} personas ({start_idx}-{end_idx-1})...")

        # Process current batch
        new_enriched = process_personas_round(
            current_batch,
            trip_context,
            args.model,
            args.temperature,
            args.max_retries,
            args.max_concurrent,
        )

        enriched_personas.extend(new_enriched)

        # Check if we have enough personas for target
        if len(enriched_personas) < target_final_count:
            print(f"    Not enough personas yet ({len(enriched_personas)} < {target_final_count}), continuing...")
            continue

        # Try to find solvable combination
        result = select_best_combination(
            enriched_personas,
            target_count=target_final_count,
            max_combinations=args.max_combinations,
        )

        if result is not None:
            best_personas, analysis = result
            print(f"    ✅ Found solvable combination!")
            print(f"       Personas processed: {len(enriched_personas)}/{len(all_personas)}")
            print(f"       Soft conflicts: {analysis['conflict_count']}")
            print(f"       Score: {analysis['score']:.2f}")

            # Save Stage 1.5 output (all enriched personas so far)
            output_1_5 = {
                "source_id": source_id,
                "initial_info": initial_info,
                "personas": enriched_personas,
                "target_final_count": target_final_count,
                "stage1_persona_count": len(all_personas),
                "stage1_5_persona_count": len(enriched_personas),
                "rounds_processed": round_idx + 1,
            }
            output_path_1_5.write_text(json.dumps(output_1_5, ensure_ascii=False, indent=2), encoding="utf-8")

            # Save Stage 1.7 output (final selected personas)
            output_1_7 = {
                "source_id": source_id,
                "initial_info": initial_info,
                "personas": best_personas,
            }
            output_path_1_7.write_text(json.dumps(output_1_7, ensure_ascii=False, indent=2), encoding="utf-8")

            # Save metadata
            if args.write_meta and meta_path_1_7:
                meta = {
                    "source_id": source_id,
                    "stage1_persona_count": len(all_personas),
                    "stage1_5_persona_count": len(enriched_personas),
                    "rounds_processed": round_idx + 1,
                    "personas_per_round": personas_per_round,
                    "target_final_count": target_final_count,
                    "final_persona_count": len(best_personas),
                    "conflict_analysis": analysis,
                    "selected_ref_ids": [p.get("ref_id") for p in best_personas],
                    "cost_savings_percent": int((1 - len(enriched_personas) / len(all_personas)) * 100),
                }
                meta_path_1_7.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            return {
                "status": "SUCCESS",
                "message": f"Solvable in round {round_idx + 1}",
                "rounds": round_idx + 1,
                "personas_processed": len(enriched_personas),
                "total_personas": len(all_personas),
                "cost_savings_percent": int((1 - len(enriched_personas) / len(all_personas)) * 100),
            }

        else:
            print(f"    No solvable combination yet, continuing...")

    # Reached max rounds without finding solvable combination
    print(f"  ⚠️  Max rounds reached without solvable combination")

    # Save what we have
    output_1_5 = {
        "source_id": source_id,
        "initial_info": initial_info,
        "personas": enriched_personas,
        "target_final_count": target_final_count,
        "stage1_persona_count": len(all_personas),
        "stage1_5_persona_count": len(enriched_personas),
        "rounds_processed": max_rounds,
        "solvable": False,
    }
    output_path_1_5.write_text(json.dumps(output_1_5, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "NO_SOLVABLE",
        "message": f"No solvable combination after {max_rounds} rounds",
        "rounds": max_rounds,
        "personas_processed": len(enriched_personas),
        "total_personas": len(all_personas),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1.5 + 1.7 Iterative Pipeline (Async API)"
    )
    ap.add_argument(
        "--stage1_dir",
        default="outputs/stage1/test/data",
        help="Stage 1 output directory",
    )
    ap.add_argument(
        "--out_dir_1_5",
        default="outputs/stage1_5_iterative/test/data",
        help="Stage 1.5 output directory",
    )
    ap.add_argument(
        "--out_dir_1_7",
        default="outputs/stage1_7_iterative/test/data",
        help="Stage 1.7 output directory",
    )
    ap.add_argument(
        "--meta_dir_1_7",
        default="outputs/stage1_7_iterative/test/meta",
        help="Stage 1.7 metadata directory",
    )
    ap.add_argument("--max_records", type=int, default=0, help="0 = all")
    ap.add_argument("--no_resume", action="store_true", help="Re-process existing outputs")
    ap.add_argument("--write_meta", action="store_true", help="Write metadata files")

    # LLM parameters
    ap.add_argument("--model", default="gpt-4.1", help="LLM model to use")
    ap.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    ap.add_argument("--max_retries", type=int, default=3, help="Max retry attempts per LLM call")

    # Async parameters
    ap.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Max concurrent async API calls (for rate limiting)",
    )

    # Iterative parameters
    ap.add_argument(
        "--personas_per_round",
        type=int,
        default=1,
        help="Number of personas to process per round (default: 1)",
    )
    ap.add_argument(
        "--max_rounds",
        type=int,
        default=10,
        help="Maximum number of rounds (default: 10)",
    )

    # Stage 1.7 parameters
    ap.add_argument(
        "--max_combinations",
        type=int,
        default=1000,
        help="Max combinations to evaluate per solvability check",
    )

    args = ap.parse_args()

    load_dotenv()

    here = Path(__file__).resolve().parent
    stage1_dir = (here / args.stage1_dir).resolve()
    out_dir_1_5 = (here / args.out_dir_1_5).resolve()
    out_dir_1_7 = (here / args.out_dir_1_7).resolve()
    meta_dir_1_7 = (here / args.meta_dir_1_7).resolve()

    out_dir_1_5.mkdir(parents=True, exist_ok=True)
    out_dir_1_7.mkdir(parents=True, exist_ok=True)
    if args.write_meta:
        meta_dir_1_7.mkdir(parents=True, exist_ok=True)

    # Load all Stage 1 outputs
    stage1_files = sorted(stage1_dir.glob("*.json"))

    if not stage1_files:
        raise SystemExit(f"No Stage 1 outputs found in {stage1_dir}")

    if args.max_records and args.max_records > 0:
        stage1_files = stage1_files[: args.max_records]

    print(f"Found {len(stage1_files)} Stage 1 outputs")
    print(f"Mode: Iterative Async API (process {args.personas_per_round} persona(s) per round, max {args.max_rounds} rounds)")
    print(f"Expected cost savings: 60-90% (process only until solvable)")
    print()

    # Process each record
    results = {"SUCCESS": 0, "SKIPPED": 0, "NO_SOLVABLE": 0, "ERROR": 0}
    total_cost_savings = []

    for stage1_file in tqdm(stage1_files, desc="Iterative Processing"):
        try:
            stage1_data = json.loads(stage1_file.read_text(encoding="utf-8"))
            result = process_single_record_iterative(
                stage1_data, out_dir_1_5, out_dir_1_7, meta_dir_1_7, args
            )

            results[result["status"]] += 1
            if "cost_savings_percent" in result:
                total_cost_savings.append(result["cost_savings_percent"])

        except Exception as e:
            results["ERROR"] += 1
            print(f"\nError processing {stage1_file.name}: {e}")

    print(f"\n✅ Iterative pipeline complete!")
    print(f"   Success: {results['SUCCESS']}")
    print(f"   Skipped: {results['SKIPPED']}")
    print(f"   No solvable: {results['NO_SOLVABLE']}")
    print(f"   Errors: {results['ERROR']}")

    if total_cost_savings:
        avg_savings = sum(total_cost_savings) / len(total_cost_savings)
        print(f"   Average cost savings: {avg_savings:.1f}%")

    print(f"\n   Stage 1.5 outputs: {out_dir_1_5}")
    print(f"   Stage 1.7 outputs: {out_dir_1_7}")


if __name__ == "__main__":
    main()
