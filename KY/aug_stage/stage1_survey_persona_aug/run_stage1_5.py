#!/usr/bin/env python3
from __future__ import annotations

"""Stage 1.5: LLM-based Alpha Survey (Batch API Mode).

This pipeline uses OpenAI Batch API for 63% cost savings:
- 50% discount from Batch API
- 26% reduction from reason축약 (10-15 words)
- Total: $53.29 for 9,200 personas (vs $143.37 baseline)

Usage:
    # Submit batch job and wait for completion
    python run_stage1_5.py

    # Submit and check later
    python run_stage1_5.py --no_wait
    python run_stage1_5.py --resume_batch_id batch_abc123xyz
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

from stage1_5 import (
    build_alpha_survey_prompt,
    run_batch_pipeline,
    check_batch_status,
    download_batch_results,
    parse_batch_results,
)


def collect_prompts_from_stage1(
    stage1_data: Dict[str, Any],
    out_dir: Path,
    no_resume: bool,
) -> List[Dict[str, Any]]:
    """Collect prompts from a single Stage 1 record.

    Args:
        stage1_data: Stage 1 output JSON
        out_dir: Output directory for Stage 1.5
        no_resume: If True, re-process existing outputs

    Returns:
        List of prompt data dicts
    """
    prompts = []

    source_id = stage1_data.get("source_id")
    if not source_id:
        return prompts

    output_path = out_dir / f"{source_id}.json"

    # Skip if already exists and not forcing re-run
    if (not no_resume) and output_path.exists():
        return prompts

    # Extract data
    initial_info = stage1_data.get("initial_info", {})
    personas = stage1_data.get("personas", [])

    if not personas:
        return prompts

    # Build trip context
    trip_context = {
        "people_number": initial_info.get("people_number", 2),
        "days": initial_info.get("days", 3),
        "budget_anchor": initial_info.get("budget_anchor"),
        "org": initial_info.get("org", "Unknown"),
        "dest": initial_info.get("dest", "Unknown"),
    }

    # Collect prompts for batch processing
    for persona in personas:
        persona_id = persona.get("ref_id", "unknown")
        prompt = build_alpha_survey_prompt(persona, trip_context)

        prompts.append({
            "persona_id": persona_id,
            "prompt": prompt,
            "source_id": source_id,
            "persona": persona,
            "trip_context": trip_context,
            "initial_info": initial_info,
            "target_final_count": stage1_data.get("target_final_count", 2),
            "stage1_persona_count": len(personas),
        })

    return prompts


def save_outputs_from_batch_results(
    batch_results: Dict[str, Dict[str, Any]],
    all_prompts: List[Dict[str, Any]],
    out_dir: Path,
) -> int:
    """Save enriched personas from batch results.

    Args:
        batch_results: Parsed batch results (custom_id -> response)
        all_prompts: List of all prompt data
        out_dir: Output directory

    Returns:
        Number of records processed
    """
    # Group by source_id
    source_outputs = {}

    for custom_id, result_data in batch_results.items():
        if not result_data["success"]:
            print(f"Warning: Failed result for {custom_id}: {result_data['error']}")
            continue

        # Find matching prompt data
        persona_id = custom_id.rsplit("_", 1)[0]  # Remove index suffix
        matching_prompt = next(
            (p for p in all_prompts if p["persona_id"] == persona_id),
            None
        )

        if not matching_prompt:
            print(f"Warning: No matching prompt for {custom_id}")
            continue

        # Enrich persona
        enriched = {
            **matching_prompt["persona"],
            "alpha_survey": result_data["data"],
        }

        source_id = matching_prompt["source_id"]
        if source_id not in source_outputs:
            source_outputs[source_id] = {
                "personas": [],
                "initial_info": matching_prompt["initial_info"],
                "target_final_count": matching_prompt["target_final_count"],
                "stage1_persona_count": matching_prompt["stage1_persona_count"],
            }

        source_outputs[source_id]["personas"].append(enriched)

    # Save outputs per source
    processed_count = 0
    for source_id, data in source_outputs.items():
        output = {
            "source_id": source_id,
            "initial_info": data["initial_info"],
            "personas": data["personas"],
            "target_final_count": data["target_final_count"],
            "stage1_persona_count": data["stage1_persona_count"],
            "stage1_5_persona_count": len(data["personas"]),
        }

        output_path = out_dir / f"{source_id}.json"
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        processed_count += 1

    return processed_count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1.5: LLM-based alpha survey (Batch API - 63% cost savings)"
    )
    ap.add_argument(
        "--stage1_dir",
        default="outputs/stage1/test/data",
        help="Stage 1 output directory",
    )
    ap.add_argument(
        "--out_dir", default="outputs/stage1_5/test/data", help="Stage 1.5 output directory"
    )
    ap.add_argument("--max_records", type=int, default=0, help="0 = all")
    ap.add_argument("--no_resume", action="store_true", help="Re-process existing outputs")

    # LLM parameters
    ap.add_argument(
        "--model", default="gpt-4.1", help="LLM model to use"
    )
    ap.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    ap.add_argument(
        "--max_retries", type=int, default=3, help="Max retry attempts per LLM call"
    )

    # Batch API settings
    ap.add_argument(
        "--batch_dir",
        default="outputs/stage1_5_batch",
        help="Directory for batch API files",
    )
    ap.add_argument(
        "--resume_batch_id",
        type=str,
        help="Resume from existing batch job ID",
    )
    ap.add_argument(
        "--no_wait",
        action="store_true",
        help="Submit batch job and exit (don't wait for completion)",
    )

    args = ap.parse_args()

    load_dotenv()

    here = Path(__file__).resolve().parent
    stage1_dir = (here / args.stage1_dir).resolve()
    out_dir = (here / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_dir = (here / args.batch_dir).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Load all Stage 1 outputs
    stage1_files = sorted(stage1_dir.glob("*.json"))

    if not stage1_files:
        raise SystemExit(f"No Stage 1 outputs found in {stage1_dir}")

    if args.max_records and args.max_records > 0:
        stage1_files = stage1_files[: args.max_records]

    print(f"Found {len(stage1_files)} Stage 1 outputs")
    print("💰 Mode: Batch API (50% discount) + Reason축약 (26% reduction) = 63% total savings")
    print()

    # Resume existing batch job
    if args.resume_batch_id:
        print(f"Resuming batch job: {args.resume_batch_id}")

        # Check status
        status = check_batch_status(args.resume_batch_id)
        print(f"Status: {status['status']}")
        print(f"Completed: {status['request_counts']['completed']}/{status['request_counts']['total']}")

        if status['status'] != 'completed':
            print("Batch job not completed yet. Waiting...")
            from stage1_5.batch_api import wait_for_batch_completion
            success = wait_for_batch_completion(args.resume_batch_id)
            if not success:
                raise SystemExit("Batch job did not complete successfully")

        # Download results
        results_file = batch_dir / f"resumed_{args.resume_batch_id}_results.jsonl"
        download_batch_results(args.resume_batch_id, results_file)

        # Parse results
        batch_results = parse_batch_results(results_file)

        # Reconstruct all_prompts from stage1 files
        print("Loading Stage 1 data to match batch results...")
        all_prompts = []
        for stage1_file in tqdm(stage1_files, desc="Loading prompts"):
            try:
                stage1_data = json.loads(stage1_file.read_text(encoding="utf-8"))
                prompts = collect_prompts_from_stage1(stage1_data, out_dir, args.no_resume)
                all_prompts.extend(prompts)
            except Exception as e:
                print(f"Error loading {stage1_file.name}: {e}")

    # Submit new batch job
    else:
        # Collect all prompts for batch processing
        print("Collecting prompts for batch processing...")
        all_prompts = []

        for stage1_file in tqdm(stage1_files, desc="Collecting prompts"):
            try:
                stage1_data = json.loads(stage1_file.read_text(encoding="utf-8"))
                prompts = collect_prompts_from_stage1(stage1_data, out_dir, args.no_resume)
                all_prompts.extend(prompts)
            except Exception as e:
                print(f"Error collecting prompts from {stage1_file.name}: {e}")

        if not all_prompts:
            raise SystemExit("No prompts collected for batch processing")

        print(f"Collected {len(all_prompts)} prompts from {len(stage1_files)} records")

        # Run batch pipeline
        batch_results = run_batch_pipeline(
            prompts=all_prompts,
            batch_dir=batch_dir,
            job_name="alpha_survey",
            model=args.model,
            temperature=args.temperature,
            wait_for_completion=(not args.no_wait),
        )

        if args.no_wait:
            batch_id_file = batch_dir / "alpha_survey_batch_id.txt"
            print(f"\n✅ Batch job submitted!")
            print(f"Batch ID saved to: {batch_id_file}")
            print(f"\nTo resume later:")
            print(f"  python run_stage1_5.py --resume_batch_id $(cat {batch_id_file})")
            return

    # Process batch results and save outputs
    print("\nProcessing batch results...")
    processed_count = save_outputs_from_batch_results(batch_results, all_prompts, out_dir)

    print(f"\n✅ Stage 1.5 complete!")
    print(f"Processed: {processed_count} records")
    print(f"Outputs: {out_dir}")
    print(f"💰 Cost savings: 63% (Batch API 50% + Reason축약 26%)")


if __name__ == "__main__":
    main()
