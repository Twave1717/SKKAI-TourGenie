"""OpenAI Batch API support for cost reduction (50% discount).

This module handles batch job creation and retrieval for the alpha survey pipeline.
Batch API provides 50% cost discount but processes within 24 hours.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from tqdm import tqdm

from .structured_output import get_alpha_survey_schema


def create_batch_request_file(
    prompts: List[Dict[str, Any]],
    output_path: Path,
    model: str = "gpt-4.1",
    temperature: float = 0.1,
) -> Path:
    """Create JSONL batch request file for OpenAI Batch API.

    Args:
        prompts: List of prompt dicts with keys: persona_id, prompt, trip_context
        output_path: Path to save JSONL file
        model: LLM model to use
        temperature: Sampling temperature

    Returns:
        Path to created JSONL file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, prompt_data in enumerate(prompts):
            request = {
                "custom_id": f"{prompt_data['persona_id']}_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a travel preference analyst. Analyze the persona's profile and provide detailed preference scores across all travel categories.",
                        },
                        {"role": "user", "content": prompt_data["prompt"]},
                    ],
                    "temperature": temperature,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "alpha_survey",
                            "schema": get_alpha_survey_schema(),
                            "strict": True,
                        },
                    },
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"Created batch request file: {output_path}")
    print(f"Total requests: {len(prompts)}")
    return output_path


def submit_batch_job(
    batch_file_path: Path,
    description: str = "Alpha Survey Batch",
) -> str:
    """Submit batch job to OpenAI.

    Args:
        batch_file_path: Path to JSONL batch request file
        description: Job description

    Returns:
        Batch job ID
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    # Upload batch file
    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    print(f"Uploaded batch file: {batch_input_file.id}")

    # Create batch job
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )

    print(f"Created batch job: {batch_job.id}")
    print(f"Status: {batch_job.status}")
    print(f"Completion window: 24h")

    return batch_job.id


def check_batch_status(batch_id: str) -> Dict[str, Any]:
    """Check batch job status.

    Args:
        batch_id: Batch job ID

    Returns:
        Batch job status dict
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)
    batch_job = client.batches.retrieve(batch_id)

    return {
        "id": batch_job.id,
        "status": batch_job.status,
        "created_at": batch_job.created_at,
        "completed_at": batch_job.completed_at,
        "failed_at": batch_job.failed_at,
        "request_counts": {
            "total": batch_job.request_counts.total,
            "completed": batch_job.request_counts.completed,
            "failed": batch_job.request_counts.failed,
        },
        "output_file_id": batch_job.output_file_id,
        "error_file_id": batch_job.error_file_id,
    }


def wait_for_batch_completion(
    batch_id: str,
    check_interval: int = 60,
    max_wait_hours: int = 25,
) -> bool:
    """Wait for batch job to complete (with progress updates).

    Args:
        batch_id: Batch job ID
        check_interval: Seconds between status checks (default: 60s)
        max_wait_hours: Maximum hours to wait (default: 25h)

    Returns:
        True if completed successfully, False if failed or timeout
    """
    max_checks = (max_wait_hours * 3600) // check_interval

    print(f"Waiting for batch job {batch_id} to complete...")
    print(f"Checking every {check_interval}s (max {max_wait_hours}h)")

    with tqdm(total=100, desc="Batch progress", unit="%") as pbar:
        last_progress = 0

        for i in range(max_checks):
            status = check_batch_status(batch_id)

            if status["status"] == "completed":
                pbar.update(100 - last_progress)
                print(f"\n✅ Batch job completed!")
                print(f"Total: {status['request_counts']['total']}")
                print(f"Completed: {status['request_counts']['completed']}")
                print(f"Failed: {status['request_counts']['failed']}")
                return True

            elif status["status"] == "failed":
                print(f"\n❌ Batch job failed!")
                return False

            elif status["status"] == "cancelled":
                print(f"\n⚠️  Batch job cancelled!")
                return False

            # Update progress bar
            total = status["request_counts"]["total"]
            completed = status["request_counts"]["completed"]
            if total > 0:
                progress = int((completed / total) * 100)
                pbar.update(progress - last_progress)
                last_progress = progress

            time.sleep(check_interval)

    print(f"\n⏱️  Timeout: Batch job did not complete within {max_wait_hours}h")
    return False


def download_batch_results(
    batch_id: str,
    output_path: Path,
) -> Optional[Path]:
    """Download batch job results.

    Args:
        batch_id: Batch job ID
        output_path: Path to save results JSONL file

    Returns:
        Path to downloaded results, or None if failed
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    # Get batch job
    batch_job = client.batches.retrieve(batch_id)

    if batch_job.status != "completed":
        print(f"Batch job not completed yet. Status: {batch_job.status}")
        return None

    if not batch_job.output_file_id:
        print("No output file available")
        return None

    # Download output file
    file_response = client.files.content(batch_job.output_file_id)
    output_path.write_bytes(file_response.content)

    print(f"Downloaded batch results: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def parse_batch_results(
    results_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Parse batch results JSONL into persona_id -> response mapping.

    Args:
        results_path: Path to batch results JSONL file

    Returns:
        Dict mapping custom_id to parsed response
    """
    results = {}

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            result = json.loads(line)
            custom_id = result.get("custom_id")

            if result.get("response", {}).get("status_code") == 200:
                # Success case
                body = result["response"]["body"]
                content = body["choices"][0]["message"]["content"]

                try:
                    parsed = json.loads(content)
                    results[custom_id] = {
                        "success": True,
                        "data": parsed,
                    }
                except json.JSONDecodeError as e:
                    results[custom_id] = {
                        "success": False,
                        "error": f"JSON parse error: {e}",
                    }
            else:
                # Error case
                error = result.get("error", {})
                results[custom_id] = {
                    "success": False,
                    "error": error.get("message", "Unknown error"),
                }

    success_count = sum(1 for r in results.values() if r["success"])
    print(f"Parsed {len(results)} results ({success_count} successful)")

    return results


def run_batch_pipeline(
    prompts: List[Dict[str, Any]],
    batch_dir: Path,
    job_name: str = "alpha_survey",
    model: str = "gpt-4.1",
    temperature: float = 0.1,
    wait_for_completion: bool = True,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Run complete batch pipeline: create, submit, wait, download, parse.

    Args:
        prompts: List of prompt dicts with keys: persona_id, prompt
        batch_dir: Directory to store batch files
        job_name: Job name (for file naming)
        model: LLM model to use
        temperature: Sampling temperature
        wait_for_completion: If True, wait for job to complete and download results

    Returns:
        Dict mapping custom_id to response, or None if not waiting
    """
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create batch request file
    request_file = batch_dir / f"{job_name}_request.jsonl"
    create_batch_request_file(prompts, request_file, model, temperature)

    # Submit batch job
    batch_id = submit_batch_job(request_file, description=f"Alpha Survey: {job_name}")

    # Save batch ID for reference
    batch_id_file = batch_dir / f"{job_name}_batch_id.txt"
    batch_id_file.write_text(batch_id)
    print(f"Batch ID saved to: {batch_id_file}")

    if not wait_for_completion:
        print("\n⏸️  Batch job submitted. Use --resume_batch_id to retrieve results later.")
        return None

    # Wait for completion
    success = wait_for_batch_completion(batch_id)
    if not success:
        print("Batch job did not complete successfully")
        return None

    # Download results
    results_file = batch_dir / f"{job_name}_results.jsonl"
    download_path = download_batch_results(batch_id, results_file)
    if not download_path:
        return None

    # Parse results
    results = parse_batch_results(download_path)

    return results
