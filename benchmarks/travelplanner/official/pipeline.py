"""Helpers that wrap the official TravelPlanner parsing/evaluation pipeline."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from datasets import load_dataset

from benchmarks.travelplanner.schemas import Prediction

from .postprocess.openai_request import (
    build_plan_format_conversion_prompt,
    prompt_chatgpt,
)


DEFAULT_MODE = "two-stage"
DEFAULT_STRATEGY = "direct"


def _model_suffix(mode: str = DEFAULT_MODE, strategy: str = DEFAULT_STRATEGY) -> str:
    if mode == "two-stage":
        return ""
    return f"_{strategy}" if mode == "sole-planning" else ""


def write_generated_plan_files(
    output_dir: Path,
    set_type: str,
    model_name: str,
    predictions: Iterable[Prediction],
    mode: str = DEFAULT_MODE,
    strategy: str = DEFAULT_STRATEGY,
) -> None:
    """Mirror the official agent output directory layout."""

    suffix = _model_suffix(mode, strategy)
    target_dir = output_dir / set_type
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, prediction in enumerate(predictions, start=1):
        json_path = target_dir / f"generated_plan_{idx}.json"
        parsed_plan = None
        raw = prediction.prediction
        if isinstance(raw, str):
            try:
                parsed_candidate = json.loads(raw)
                if isinstance(parsed_candidate, list):
                    parsed_plan = parsed_candidate
            except json.JSONDecodeError:
                parsed_plan = None
        payload = [
            {
                f"{model_name}{suffix}_{mode}_results": raw,
                f"{model_name}{suffix}_{mode}_results_logs": None,
                f"{model_name}{suffix}_{mode}_action_logs": [],
            }
        ]
        if parsed_plan is not None:
            payload[-1][f"{model_name}{suffix}_{mode}_parsed_results"] = parsed_plan
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")


def run_official_parsing(
    output_dir: Path,
    tmp_dir: Path,
    set_type: str,
    model_name: str,
    mode: str = DEFAULT_MODE,
    strategy: str = DEFAULT_STRATEGY,
    parser_model: str = "gpt-4-1106-preview",
    total_examples: Optional[int] = None,
    worker_count: int = 1,
) -> None:
    """Invoke the official GPT-based parsing pipeline."""

    suffix = _model_suffix(mode, strategy)
    prompts = build_plan_format_conversion_prompt(
        directory=str(output_dir),
        set_type=set_type,
        model_name=model_name,
        strategy=strategy,
        mode=mode,
        total_examples=total_examples,
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_txt = tmp_dir / f"{set_type}_{model_name}{suffix}_{mode}.txt"

    results: List[str] = [""] * len(prompts)
    total_cost = 0.0
    worker_count = max(1, min(worker_count, len(prompts))) if prompts else 1

    def _parse_single(idx: int, prompt: str):
        if not prompt:
            return idx, str(idx), 0.0
        per_prompt_tmp = tmp_dir / f"{set_type}_{model_name}{suffix}_{mode}_{idx}.txt"
        response, _, price = prompt_chatgpt(
            "You are a helpful assistant.",
            user_input=prompt,
            temperature=0,
            save_path=str(per_prompt_tmp),
            index=idx,
            model_name=parser_model,
        )
        return idx, response, price

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_parse_single, idx, prompt): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="official-parser",
            unit="plan",
        ):
            idx, response, price = future.result()
            if idx < len(results):
                results[idx] = response
            total_cost += price

    tmp_txt.write_text("\n".join(results), encoding="utf-8")

    def _try_parse_plan(text: str) -> Optional[List[dict]]:
        candidate = text.strip()
        if not candidate:
            return None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                return None
        return parsed if isinstance(parsed, list) else None

    # Update generated_plan files with parsed results
    target_dir = output_dir / set_type
    if total_examples is None:
        dataset = load_dataset("osunlp/TravelPlanner", set_type)[set_type]
        total = len(dataset)
    else:
        total = total_examples

    for idx in range(1, total + 1):
        json_path = target_dir / f"generated_plan_{idx}.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        raw = results[idx - 1] if idx - 1 < len(results) else ""
        parsed_plan: Optional[List[dict]] = None
        content = raw or ""
        if content and "```" in content:
            try:
                if "```json" in content:
                    snippet = content.split("```json", 1)[1].split("```", 1)[0]
                else:
                    snippet = content.split("```", 1)[1].split("```", 1)[0]
                parsed_plan = _try_parse_plan(snippet)
            except Exception:
                parsed_plan = None
        if parsed_plan is None and content:
            if "\t" in content:
                content = content.split("\t", 1)[1]
            parsed_plan = _try_parse_plan(content)
        payload[-1][f"{model_name}{suffix}_{mode}_parsed_results"] = parsed_plan
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")


def load_parsed_plans(
    output_dir: Path,
    set_type: str,
    model_name: str,
    mode: str = DEFAULT_MODE,
    strategy: str = DEFAULT_STRATEGY,
    total_examples: Optional[int] = None,
) -> List[Optional[List[dict]]]:
    suffix = _model_suffix(mode, strategy)
    target_dir = output_dir / set_type
    plans: List[Optional[List[dict]]] = []

    if total_examples is None:
        dataset = load_dataset("osunlp/TravelPlanner", set_type)[set_type]
        total = len(dataset)
    else:
        total = total_examples

    for idx in range(1, total + 1):
        json_path = target_dir / f"generated_plan_{idx}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing generated_plan_{idx}.json in {target_dir}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        plan = payload[-1].get(f"{model_name}{suffix}_{mode}_parsed_results")
        plans.append(plan)
    return plans
