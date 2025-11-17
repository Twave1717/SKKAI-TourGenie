#!/usr/bin/env python3
"""
Convert TripCraft natural-language plan outputs into the evaluation JSONL format.

This script reads the generated plan files (e.g. results/tripcraft/<run>/<day>/gpt4o_orig_generated_plan_*.json),
grabs the corresponding metadata rows from the original CSVs, and re-queries OpenAI to turn each natural-language
plan back into the structured JSON that TripCraft's evaluators expect.

Usage:
    python scripts/tripcraft_postprocess.py \
        --input-root results/tripcraft/gpt41mini_test_mini \
        --output-jsonl results/tripcraft/gpt41mini_test_mini.jsonl \
        --csv-3day benchmarks/TripCraft/tripcraft/tripcraft_3day.csv \
        --csv-5day benchmarks/TripCraft/tripcraft/tripcraft_5day.csv \
        --csv-7day benchmarks/TripCraft/tripcraft/tripcraft_7day.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List

import openai
import pandas as pd
import numpy as np
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a meticulous assistant that converts natural-language travel itineraries into JSON.\n"
    "Given a single trip description, return a JSON array where each element corresponds to one day and "
    "contains the following string fields: 'days' (integer), 'current_city', 'transportation', 'breakfast', "
    "'attraction' (semicolon-separated list), 'lunch', 'dinner', 'accommodation', 'event', "
    "'point_of_interest_list'. Use '-' when information is missing. Do not include any prose or code fences."
)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _strip_code_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = out[out.find("\n") + 1 :]
    if out.endswith("```"):
        out = out[: out.rfind("```")]
    return out.strip()


def _call_openai(model: str, plan_text: str, retries: int = 5, delay: float = 2.0) -> List[Dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Convert the following travel plan into the required JSON format.\n"
                "Plan:\n"
                f"{plan_text}"
            ),
        },
    ]
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(_strip_code_fences(content))
            if not isinstance(parsed, list):
                raise ValueError("Model response is not a JSON array.")
            return parsed
        except Exception as exc:  # pylint: disable=broad-except
            last_err = exc
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to convert plan after {retries} attempts: {last_err}") from last_err


def _parse_structured_fields(row: pd.Series) -> Dict:
    fields = [
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
    ]
    result: Dict = {}
    for key in fields:
        value = row.get(key)
        if pd.isna(value):
            result[key] = None
            continue
        if isinstance(value, str):
            value_str = value.strip()
            if value_str.startswith("{") or value_str.startswith("["):
                try:
                    result[key] = ast.literal_eval(value_str)
                    continue
                except (ValueError, SyntaxError):
                    pass
            result[key] = value
        elif isinstance(value, (int, float)):
            if key in {"days", "visiting_city_number", "people_number"}:
                result[key] = int(value)
            elif key == "budget":
                result[key] = float(value)
            else:
                result[key] = value
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def _load_plan_text(plan_file: Path) -> str:
    data = json.loads(plan_file.read_text())
    if not data or not isinstance(data, list):
        raise ValueError(f"Unexpected payload in {plan_file}")
    payload = data[-1]
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected dict payload in {plan_file}")
    for key, value in payload.items():
        if key.endswith("_sole-planning_results"):
            return value
    raise KeyError(f"No '*_sole-planning_results' key found in {plan_file}")


def _iter_plan_files(day_dir: Path) -> Iterable[Path]:
    return sorted(day_dir.glob("gpt4*_generated_plan_*.json"), key=lambda p: int(p.stem.split("_")[-1]))


def convert_runs(
    input_root: Path,
    output_path: Path,
    csv_map: Dict[str, Path],
    model: str,
) -> None:
    records = []
    idx_counter = 1
    for day, csv_path in csv_map.items():
        day_dir = input_root / day
        if not day_dir.exists():
            continue
        df = _load_csv(csv_path)
        plan_files = list(_iter_plan_files(day_dir))
        if not plan_files:
            continue
        if len(plan_files) > len(df):
            raise ValueError(f"{day}: more plan files ({len(plan_files)}) than rows in CSV ({len(df)})")

        tqdm_desc = f"Converting {day}"
        for plan_file in tqdm(plan_files, desc=tqdm_desc):
            plan_idx = int(plan_file.stem.split("_")[-1]) - 1
            row = df.iloc[plan_idx]
            plan_text = _load_plan_text(plan_file)
            structured_plan = _call_openai(model=model, plan_text=plan_text)
            record = {
                "idx": idx_counter,
                "JSON": _parse_structured_fields(row),
                "persona": row.get("persona", ""),
                "plan": structured_plan,
            }
            records.append(record)
            idx_counter += 1

    if not records:
        raise RuntimeError(f"No plan files found under {input_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TripCraft plan outputs into evaluation JSONL.")
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing day subfolders.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Destination JSONL path.")
    parser.add_argument("--csv-3day", type=Path, required=True, help="CSV used for 3-day generation.")
    parser.add_argument("--csv-5day", type=Path, required=True, help="CSV used for 5-day generation.")
    parser.add_argument("--csv-7day", type=Path, required=True, help="CSV used for 7-day generation.")
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4.1-mini",
        help="Model used for plan-to-JSON conversion (default: gpt-4.1-mini).",
    )
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY must be set before running this script.")

    csv_map = {
        "3day": args.csv_3day,
        "5day": args.csv_5day,
        "7day": args.csv_7day,
    }
    convert_runs(input_root=args.input_root, output_path=args.output_jsonl, csv_map=csv_map, model=args.openai_model)
    print(f"[TripCraft] Wrote evaluation JSONL to {args.output_jsonl}")


if __name__ == "__main__":
    main()
