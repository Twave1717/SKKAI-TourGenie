#!/usr/bin/env python3
"""Convert each TripCraft CSV row into a standalone JSON file.

Reads the TripCraft CSVs under ``benchmarks/TripCraft/tripcraft`` and, for each
file, creates a sibling folder with the CSV stem. Each row is written as
``<stem>/<stem>_<row_number>.json`` using best-effort parsing to turn
stringified data into proper JSON where possible.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TRIPCRAFT_DIR = REPO_ROOT / "benchmarks" / "TripCraft" / "tripcraft"


def parse_value(value: str) -> Any:
    """Best-effort conversion from CSV string to a JSON-serialisable object."""
    if value is None:
        return None

    stripped = value.strip()
    if stripped == "":
        return ""

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:
            continue

    return value


def convert_csv(csv_path: Path) -> None:
    stem = csv_path.stem
    out_dir = csv_path.parent / stem
    out_dir.mkdir(exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, start=1):
            parsed_row = {key: parse_value(val) for key, val in row.items()}
            out_path = out_dir / f"{stem}_{idx}.json"
            with out_path.open("w", encoding="utf-8") as json_file:
                json.dump(parsed_row, json_file, ensure_ascii=False, indent=2)


def main() -> None:
    csv_files = sorted(TRIPCRAFT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {TRIPCRAFT_DIR}")
        return

    for csv_file in csv_files:
        convert_csv(csv_file)


if __name__ == "__main__":
    main()
