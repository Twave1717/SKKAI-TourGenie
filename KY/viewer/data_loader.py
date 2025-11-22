import ast
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "benchmarks" / "TripCraft" / "tripcraft"


def list_csv_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return available TripCraft CSV paths sorted by name."""
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


def safe_literal_eval(value: str) -> Optional[Any]:
    """Safely evaluate a Python literal-like string; return None on failure."""
    if not value or not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Load CSV rows and parse structured columns."""
    trips: List[Dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row["__index__"] = idx
            row["annotation_plan_parsed"] = safe_literal_eval(row.get("annotation_plan", ""))
            row["reference_information_parsed"] = safe_literal_eval(row.get("reference_information", ""))
            trips.append(row)
    return trips


def build_plan_text(row: Dict[str, Any]) -> str:
    """Construct a readable text block for a trip row."""
    parts = [
        f"Origin: {row.get('org', '')}",
        f"Destination: {row.get('dest', '')}",
        f"Dates: {row.get('date', '')}",
        f"People: {row.get('people_number', '')}",
        f"Budget: {row.get('budget', '')}",
        f"Persona: {row.get('persona', '')}",
        f"Query: {row.get('query', '')}",
    ]

    plan = row.get("annotation_plan_parsed")
    if plan and isinstance(plan, list):
        for day in plan:
            day_num = day.get("days", "")
            city = day.get("current_city", "")
            segment = f"Day {day_num} - {city}: "
            for key in ("transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation", "event"):
                if day.get(key):
                    segment += f"{key}={day[key]} | "
            parts.append(segment.rstrip(" | "))
    else:
        raw = row.get("annotation_plan", "")
        if raw:
            parts.append(f"Plan: {raw}")
    return "\n".join([p for p in parts if p])


def itinerary_table(plan: Any) -> Optional[pd.DataFrame]:
    """Convert parsed itinerary to a DataFrame for display."""
    if not plan or not isinstance(plan, list):
        return None
    rows = []
    for day in plan:
        rows.append(
            {
                "day": day.get("days", ""),
                "city": day.get("current_city", ""),
                "transport": day.get("transportation", ""),
                "breakfast": day.get("breakfast", ""),
                "attraction": day.get("attraction", ""),
                "lunch": day.get("lunch", ""),
                "dinner": day.get("dinner", ""),
                "accommodation": day.get("accommodation", ""),
                "event": day.get("event", ""),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DATA_DIR",
    "ROOT_DIR",
    "build_plan_text",
    "itinerary_table",
    "list_csv_files",
    "load_rows",
    "safe_literal_eval",
]
