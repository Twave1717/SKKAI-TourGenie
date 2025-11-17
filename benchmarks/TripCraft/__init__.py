"""
Utility helpers for running the TripCraft benchmark inside the unified Agentic AI repo.

This module provides a thin shim between the original TripCraft release
(`benchmarks/TripCraft`) and the higher level scripts / CLIs used elsewhere in the
project.  All heavy lifting (data loading, planning, evaluation) still happens
inside the upstream codebase.
"""

from __future__ import annotations

from pathlib import Path


TRIPCRAFT_ROOT = Path(__file__).resolve().parents[1] / "TripCraft"
TRIPCRAFT_DB_DIR = TRIPCRAFT_ROOT / "TripCraft_database"
TRIPCRAFT_RUN_SCRIPT = TRIPCRAFT_ROOT / "run.sh"
TRIPCRAFT_EVAL_SCRIPT = TRIPCRAFT_ROOT / "evaluation" / "eval.py"
TRIPCRAFT_QUAL_SCRIPT = TRIPCRAFT_ROOT / "evaluation" / "qualitative_metrics.py"


def default_csv_for_day(day: str, base_dir: Path | None = None) -> Path:
    """Return the default CSV file shipped with TripCraft for the requested day split."""
    root = base_dir or TRIPCRAFT_ROOT
    mapping = {
        "3day": root / "tripcraft" / "tripcraft_3day.csv",
        "5day": root / "tripcraft" / "tripcraft_5day.csv",
        "7day": root / "tripcraft" / "tripcraft_7day.csv",
    }
    if day not in mapping:
        raise ValueError(f"Unsupported TripCraft day split '{day}'. Choose from {sorted(mapping)}.")
    return mapping[day]
