#!/usr/bin/env python3
"""
TripCraft evaluation & leaderboard updater runnable entirely from the TripCraft conda environment.

Usage:
    python scripts/tripcraft_eval.py \
        --submission results/tripcraft/gpt41mini_test_mini.jsonl \
        --provider openai \
        --model gpt-4.1-mini \
        --workflow test-mini \
        --result-label gpt41mini_test_mini
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIPCRAFT_ROOT = REPO_ROOT / "benchmarks" / "TripCraft"

if str(TRIPCRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRIPCRAFT_ROOT))

EVAL_DIR = TRIPCRAFT_ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from evaluation.eval import eval_score


DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tripcraft"
DEFAULT_LEADERBOARD = REPO_ROOT / "leaderboards" / "TripCraft" / "main.md"
TABLE_HEADER = [
    "Rank",
    "Provider",
    "Model",
    "Updated",
    "Delivery Rate",
    "Commonsense Constraint Micro Pass Rate",
    "Commonsense Constraint Macro Pass Rate",
    "Hard Constraint Micro Pass Rate",
    "Hard Constraint Macro Pass Rate",
    "Final Pass Rate",
    "Results",
]


def _load_jsonl(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _write_subset(entries: List[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp_path = Path(tmp.name)
    try:
        for entry in entries:
            tmp.write(json.dumps(entry))
            tmp.write("\n")
    finally:
        tmp.close()
    return tmp_path


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _update_leaderboard(row: Dict[str, str], leaderboard_path: Path) -> None:
    if not leaderboard_path.exists():
        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        with leaderboard_path.open("w", encoding="utf-8") as fout:
            fout.write("# TripCraft Leaderboard\n\n")
            header = "| " + " | ".join(TABLE_HEADER) + " |\n"
            divider = "| " + " | ".join(["---"] * len(TABLE_HEADER)) + " |\n"
            fout.write(header)
            fout.write(divider)
            fout.write("| " + " | ".join(["-"] * len(TABLE_HEADER)) + " |\n")

    with leaderboard_path.open("r", encoding="utf-8") as fin:
        lines = fin.readlines()

    data_lines = [line for line in lines if line.startswith("|")][2:]  # skip header + divider
    rows: List[Dict[str, str]] = []
    for line in data_lines:
        parts = [part.strip() for part in line.strip().split("|")[1:-1]]
        if not parts or parts[0] == "-":
            continue
        rows.append(dict(zip(TABLE_HEADER, parts)))

    rows.append(row)

    def _extract_float(value: str) -> float:
        try:
            return float(value.rstrip("%"))
        except ValueError:
            return 0.0

    rows.sort(
        key=lambda r: (
            _extract_float(r["Final Pass Rate"]),
            _extract_float(r["Hard Constraint Macro Pass Rate"]),
            _extract_float(r["Hard Constraint Micro Pass Rate"]),
            _extract_float(r["Commonsense Constraint Macro Pass Rate"]),
            _extract_float(r["Commonsense Constraint Micro Pass Rate"]),
            _extract_float(r["Delivery Rate"]),
        ),
        reverse=True,
    )

    output_lines = lines[:2]  # header + divider
    for idx, r in enumerate(rows, start=1):
        r["Rank"] = str(idx)
        output_lines.append("| " + " | ".join(r.get(col, "") for col in TABLE_HEADER) + " |\n")

    with leaderboard_path.open("w", encoding="utf-8") as fout:
        fout.write("# TripCraft Leaderboard\n\n")
        fout.write("| " + " | ".join(TABLE_HEADER) + " |\n")
        fout.write("| " + " | ".join(["---"] * len(TABLE_HEADER)) + " |\n")
        for line in output_lines[2:]:
            fout.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TripCraft submissions without Poetry.")
    parser.add_argument("--submission", type=Path, required=True, help="Path to TripCraft JSONL file.")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--workflow", type=str, default="direct")
    parser.add_argument("--result-label", type=str, required=True, help="Label used for metrics/leaderboard entries.")
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=DEFAULT_LEADERBOARD,
        help="Leaderboard Markdown file to update.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where metrics JSON will be written.",
    )
    args = parser.parse_args()

    entries = _load_jsonl(args.submission)
    if not entries:
        raise RuntimeError("Submission JSONL is empty.")

    overall_scores, details = eval_score("custom", file_path=str(args.submission))

    per_day_scores: Dict[int, Dict[str, float]] = {}
    for day in (3, 5, 7):
        subset = [entry for entry in entries if entry["JSON"].get("days") == day]
        if not subset:
            continue
        subset_path = _write_subset(subset)
        try:
            day_scores, _ = eval_score("custom", file_path=str(subset_path))
            per_day_scores[day] = day_scores
        finally:
            subset_path.unlink(missing_ok=True)

    metrics_dir = args.results_dir / args.result_label
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    metrics_payload = {
        "provider": args.provider,
        "model": args.model,
        "workflow": args.workflow,
        "submission": str(args.submission),
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "overall": overall_scores,
        "per_day": per_day_scores,
        "details": details,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    try:
        results_rel = metrics_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        results_rel = metrics_path.as_posix()

    row = {
        "Rank": "",
        "Provider": args.provider,
        "Model": f"{args.model} ({args.workflow})",
        "Updated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "Delivery Rate": _format_pct(overall_scores["Delivery Rate"]),
        "Commonsense Constraint Micro Pass Rate": _format_pct(
            overall_scores["Commonsense Constraint Micro Pass Rate"]
        ),
        "Commonsense Constraint Macro Pass Rate": _format_pct(
            overall_scores["Commonsense Constraint Macro Pass Rate"]
        ),
        "Hard Constraint Micro Pass Rate": _format_pct(overall_scores["Hard Constraint Micro Pass Rate"]),
        "Hard Constraint Macro Pass Rate": _format_pct(overall_scores["Hard Constraint Macro Pass Rate"]),
        "Final Pass Rate": _format_pct(overall_scores["Final Pass Rate"]),
        "Results": f"[metrics]({results_rel})",
    }
    _update_leaderboard(row, args.leaderboard)
    print(f"[TripCraft] Metrics written to {metrics_path}")
    print(f"[TripCraft] Leaderboard updated at {args.leaderboard}")


if __name__ == "__main__":
    main()
