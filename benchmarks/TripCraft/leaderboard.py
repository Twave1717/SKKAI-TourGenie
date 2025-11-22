from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

TRIPCRAFT_LEADERBOARD_DIR = Path("leaderboards/TripCraft")
TRIPCRAFT_LEADERBOARD_PATH = TRIPCRAFT_LEADERBOARD_DIR / "main.md"

TRIPCRAFT_METRIC_KEYS = [
    "Delivery Rate",
    "Commonsense Constraint Micro Pass Rate",
    "Commonsense Constraint Macro Pass Rate",
    "Hard Constraint Micro Pass Rate",
    "Hard Constraint Macro Pass Rate",
    "Final Pass Rate",
]

TRIPCRAFT_SORT_PRIORITY = [
    "Final Pass Rate",
    "Hard Constraint Macro Pass Rate",
    "Hard Constraint Micro Pass Rate",
    "Commonsense Constraint Macro Pass Rate",
    "Commonsense Constraint Micro Pass Rate",
    "Delivery Rate",
]


def build_header(metric_keys: List[str], title: str = "# TripCraft Leaderboard") -> List[str]:
    columns = " | ".join(
        [
            "Rank",
            "Provider",
            "Model",
            "Updated",
            *metric_keys,
            "Results",
        ]
    )
    separator = "| " + " | ".join(["---"] * (4 + len(metric_keys) + 1)) + " |"
    return [
        title,
        "",
        f"| {columns} |",
        separator,
    ]


def _format_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _row_key(provider: str, model: str, label: str | None) -> Tuple[str, str, str]:
    return (
        (provider or "").strip().lower(),
        (model or "").strip().lower(),
        (label or "").strip().lower(),
    )


def _parse_results_cell(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and "](" in value and value.endswith(")"):
        start = value.index("](") + 2
        return value[start:-1]
    return value


def _extract_result_label(results_cell: str) -> str | None:
    start = results_cell.rfind("(")
    end = results_cell.rfind(")")
    if start != -1 and end != -1 and end > start:
        candidate = results_cell[start + 1 : end]
        if candidate and "/" not in candidate:
            return candidate
    return None


def _parse_rows(lines: List[str], metric_keys: List[str]) -> List[Dict]:
    entries: List[Dict] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| Rank"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 4 + len(metric_keys) + 1:
            continue
        provider = cells[1]
        model = cells[2]
        updated = cells[3]
        metric_values = cells[4 : 4 + len(metric_keys)]
        results_cell = cells[4 + len(metric_keys)]
        metrics = {
            metric_keys[idx]: float(value) if value not in {"-", ""} else None
            for idx, value in enumerate(metric_values)
        }
        entries.append(
            {
                "provider": provider,
                "model": model,
                "updated": updated,
                "metrics": metrics,
                "results_path": _parse_results_cell(results_cell),
                "result_label": _extract_result_label(results_cell),
            }
        )
    return entries


def _sort_entries(entries: List[Dict]) -> List[Dict]:
    def key(entry: Dict) -> Tuple:
        metrics = entry.get("metrics", {})
        return tuple(
            -(metrics.get(metric) or 0.0) for metric in TRIPCRAFT_SORT_PRIORITY
        ) + (
            entry.get("provider", "").lower(),
            entry.get("model", "").lower(),
        )

    return sorted(entries, key=key)


def update_tripcraft_leaderboard(
    provider: str,
    model: str,
    metrics: Dict[str, float | None],
    results_path: str,
    *,
    metric_keys: List[str] | None = None,
    result_label: str | None = None,
) -> None:
    metric_keys = metric_keys or TRIPCRAFT_METRIC_KEYS
    TRIPCRAFT_LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    if TRIPCRAFT_LEADERBOARD_PATH.exists():
        lines = TRIPCRAFT_LEADERBOARD_PATH.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    header = build_header(metric_keys)
    data_lines = lines[2:] if len(lines) > 2 else []
    entries = _parse_rows(data_lines, metric_keys)

    key = _row_key(provider, model, result_label)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    row_entry = {
        "provider": provider,
        "model": model,
        "updated": timestamp,
        "metrics": metrics,
        "results_path": results_path,
        "result_label": result_label,
    }

    replaced = False
    for idx, entry in enumerate(entries):
        if _row_key(entry["provider"], entry["model"], entry.get("result_label")) == key:
            entries[idx] = row_entry
            replaced = True
            break
    if not replaced:
        entries.append(row_entry)

    sorted_entries = _sort_entries(entries)

    table_lines = header.copy()
    for rank, entry in enumerate(sorted_entries, start=1):
        metric_values = [_format_metric(entry["metrics"].get(metric)) for metric in metric_keys]
        label = entry.get("result_label")
        link_display = "결과"
        link_path = entry["results_path"]
        if label:
            link_display = f"{link_display} ({label})"
        row = " | ".join(
            [
                str(rank),
                entry["provider"],
                entry["model"],
                entry["updated"],
                *metric_values,
                f"[{link_display}]({link_path})",
            ]
        )
        table_lines.append(f"| {row} |")

    TRIPCRAFT_LEADERBOARD_PATH.write_text("\n".join(table_lines), encoding="utf-8")
