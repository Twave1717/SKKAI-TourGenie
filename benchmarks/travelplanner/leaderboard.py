from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


LEADERBOARD_DIR = Path("leaderboards/TravelPlanner")
LEADERBOARD_PATH = LEADERBOARD_DIR / "main.md"
MINI_LEADERBOARD_PATH = LEADERBOARD_DIR / "mini.md"
READ_ME_PATH = LEADERBOARD_DIR / "README.md"

METRIC_SORT_PRIORITY = [
    "Final Pass Rate",
    "Hard Constraint Macro Pass Rate",
    "Hard Constraint Micro Pass Rate",
    "Commonsense Constraint Macro Pass Rate",
    "Commonsense Constraint Micro Pass Rate",
    "Delivery Rate",
]

DEFAULT_METRIC_KEYS = [
    "Delivery Rate",
    "Commonsense Constraint Micro Pass Rate",
    "Commonsense Constraint Macro Pass Rate",
    "Hard Constraint Micro Pass Rate",
    "Hard Constraint Macro Pass Rate",
    "Final Pass Rate",
]


def _table_body_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return lines


def _sync_travelplanner_readme() -> None:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    main_lines = _table_body_lines(LEADERBOARD_PATH)
    mini_lines = _table_body_lines(MINI_LEADERBOARD_PATH)

    content: List[str] = ["# TravelPlanner Leaderboards", ""]

    content.append("## Main Leaderboard")
    content.append("")
    if main_lines:
        content.extend(main_lines)
    else:
        content.append("_No submissions yet._")
    content.append("")

    content.append("## Mini Leaderboard")
    content.append("")
    if mini_lines:
        content.extend(mini_lines)
    else:
        content.append("_No submissions yet._")
    content.append("")

    READ_ME_PATH.write_text("\n".join(content), encoding="utf-8")


def build_header(metric_keys: List[str], title: str = "# TravelPlanner Leaderboard") -> List[str]:
    columns = " | ".join([
        "Rank",
        "Provider",
        "Model",
        "Updated",
        *metric_keys,
        "Results",
    ])
    separator = "| " + " | ".join(["---"] * (3 + len(metric_keys) + 1)) + " |"
    return [
        title,
        "",
        f"| {columns} |",
        separator,
    ]


HEADER_LINES = build_header(DEFAULT_METRIC_KEYS, title="# TravelPlanner Main Leaderboard")


def _ensure_header(lines: List[str], header_lines: List[str]) -> List[str]:
    if not lines:
        return header_lines.copy()
    if lines[0].strip() != header_lines[0]:
        return header_lines.copy()
    return lines


def _format_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _extract_result_label(results_cell: str) -> str:
    start = results_cell.find("[")
    end = results_cell.find("]", start + 1) if start != -1 else -1
    if start == -1 or end == -1:
        return ""
    display = results_cell[start + 1 : end]
    if display.endswith(")") and "(" in display:
        candidate = display[display.rfind("(") + 1 : -1]
        if "/" not in candidate:
            return candidate
    return ""


def _row_key(provider: str, model: str, result_label: str | None = None) -> Tuple[str, str, str]:
    return (
        (provider or "").strip().lower(),
        (model or "").strip().lower(),
        (result_label or "").strip().lower(),
    )


def _parse_results_cell(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and "](" in value and value.endswith(")"):
        start = value.index("](") + 2
        return value[start:-1]
    return value


def _build_row(
    rank: int,
    provider: str,
    model: str,
    updated: str,
    metrics: Dict[str, float | None],
    results_path: str,
    metric_keys: List[str],
    result_label: str | None = None,
) -> str:
    metric_values = [_format_metric(metrics.get(key)) for key in metric_keys]
    link_text = results_path if not result_label else f"{results_path}({result_label})"
    columns = " | ".join(
        [str(rank), provider, model, updated, *metric_values, f"[{link_text}]({results_path})"]
    )
    return f"| {columns} |"


def _parse_data_rows(lines: List[str], metric_keys: List[str]) -> List[Dict]:
    entries: List[Dict] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| Rank"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        has_rank = cells[0].isdigit()
        offset = 1 if has_rank else 0
        expected = offset + 3 + len(metric_keys) + 1
        if len(cells) < expected:
            continue
        provider = cells[offset + 0]
        if provider.lower() in {"provider", "---"}:
            continue
        model = cells[offset + 1]
        if model.lower() == "model":
            continue
        updated = cells[offset + 2]
        metrics_segment = cells[offset + 3 : offset + 3 + len(metric_keys)]
        results_cell = cells[offset + 3 + len(metric_keys)]
        metrics = {
            metric_keys[idx]: _parse_metric_value(value) for idx, value in enumerate(metrics_segment)
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


def _sort_entries(entries: List[Dict], metric_keys: List[str]) -> List[Dict]:
    def sort_key(entry: Dict) -> Tuple:
        scores: List[float] = []
        for metric_name in METRIC_SORT_PRIORITY:
            if metric_name in metric_keys:
                value = entry["metrics"].get(metric_name)
            else:
                value = None
            scores.append(value if value is not None else float("-inf"))
        return tuple(scores + [entry["updated"]])

    return sorted(entries, key=sort_key, reverse=True)


def _parse_metric_value(value: str) -> float | None:
    value = value.strip()
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def update_leaderboard(
    provider: str,
    model: str,
    run_dir: Path,
    metrics: Dict[str, float],
    leaderboard_path: Path = LEADERBOARD_PATH,
    header_lines: List[str] | None = None,
    metric_keys: List[str] | None = None,
    result_label: str | None = None,
) -> None:
    model_display = model or "default"
    run_dir = run_dir.resolve()
    updated = datetime.now().strftime("%Y-%m-%d %H:%M")

    metric_keys = metric_keys or DEFAULT_METRIC_KEYS
    header = header_lines or build_header(metric_keys)

    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    if leaderboard_path.exists():
        lines = leaderboard_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    lines = _ensure_header(lines, header)

    entries = [
        entry
        for entry in _parse_data_rows(lines, metric_keys)
        if _row_key(entry["provider"], entry["model"], entry["result_label"])
        != _row_key(provider, model_display, result_label)
    ]

    entries.append(
        {
            "provider": provider,
            "model": model_display,
            "updated": updated,
            "metrics": metrics,
            "results_path": _rel_path(run_dir),
            "result_label": result_label,
        }
    )

    sorted_entries = _sort_entries(entries, metric_keys)

    output_lines = header.copy()
    output_lines.append("")
    for rank, entry in enumerate(sorted_entries, start=1):
        output_lines.append(
            _build_row(
                rank,
                entry["provider"],
                entry["model"],
                entry["updated"],
                entry["metrics"],
                entry["results_path"],
                metric_keys,
                entry["result_label"],
            )
        )

    leaderboard_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    if leaderboard_path.resolve() in {LEADERBOARD_PATH.resolve(), MINI_LEADERBOARD_PATH.resolve()}:
        _sync_travelplanner_readme()
def _rel_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()
