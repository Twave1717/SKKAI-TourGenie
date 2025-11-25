"""
TripCraft CSV 계획 파서 + 요약 포매터.
- CSV 단일 행을 구조화 plan_payload로 변환
- formatted_plans 생성에 필요한 포매팅 함수 포함
"""

import ast
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_project_root() -> Path:
    candidates = [Path.cwd()] + list(Path(__file__).resolve().parents)
    for base in candidates:
        if (base / "KY" / "multi_persona").exists() or (base / "pyproject.toml").exists():
            return base
    return Path(__file__).resolve().parents[2]


ROOT_DIR = find_project_root()
DATA_DIR = ROOT_DIR / "benchmarks" / "TripCraft" / "tripcraft"


def list_csv_files(data_dir: Path = DATA_DIR) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


def _maybe_literal(val: Any):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    trips: List[Dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row["__index__"] = idx
            row["annotation_plan_parsed"] = _maybe_literal(row.get("annotation_plan", ""))
            row["reference_information_parsed"] = _maybe_literal(row.get("reference_information", ""))
            trips.append(row)
    return trips


def parse_annotation_plan(raw: Any) -> List[Dict[str, Any]]:
    parsed = _maybe_literal(raw)
    if isinstance(parsed, list):
        normalized = []
        for day in parsed:
            if not isinstance(day, dict):
                continue
            normalized.append(
                {
                    "day": day.get("days"),
                    "current_city": day.get("current_city"),
                    "transportation": day.get("transportation"),
                    "breakfast": day.get("breakfast"),
                    "lunch": day.get("lunch"),
                    "dinner": day.get("dinner"),
                    "attraction": day.get("attraction"),
                    "accommodation": day.get("accommodation"),
                    "event": day.get("event"),
                    "point_of_interest_list": day.get("point_of_interest_list"),
                }
            )
        return normalized
    return []


def load_tripcraft_plan(path: str, row_index: int = 0) -> Dict[str, Any]:
    """
    TripCraft csv에서 지정 인덱스의 계획을 구조화된 dict로 반환.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Plan file not found: {path}")
    rows = load_rows(p)
    if row_index >= len(rows):
        raise IndexError(f"row_index {row_index} out of range for file with {len(rows)} rows")
    row = rows[row_index]

    payload: Dict[str, Any] = {
        "meta": {
            "org": row.get("org"),
            "dest": row.get("dest"),
            "days": row.get("days"),
            "visiting_city_number": row.get("visiting_city_number"),
            "date": _maybe_literal(row.get("date")),
            "people_number": row.get("people_number"),
            "budget": row.get("budget"),
            "level": row.get("level"),
            "persona": row.get("persona"),
            "local_constraint": _maybe_literal(row.get("local_constraint")),
            "query": row.get("query"),
        },
        "itinerary": parse_annotation_plan(row.get("annotation_plan")),
        "references": {
            "reference_information": row.get("reference_information"),
            "reference_information_2": row.get("reference_information_2"),
            "reference_information_3": row.get("reference_information_3"),
        },
        "raw_row": row,
    }
    return payload


def poi_md_table(poi_str: str) -> str:
    entries = [seg.strip() for seg in (poi_str or "").split(";") if seg.strip()]
    if not entries:
        return ""
    lines = ["| POI |", "|---|"]
    for seg in entries:
        lines.append(f"| {seg} |")
    return "\n".join(lines)


def format_trip(row: Dict[str, Any]) -> str:
    ap = row.get("annotation_plan_parsed") or []
    parts: List[str] = []
    parts.append(f"## 여행 {row['__index__'] + 1}: {row.get('org')} → {row.get('dest')}")
    parts.append(f"- 날짜: {row.get('date')}  \n- 인원: {row.get('people_number')}명  \n- 예산: {row.get('budget')}")
    for day in ap:
        parts.append(f"\n### Day {day.get('days')}")
        parts.append(f"- 구간: {day.get('current_city')}")
        parts.append(f"- 교통: {day.get('transportation')}")
        parts.append(f"- 관광: {day.get('attraction')}")
        parts.append(f"- 식사: 아침 {day.get('breakfast')} / 점심 {day.get('lunch')} / 저녁 {day.get('dinner')}")
        parts.append(f"- 숙박: {day.get('accommodation')}")
        poi_md = poi_md_table(day.get("point_of_interest_list", ""))
        if poi_md:
            parts.append("\nPOI\n" + poi_md)
    return "\n".join(parts)
