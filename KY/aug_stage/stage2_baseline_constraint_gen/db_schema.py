from __future__ import annotations

"""DB schema helpers for TravelPlanner Stage 2.

Stage 2 needs two kinds of DB context:

1) Per-instance DB snapshot (cheap, derived from TravelPlanner row's `reference_information`)
   - allowed enum values actually present for that instance
   - rough numeric ranges to help pick realistic thresholds
   - a few attraction name samples for keyword-grounding

2) Global schema (cheap, derived from CSV headers in the TravelPlanner DB folder)
   - table/column listing (Text-to-SQL style)
   - field-alias map: the ONLY allowed `field` names in constraints

We never call external APIs.
"""

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------
# TravelPlanner paper enums (verbatim strings)
# ---------------------------

CUISINES = ["Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"]
ROOM_RULES = ["No parties", "No smoking", "No children under 10", "No pets", "No visitors"]
ROOM_TYPES = ["Entire Room", "Private Room", "Shared Room", "No Shared Room"]


# ---------------------------
# reference_information -> instance snapshot
# ---------------------------

def _safe_literal(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in "[{(":
            try:
                return ast.literal_eval(s)
            except Exception:
                return v
    return v


def parse_reference_information(ref: Any) -> List[Dict[str, str]]:
    """Parse TravelPlanner `reference_information` into a list of {Description, Content}."""
    if ref is None:
        return []
    if isinstance(ref, list):
        return [x for x in ref if isinstance(x, dict)]
    if not isinstance(ref, str):
        return []
    parsed = _safe_literal(ref)
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


_FLOAT_RE = re.compile(r"\b(\d+\.\d+)\b")
_INT_RE = re.compile(r"(?<!\.)\b(\d+)\b(?!\.)")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")

def _is_num(tok: str) -> bool:
    return bool(_NUM_RE.match(tok))

def _find_all_floats(s: str) -> List[float]:
    out: List[float] = []
    for m in _FLOAT_RE.finditer(s):
        try:
            out.append(float(m.group(1)))
        except Exception:
            pass
    return out


def _find_all_ints(s: str) -> List[int]:
    out: List[int] = []
    for m in _INT_RE.finditer(s):
        try:
            out.append(int(m.group(1)))
        except Exception:
            pass
    return out


def _rng(xs: Sequence[float]) -> Optional[Dict[str, float]]:
    if not xs:
        return None
    return {"min": float(min(xs)), "max": float(max(xs))}


def _extract_attraction_names(content: str, limit: int = 30) -> List[str]:
    """Heuristic: attraction rows usually contain lat/lon floats; take prefix before first float as name."""
    names: List[str] = []
    for ln in content.splitlines():
        t = ln.strip()
        if not t or t.lower().startswith(("name", "description")):
            continue
        m = _FLOAT_RE.search(t)
        if not m:
            continue
        name = t[: m.start()].strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _extract_enum_hits(content: str, enums: List[str]) -> List[str]:
    return [e for e in enums if re.search(rf"\b{re.escape(e)}\b", content)]


def _extract_restaurant_stats(content: str) -> Dict[str, Any]:
    avg_costs: List[float] = []
    ratings: List[float] = []
    rows = 0
    for ln in content.splitlines():
        t = ln.strip()
        if not t or t.lower().startswith(("name", "restaurants", "description")):
            continue
        rows += 1
        ints = _find_all_ints(t)
        fls = _find_all_floats(t)
        if ints:
            avg_costs.append(float(ints[0]))
        if fls:
            ratings.append(float(fls[0]))
    return {
        "rows": max(0, rows - 1),  # cheap header compensation
        "avg_cost_range": _rng(avg_costs),
        "rating_range": _rng(ratings),
        "cuisines_available": _extract_enum_hits(content, CUISINES),
    }


def _extract_accommodation_stats(content: str) -> Dict[str, Any]:
    prices: List[float] = []
    reviews: List[float] = []
    min_nights: List[float] = []
    max_occ: List[int] = []
    rows = 0
    for ln in content.splitlines():
        t = ln.strip()
        if not t or t.lower().startswith(("name", "accommod", "description")):
            continue
        rows += 1
        fls = _find_all_floats(t)
        ints = _find_all_ints(t)
        if fls:
            prices.append(fls[0])
            if len(fls) >= 2:
                reviews.append(fls[-1])
            if len(fls) >= 3:
                min_nights.append(fls[1])
        if ints:
            max_occ.append(ints[-1])
    return {
        "rows": max(0, rows - 1),
        "room_rules_available": _extract_enum_hits(content, ROOM_RULES),
        "room_types_available": _extract_enum_hits(content, ROOM_TYPES),
        "price_range": _rng(prices),
        "review_range": _rng(reviews),
        "minimum_nights_range": _rng(min_nights),
        "maximum_occupancy_range": ({"min": int(min(max_occ)), "max": int(max(max_occ))} if max_occ else None),
    }


def _extract_flight_stats(content: str) -> Dict[str, Any]:
    """
    Robust parser for TravelPlanner flight table text.

    Key idea:
    - Many formats contain DepTime/ArrTime as HH:MM.
    - The token right before the first HH:MM is usually Price.
    This avoids accidentally picking '2022' from FlightDate.
    """
    prices: List[float] = []
    durations_min: List[int] = []
    distances: List[float] = []
    rows = 0

    for ln in content.splitlines():
        t = ln.strip()
        if not t:
            continue

        low = t.lower()
        # skip header-like lines
        if low.startswith(("flight", "description", "flightnumber", "flight_number")):
            continue

        toks = t.split()
        time_idxs = [i for i, tok in enumerate(toks) if _TIME_RE.match(tok)]
        if len(time_idxs) < 2:
            # not a real data row
            continue

        # price is right before first time token (DepTime)
        i0 = time_idxs[0]
        if i0 >= 1 and _is_num(toks[i0 - 1]):
            prices.append(float(toks[i0 - 1]))
            rows += 1
        else:
            # can't find a reliable price, skip row
            continue

        # duration like "1 hours 40 minutes"
        m = re.search(r"(\d+)\s*hours?\s*(\d+)\s*minutes?", t, flags=re.I)
        if m:
            durations_min.append(int(m.group(1)) * 60 + int(m.group(2)))

        # distance is often the last numeric token
        for tok in reversed(toks):
            if _is_num(tok):
                distances.append(float(tok))
                break

    return {
        "rows": rows,
        "price_range": _rng(prices),
        "duration_minutes_range": _rng(durations_min),
        "distance_range": _rng(distances),
    }


def _extract_ground_transport_stats(content: str) -> Dict[str, Any]:
    # Example: "self-driving, from A to B, duration: 6 hours 47 mins, distance: 693 km, cost: 34"
    cost = None
    duration_min = None
    m = re.search(r"cost:\s*(\d+(?:\.\d+)?)", content, flags=re.I)
    if m:
        try:
            cost = float(m.group(1))
        except Exception:
            pass
    m = re.search(r"duration:\s*(\d+)\s*hours?\s*(\d+)\s*mins?", content, flags=re.I)
    if m:
        try:
            duration_min = int(m.group(1)) * 60 + int(m.group(2))
        except Exception:
            pass
    return {"cost": cost, "duration_minutes": duration_min}


def build_db_schema_snapshot(reference_information: Any) -> Dict[str, Any]:
    """Return a compact, prompt-friendly schema snapshot for this instance."""
    items = parse_reference_information(reference_information)

    out: Dict[str, Any] = {
        "allowed_values": {
            "cuisines": [],
            "room_rules": [],
            "room_types": [],
            "ground_modes": ["flight", "self-driving", "taxi"],
        },
        "ranges": {},
        "counts": {
            "reference_items": len(items),
            "restaurant_rows": 0,
            "accommodation_rows": 0,
            "attraction_rows": 0,
            "flight_rows": 0,
        },
        "attraction_name_samples": [],
    }

    for it in items:
        desc = str(it.get("Description") or "")
        content = str(it.get("Content") or "")
        dlow = desc.lower()

        if "restaurants" in dlow:
            rs = _extract_restaurant_stats(content)
            out["counts"]["restaurant_rows"] += int(rs.get("rows") or 0)
            out["allowed_values"]["cuisines"] = sorted(set(out["allowed_values"]["cuisines"]) | set(rs.get("cuisines_available") or []))
            if rs.get("avg_cost_range"):
                out["ranges"]["restaurant.avg_cost"] = rs["avg_cost_range"]
            if rs.get("rating_range"):
                out["ranges"]["restaurant.rating"] = rs["rating_range"]

        elif "accommod" in dlow:
            ac = _extract_accommodation_stats(content)
            out["counts"]["accommodation_rows"] += int(ac.get("rows") or 0)
            out["allowed_values"]["room_rules"] = sorted(set(out["allowed_values"]["room_rules"]) | set(ac.get("room_rules_available") or []))
            out["allowed_values"]["room_types"] = sorted(set(out["allowed_values"]["room_types"]) | set(ac.get("room_types_available") or []))
            if ac.get("price_range"):
                out["ranges"]["accommodation.price"] = ac["price_range"]
            if ac.get("review_range"):
                out["ranges"]["accommodation.review"] = ac["review_range"]
            if ac.get("minimum_nights_range"):
                out["ranges"]["accommodation.minimum_nights"] = ac["minimum_nights_range"]
            if ac.get("maximum_occupancy_range"):
                out["ranges"]["accommodation.maximum_occupancy"] = ac["maximum_occupancy_range"]

        elif "attractions" in dlow:
            names = _extract_attraction_names(content, limit=30)
            out["counts"]["attraction_rows"] += max(0, len(names))
            out["attraction_name_samples"] = names

        elif "flight" in dlow:
            fs = _extract_flight_stats(content)
            out["counts"]["flight_rows"] += int(fs.get("rows") or 0)

            if fs.get("distance_range"):
                cur = out["ranges"].get("flight.distance")
                dr = fs["distance_range"]
                out["ranges"]["flight.distance"] = dr if not cur else {
                    "min": min(cur["min"], dr["min"]),
                    "max": max(cur["max"], dr["max"]),
                }

            if fs.get("price_range"):
                cur = out["ranges"].get("flight.price")
                pr = fs["price_range"]
                out["ranges"]["flight.price"] = pr if not cur else {"min": min(cur["min"], pr["min"]), "max": max(cur["max"], pr["max"])}

            if fs.get("duration_minutes_range"):
                cur = out["ranges"].get("flight.duration_minutes")
                dr = fs["duration_minutes_range"]
                out["ranges"]["flight.duration_minutes"] = dr if not cur else {"min": min(cur["min"], dr["min"]), "max": max(cur["max"], dr["max"])}

        elif "self-driving" in dlow or "taxi" in dlow:
            gt = _extract_ground_transport_stats(content)
            if "self-driving" in dlow:
                if gt.get("cost") is not None:
                    out["ranges"]["ground.self_driving_cost"] = {"min": gt["cost"], "max": gt["cost"]}
                if gt.get("duration_minutes") is not None:
                    out["ranges"]["ground.self_driving_duration_minutes"] = {"min": gt["duration_minutes"], "max": gt["duration_minutes"]}
            if "taxi" in dlow:
                if gt.get("cost") is not None:
                    out["ranges"]["ground.taxi_cost"] = {"min": gt["cost"], "max": gt["cost"]}
                if gt.get("duration_minutes") is not None:
                    out["ranges"]["ground.taxi_duration_minutes"] = {"min": gt["duration_minutes"], "max": gt["duration_minutes"]}

    out["counts"].update(
        {
            "has_restaurants": any("restaurants" in str(x.get("Description") or "").lower() for x in items),
            "has_accommodations": any("accommod" in str(x.get("Description") or "").lower() for x in items),
            "has_attractions": any("attractions" in str(x.get("Description") or "").lower() for x in items),
            "has_flights": any("flight" in str(x.get("Description") or "").lower() for x in items),
            "has_ground_transport": any(("self-driving" in str(x.get("Description") or "").lower()) or ("taxi" in str(x.get("Description") or "").lower()) for x in items),
        }
    )

    # keep compact
    out["allowed_values"]["cuisines"] = out["allowed_values"]["cuisines"][:10]
    out["allowed_values"]["room_rules"] = out["allowed_values"]["room_rules"][:10]
    out["allowed_values"]["room_types"] = out["allowed_values"]["room_types"][:10]
    out["attraction_name_samples"] = out["attraction_name_samples"][:25]
    return out


# ---------------------------
# Global schema (CSV headers) helpers
# ---------------------------

_DB_CANDIDATES: Dict[str, List[str]] = {
    # official TravelPlanner DB layout (root/{category}/file.csv)
    "restaurants": ["restaurants/clean_restaurant_2022.csv", "clean_restaurant_2022.csv"],
    "accommodations": ["accommodations/clean_accommodations_2022.csv", "clean_accommodations_2022.csv"],
    "attractions": ["attractions/attractions.csv", "attractions.csv"],
    "flights": ["flights/clean_Flights_2022.csv", "clean_Flights_2022.csv"],
    "distance": ["googleDistanceMatrix/distance.csv", "distance.csv"],
}

_DB_FILENAMES: Dict[str, str] = {
    "restaurants": "clean_restaurant_2022.csv",
    "accommodations": "clean_accommodations_2022.csv",
    "attractions": "attractions.csv",
    "flights": "clean_Flights_2022.csv",
    "distance": "distance.csv",
}


def _resolve_db_file(db_root: Path, table: str) -> Optional[Path]:
    """Resolve a table CSV path from db_root.

    - Tries known relative candidates first (works for the official DB layout).
    - Falls back to a recursive filename search (works for custom layouts).
    """
    for rel in _DB_CANDIDATES.get(table, []):
        p = db_root / rel
        if p.exists():
            return p

    # Fallback: search by filename under db_root
    fname = _DB_FILENAMES.get(table)
    if not fname:
        return None
    hits = list(db_root.rglob(fname))
    if not hits:
        return None
    hits.sort(key=lambda x: len(str(x)))
    return hits[0]


def _read_csv_header(path: Path) -> List[str]:
    """Read only the header row of a CSV (cheap even for large files)."""
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        row = next(csv.reader(f))
    cols: List[str] = []
    for i, c in enumerate(row):
        c = (c or "").strip()
        if i == 0 and (c == "" or c.lower() in {"index", "unnamed: 0"}):
            continue
        cols.append(c)
    return cols


def build_global_db_schema(db_root: Path) -> Dict[str, Any]:
    """Build a global DB schema summary from TravelPlanner CSV headers.

    Used ONLY for prompting (schema awareness), not for query execution.
    """
    db_root = Path(db_root)

    tables: List[Dict[str, Any]] = []
    for t in ["restaurants", "accommodations", "attractions", "flights", "distance"]:
        p = _resolve_db_file(db_root, t)
        if p and p.exists():
            tables.append({"table": t, "path": str(p), "columns": _read_csv_header(p)})
        else:
            tables.append({"table": t, "path": None, "columns": [], "missing": True})

    # Field aliases (what Stage2 constraints use) -> (table, column or computed)
    field_aliases: List[Dict[str, str]] = [
        # Restaurants
        {"field": "restaurant.cuisine", "table": "restaurants", "column": "Cuisines"},
        {"field": "restaurant.avg_cost", "table": "restaurants", "column": "Average Cost"},
        {"field": "restaurant.rating", "table": "restaurants", "column": "Aggregate Rating"},
        {"field": "restaurant.name_keyword", "table": "restaurants", "column": "Name"},
        {"field": "restaurant.count", "table": "restaurants", "column": "<computed count rows for the scenario>"},

        # Accommodations
        {"field": "accommodation.room_type", "table": "accommodations", "column": "room type"},
        {"field": "accommodation.house_rule", "table": "accommodations", "column": "house_rules"},
        {"field": "accommodation.price", "table": "accommodations", "column": "price"},
        {"field": "accommodation.review", "table": "accommodations", "column": "review rate number"},
        {"field": "accommodation.minimum_nights", "table": "accommodations", "column": "minimum nights"},
        {"field": "accommodation.maximum_occupancy", "table": "accommodations", "column": "maximum occupancy"},
        {"field": "accommodation.name_keyword", "table": "accommodations", "column": "NAME"},
        {"field": "accommodation.count", "table": "accommodations", "column": "<computed count rows for the scenario>"},

        # Attractions
        {"field": "attraction.name_keyword", "table": "attractions", "column": "Name"},
        {"field": "attraction.address_keyword", "table": "attractions", "column": "Address"},
        {"field": "attraction.count", "table": "attractions", "column": "<computed count rows for the scenario>"},

        # Flights
        {"field": "flight.price", "table": "flights", "column": "Price"},
        {"field": "flight.duration_minutes", "table": "flights", "column": "ActualElapsedTime"},
        {"field": "flight.depart_time", "table": "flights", "column": "DepTime"},
        {"field": "flight.arrive_time", "table": "flights", "column": "ArrTime"},
        {"field": "flight.distance", "table": "flights", "column": "Distance"},
        {"field": "flight.count", "table": "flights", "column": "<computed count rows for the scenario>"},

        # Ground transport
        {"field": "ground.mode", "table": "distance", "column": "<computed {flight|self-driving|taxi}>"},  # virtual
        {"field": "ground.cost", "table": "distance", "column": "cost"},
        {"field": "ground.duration_minutes", "table": "distance", "column": "duration"},
        {"field": "ground.distance", "table": "distance", "column": "distance"},
    ]
    return {"db_root": str(db_root), "tables": tables, "field_aliases": field_aliases}


def global_schema_to_text(schema: Dict[str, Any]) -> str:
    """Convert schema dict into a prompt-friendly text block (Text-to-SQL style)."""
    lines: List[str] = []
    lines.append("Given the following TravelPlanner database schema (CSV tables):")
    for t in schema.get("tables", []):
        cols = t.get("columns") or []
        if cols:
            lines.append(f"- {t['table']}: " + ", ".join(cols))
        else:
            lines.append(f"- {t['table']}: (missing)")

    lines.append("")
    lines.append("Field aliases (use these exact `field` names in constraints):")
    for a in schema.get("field_aliases", []):
        lines.append(f"- {a['field']} -> {a['table']}.{a['column']}")
    return "\n".join(lines)


def load_or_build_global_schema(db_root: Path, cache_path: Path) -> Dict[str, Any]:
    """Load cached schema JSON if compatible, else rebuild and write."""
    db_root = Path(db_root)
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("db_root") == str(db_root):
                return cached
        except Exception:
            pass

    schema = build_global_db_schema(db_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema
