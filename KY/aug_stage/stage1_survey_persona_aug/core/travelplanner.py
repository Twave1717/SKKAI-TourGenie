from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def safe_literal(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in "[{(":
            try:
                return ast.literal_eval(s)
            except Exception:
                return v
    return v


def as_list(v: Any) -> List[str]:
    v = safe_literal(v)
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, tuple):
        return [str(x) for x in list(v)]
    if v in (None, ""):
        return []
    return [str(v)]


def as_dict(v: Any) -> Dict[str, Any]:
    v = safe_literal(v)
    return v if isinstance(v, dict) else {}


def stable_source_id(row: Dict[str, Any]) -> str:
    rid = row.get("id")
    if isinstance(rid, str) and rid.strip():
        return rid.strip()
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:16]
    return f"row_{h}"


_BUDGET_RE = re.compile(r"\$\s*([0-9][0-9,]*)")

# People number patterns (ordered by specificity)
_PEOPLE_PATTERNS = [
    # Exact numbers first
    (re.compile(r'\b(?:group|party|team)\s+of\s+(\d+)\s+(?:people|persons|individuals|travelers)', re.I), 'group_of_N'),
    (re.compile(r'\b(\d+)\s+(?:people|persons|individuals|travelers)\b', re.I), 'N_people'),
    (re.compile(r'\bfor\s+(\d+)\s+(?:people|persons|individuals|travelers)', re.I), 'for_N'),

    # Named numbers
    (re.compile(r'\b(?:eight|8)\s+(?:people|persons|individuals|travelers)\b', re.I), 8),
    (re.compile(r'\b(?:seven|7)\s+(?:people|persons|individuals|travelers)\b', re.I), 7),
    (re.compile(r'\b(?:six|6)\s+(?:people|persons|individuals|travelers)\b', re.I), 6),
    (re.compile(r'\b(?:five|5)\s+(?:people|persons|individuals|travelers)\b', re.I), 5),
    (re.compile(r'\b(?:four|4)\s+(?:people|persons|individuals|travelers)\b', re.I), 4),
    (re.compile(r'\b(?:three|3)\s+(?:people|persons|individuals|travelers)\b', re.I), 3),
    (re.compile(r'\b(?:two|2)\s+(?:people|persons|individuals|travelers)\b', re.I), 2),

    # Special phrases
    (re.compile(r'\b(?:solo|one\s+person|single\s+traveler|one\s+individual)\b', re.I), 1),
    (re.compile(r'\b(?:couple|pair|duo|two\s+of\s+us)\b', re.I), 2),
    (re.compile(r'\bfamily\s+of\s+(\d+)\b', re.I), 'family_of_N'),

    # For us/me patterns
    (re.compile(r'\bfor\s+(?:me|myself)\b', re.I), 1),
    (re.compile(r'\bfor\s+us\b', re.I), 2),  # Default "us" to 2 if no other info
]


def extract_people_from_query(query: str) -> Optional[int]:
    """Extract people_number from query text using regex patterns.

    Returns:
        Number of people (1-8+) or None if cannot determine
    """
    if not query:
        return None

    for pattern, result in _PEOPLE_PATTERNS:
        match = pattern.search(query)
        if match:
            if isinstance(result, int):
                return result
            elif result in ['group_of_N', 'N_people', 'for_N', 'family_of_N']:
                try:
                    num = int(match.group(1))
                    if 1 <= num <= 20:  # Sanity check
                        return num
                except (ValueError, IndexError):
                    continue

    return None


def extract_budget(row: Dict[str, Any]) -> Tuple[Optional[int], str]:
    """Return (budget_anchor, source). Source in: field|query|none."""
    v = row.get("budget")
    if v is not None and v != "":
        try:
            return int(float(v)), "field"
        except Exception:
            pass

    q = row.get("query")
    if isinstance(q, str) and q:
        m = _BUDGET_RE.search(q)
        if m:
            num = m.group(1).replace(",", "")
            try:
                return int(num), "query"
            except Exception:
                pass

    return None, "none"


def rewrite_query(org: Any, dest: Any, days: int, dates: List[str], people: int, budget_anchor: Optional[int]) -> str:
    """Deterministic, self-consistent query generator (avoid contradictions in original)."""
    start = dates[0] if dates else ""
    end = dates[-1] if dates else ""
    if people <= 1:
        head = f"Please create a travel plan for me where I'll be departing from {org} and heading to {dest} for a {days}-day trip"
    else:
        head = f"Please create a travel plan for our group of {people} people where we'll be departing from {org} and heading to {dest} for a {days}-day trip"
    if start and end:
        head += f" from {start} to {end}."
    else:
        head += "."
    if budget_anchor is not None:
        head += f" Please help us keep this journey within a total budget of ${budget_anchor}."
    return head


def normalize_initial_info(
    row: Dict[str, Any],
    *,
    people_choices: List[int],
    rng,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize a TravelPlanner row into Stage1 initial_info + normalization_meta.

    Priority for people_number extraction:
    1. people_number field (if present and not None)
    2. Parse from query text
    3. Default to 1

    If people_number == 1, upsample to random choice from people_choices.
    """
    # Try field first
    people_from_field = row.get("people_number")
    people_source = "field"

    # If field is None or missing, parse from query
    if people_from_field is None:
        query = row.get("query", "")
        people_from_query = extract_people_from_query(query)
        if people_from_query is not None:
            orig_people = people_from_query
            people_source = "query"
        else:
            orig_people = 1
            people_source = "default"
    else:
        orig_people = int(people_from_field)

    # Upsampling logic: if 1 person, randomly choose from people_choices
    people = orig_people if orig_people >= 2 else rng.choice(people_choices)
    people_upsampled = (people != orig_people)

    budget_anchor, budget_src = extract_budget(row)
    orig_budget = budget_anchor

    # If we upsample group size, scale total budget proportionally (keep pppn roughly stable).
    if budget_anchor is not None and orig_people > 0 and people != orig_people:
        budget_anchor = int(round(budget_anchor * (people / orig_people)))

    dates = as_list(row.get("date"))
    query = rewrite_query(row.get("org"), row.get("dest"), int(row.get("days") or 1), dates, people, budget_anchor)

    local_anchor = as_dict(row.get("local_constraint"))
    # Ensure canonical keys exist (TravelPlanner schema-like)
    if "house rule" not in local_anchor:
        local_anchor["house rule"] = None
    if "cuisine" not in local_anchor:
        local_anchor["cuisine"] = None
    if "room type" not in local_anchor:
        local_anchor["room type"] = None
    if "transportation" not in local_anchor:
        local_anchor["transportation"] = None

    initial_info = {
        "org": row.get("org"),
        "dest": row.get("dest"),
        "days": int(row.get("days") or 1),
        "visiting_city_number": int(row.get("visiting_city_number") or 1),
        "date": dates,
        "people_number": people,
        "query": query,
        "budget_anchor": budget_anchor,
        "local_constraint_anchor": local_anchor,
        "level": row.get("level"),
    }

    meta = {
        "people_original": orig_people,
        "people_final": people,
        "people_upsampled": people_upsampled,
        "people_source": people_source,  # field | query | default
        "budget_original": orig_budget,
        "budget_final": budget_anchor,
        "budget_source": budget_src,
        "query_rewritten": True,
    }
    return initial_info, meta


def grounding_anchors(initial_info: Dict[str, Any]) -> List[str]:
    anchors: List[str] = []
    if initial_info.get("dest"):
        anchors.append(str(initial_info["dest"]))
    d = initial_info.get("date") or []
    if isinstance(d, list) and len(d) >= 2:
        anchors.append(f"Dates: {d[0]} to {d[-1]}")
    pn = initial_info.get("people_number")
    if pn is not None:
        anchors.append(f"Group size: {pn} people")
    ba = initial_info.get("budget_anchor")
    if ba is not None:
        anchors.append(f"Budget anchor: ${ba}")
    return anchors[:4]
