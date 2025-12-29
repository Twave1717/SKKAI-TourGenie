from __future__ import annotations

"""Stravl Travel Preference Data helpers.

This module focuses on:
- robust header access (BOM/whitespace/case)
- decoding the FORM_* coded values into readable labels

It is intentionally dependency-free (stdlib only).

Stravl's public documentation describes these fields (FORM_A/B/C/F/G/H/I/J).
"""

import ast
import re
from typing import Any, Dict, List, Optional


# ---------------------------
# Decoding tables (per Stravl docs)
# ---------------------------

AGE_RANGES = {
    0: "0-19",
    1: "20-39",
    2: "40-59",
    3: "60+",
}

BUDGET_PPPN = {
    0: "$0-$49",
    1: "$50-$99",
    2: "$100-$249",
    3: "$300+",
}

SEASON = {
    0: "Winter",
    1: "Spring",
    2: "Summer",
    3: "Fall",
}

EXPERIENCE = {
    0: "Beach",
    1: "Adventure",
    2: "Nature",
    3: "Culture",
    4: "Nightlife",
    5: "History",
    6: "Shopping",
    7: "Cuisine",
}

SCENERY = {
    0: "Urban",
    1: "Rural",
    2: "Sea",
    3: "Mountain",
    4: "Lake",
    5: "Desert",
    6: "Plains",
    7: "Jungle",
}

ACTIVITY = {
    0: "Chill & Relaxed",
    1: "Balanced",
    2: "Active",
}

SAFETY = {
    0: "Very Safety Conscious",
    1: "Balanced",
    2: "Ready for Anything",
}

POPULARITY = {
    0: "Off the Beaten Path",
    1: "Classic Spot",
    2: "Mainstream & Trendy",
}


def norm_key(k: Any) -> str:
    """Normalize CSV header keys for robust lookup.

    - strips whitespace
    - removes a leading UTF-8 BOM (\ufeff)
    - lowercases
    - collapses internal whitespace
    """
    if k is None:
        return ""
    s = str(k).replace("\ufeff", "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def get_ci(row: Dict[str, Any], key: str) -> Optional[str]:
    """Case-insensitive + BOM-tolerant dict access."""
    nk = norm_key(key)
    for k, v in row.items():
        if norm_key(k) == nk:
            return v
    return None


def parse_int(v: Any) -> Optional[int]:
    if v in (None, ""):
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    try:
        return int(s)
    except Exception:
        return None


def parse_int_list(v: Any) -> List[int]:
    """Parse a list-like cell into a list of ints.

    Observed patterns:
    - "[0, 1, 7]" (python-like)
    - "0" (single value)
    - "" / None
    """
    if v in (None, ""):
        return []
    if isinstance(v, list):
        out: List[int] = []
        for x in v:
            xi = parse_int(x)
            if xi is not None:
                out.append(xi)
        return out

    s = str(v).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            return parse_int_list(parsed)
        except Exception:
            return []
    # Single value fallback
    i = parse_int(s)
    return [] if i is None else [i]


def decode_form_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a Stravl CSV row (FORM_* fields) to readable labels.

    Returns a compact dict; values may be None / [] if missing.
    """
    a = parse_int_list(get_ci(row, "FORM_A"))
    b_list = parse_int_list(get_ci(row, "FORM_B"))
    c_list = parse_int_list(get_ci(row, "FORM_C"))
    f = parse_int_list(get_ci(row, "FORM_F"))
    g = parse_int_list(get_ci(row, "FORM_G"))
    h_list = parse_int_list(get_ci(row, "FORM_H"))
    i_list = parse_int_list(get_ci(row, "FORM_I"))
    j_list = parse_int_list(get_ci(row, "FORM_J"))

    b = b_list[0] if b_list else None
    c = c_list[0] if c_list else None
    h = h_list[0] if h_list else None
    i = i_list[0] if i_list else None
    j = j_list[0] if j_list else None

    return {
        "age_ranges": [AGE_RANGES[x] for x in a if x in AGE_RANGES][:3],
        "budget_pppn": BUDGET_PPPN.get(b),
        "season": SEASON.get(c),
        "experiences": [EXPERIENCE[x] for x in f if x in EXPERIENCE][:4],
        "scenery": [SCENERY[x] for x in g if x in SCENERY][:4],
        "activity_level": ACTIVITY.get(h),
        "safety_conscious": SAFETY.get(i),
        "destination_popularity": POPULARITY.get(j),
    }
