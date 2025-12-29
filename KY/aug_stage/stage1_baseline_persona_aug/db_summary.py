from __future__ import annotations

"""Closed-world DB summary extractor for TravelPlanner `reference_information`.

TravelPlanner rows include a large `reference_information` field (stringified list of
tables: attractions, restaurants, accommodations, flights, etc). We do NOT want to
dump that whole blob into Stage1 prompts, but we *do* want compact signals for:

- debugging (what DB evidence exists)
- grounding (which cuisines / room rules / room types appear in the DB tables)

This module extracts a small summary dict.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Set


# TravelPlanner paper enum strings (keep verbatim)
CUISINE_ENUM = ["Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"]
ROOM_RULE_ENUM = ["No parties", "No smoking", "No children under 10", "No pets", "No visitors"]
ROOM_TYPE_MAP = {
    "entire home/apt": "Entire Room",
    "private room": "Private Room",
    "shared room": "Shared Room",
}


def _safe_literal(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in "[{(":
            try:
                return ast.literal_eval(s)
            except Exception:
                return v
    return v


def extract_db_summary(reference_information: Any) -> Dict[str, Any]:
    """Return a compact summary dict (all values JSON-serializable)."""
    info = _safe_literal(reference_information)
    if not isinstance(info, list):
        return {
            "ref_types": [],
            "cuisines_available": [],
            "room_rules_available": [],
            "room_types_available": [],
            "counts": {},
        }

    ref_types: List[str] = []
    cuisines: Set[str] = set()
    room_rules: Set[str] = set()
    room_types: Set[str] = set()
    counts: Dict[str, int] = {}

    for entry in info:
        if not isinstance(entry, dict):
            continue
        desc = str(entry.get("Description") or "")
        content = str(entry.get("Content") or "")
        if desc:
            ref_types.append(desc)

        # crude type buckets
        bucket = "other"
        dlow = desc.lower()
        if "restaurants" in dlow:
            bucket = "restaurants"
        elif "accommodations" in dlow:
            bucket = "accommodations"
        elif "attractions" in dlow:
            bucket = "attractions"
        elif dlow.startswith("flight from") or "flight" in dlow:
            bucket = "flights"
        elif "taxi" in dlow:
            bucket = "taxi"
        elif "self-driving" in dlow:
            bucket = "self_driving"

        # approximate row counts (lines minus header)
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if bucket in ("restaurants", "accommodations", "attractions", "flights"):
            counts[bucket] = counts.get(bucket, 0) + max(0, len(lines) - 1)
        else:
            counts[bucket] = counts.get(bucket, 0) + (1 if content.strip() else 0)

        # enum extraction by substring match (robust to table formatting)
        for c in CUISINE_ENUM:
            if re.search(r"\b" + re.escape(c) + r"\b", content, flags=re.IGNORECASE):
                cuisines.add(c)
        for rr in ROOM_RULE_ENUM:
            if rr.lower() in content.lower():
                room_rules.add(rr)
        for raw, norm in ROOM_TYPE_MAP.items():
            if raw in content.lower():
                room_types.add(norm)

    return {
        "ref_types": sorted(set(ref_types)),
        "cuisines_available": sorted(cuisines),
        "room_rules_available": sorted(room_rules),
        "room_types_available": sorted(room_types),
        "counts": counts,
    }
