from __future__ import annotations

"""Semantic validation for Stage 2 outputs.

OpenAI Structured Outputs guarantee JSON *shape*, but not instance-level semantics.
This validator catches common issues and produces error messages suitable for a
regeneration loop.

We support two modes:
- strict_values=True : try to keep constraints feasible w.r.t. the per-instance snapshot
- strict_values=False: only enforce global enums / type sanity (less restrictive)

NOTE: We intentionally keep validation lightweight (no DB queries).
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# Global enums (TravelPlanner paper strings)
CUISINES = {"Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"}
ROOM_RULES = {"No parties", "No smoking", "No children under 10", "No pets", "No visitors"}
ROOM_TYPES = {"Entire Room", "Private Room", "Shared Room", "No Shared Room"}
GROUND_MODES = {"flight", "self-driving", "taxi"}

_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


# Conservative global numeric sanity (rough, DB-wide).
# Used only when instance ranges are missing.
GLOBAL_NUM_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "restaurant.avg_cost": (0, 400),
    "restaurant.rating": (0, 5),
    "restaurant.count": (0, None),
    "accommodation.price": (0, 2000),
    "accommodation.review": (0, 5),
    "accommodation.minimum_nights": (0, 60),
    "accommodation.maximum_occupancy": (1, 20),
    "accommodation.count": (0, None),
    "attraction.count": (0, None),
    "flight.price": (0, 6000),
    "flight.duration_minutes": (0, 24 * 60),
    "flight.distance": (0, 20000),
    "flight.count": (0, None),
    "ground.cost": (0, 20000),
    "ground.duration_minutes": (0, 24 * 60),
    "ground.distance": (0, 20000),
}


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _get_range(db_snapshot: Dict[str, Any], field: str) -> Optional[Tuple[float, float]]:
    r = (db_snapshot.get("ranges") or {}).get(field)
    if isinstance(r, dict) and isinstance(r.get("min"), (int, float)) and isinstance(r.get("max"), (int, float)):
        return float(r["min"]), float(r["max"])
    g = GLOBAL_NUM_RANGES.get(field)
    if g and g[0] is not None and g[1] is not None:
        return float(g[0]), float(g[1])
    return None


def _alpha_beta_ok(alpha: Any, beta: Any) -> bool:
    if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
        return True
    # Allow small drift; prompt recommends beta = 1 - alpha
    return abs((1.0 - float(alpha)) - float(beta)) <= 0.15


def _numeric_feasible(op: str, val: Any, rng: Tuple[float, float], *, hard: bool, reject_noop: bool) -> Optional[str]:
    mn, mx = rng

    def bad(msg: str) -> str:
        return msg

    if op == "between":
        ...
        return None

    if not isinstance(val, (int, float)):
        return bad(f"Numeric value required for op={op}. Got: {val} ({type(val).__name__})")
    x = float(val)

    if op in ("<=", "<"):
        if hard and x < mn:
            return bad(f"Threshold {x} is below snapshot min {mn} (infeasible).")
        if reject_noop and hard and x > mx * 3:
            return bad(f"Threshold {x} is too loose vs snapshot max {mx} (likely unit mistake / no-op).")
        return None

    if op in (">=", ">"):
        if hard and x > mx:
            return bad(f"Threshold {x} is above snapshot max {mx} (infeasible).")
        if reject_noop and hard and mn > 0 and x < mn / 3:
            return bad(f"Threshold {x} is too loose vs snapshot min {mn} (likely no-op).")
        return None

    if op in ("==", "!="):
        if hard and (x < mn or x > mx):
            return bad(f"Value {x} is outside snapshot range [{mn},{mx}].")
        return None

    return None


def validate_stage2(
    persona_requirements: List[Dict[str, Any]],
    n_personas: int,
    db_snapshot: Dict[str, Any],
    *,
    strict_values: bool = False,
    reject_noop: bool = False,
    drop_bad: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate (or sanitize) persona_requirements.

    Returns (errors, sanitized_requirements, dropped_constraints).
    - errors: fatal issues (schema-level) that should trigger regeneration.
    - sanitized_requirements: filtered copy when drop_bad=True, else original.
    - dropped_constraints: removed constraints with reasons when drop_bad=True.
    """
    errors: List[str] = []
    dropped: List[Dict[str, Any]] = []

    # Fatal schema-level checks
    if len(persona_requirements) != n_personas:
        errors.append(f"Expected persona_requirements length {n_personas}, got {len(persona_requirements)}.")
    idxs = [pr.get("persona_index") for pr in persona_requirements]
    if any(not isinstance(i, int) for i in idxs):
        errors.append("All persona_index must be integers.")
    else:
        want = set(range(n_personas))
        got = set(i for i in idxs if isinstance(i, int))
        if got != want:
            errors.append(f"persona_index set must be exactly {sorted(want)}; got {sorted(got)}.")

    allowed = db_snapshot.get("allowed_values") or {}
    allowed_cuisines = set(allowed.get("cuisines") or [])
    allowed_room_rules = set(allowed.get("room_rules") or [])
    allowed_room_types = set(allowed.get("room_types") or [])
    allowed_modes = set(allowed.get("ground_modes") or []) or GROUND_MODES

    counts = db_snapshot.get("counts") or {}
    row_counts = {
        "restaurant.count": int(counts.get("restaurant_rows") or 0),
        "accommodation.count": int(counts.get("accommodation_rows") or 0),
        "attraction.count": int(counts.get("attraction_rows") or 0),
        "flight.count": int(counts.get("flight_rows") or 0),
    }

    numeric_fields = {
        "restaurant.avg_cost",
        "restaurant.rating",
        "restaurant.count",
        "accommodation.price",
        "accommodation.review",
        "accommodation.minimum_nights",
        "accommodation.maximum_occupancy",
        "accommodation.count",
        "attraction.count",
        "flight.price",
        "flight.duration_minutes",
        "flight.distance",
        "flight.count",
        "ground.cost",
        "ground.duration_minutes",
        "ground.distance",
    }

    keyword_fields = {
        "attraction.name_keyword",
        "attraction.address_keyword",
        "restaurant.name_keyword",
        "accommodation.name_keyword",
    }

    sanitized: List[Dict[str, Any]] = []

    for pr in persona_requirements:
        sr = pr.get("structured_requirement") or {}
        new_sr: Dict[str, Any] = {"hard_constraints": [], "soft_constraints": []}

        for bucket in ("hard_constraints", "soft_constraints"):
            cons = sr.get(bucket) or []
            if not isinstance(cons, list):
                errors.append(f"{bucket} must be a list.")
                continue

            for c in cons:
                if not isinstance(c, dict):
                    if drop_bad:
                        dropped.append({"reason": "constraint_not_object", "constraint": c, "bucket": bucket, "persona_index": pr.get("persona_index")})
                        continue
                    errors.append("Constraint items must be objects.")
                    continue

                field = c.get("field")
                op = c.get("op")
                val = c.get("value")
                hard = c.get("hard")
                alpha = c.get("alpha")
                beta = c.get("beta")
                desc = c.get("description")

                def drop(reason: str) -> None:
                    dropped.append(
                        {
                            "reason": reason,
                            "field": field,
                            "op": op,
                            "value": val,
                            "bucket": bucket,
                            "persona_index": pr.get("persona_index"),
                        }
                    )

                bad = False

                if not isinstance(desc, str) or len(desc.strip()) < 5:
                    bad = True
                    drop("missing_description") if drop_bad else errors.append(f"Constraint.description must be a non-empty string (len>=5). field={field}")

                if not _alpha_beta_ok(alpha, beta):
                    bad = True
                    drop("alpha_beta_inconsistent") if drop_bad else errors.append(f"alpha/beta inconsistent (field={field}, alpha={alpha}, beta={beta}). Use beta≈1-alpha.")

                if hard is True and isinstance(alpha, (int, float)) and float(alpha) < 0.80:
                    bad = True
                    drop("hard_alpha_too_low") if drop_bad else errors.append(f"Hard constraint alpha too low (field={field}, alpha={alpha}).")
                if hard is True and isinstance(beta, (int, float)) and float(beta) > 0.25:
                    bad = True
                    drop("hard_beta_too_high") if drop_bad else errors.append(f"Hard constraint beta too high (field={field}, beta={beta}).")
                if hard is False and isinstance(alpha, (int, float)) and float(alpha) > 0.90:
                    bad = True
                    drop("soft_alpha_too_high") if drop_bad else errors.append(f"Soft constraint alpha too high (field={field}, alpha={alpha}).")

                if field == "restaurant.cuisine":
                    allowed_set = allowed_cuisines if (strict_values and allowed_cuisines) else CUISINES
                    for x in _as_list(val):
                        if isinstance(x, str) and x not in allowed_set:
                            bad = True
                            drop("cuisine_not_allowed") if drop_bad else errors.append(f"Invalid cuisine '{x}'. Allowed: {sorted(allowed_set)}")

                if field == "accommodation.room_type":
                    allowed_set = allowed_room_types if (strict_values and allowed_room_types) else ROOM_TYPES
                    for x in _as_list(val):
                        if isinstance(x, str) and x not in allowed_set:
                            bad = True
                            drop("room_type_not_allowed") if drop_bad else errors.append(f"Invalid room type '{x}'. Allowed: {sorted(allowed_set)}")

                if field == "accommodation.house_rule":
                    allowed_set = allowed_room_rules if (strict_values and allowed_room_rules) else ROOM_RULES
                    for x in _as_list(val):
                        if isinstance(x, str) and x not in allowed_set:
                            bad = True
                            drop("house_rule_not_allowed") if drop_bad else errors.append(f"Invalid room rule '{x}'. Allowed: {sorted(allowed_set)}")

                if field == "ground.mode":
                    for x in _as_list(val):
                        if isinstance(x, str) and x not in allowed_modes:
                            bad = True
                            drop("ground_mode_not_allowed") if drop_bad else errors.append(f"Invalid ground mode '{x}'. Allowed: {sorted(allowed_modes)}")

                if field in ("flight.depart_time", "flight.arrive_time"):
                    if op == "between":
                        if not isinstance(val, list) or len(val) != 2 or any(not isinstance(t, str) or not _TIME_RE.match(t) for t in val):
                            bad = True
                            drop("time_format") if drop_bad else errors.append(f"Time 'between' must use [HH:MM, HH:MM]. Got: {val}")
                    else:
                        if not isinstance(val, str) or not _TIME_RE.match(val):
                            bad = True
                            drop("time_format") if drop_bad else errors.append(f"Invalid time '{val}'. Must be HH:MM")

                if isinstance(field, str) and field in numeric_fields:
                    if strict_values and field in row_counts and isinstance(val, (int, float)):
                        if op in (">=", ">") and float(val) > float(row_counts[field]) and hard is True:
                            bad = True
                            drop("count_infeasible") if drop_bad else errors.append(f"Count constraint infeasible: {field} {op} {val} but snapshot has only {row_counts[field]} rows.")

                    rng = _get_range(db_snapshot, field)
                    if rng:
                        err = _numeric_feasible(str(op), val, rng, hard=bool(hard), reject_noop=reject_noop)
                        if err:
                            bad = True
                            drop("numeric_infeasible_or_noop") if drop_bad else errors.append(f"{err} (field={field}, op={op})")

                if field in keyword_fields:
                    for x in _as_list(val):
                        if not isinstance(x, str) or len(x.strip()) < 2:
                            bad = True
                            drop("keyword_not_string") if drop_bad else errors.append(f"Invalid keyword '{x}'. Use non-empty strings (len>=2).")

                if drop_bad and not bad:
                    new_sr[bucket].append(c)
                if (not drop_bad) and not bad:
                    new_sr[bucket].append(c)

        sanitized.append({"persona_index": pr.get("persona_index"), "structured_requirement": new_sr})

    return errors, (persona_requirements if not drop_bad else sanitized), dropped
