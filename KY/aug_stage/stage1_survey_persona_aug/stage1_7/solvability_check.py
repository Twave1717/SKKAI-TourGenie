"""Solvability checking for persona conflict groups."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def extract_alpha_values(persona: Dict[str, Any]) -> Dict[str, float]:
    """Extract flat alpha values from enriched persona.

    Args:
        persona: Persona with alpha_survey field

    Returns:
        Dict mapping "category.field" -> alpha value
    """
    alpha_survey = persona.get("alpha_survey", {})
    alpha_values = {}

    for category, fields in alpha_survey.items():
        if not isinstance(fields, dict):
            continue
        for field_name, field_data in fields.items():
            if not isinstance(field_data, dict):
                continue
            key = f"{category}.{field_name}"
            alpha_values[key] = field_data.get("importance_score", 0.0)

    return alpha_values


def check_field_conflict(
    personas: List[Dict[str, Any]], field: str
) -> Tuple[bool, Dict[str, Any]]:
    """Check if a specific field has conflict across personas.

    Args:
        personas: List of enriched personas
        field: Field key like "accommodations.price"

    Returns:
        Tuple of (has_conflict, conflict_info)
    """
    category, field_name = field.split(".", 1)

    # Extract values and alphas for this field
    persona_data = []
    for i, persona in enumerate(personas):
        alpha_survey = persona.get("alpha_survey", {})
        field_data = alpha_survey.get(category, {}).get(field_name, {})

        value = field_data.get("value")
        alpha = field_data.get("importance_score", 0.0)

        persona_data.append(
            {
                "persona_idx": i,
                "persona_id": persona.get("ref_id", f"persona_{i}"),
                "value": value,
                "alpha": alpha,
            }
        )

    # Check for hard constraint conflicts (α≥9)
    hard_constraints = [p for p in persona_data if p["alpha"] >= 9]

    if len(hard_constraints) >= 2:
        # Check if hard constraints conflict
        values = [p["value"] for p in hard_constraints]
        unique_values = set(str(v) for v in values if v is not None)

        if len(unique_values) > 1:
            # Hard constraint conflict - NOT SOLVABLE
            return False, {
                "field": field,
                "conflict_type": "hard_constraint_conflict",
                "personas": hard_constraints,
                "reason": "Multiple personas have conflicting hard constraints (α≥9)",
            }

    # Check for soft constraint conflicts (4≤α<9)
    soft_constraints = [p for p in persona_data if 4 <= p["alpha"] < 9]

    if len(soft_constraints) >= 2:
        # Check if soft constraints differ
        values = [p["value"] for p in soft_constraints]
        unique_values = set(str(v) for v in values if v is not None)

        if len(unique_values) > 1:
            # Soft constraint conflict - SOLVABLE
            return True, {
                "field": field,
                "conflict_type": "soft_constraint_conflict",
                "personas": soft_constraints,
                "reason": "Multiple personas have different soft preferences (4≤α<9)",
            }

    # No meaningful conflict
    return False, {
        "field": field,
        "conflict_type": "no_conflict",
        "reason": "No conflicting preferences",
    }


def analyze_conflict_dimensions(personas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze all conflict dimensions for a persona group.

    Args:
        personas: List of enriched personas

    Returns:
        Analysis dict with conflict counts and details
    """
    if not personas:
        return {
            "is_solvable": False,
            "reason": "No personas provided",
            "hard_conflicts": [],
            "soft_conflicts": [],
            "no_conflicts": [],
        }

    # Get all possible fields from first persona
    first_persona = personas[0]
    alpha_survey = first_persona.get("alpha_survey", {})

    all_fields = []
    for category, fields in alpha_survey.items():
        if isinstance(fields, dict):
            for field_name in fields.keys():
                all_fields.append(f"{category}.{field_name}")

    # Check each field
    hard_conflicts = []
    soft_conflicts = []
    no_conflicts = []

    for field in all_fields:
        has_conflict, conflict_info = check_field_conflict(personas, field)

        if conflict_info["conflict_type"] == "hard_constraint_conflict":
            hard_conflicts.append(conflict_info)
        elif conflict_info["conflict_type"] == "soft_constraint_conflict":
            soft_conflicts.append(conflict_info)
        else:
            no_conflicts.append(conflict_info)

    # Determine solvability
    is_solvable = len(hard_conflicts) == 0 and len(soft_conflicts) >= 2

    if len(hard_conflicts) > 0:
        reason = f"Has {len(hard_conflicts)} hard constraint conflicts (NOT solvable)"
    elif len(soft_conflicts) < 2:
        reason = f"Only {len(soft_conflicts)} soft conflicts (need ≥2 for interesting conflict)"
    else:
        reason = f"Solvable: 0 hard conflicts, {len(soft_conflicts)} soft conflicts"

    return {
        "is_solvable": is_solvable,
        "reason": reason,
        "hard_conflicts": hard_conflicts,
        "soft_conflicts": soft_conflicts,
        "no_conflicts": no_conflicts,
        "conflict_count": len(soft_conflicts),
    }


def check_solvable_conflict(personas: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Quick check if a persona group has solvable conflict.

    Args:
        personas: List of enriched personas

    Returns:
        Tuple of (is_solvable, reason)
    """
    analysis = analyze_conflict_dimensions(personas)
    return analysis["is_solvable"], analysis["reason"]
