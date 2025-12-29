"""Parser and validator for LLM alpha survey responses."""

from __future__ import annotations

from typing import Any, Dict, Optional


# Expected field structure per category
EXPECTED_FIELDS = {
    "accommodations": [
        "price",
        "rating",
        "room_type",
        "house_rule",
        "minimum_nights",
        "maximum_occupancy",
        "review_score",
        "breakfast_included",
    ],
    "restaurants": [
        "price",
        "rating",
        "cuisine_type",
        "dietary_restrictions",
        "ambiance",
        "location_proximity",
    ],
    "flights": [
        "price",
        "stops",
        "class",
        "departure_time",
        "arrival_time",
        "airline_preference",
        "baggage_allowance",
    ],
    "attractions": [
        "rating",
        "popularity",
        "entry_fee",
        "activity_type",
        "accessibility",
    ],
    "intercity_travel": [
        "mode",
        "price",
        "duration",
        "comfort",
        "scenic_route",
    ],
}


def parse_alpha_response(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse and validate LLM alpha survey response.

    Args:
        response: Raw JSON response from LLM

    Returns:
        Validated and normalized alpha survey dict, or None if invalid
    """
    if not response or not isinstance(response, dict):
        return None

    parsed = {}
    total_fields = 0
    valid_fields = 0

    for category, expected_fields in EXPECTED_FIELDS.items():
        if category not in response:
            print(f"Warning: Missing category '{category}'")
            continue

        category_data = response[category]
        if not isinstance(category_data, dict):
            print(f"Warning: Category '{category}' is not a dict")
            continue

        parsed[category] = {}

        for field in expected_fields:
            total_fields += 1

            if field not in category_data:
                print(f"Warning: Missing field '{category}.{field}'")
                # Use default values
                parsed[category][field] = {
                    "value": None,
                    "importance_score": 0,
                    "reason": "Not specified by LLM",
                }
                continue

            field_data = category_data[field]
            if not isinstance(field_data, dict):
                print(f"Warning: Field '{category}.{field}' is not a dict")
                parsed[category][field] = {
                    "value": None,
                    "importance_score": 0,
                    "reason": "Invalid format",
                }
                continue

            # Validate required keys
            value = field_data.get("value")
            importance_score = field_data.get("importance_score")
            reason = field_data.get("reason", "")

            # Validate importance_score (0-10)
            if importance_score is None:
                print(f"Warning: Missing importance_score for '{category}.{field}'")
                importance_score = 0
            else:
                try:
                    importance_score = float(importance_score)
                    importance_score = max(0.0, min(10.0, importance_score))
                except (ValueError, TypeError):
                    print(f"Warning: Invalid importance_score for '{category}.{field}': {importance_score}")
                    importance_score = 0.0

            # Store validated field
            parsed[category][field] = {
                "value": value,
                "importance_score": importance_score,
                "reason": str(reason) if reason else "",
            }
            valid_fields += 1

    # Check if we have reasonable coverage (at least 70% of fields)
    coverage = valid_fields / total_fields if total_fields > 0 else 0
    if coverage < 0.7:
        print(f"Warning: Low field coverage ({coverage:.1%}), response may be incomplete")

    return parsed


def extract_alpha_values(parsed: Dict[str, Any]) -> Dict[str, float]:
    """Extract just the alpha (importance) values from parsed response.

    Args:
        parsed: Validated alpha survey dict

    Returns:
        Flat dict mapping "category.field" -> alpha value
    """
    alpha_values = {}

    for category, fields in parsed.items():
        for field_name, field_data in fields.items():
            key = f"{category}.{field_name}"
            alpha_values[key] = field_data.get("importance_score", 0.0)

    return alpha_values


def categorize_by_importance(alpha_values: Dict[str, float]) -> Dict[str, list]:
    """Categorize fields by importance level.

    Args:
        alpha_values: Flat dict of field -> alpha value

    Returns:
        Dict with categories: must_have, should_have, could_have, indifferent
    """
    categories = {
        "must_have": [],  # 9-10
        "should_have": [],  # 7-8
        "could_have": [],  # 4-6
        "indifferent": [],  # 0-3
    }

    for field, alpha in alpha_values.items():
        if alpha >= 9:
            categories["must_have"].append(field)
        elif alpha >= 7:
            categories["should_have"].append(field)
        elif alpha >= 4:
            categories["could_have"].append(field)
        else:
            categories["indifferent"].append(field)

    return categories
