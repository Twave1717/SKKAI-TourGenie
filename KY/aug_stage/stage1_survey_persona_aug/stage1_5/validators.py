"""Validators for alpha survey preferences against TravelPlanner database.

This module contains validation functions for all 20 questions.
Each validator takes alpha survey preferences and actual selected items (from DB) and returns
a compliance score (0-1) or boolean pass/fail.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path


# =============================================================================
# SECTION 1: ACCOMMODATIONS VALIDATORS
# =============================================================================


def validate_accommodation_price(
    preference: str,
    actual_prices: List[float]
) -> Tuple[bool, str]:
    """Q1-1: Accommodation price tier preference.

    Args:
        preference: Price tier string (e.g., "Economy (Under $80)", "Budget ($80-90)", etc.)
        actual_prices: List of accommodation prices selected

    Returns:
        (pass/fail, reason)
    """
    if not actual_prices:
        return True, "No accommodation data"

    if "flexible" in preference.lower() or "no preference" in preference.lower():
        return True, "Flexible pricing preference"

    avg_price = sum(actual_prices) / len(actual_prices)

    # Extract price range from preference string
    import re
    # Match patterns like "Under $80", "$80-90", "Over $400"
    if "under" in preference.lower():
        match = re.search(r'\$(\d+)', preference)
        if match:
            max_price = float(match.group(1))
            if avg_price <= max_price:
                return True, f"Avg ${avg_price:.0f} under ${max_price:.0f}"
            else:
                return False, f"Avg ${avg_price:.0f} exceeds ${max_price:.0f}"
    elif "over" in preference.lower():
        match = re.search(r'\$(\d+)', preference)
        if match:
            min_price = float(match.group(1))
            if avg_price >= min_price:
                return True, f"Avg ${avg_price:.0f} over ${min_price:.0f}"
            else:
                return False, f"Avg ${avg_price:.0f} below ${min_price:.0f}"
    else:
        # Range format: $80-90
        matches = re.findall(r'\$(\d+)', preference)
        if len(matches) >= 2:
            min_price, max_price = float(matches[0]), float(matches[1])
            if min_price <= avg_price <= max_price:
                return True, f"Avg ${avg_price:.0f} in range ${min_price:.0f}-${max_price:.0f}"
            else:
                return False, f"Avg ${avg_price:.0f} outside ${min_price:.0f}-${max_price:.0f}"

    return True, "Cannot parse price preference"


def validate_accommodation_rating(
    min_rating: float,
    actual_ratings: List[float]
) -> Tuple[bool, str]:
    """Q1-2: Minimum accommodation star rating.

    Args:
        min_rating: Minimum acceptable rating (0-5)
        actual_ratings: List of accommodation ratings

    Returns:
        (pass/fail, reason)
    """
    if not actual_ratings:
        return True, "No accommodation data"

    if min_rating <= 0:
        return True, "No minimum rating requirement"

    min_actual = min(actual_ratings)

    if min_actual >= min_rating:
        return True, f"All ratings >= {min_rating:.1f} (lowest: {min_actual:.1f})"
    else:
        return False, f"Some ratings below {min_rating:.1f} (lowest: {min_actual:.1f})"


def validate_room_type(
    preference: str,
    actual_room_types: List[str]
) -> Tuple[bool, str]:
    """Q1-3: Room type requirement.

    Args:
        preference: "entire home/apt", "private room", "shared room", "any/flexible"
        actual_room_types: List of room types selected

    Returns:
        (pass/fail, reason)
    """
    if not actual_room_types:
        return True, "No accommodation data"

    pref_lower = preference.lower()

    if "any" in pref_lower or "flexible" in pref_lower:
        return True, "Flexible room type"

    # Keyword matching
    keywords = {
        "entire home/apt": ["entire", "apartment", "home"],
        "private room": ["private"],
        "shared room": ["shared", "dormitory", "hostel"]
    }

    required_keywords = None
    for key, kws in keywords.items():
        if any(kw in pref_lower for kw in kws):
            required_keywords = kws
            break

    if required_keywords:
        matches = 0
        for room_type in actual_room_types:
            room_lower = room_type.lower()
            if any(kw in room_lower for kw in required_keywords):
                matches += 1

        if matches == len(actual_room_types):
            return True, f"All {len(actual_room_types)} rooms match {preference}"
        else:
            return False, f"Only {matches}/{len(actual_room_types)} rooms match {preference}"

    return True, "Unknown preference"


def validate_house_rules(
    required_rules: List[str],
    actual_house_rules: List[str]
) -> Tuple[bool, str]:
    """Q1-4: House rules requirements.

    Args:
        required_rules: List of required/banned rules (e.g., ["Non-smoking", "Must allow Pets"])
        actual_house_rules: List of house rules from selected accommodations

    Returns:
        (pass/fail, reason)
    """
    if not required_rules or "no specific requirements" in str(required_rules).lower():
        return True, "No specific house rules requirements"

    if not actual_house_rules:
        return True, "No house rules data"

    # Combine all rules into searchable text
    all_rules_text = " ".join([str(r).lower() for r in actual_house_rules])

    violations = []
    for rule in required_rules:
        rule_lower = rule.lower()

        # Handle "Must" vs "Must forbid"
        if "must allow" in rule_lower:
            keyword = rule_lower.split("must allow")[1].strip()
            if keyword not in all_rules_text:
                violations.append(f"Missing: {rule}")
        elif "must forbid" in rule_lower:
            keyword = rule_lower.split("must forbid")[1].strip()
            if keyword in all_rules_text:
                violations.append(f"Violated: {rule}")
        elif "must be" in rule_lower:
            keyword = rule_lower.split("must be")[1].strip()
            if keyword not in all_rules_text:
                violations.append(f"Missing: {rule}")

    if violations:
        return False, f"Rule violations: {', '.join(violations)}"
    else:
        return True, f"All {len(required_rules)} house rules satisfied"


def validate_minimum_nights(
    flexibility: str,
    actual_min_nights: List[int],
    trip_days: int
) -> Tuple[bool, str]:
    """Q1-5: Minimum nights flexibility.

    Args:
        flexibility: "very flexible (3+ nights)", "somewhat flexible (max 2)", "not flexible (1 night)", "no preference"
        actual_min_nights: List of minimum nights requirements
        trip_days: Total trip duration in days

    Returns:
        (pass/fail, reason)
    """
    if not actual_min_nights:
        return True, "No accommodation data"

    if "no preference" in flexibility.lower():
        return True, "No preference"

    max_min_nights = max(actual_min_nights)

    if "very flexible" in flexibility.lower() or "3+ nights" in flexibility.lower():
        # Can stay 3+ nights in one place
        if max_min_nights <= trip_days:
            return True, f"Max min-nights {max_min_nights} acceptable for {trip_days}-day trip"
        else:
            return False, f"Max min-nights {max_min_nights} exceeds trip duration {trip_days}"

    elif "somewhat flexible" in flexibility.lower() or "max 2" in flexibility.lower():
        # Max 2 nights per place
        if max_min_nights <= 2:
            return True, f"Max min-nights {max_min_nights} within 2-night limit"
        else:
            return False, f"Max min-nights {max_min_nights} exceeds 2-night limit"

    elif "not flexible" in flexibility.lower() or "1 night" in flexibility.lower():
        # 1 night max
        if max_min_nights <= 1:
            return True, f"Max min-nights {max_min_nights} = 1 night"
        else:
            return False, f"Max min-nights {max_min_nights} exceeds 1-night limit"

    return True, "Unknown preference"


def validate_maximum_occupancy(
    required_capacity: int,
    actual_capacities: List[int]
) -> Tuple[bool, str]:
    """Q1-6: Maximum occupancy requirement.

    Args:
        required_capacity: Minimum required capacity
        actual_capacities: List of maximum occupancy values

    Returns:
        (pass/fail, reason)
    """
    if not actual_capacities:
        return True, "No accommodation data"

    if required_capacity <= 0:
        return True, "No capacity requirement"

    min_capacity = min(actual_capacities)

    if min_capacity >= required_capacity:
        return True, f"All rooms accommodate {required_capacity}+ people (min: {min_capacity})"
    else:
        return False, f"Some rooms too small (min capacity: {min_capacity}, need: {required_capacity})"


def validate_accommodation_review_score(
    min_score: float,
    actual_scores: List[float]
) -> Tuple[bool, str]:
    """Q1-7: Minimum accommodation review score.

    Args:
        min_score: Minimum acceptable review score (out of 5.0)
        actual_scores: List of review scores

    Returns:
        (pass/fail, reason)
    """
    if not actual_scores:
        return True, "No accommodation data"

    if min_score <= 0:
        return True, "No minimum score requirement"

    min_actual = min(actual_scores)

    if min_actual >= min_score:
        return True, f"All scores >= {min_score:.1f} (lowest: {min_actual:.1f})"
    else:
        return False, f"Some scores below {min_score:.1f} (lowest: {min_actual:.1f})"


# =============================================================================
# SECTION 2: RESTAURANTS VALIDATORS
# =============================================================================


def validate_restaurant_price(
    preference: str,
    actual_prices: List[float]
) -> Tuple[bool, str]:
    """Q2-1: Restaurant price tier preference.

    Args:
        preference: Price tier string (e.g., "Economy (Under $10)", etc.)
        actual_prices: List of restaurant average costs per person

    Returns:
        (pass/fail, reason)
    """
    if not actual_prices:
        return True, "No restaurant data"

    if "flexible" in preference.lower() or "no preference" in preference.lower():
        return True, "Flexible pricing preference"

    avg_price = sum(actual_prices) / len(actual_prices)

    # Extract price range from preference string
    import re
    if "under" in preference.lower():
        match = re.search(r'\$(\d+)', preference)
        if match:
            max_price = float(match.group(1))
            if avg_price <= max_price:
                return True, f"Avg ${avg_price:.0f} under ${max_price:.0f}"
            else:
                return False, f"Avg ${avg_price:.0f} exceeds ${max_price:.0f}"
    elif "over" in preference.lower():
        match = re.search(r'\$(\d+)', preference)
        if match:
            min_price = float(match.group(1))
            if avg_price >= min_price:
                return True, f"Avg ${avg_price:.0f} over ${min_price:.0f}"
            else:
                return False, f"Avg ${avg_price:.0f} below ${min_price:.0f}"
    else:
        matches = re.findall(r'\$(\d+)', preference)
        if len(matches) >= 2:
            min_price, max_price = float(matches[0]), float(matches[1])
            if min_price <= avg_price <= max_price:
                return True, f"Avg ${avg_price:.0f} in range ${min_price:.0f}-${max_price:.0f}"
            else:
                return False, f"Avg ${avg_price:.0f} outside ${min_price:.0f}-${max_price:.0f}"

    return True, "Cannot parse price preference"


def validate_restaurant_rating(
    min_rating: float,
    actual_ratings: List[float]
) -> Tuple[bool, str]:
    """Q2-2: Minimum restaurant rating.

    Args:
        min_rating: Minimum acceptable rating (out of 5.0)
        actual_ratings: List of restaurant aggregate ratings

    Returns:
        (pass/fail, reason)
    """
    if not actual_ratings:
        return True, "No restaurant data"

    if min_rating <= 0:
        return True, "No minimum rating requirement"

    min_actual = min(actual_ratings)

    if min_actual >= min_rating:
        return True, f"All ratings >= {min_rating:.1f} (lowest: {min_actual:.1f})"
    else:
        return False, f"Some ratings below {min_rating:.1f} (lowest: {min_actual:.1f})"


def validate_cuisine_type(
    mandatory_cuisines: List[str],
    banned_cuisines: List[str],
    actual_cuisines: List[str]
) -> Tuple[bool, str]:
    """Q2-3: Cuisine type requirements (mandatory/banned).

    Args:
        mandatory_cuisines: List of cuisines that MUST be included
        banned_cuisines: List of cuisines that MUST be avoided
        actual_cuisines: List of selected restaurant cuisines

    Returns:
        (pass/fail, reason)
    """
    if not actual_cuisines:
        return True, "No restaurant data"

    # Combine all cuisines into searchable text
    all_cuisines_lower = " ".join([c.lower() for c in actual_cuisines])

    # Check mandatory cuisines
    if mandatory_cuisines and "no strict restrictions" not in str(mandatory_cuisines).lower():
        missing = []
        for cuisine in mandatory_cuisines:
            if cuisine.lower() not in all_cuisines_lower:
                missing.append(cuisine)

        if missing:
            return False, f"Missing mandatory cuisines: {', '.join(missing)}"

    # Check banned cuisines
    if banned_cuisines and "no strict restrictions" not in str(banned_cuisines).lower():
        violations = []
        for cuisine in banned_cuisines:
            if cuisine.lower() in all_cuisines_lower:
                violations.append(cuisine)

        if violations:
            return False, f"Banned cuisines found: {', '.join(violations)}"

    return True, "All cuisine requirements satisfied"


# =============================================================================
# SOFT CONSTRAINT VALIDATORS (Indirect Verification)
# =============================================================================


def validate_dining_ambiance(
    preference: str, restaurant_names: List[str], cuisines_list: List[str]
) -> Tuple[bool, str]:
    """Q2-4: Dining atmosphere via name/cuisine pattern matching.

    Args:
        preference: "casual", "family_friendly", "lively", "fine_dining", "no_preference"
        restaurant_names: List of restaurant names
        cuisines_list: List of cuisines

    Returns:
        (pass/fail, reason)
    """
    if preference.lower() in ["no_preference", "no preference"]:
        return True, "No preference"

    # Pattern matching heuristics
    patterns = {
        "casual": ["street", "food truck", "fast food", "cafe", "diner"],
        "family_friendly": ["family", "buffet", "casual", "american", "italian"],
        "lively": ["bar", "pub", "grill", "sports", "brewery", "tavern"],
        "fine_dining": ["fine", "steakhouse", "french", "italian", "seafood", "sushi"],
    }

    pref_lower = preference.lower().replace(" / ", "_").replace("-", "_").replace(" ", "_")
    if pref_lower not in patterns:
        return True, f"Unknown preference: {preference}"

    search_patterns = patterns[pref_lower]
    combined_text = " ".join([n.lower() for n in restaurant_names] + [c.lower() for c in cuisines_list])

    matches = sum(1 for p in search_patterns if p in combined_text)

    if matches > 0:
        return True, f"Found {matches} matching pattern(s)"
    else:
        return False, f"No {preference} atmosphere detected"


def validate_location_proximity(
    preference: str,
    hotel_coords: List[Tuple[float, float]],
    restaurant_coords: List[Tuple[float, float]],
    distance_km_threshold: float = 1.0  # Walking distance ~10 min
) -> Tuple[bool, str]:
    """Q2-5: Location convenience via distance calculation.

    Args:
        preference: "walking_distance", "short_drive", "willing_to_travel", "no_preference"
        hotel_coords: List of (lat, lon) for hotels
        restaurant_coords: List of (lat, lon) for restaurants
        distance_km_threshold: Max distance for "walking distance" in km

    Returns:
        (pass/fail, reason)
    """
    if preference.lower() in ["no_preference", "no preference", "willing_to_travel", "willing to travel for good food"]:
        return True, "No strict proximity requirement"

    if not hotel_coords or not restaurant_coords:
        return True, "Insufficient coordinate data"

    # Calculate min distance from any hotel to any restaurant
    min_dist = float('inf')
    for h_lat, h_lon in hotel_coords:
        for r_lat, r_lon in restaurant_coords:
            # Haversine distance (simplified)
            dist = ((h_lat - r_lat)**2 + (h_lon - r_lon)**2)**0.5 * 111  # Approx km
            min_dist = min(min_dist, dist)

    thresholds = {
        "walking_distance": 1.0,
        "must be walking distance (<10 min)": 1.0,
        "short_drive": 5.0,
        "short drive/transit ok (<20 min)": 5.0,
    }

    threshold = thresholds.get(preference.lower(), 1.0)

    if min_dist <= threshold:
        return True, f"Min distance: {min_dist:.2f} km (within {threshold} km)"
    else:
        return False, f"Min distance: {min_dist:.2f} km (exceeds {threshold} km)"


# =============================================================================
# SECTION 3: FLIGHTS VALIDATORS
# =============================================================================


def validate_flight_price(
    preference: str,
    actual_prices: List[float]
) -> Tuple[bool, str]:
    """Q3-1: Flight price tier preference.

    Args:
        preference: "super saver", "economy standard", "premium economy", "business/first", "flexible"
        actual_prices: List of flight prices

    Returns:
        (pass/fail, reason)
    """
    if not actual_prices:
        return True, "No flight data"

    if "flexible" in preference.lower() or "no preference" in preference.lower():
        return True, "Flexible pricing preference"

    avg_price = sum(actual_prices) / len(actual_prices)
    pref_lower = preference.lower()

    # Price tier thresholds (heuristic)
    if "super saver" in pref_lower or "cheapest" in pref_lower:
        # Expect lowest 20% of prices
        if avg_price < 200:
            return True, f"Avg ${avg_price:.0f} is budget-friendly"
        else:
            return False, f"Avg ${avg_price:.0f} too high for super saver"

    elif "economy standard" in pref_lower or "average" in pref_lower:
        # Mid-range $200-500
        if 200 <= avg_price <= 500:
            return True, f"Avg ${avg_price:.0f} in economy range"
        else:
            return False, f"Avg ${avg_price:.0f} outside economy range ($200-500)"

    elif "premium economy" in pref_lower or "comfort" in pref_lower:
        # $500-1000
        if 500 <= avg_price <= 1000:
            return True, f"Avg ${avg_price:.0f} in premium economy range"
        else:
            return False, f"Avg ${avg_price:.0f} outside premium range ($500-1000)"

    elif "business" in pref_lower or "first" in pref_lower or "luxury" in pref_lower:
        # $1000+
        if avg_price >= 1000:
            return True, f"Avg ${avg_price:.0f} in business/first range"
        else:
            return False, f"Avg ${avg_price:.0f} below business/first range ($1000+)"

    return True, "Unknown preference"


def validate_flight_layovers(
    preference: str,
    flights_data: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Q3-2: Layover tolerance via flight data analysis.

    Args:
        preference: "direct_only", "one_stop", "multiple_stops", "no_preference"
        flights_data: List of flight dicts with origin, dest, stops

    Returns:
        (pass/fail, reason)
    """
    if preference.lower() in ["no_preference", "no preference"]:
        return True, "No preference"

    if not flights_data:
        return True, "No flight data"

    # Count flights with layovers (if same origin-dest pair has multiple entries = layover)
    route_counts = {}
    for flight in flights_data:
        route = (flight.get("OriginCityName", ""), flight.get("DestCityName", ""))
        route_counts[route] = route_counts.get(route, 0) + 1

    max_stops = max(route_counts.values()) - 1  # -1 because first flight = 0 stops

    limits = {
        "direct_only": 0,
        "direct flights only": 0,
        "one_stop": 1,
        "direct preferred, 1 stop acceptable": 1,
        "multiple_stops": 2,
        "multiple stops acceptable (for savings)": 2,
    }

    limit = limits.get(preference.lower(), 0)

    if max_stops <= limit:
        return True, f"Max stops: {max_stops} (within {limit})"
    else:
        return False, f"Max stops: {max_stops} (exceeds {limit})"


def validate_baggage_allowance(
    preference: str,
    flight_prices: List[float],
    avg_price_threshold: float = 150.0  # Below this = likely budget airline
) -> Tuple[bool, str]:
    """Q3-5: Baggage allowance via flight price heuristic.

    Args:
        preference: "checked_bag", "carry_on_only", "no_preference"
        flight_prices: List of flight prices
        avg_price_threshold: Below this price, assume no checked bag included

    Returns:
        (pass/fail, reason)
    """
    if preference.lower() in ["no_preference", "no preference", "carry_on_only", "carry-on only is fine"]:
        return True, "Flexible on baggage"

    if not flight_prices:
        return True, "No flight price data"

    avg_price = sum(flight_prices) / len(flight_prices)

    if preference.lower() in ["checked_bag", "must include checked bag"]:
        if avg_price >= avg_price_threshold:
            return True, f"Avg price ${avg_price:.0f} likely includes baggage"
        else:
            return False, f"Avg price ${avg_price:.0f} likely budget airline (no baggage)"

    return True, "Unknown preference"


def validate_departure_time(
    preference: str,
    actual_departure_times: List[str]
) -> Tuple[bool, str]:
    """Q3-3: Preferred departure time window.

    Args:
        preference: "morning (06:00-12:00)", "afternoon (12:00-18:00)", "evening/night (18:00+)", "flexible"
        actual_departure_times: List of departure times (format: "HH:MM" or datetime strings)

    Returns:
        (pass/fail, reason)
    """
    if not actual_departure_times:
        return True, "No flight data"

    if "flexible" in preference.lower() or "no preference" in preference.lower():
        return True, "Flexible departure time"

    # Parse times to hour values
    def parse_hour(time_str: str) -> int:
        """Extract hour from time string."""
        import re
        # Try to extract HH:MM pattern
        match = re.search(r'(\d{1,2}):(\d{2})', str(time_str))
        if match:
            return int(match.group(1))
        # Try to extract just hour
        match = re.search(r'(\d{1,2})', str(time_str))
        if match:
            return int(match.group(1))
        return -1

    hours = [parse_hour(t) for t in actual_departure_times]
    hours = [h for h in hours if 0 <= h <= 23]

    if not hours:
        return True, "Cannot parse departure times"

    pref_lower = preference.lower()

    # Define time windows
    if "morning" in pref_lower:
        # 06:00-12:00
        in_window = sum(1 for h in hours if 6 <= h < 12)
        if in_window == len(hours):
            return True, f"All {len(hours)} flights in morning window (06:00-12:00)"
        else:
            return False, f"Only {in_window}/{len(hours)} flights in morning window"

    elif "afternoon" in pref_lower:
        # 12:00-18:00
        in_window = sum(1 for h in hours if 12 <= h < 18)
        if in_window == len(hours):
            return True, f"All {len(hours)} flights in afternoon window (12:00-18:00)"
        else:
            return False, f"Only {in_window}/{len(hours)} flights in afternoon window"

    elif "evening" in pref_lower or "night" in pref_lower:
        # 18:00+
        in_window = sum(1 for h in hours if h >= 18)
        if in_window == len(hours):
            return True, f"All {len(hours)} flights in evening/night window (18:00+)"
        else:
            return False, f"Only {in_window}/{len(hours)} flights in evening/night window"

    return True, "Unknown preference"


def validate_arrival_time(
    preference: str,
    actual_arrival_times: List[str]
) -> Tuple[bool, str]:
    """Q3-4: Preferred arrival time window.

    Args:
        preference: "morning (arrive early)", "afternoon (check-in time)", "evening/night (sleep immediately)", "flexible"
        actual_arrival_times: List of arrival times (format: "HH:MM" or datetime strings)

    Returns:
        (pass/fail, reason)
    """
    if not actual_arrival_times:
        return True, "No flight data"

    if "flexible" in preference.lower() or "no preference" in preference.lower():
        return True, "Flexible arrival time"

    # Parse times to hour values
    def parse_hour(time_str: str) -> int:
        """Extract hour from time string."""
        import re
        match = re.search(r'(\d{1,2}):(\d{2})', str(time_str))
        if match:
            return int(match.group(1))
        match = re.search(r'(\d{1,2})', str(time_str))
        if match:
            return int(match.group(1))
        return -1

    hours = [parse_hour(t) for t in actual_arrival_times]
    hours = [h for h in hours if 0 <= h <= 23]

    if not hours:
        return True, "Cannot parse arrival times"

    pref_lower = preference.lower()

    # Define time windows
    if "morning" in pref_lower or "early" in pref_lower:
        # 06:00-12:00
        in_window = sum(1 for h in hours if 6 <= h < 12)
        if in_window == len(hours):
            return True, f"All {len(hours)} flights arrive in morning (06:00-12:00)"
        else:
            return False, f"Only {in_window}/{len(hours)} flights arrive in morning"

    elif "afternoon" in pref_lower or "check-in" in pref_lower:
        # 12:00-18:00
        in_window = sum(1 for h in hours if 12 <= h < 18)
        if in_window == len(hours):
            return True, f"All {len(hours)} flights arrive in afternoon (12:00-18:00)"
        else:
            return False, f"Only {in_window}/{len(hours)} flights arrive in afternoon"

    elif "evening" in pref_lower or "night" in pref_lower or "sleep" in pref_lower:
        # 18:00+
        in_window = sum(1 for h in hours if h >= 18)
        if in_window == len(hours):
            return True, f"All {len(hours)} flights arrive evening/night (18:00+)"
        else:
            return False, f"Only {in_window}/{len(hours)} flights arrive evening/night"

    return True, "Unknown preference"


# =============================================================================
# SECTION 4: INTER-CITY TRAVEL VALIDATORS
# =============================================================================


def validate_intercity_duration(
    max_duration_preference: str,
    actual_durations: List[float]  # In hours
) -> Tuple[bool, str]:
    """Q4-1: Maximum inter-city travel duration tolerance.

    Args:
        max_duration_preference: "under 2 hours", "2-4 hours", "4+ hours is fine", "no preference"
        actual_durations: List of actual travel durations in hours

    Returns:
        (pass/fail, reason)
    """
    if not actual_durations:
        return True, "No inter-city travel data"

    if "no preference" in max_duration_preference.lower():
        return True, "No duration preference"

    max_actual = max(actual_durations)
    pref_lower = max_duration_preference.lower()

    if "under 2" in pref_lower:
        if max_actual <= 2.0:
            return True, f"Max duration {max_actual:.1f}h within 2h limit"
        else:
            return False, f"Max duration {max_actual:.1f}h exceeds 2h limit"

    elif "2" in pref_lower and "4" in pref_lower:
        # "2-4 hours"
        if max_actual <= 4.0:
            return True, f"Max duration {max_actual:.1f}h within 4h limit"
        else:
            return False, f"Max duration {max_actual:.1f}h exceeds 4h limit"

    elif "4+" in pref_lower or "4 hours is fine" in pref_lower:
        # Any duration is acceptable
        return True, f"Max duration {max_actual:.1f}h (4+ hours acceptable)"

    return True, "Unknown preference"


# =============================================================================
# HARD CONSTRAINT VALIDATORS (DB Derivable)
# =============================================================================


def validate_total_flight_duration(
    max_duration_minutes: int,
    flight_durations: List[int]  # In minutes
) -> Tuple[bool, str]:
    """NEW-3: Total flight duration tolerance.

    Args:
        max_duration_minutes: Maximum acceptable total flight time
        flight_durations: List of flight durations in minutes

    Returns:
        (pass/fail, reason)
    """
    if not flight_durations:
        return True, "No flight data"

    total_duration = sum(flight_durations)

    if total_duration <= max_duration_minutes:
        return True, f"Total {total_duration} min within {max_duration_minutes} min"
    else:
        return False, f"Total {total_duration} min exceeds {max_duration_minutes} min"


def validate_flight_distance_tolerance(
    max_distance_km: float,
    flight_distances: List[float]
) -> Tuple[bool, str]:
    """NEW-5: Flight distance tolerance.

    Args:
        max_distance_km: Maximum acceptable flight distance per segment
        flight_distances: List of flight distances in km

    Returns:
        (pass/fail, reason)
    """
    if not flight_distances:
        return True, "No flight data"

    max_actual = max(flight_distances)

    if max_actual <= max_distance_km:
        return True, f"Max segment {max_actual:.0f} km within {max_distance_km:.0f} km"
    else:
        return False, f"Max segment {max_actual:.0f} km exceeds {max_distance_km:.0f} km"


def validate_geographic_clustering(
    preference: str,
    poi_coords: List[Tuple[float, float]]  # (lat, lon)
) -> Tuple[bool, str]:
    """NEW-7: Geographic clustering preference.

    Args:
        preference: "clustered", "scattered", "no_preference"
        poi_coords: List of (lat, lon) for POIs

    Returns:
        (pass/fail, reason)
    """
    if preference.lower() in ["no_preference", "no preference"]:
        return True, "No preference"

    if len(poi_coords) < 2:
        return True, "Insufficient POI data"

    # Calculate standard deviation of coordinates
    lats = [coord[0] for coord in poi_coords]
    lons = [coord[1] for coord in poi_coords]

    lat_std = (sum((x - sum(lats)/len(lats))**2 for x in lats) / len(lats))**0.5
    lon_std = (sum((x - sum(lons)/len(lons))**2 for x in lons) / len(lons))**0.5

    avg_std = (lat_std + lon_std) / 2

    # Threshold: <0.1 degrees (~11 km) = clustered, >0.5 degrees = scattered
    if preference.lower() in ["clustered", "one area"]:
        if avg_std < 0.1:
            return True, f"Std dev {avg_std:.3f}° suggests clustered"
        else:
            return False, f"Std dev {avg_std:.3f}° too scattered"

    elif preference.lower() in ["scattered", "exploration"]:
        if avg_std > 0.5:
            return True, f"Std dev {avg_std:.3f}° suggests scattered"
        else:
            return False, f"Std dev {avg_std:.3f}° too clustered"

    return True, "Unknown preference"


def validate_cuisine_diversity(
    preference: str,
    cuisines_list: List[str]
) -> Tuple[bool, str]:
    """NEW-8: Cuisine diversity preference.

    Args:
        preference: "diverse", "consistent", "no_preference"
        cuisines_list: List of restaurant cuisines

    Returns:
        (pass/fail, reason)
    """
    if preference.lower() in ["no_preference", "no preference"]:
        return True, "No preference"

    if not cuisines_list:
        return True, "No cuisine data"

    # Flatten cuisines (they may be comma-separated)
    all_cuisines = []
    for c in cuisines_list:
        all_cuisines.extend([x.strip() for x in c.split(",")])

    unique_cuisines = len(set(all_cuisines))
    total_cuisines = len(all_cuisines)

    diversity_ratio = unique_cuisines / total_cuisines if total_cuisines > 0 else 0

    # Threshold: >0.7 = diverse, <0.3 = consistent
    if preference.lower() in ["diverse", "try many"]:
        if diversity_ratio > 0.7:
            return True, f"Diversity {diversity_ratio:.2f} suggests variety"
        else:
            return False, f"Diversity {diversity_ratio:.2f} too consistent"

    elif preference.lower() in ["consistent", "stick to favorites"]:
        if diversity_ratio < 0.3:
            return True, f"Diversity {diversity_ratio:.2f} suggests consistency"
        else:
            return False, f"Diversity {diversity_ratio:.2f} too diverse"

    return True, "Unknown preference"


def validate_price_consistency(
    preference: str,
    accommodation_prices: List[float],
    restaurant_prices: List[float]
) -> Tuple[bool, str]:
    """Q1-8 (NEW-9): Price consistency preference.

    Args:
        preference: "consistent", "moderate_variation", "high_variation", "no_preference"
        accommodation_prices: List of accommodation prices
        restaurant_prices: List of restaurant prices per person

    Returns:
        (pass/fail, reason)
    """
    if preference.lower() in ["no_preference", "no preference"]:
        return True, "No preference"

    # Combine all prices
    all_prices = accommodation_prices + restaurant_prices

    if len(all_prices) < 2:
        return True, "Insufficient price data"

    # Calculate coefficient of variation (CV)
    mean_price = sum(all_prices) / len(all_prices)
    variance = sum((p - mean_price) ** 2 for p in all_prices) / len(all_prices)
    std_dev = variance ** 0.5
    cv = (std_dev / mean_price) * 100 if mean_price > 0 else 0

    # Thresholds: <20% = consistent, 20-40% = moderate, >40% = high variation
    pref_lower = preference.lower().replace(" ", "_")

    if pref_lower in ["consistent", "consistent_(similar_prices)", "consistent (similar prices for all accommodations/restaurants)"]:
        if cv < 20:
            return True, f"CV {cv:.1f}% suggests consistent pricing"
        else:
            return False, f"CV {cv:.1f}% too variable for consistent preference"

    elif pref_lower in ["moderate_variation", "moderate variation", "moderate variation (some ups and downs acceptable)"]:
        if 20 <= cv <= 40:
            return True, f"CV {cv:.1f}% suggests moderate variation"
        else:
            return False, f"CV {cv:.1f}% outside moderate range (20-40%)"

    elif pref_lower in ["high_variation", "high variation ok", "high_variation_ok_(best_value)", "high variation ok (i prioritize best value, even if prices vary)"]:
        if cv > 40:
            return True, f"CV {cv:.1f}% suggests high variation"
        else:
            return False, f"CV {cv:.1f}% too consistent for high variation preference"

    return True, "Unknown preference"


def validate_dining_frequency(
    preference: str,
    num_restaurants: int,
    num_days: int
) -> Tuple[bool, str]:
    """Q2-6 (NEW-10): Dining frequency preference.

    Args:
        preference: "1_meal_day", "2_meals_day", "3_meals_day", "flexible"
        num_restaurants: Number of restaurants selected
        num_days: Number of trip days

    Returns:
        (pass/fail, reason)
    """
    if preference.lower() in ["flexible", "flexible (depends on the day)"]:
        return True, "Flexible preference"

    if num_days == 0:
        return True, "Invalid trip duration"

    actual_meals_per_day = num_restaurants / num_days

    pref_lower = preference.lower().replace(" ", "_").replace("/", "_")

    # Tolerance: ±0.5 meals per day
    if pref_lower in ["1_meal_day", "1 meal/day", "1 meal/day (mostly self-catering or hotel breakfast)"]:
        if actual_meals_per_day <= 1.5:
            return True, f"{actual_meals_per_day:.1f} meals/day matches 1 meal/day preference"
        else:
            return False, f"{actual_meals_per_day:.1f} meals/day exceeds 1 meal/day preference"

    elif pref_lower in ["2_meals_day", "2 meals/day", "2 meals/day (typical dining out)"]:
        if 1.5 <= actual_meals_per_day <= 2.5:
            return True, f"{actual_meals_per_day:.1f} meals/day matches 2 meals/day preference"
        else:
            return False, f"{actual_meals_per_day:.1f} meals/day outside 2 meals/day range"

    elif pref_lower in ["3_meals_day", "3 meals/day", "3 meals/day (all meals at restaurants)"]:
        if actual_meals_per_day >= 2.5:
            return True, f"{actual_meals_per_day:.1f} meals/day matches 3 meals/day preference"
        else:
            return False, f"{actual_meals_per_day:.1f} meals/day below 3 meals/day preference"

    return True, "Unknown preference"
