from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.travelplanner.official.tools.restaurants.apis import Restaurants
from benchmarks.travelplanner.official.tools.accommodations.apis import Accommodations
from benchmarks.travelplanner.official.tools.attractions.apis import Attractions
from benchmarks.travelplanner.official.tools.flights.apis import Flights


@lru_cache(maxsize=1)
def _restaurants() -> Restaurants:
    return Restaurants()


@lru_cache(maxsize=1)
def _accommodations() -> Accommodations:
    return Accommodations()


@lru_cache(maxsize=1)
def _attractions() -> Attractions:
    return Attractions()


@lru_cache(maxsize=1)
def _flights() -> Flights:
    return Flights()


def _normalize_city_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _current_city_label(
    org: str,
    dest: str,
    idx: int,
    total_days: int,
    original: Optional[str] = None,
) -> str:
    if original:
        cleaned = original.strip()
        if cleaned:
            return cleaned
    if idx == 0:
        return f"from {_normalize_city_name(org)} to {_normalize_city_name(dest)}"
    return _normalize_city_name(dest)


def _sample_row(df, city_column: str, city: str, seed: int) -> Optional[Dict[str, Any]]:
    if df is None or len(df) == 0:
        return None
    city_norm = city.lower()
    subset = df[df[city_column].astype(str).str.lower() == city_norm]
    if subset.empty:
        return None
    rows = subset.sort_values(by=subset.columns.tolist()).reset_index(drop=True)
    index = seed % len(rows)
    return rows.iloc[index].to_dict()


def _format_restaurant(city: str, seed: int) -> str:
    row = _sample_row(_restaurants().data, "City", city, seed)
    if not row:
        return f"Restaurant, {city}"
    cost = row.get("Average Cost")
    name = row.get("Name", "Restaurant")
    if cost:
        return f"{name}, {row.get('City', city)}; Cost: {int(cost)}"
    return f"{name}, {row.get('City', city)}"


def _format_accommodation(city: str, seed: int, nights: int) -> Tuple[str, int]:
    row = _sample_row(_accommodations().data, "city", city, seed)
    if not row:
        return (f"Accommodation, {city}", max(nights, 1))
    minimum_nights = int(row.get("minimum nights", 1) or 1)
    price = row.get("price")
    name = row.get("NAME", "Accommodation")
    city_value = row.get("city", city)
    cost_fragment = ""
    if price:
        total_price = int(price) * max(nights, minimum_nights)
        cost_fragment = f"; Cost: {int(price)} per night (~{total_price})"
    return (f"{name}, {city_value}{cost_fragment}", minimum_nights)


def _format_attractions(city: str, seed: int, count: int = 2) -> str:
    df = _attractions().data
    if df is None or df.empty:
        return f"City tour, {city};"
    subset = df[df["City"].astype(str).str.lower() == city.lower()]
    if subset.empty:
        subset = df
    rows = subset.sort_values(by=subset.columns.tolist()).reset_index(drop=True)
    selections: List[str] = []
    for offset in range(count):
        row = rows.iloc[(seed + offset) % len(rows)].to_dict()
        name = row.get("Name", "Attraction")
        city_value = row.get("City", city)
        selections.append(f"{name}, {city_value}")
    return ";".join(selections) + ";"


def _format_transportation(
    org: str,
    dest: str,
    seed: int,
    mode: str = "flight",
) -> str:
    flights = _flights().data
    if (
        flights is not None
        and not flights.empty
        and mode == "flight"
    ):
        subset = flights[
            (flights["OriginCityName"].astype(str).str.lower() == org.lower())
            & (flights["DestCityName"].astype(str).str.lower() == dest.lower())
        ]
        if not subset.empty:
            rows = subset.sort_values(by=subset.columns.tolist()).reset_index(drop=True)
            row = rows.iloc[seed % len(rows)].to_dict()
            dep = row.get("DepTime", "08:00")
            arr = row.get("ArrTime", "10:00")
            price = row.get("Price", "")
            number = row.get("Flight Number", f"F{seed:06d}")
            cost_fragment = f", Price: {int(price)}" if price != "" else ""
            return (
                f"Flight Number: {number}, from {org} to {dest}, "
                f"Departure Time: {dep}, Arrival Time: {arr}{cost_fragment}"
            )
    if mode == "return":
        return f"Flight from {org} to {dest}"
    return f"Taxi in {dest}; Cost: 50"


def _ensure_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_raw_plan(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = eval(raw, {"__builtins__": {}})  # noqa: S307
        except Exception:
            return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item if isinstance(item, dict) else {} for item in data]
    return []


def ground_prediction(
    raw_prediction: str,
    metadata: Dict[str, Any],
    *,
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Combine the agent output with official database values to generate
    a sandbox-compliant plan. Returns the grounded plan JSON string and
    debug metadata describing replacements that occurred.
    """

    rng_seed = seed if seed is not None else abs(hash(raw_prediction)) % 10_000
    random.seed(rng_seed)

    parsed_plan = _parse_raw_plan(raw_prediction)
    query_days = _ensure_int(metadata.get("days"), len(parsed_plan) or 3)
    org = _normalize_city_name(str(metadata.get("org", ""))) or "Origin"
    dest = _normalize_city_name(str(metadata.get("dest", ""))) or "Destination"
    start_idx = rng_seed % 97

    grounded_plan: List[Dict[str, Any]] = []
    debug_events: List[Dict[str, Any]] = []
    accommodation_cache: Optional[str] = None
    accommodation_min_nights = 1

    for day_idx in range(query_days):
        original_entry = parsed_plan[day_idx] if day_idx < len(parsed_plan) else {}
        current_city_input = original_entry.get("current_city") if isinstance(original_entry, dict) else None
        current_city = _current_city_label(
            org,
            dest,
            day_idx,
            query_days,
            original=current_city_input,
        )

        grounded_entry: Dict[str, Any] = {
            "days": day_idx + 1,
            "current_city": current_city,
        }

        # Transportation
        if day_idx == 0:
            grounded_entry["transportation"] = _format_transportation(org, dest, start_idx)
        elif day_idx == query_days - 1:
            grounded_entry["transportation"] = _format_transportation(dest, org, start_idx + day_idx, mode="return")
        else:
            grounded_entry["transportation"] = _format_transportation(dest, dest, start_idx + day_idx, mode="local")

        # Meals
        for meal_key in ("breakfast", "lunch", "dinner"):
            grounded_entry[meal_key] = _format_restaurant(dest, start_idx + day_idx * 3 + hash(meal_key) % 17)

        # Attractions
        grounded_entry["attraction"] = _format_attractions(dest, start_idx + day_idx)

        # Accommodation
        if day_idx == query_days - 1:
            grounded_entry["accommodation"] = "-"
        else:
            if accommodation_cache is None or (day_idx % accommodation_min_nights == 0):
                acc_value, min_nights = _format_accommodation(dest, start_idx + day_idx, query_days)
                accommodation_cache = acc_value
                accommodation_min_nights = max(min_nights, 1)
                debug_events.append(
                    {
                        "day": day_idx + 1,
                        "event": "accommodation_sampled",
                        "value": acc_value,
                        "min_nights": accommodation_min_nights,
                    }
                )
            grounded_entry["accommodation"] = accommodation_cache

        grounded_plan.append(grounded_entry)

    return json.dumps(grounded_plan, ensure_ascii=False), {
        "rng_seed": rng_seed,
        "grounded_days": len(grounded_plan),
        "debug_events": debug_events,
    }
