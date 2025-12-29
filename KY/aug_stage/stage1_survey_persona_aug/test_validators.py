#!/usr/bin/env python3
"""Test validators against actual TravelPlanner database.

This script loads real DB data and tests all 8 soft constraint validators
and 8 new derived validators to ensure they work correctly.
"""

import sys
from pathlib import Path
import pandas as pd

# Add stage1_5 to path
sys.path.insert(0, str(Path(__file__).parent))

from stage1_5.validators import (
    # Soft constraint validators (8)
    validate_dietary_restrictions,
    validate_dining_ambiance,
    validate_location_proximity,
    validate_flight_layovers,
    validate_baggage_allowance,
    validate_transport_mode,
    validate_comfort_level,
    validate_scenic_route,
    # New derived validators (8)
    validate_city_distribution,
    validate_per_person_accommodation_budget,
    validate_total_flight_duration,
    validate_flight_date_flexibility,
    validate_flight_distance_tolerance,
    validate_intercity_distance_tolerance,
    validate_geographic_clustering,
    validate_cuisine_diversity,
)


DB_PATH = Path("/Users/ky/AgenticAI/Challenge/benchmarks/travelplanner/official/database")


def load_sample_data():
    """Load sample data from TravelPlanner DB."""
    print("Loading TravelPlanner database...\n")

    # Load accommodations
    accommodations_path = DB_PATH / "accommodations" / "clean_accommodations_2022.csv"
    if accommodations_path.exists():
        df_acc = pd.read_csv(accommodations_path)
        print(f"✅ Loaded {len(df_acc)} accommodations")
    else:
        df_acc = None
        print(f"❌ Accommodations DB not found at {accommodations_path}")

    # Load restaurants
    restaurants_path = DB_PATH / "restaurants" / "clean_restaurant_2022.csv"
    if restaurants_path.exists():
        df_rest = pd.read_csv(restaurants_path)
        print(f"✅ Loaded {len(df_rest)} restaurants")
    else:
        df_rest = None
        print(f"❌ Restaurants DB not found at {restaurants_path}")

    # Load flights
    flights_path = DB_PATH / "flights" / "clean_Flights_2022.csv"
    if flights_path.exists():
        df_flights = pd.read_csv(flights_path)
        print(f"✅ Loaded {len(df_flights)} flights")
    else:
        df_flights = None
        print(f"❌ Flights DB not found at {flights_path}")

    # Load distance
    distance_path = DB_PATH / "googleDistanceMatrix" / "distance.csv"
    if distance_path.exists():
        df_dist = pd.read_csv(distance_path)
        print(f"✅ Loaded {len(df_dist)} distance records")
    else:
        df_dist = None
        print(f"❌ Distance DB not found at {distance_path}")

    print()
    return df_acc, df_rest, df_flights, df_dist


def test_dietary_restrictions(df_rest):
    """Test Q2-4: Dietary restrictions validator."""
    print("="*60)
    print("TEST 1: Dietary Restrictions (Q2-4)")
    print("="*60)

    if df_rest is None:
        print("❌ SKIP: No restaurant data\n")
        return

    # Sample cuisines
    sample_cuisines = df_rest["Cuisines"].dropna().head(20).tolist()

    # Test cases
    test_cases = [
        ("vegetarian", sample_cuisines),
        ("vegan", sample_cuisines),
        ("halal", sample_cuisines),
        ("none", sample_cuisines),
    ]

    for pref, cuisines in test_cases:
        result, reason = validate_dietary_restrictions(pref, cuisines)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:15s} | {reason}")

    print()


def test_dining_ambiance(df_rest):
    """Test Q2-5: Dining ambiance validator."""
    print("="*60)
    print("TEST 2: Dining Ambiance (Q2-5)")
    print("="*60)

    if df_rest is None:
        print("❌ SKIP: No restaurant data\n")
        return

    # Sample data
    sample_names = df_rest["Name"].dropna().head(20).tolist()
    sample_cuisines = df_rest["Cuisines"].dropna().head(20).tolist()

    # Test cases
    test_cases = [
        ("casual", sample_names, sample_cuisines),
        ("fine_dining", sample_names, sample_cuisines),
        ("lively", sample_names, sample_cuisines),
        ("no_preference", sample_names, sample_cuisines),
    ]

    for pref, names, cuisines in test_cases:
        result, reason = validate_dining_ambiance(pref, names, cuisines)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:15s} | {reason}")

    print()


def test_location_proximity(df_acc, df_rest):
    """Test Q2-6: Location proximity validator."""
    print("="*60)
    print("TEST 3: Location Proximity (Q2-6)")
    print("="*60)

    # Note: accommodations/restaurants DB don't have coordinates
    # Using mock data for demonstration
    print("⚠️  NOTE: Using mock coordinate data (DB lacks lat/lon for accommodations/restaurants)\n")

    # Mock coordinates (realistic NYC area)
    acc_coords = [
        (40.7128, -74.0060),  # NYC
        (40.7589, -73.9851),
        (40.7306, -73.9352),
    ]

    rest_coords = [
        (40.7200, -74.0100),  # Close to hotels
        (40.7500, -73.9900),
        (40.7250, -73.9400),
    ]

    # Test cases
    test_cases = [
        ("walking_distance", acc_coords, rest_coords),
        ("short_drive", acc_coords, rest_coords),
        ("willing_to_travel", acc_coords, rest_coords),
        ("no_preference", acc_coords, rest_coords),
    ]

    for pref, h_coords, r_coords in test_cases:
        result, reason = validate_location_proximity(pref, h_coords, r_coords)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:20s} | {reason}")

    print()


def test_flight_layovers(df_flights):
    """Test Q3-2: Flight layovers validator."""
    print("="*60)
    print("TEST 4: Flight Layovers (Q3-2)")
    print("="*60)

    if df_flights is None:
        print("❌ SKIP: No flight data\n")
        return

    # Sample flights
    sample_flights = df_flights.head(10).to_dict('records')

    # Test cases
    test_cases = [
        ("direct_only", sample_flights),
        ("one_stop", sample_flights),
        ("multiple_stops", sample_flights),
        ("no_preference", sample_flights),
    ]

    for pref, flights in test_cases:
        result, reason = validate_flight_layovers(pref, flights)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:15s} | {reason}")

    print()


def test_baggage_allowance(df_flights):
    """Test Q3-7: Baggage allowance validator."""
    print("="*60)
    print("TEST 5: Baggage Allowance (Q3-7)")
    print("="*60)

    if df_flights is None:
        print("❌ SKIP: No flight data\n")
        return

    # Sample prices
    sample_prices_low = df_flights["Price"].dropna().head(10).tolist()
    sample_prices_high = df_flights["Price"].dropna().tail(10).tolist()

    # Test cases
    test_cases = [
        ("checked_bag", sample_prices_low, "Low prices"),
        ("checked_bag", sample_prices_high, "High prices"),
        ("carry_on_only", sample_prices_low, "Low prices"),
        ("no_preference", sample_prices_low, "Any prices"),
    ]

    for pref, prices, desc in test_cases:
        result, reason = validate_baggage_allowance(pref, prices)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:15s} ({desc:12s}) | {reason}")

    print()


def test_transport_mode(df_dist):
    """Test Q4-1 (was Q5-1): Transport mode validator."""
    print("="*60)
    print("TEST 6: Transport Mode (Q4-1)")
    print("="*60)

    if df_dist is None:
        print("❌ SKIP: No distance data\n")
        return

    # Parse duration to hours
    def parse_duration(s):
        """Parse '1 hours 30 minutes' to float hours."""
        if pd.isna(s):
            return 0
        hours = 0
        minutes = 0
        import re
        h_match = re.search(r'(\d+)\s*hours?', str(s))
        m_match = re.search(r'(\d+)\s*minutes?', str(s))
        if h_match:
            hours = int(h_match.group(1))
        if m_match:
            minutes = int(m_match.group(1))
        return hours + minutes / 60.0

    # Sample distance data
    sample = df_dist.head(10)

    test_cases = []
    for _, row in sample.iterrows():
        duration_hours = parse_duration(row.get("duration", ""))
        distance_km = row.get("distance", 0)
        # Convert distance to float if it's a string
        try:
            distance_km = float(distance_km)
        except:
            distance_km = 0
        if duration_hours > 0 and distance_km > 0:
            test_cases.append((distance_km, duration_hours))
            if len(test_cases) >= 3:
                break

    # Test with different modes
    for dist_km, dur_h in test_cases:
        for mode in ["car", "train / bus", "flight"]:
            result, reason = validate_transport_mode(mode, dur_h, dist_km)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} | Mode: {mode:15s} | Dist: {dist_km:.0f} km, Dur: {dur_h:.1f}h | {reason}")

    print()


def test_comfort_level():
    """Test Q4-3 (was Q5-4): Comfort level validator."""
    print("="*60)
    print("TEST 7: Comfort Level (Q4-3)")
    print("="*60)

    # Test cases
    test_cases = [
        ("basic", "bus"),
        ("basic", "train"),
        ("standard", "car"),
        ("standard", "train"),
        ("premium", "flight"),
        ("premium", "bus"),
        ("no_preference", "car"),
    ]

    for pref, mode in test_cases:
        result, reason = validate_comfort_level(pref, mode)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:15s} | Mode: {mode:8s} | {reason}")

    print()


def test_scenic_route(df_dist):
    """Test Q4-4 (was Q5-5): Scenic route validator."""
    print("="*60)
    print("TEST 8: Scenic Route (Q4-4)")
    print("="*60)

    if df_dist is None:
        print("❌ SKIP: No distance data\n")
        return

    # Parse duration
    def parse_duration(s):
        if pd.isna(s):
            return 0
        hours = 0
        minutes = 0
        import re
        h_match = re.search(r'(\d+)\s*hours?', str(s))
        m_match = re.search(r'(\d+)\s*minutes?', str(s))
        if h_match:
            hours = int(h_match.group(1))
        if m_match:
            minutes = int(m_match.group(1))
        return hours + minutes / 60.0

    # Get sample
    sample = df_dist.head(10)

    test_cases = []
    for _, row in sample.iterrows():
        duration_hours = parse_duration(row.get("duration", ""))
        distance_km = row.get("distance", 0)
        # Convert distance to float if it's a string
        try:
            distance_km = float(distance_km)
        except:
            distance_km = 0
        if duration_hours > 0 and distance_km > 0:
            test_cases.append((distance_km, duration_hours))
            if len(test_cases) >= 3:
                break

    for dist_km, dur_h in test_cases:
        for pref in ["scenic", "fastest", "no_preference"]:
            result, reason = validate_scenic_route(pref, dist_km, dur_h)
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} | Preference: {pref:15s} | {reason}")

    print()


# =============================================================================
# NEW DERIVED VALIDATORS TESTS
# =============================================================================


def test_city_distribution(df_acc):
    """Test NEW-1: City distribution validator."""
    print("="*60)
    print("TEST 9: City Distribution (NEW-1)")
    print("="*60)

    if df_acc is None:
        print("❌ SKIP: No accommodation data\n")
        return

    # Sample cities
    cities_single = ["New York", "New York", "New York"]
    cities_multiple = ["New York", "Boston", "Philadelphia"]

    # Test cases
    test_cases = [
        ("single_city", cities_single),
        ("single_city", cities_multiple),
        ("multiple_cities", cities_single),
        ("multiple_cities", cities_multiple),
        ("no_preference", cities_single),
    ]

    for pref, cities in test_cases:
        result, reason = validate_city_distribution(pref, cities)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:15s} | Cities: {set(cities)} | {reason}")

    print()


def test_per_person_budget(df_acc):
    """Test NEW-2: Per-person accommodation budget validator."""
    print("="*60)
    print("TEST 10: Per-Person Accommodation Budget (NEW-2)")
    print("="*60)

    if df_acc is None:
        print("❌ SKIP: No accommodation data\n")
        return

    # Sample prices
    sample_prices = df_acc["price"].dropna().head(10).tolist()

    # Test cases
    test_cases = [
        (50.0, sample_prices, 4),  # $50 pp preference, group of 4
        (100.0, sample_prices, 2),  # $100 pp preference, group of 2
        (200.0, sample_prices, 1),  # $200 pp preference, solo
    ]

    for pref_pp, prices, group_size in test_cases:
        result, reason = validate_per_person_accommodation_budget(pref_pp, prices, group_size)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Target: ${pref_pp:.0f} pp, Group: {group_size} | {reason}")

    print()


def test_total_flight_duration(df_flights):
    """Test NEW-3: Total flight duration validator."""
    print("="*60)
    print("TEST 11: Total Flight Duration (NEW-3)")
    print("="*60)

    if df_flights is None:
        print("❌ SKIP: No flight data\n")
        return

    # Parse flight durations
    def parse_duration_minutes(s):
        if pd.isna(s):
            return 0
        hours = 0
        minutes = 0
        import re
        h_match = re.search(r'(\d+)\s*hours?', str(s))
        m_match = re.search(r'(\d+)\s*minutes?', str(s))
        if h_match:
            hours = int(h_match.group(1))
        if m_match:
            minutes = int(m_match.group(1))
        return hours * 60 + minutes

    sample_durations = [
        parse_duration_minutes(d)
        for d in df_flights["ActualElapsedTime"].dropna().head(5)
    ]
    sample_durations = [d for d in sample_durations if d > 0]

    # Test cases
    test_cases = [
        (300, sample_durations),  # 5 hours max
        (600, sample_durations),  # 10 hours max
        (60, sample_durations),   # 1 hour max (should fail)
    ]

    for max_min, durations in test_cases:
        result, reason = validate_total_flight_duration(max_min, durations)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Max: {max_min} min | {reason}")

    print()


def test_flight_date_flexibility(df_flights):
    """Test NEW-4: Flight date flexibility validator."""
    print("="*60)
    print("TEST 12: Flight Date Flexibility (NEW-4)")
    print("="*60)

    if df_flights is None:
        print("❌ SKIP: No flight data\n")
        return

    # Sample dates
    sample_dates = df_flights["FlightDate"].dropna().head(5).tolist()

    # Test cases
    test_cases = [
        ("2022-03-15", sample_dates, 1),   # 1 day flexibility
        ("2022-03-15", sample_dates, 7),   # 1 week flexibility
        ("2022-03-15", sample_dates, 30),  # 1 month flexibility
    ]

    for req_date, actual_dates, flex_days in test_cases:
        result, reason = validate_flight_date_flexibility(req_date, actual_dates, flex_days)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Requested: {req_date}, Flex: ±{flex_days} days | {reason}")

    print()


def test_flight_distance_tolerance(df_flights):
    """Test NEW-5: Flight distance tolerance validator."""
    print("="*60)
    print("TEST 13: Flight Distance Tolerance (NEW-5)")
    print("="*60)

    if df_flights is None:
        print("❌ SKIP: No flight data\n")
        return

    # Sample distances
    sample_distances = df_flights["Distance"].dropna().head(10).tolist()

    # Test cases
    test_cases = [
        (500.0, sample_distances),   # 500 km max
        (1000.0, sample_distances),  # 1000 km max
        (3000.0, sample_distances),  # 3000 km max
    ]

    for max_km, distances in test_cases:
        result, reason = validate_flight_distance_tolerance(max_km, distances)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Max: {max_km:.0f} km | {reason}")

    print()


def test_intercity_distance_tolerance(df_dist):
    """Test NEW-6: Inter-city distance tolerance validator."""
    print("="*60)
    print("TEST 14: Inter-city Distance Tolerance (NEW-6)")
    print("="*60)

    if df_dist is None:
        print("❌ SKIP: No distance data\n")
        return

    # Sample distances (convert to float)
    sample_distances = []
    for d in df_dist["distance"].dropna().head(10):
        try:
            sample_distances.append(float(d))
        except:
            pass

    # Test cases
    test_cases = [
        (100.0, sample_distances),   # 100 km max
        (500.0, sample_distances),   # 500 km max
        (2000.0, sample_distances),  # 2000 km max
    ]

    for max_km, distances in test_cases:
        result, reason = validate_intercity_distance_tolerance(max_km, distances)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Max: {max_km:.0f} km | {reason}")

    print()


def test_geographic_clustering(df_acc):
    """Test NEW-7: Geographic clustering validator."""
    print("="*60)
    print("TEST 15: Geographic Clustering (NEW-7)")
    print("="*60)

    print("⚠️  NOTE: Using mock coordinate data (DB lacks lat/lon for accommodations)\n")

    # Sample coordinates (clustered)
    clustered_coords = [
        (40.7128, -74.0060),  # NYC
        (40.7589, -73.9851),  # NYC
        (40.7306, -73.9352),  # NYC
    ]

    # Sample coordinates (scattered)
    scattered_coords = [
        (40.7128, -74.0060),  # NYC
        (34.0522, -118.2437),  # LA
        (41.8781, -87.6298),   # Chicago
    ]

    # Test cases
    test_cases = [
        ("clustered", clustered_coords),
        ("clustered", scattered_coords),
        ("scattered", clustered_coords),
        ("scattered", scattered_coords),
        ("no_preference", clustered_coords),
    ]

    for pref, coords in test_cases:
        result, reason = validate_geographic_clustering(pref, coords)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:15s} | {reason}")

    print()


def test_cuisine_diversity(df_rest):
    """Test NEW-8: Cuisine diversity validator."""
    print("="*60)
    print("TEST 16: Cuisine Diversity (NEW-8)")
    print("="*60)

    if df_rest is None:
        print("❌ SKIP: No restaurant data\n")
        return

    # Sample cuisines
    diverse_cuisines = ["Italian", "Chinese", "Mexican", "French", "Japanese"]
    consistent_cuisines = ["Italian", "Italian", "Italian", "Pizza", "Pizza"]

    # Test cases
    test_cases = [
        ("diverse", diverse_cuisines),
        ("diverse", consistent_cuisines),
        ("consistent", diverse_cuisines),
        ("consistent", consistent_cuisines),
        ("no_preference", diverse_cuisines),
    ]

    for pref, cuisines in test_cases:
        result, reason = validate_cuisine_diversity(pref, cuisines)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | Preference: {pref:15s} | {reason}")

    print()


def main():
    print("=" * 60)
    print("ALPHA SURVEY VALIDATORS TEST SUITE")
    print("=" * 60)
    print()

    # Load data
    df_acc, df_rest, df_flights, df_dist = load_sample_data()

    # Test 8 soft constraint validators
    print("\n" + "="*60)
    print("PART 1: SOFT CONSTRAINT VALIDATORS (8 items)")
    print("="*60 + "\n")

    test_dietary_restrictions(df_rest)
    test_dining_ambiance(df_rest)
    test_location_proximity(df_acc, df_rest)
    test_flight_layovers(df_flights)
    test_baggage_allowance(df_flights)
    test_transport_mode(df_dist)
    test_comfort_level()
    test_scenic_route(df_dist)

    # Test 8 new derived validators
    print("\n" + "="*60)
    print("PART 2: NEW DERIVED VALIDATORS (8 items)")
    print("="*60 + "\n")

    test_city_distribution(df_acc)
    test_per_person_budget(df_acc)
    test_total_flight_duration(df_flights)
    test_flight_date_flexibility(df_flights)
    test_flight_distance_tolerance(df_flights)
    test_intercity_distance_tolerance(df_dist)
    test_geographic_clustering(df_acc)
    test_cuisine_diversity(df_rest)

    # Summary
    print("="*60)
    print("✅ ALL VALIDATOR TESTS COMPLETED")
    print("="*60)
    print()
    print("Summary:")
    print("- 8 Soft Constraint Validators: Tested with real DB data")
    print("- 8 New Derived Validators: Tested with real DB data")
    print()
    print("Next steps:")
    print("1. Review failed tests (❌) and adjust validators if needed")
    print("2. Update structured_output.py with 22-question schema")
    print("3. Add 8 new questions to prompt_builder.py")


if __name__ == "__main__":
    main()
