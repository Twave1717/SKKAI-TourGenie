#!/usr/bin/env python3
"""Find new verifiable questions from unused DB fields."""

import pandas as pd
from pathlib import Path
import re

DB_PATH = Path("/Users/ky/AgenticAI/Challenge/benchmarks/travelplanner/official/database")


def analyze_unused_fields():
    """Identify DB fields not yet used in constraints."""

    print("="*80)
    print("UNUSED DB FIELDS ANALYSIS")
    print("="*80)
    print()

    # Load DBs
    df_acc = pd.read_csv(DB_PATH / "accommodations" / "clean_accommodations_2022.csv")
    df_rest = pd.read_csv(DB_PATH / "restaurants" / "clean_restaurant_2022.csv")
    df_flights = pd.read_csv(DB_PATH / "flights" / "clean_Flights_2022.csv", nrows=10000)
    df_dist = pd.read_csv(DB_PATH / "googleDistanceMatrix" / "distance.csv")

    print("Current 18 questions use these fields:")
    print()
    print("Accommodations:")
    print("  ✅ price (Q1-1)")
    print("  ✅ review rate number (Q1-2, Q1-7)")
    print("  ✅ room type (Q1-3)")
    print("  ✅ house_rules (Q1-4)")
    print("  ✅ minimum nights (Q1-5)")
    print("  ✅ maximum occupancy (Q1-6)")
    print("  ❓ NAME - UNUSED")
    print("  ❓ city - FIXED (not a constraint)")
    print()
    print("Restaurants:")
    print("  ✅ Average Cost (Q2-1)")
    print("  ✅ Aggregate Rating (Q2-2)")
    print("  ✅ Cuisines (Q2-3)")
    print("  ⚠️  Name - PARTIAL (Q2-4 ambiance pattern)")
    print("  ❓ City - FIXED")
    print()
    print("Flights:")
    print("  ✅ Price (Q3-1)")
    print("  ⚠️  Route (Q3-2 layovers)")
    print("  ✅ DepTime (Q3-3)")
    print("  ✅ ArrTime (Q3-4)")
    print("  ⚠️  ActualElapsedTime - PARTIAL (NEW-3 total duration)")
    print("  ✅ Distance (NEW-5)")
    print("  ❓ FlightNumber - MEANINGLESS")
    print("  ❓ FlightDate - FIXED")
    print("  ❓ OriginCityName - FIXED")
    print("  ❓ DestCityName - FIXED")
    print()
    print("Distance:")
    print("  ✅ duration (Q4-1)")
    print("  ❓ distance - FIXED (cities are fixed)")
    print("  ❓ origin - FIXED")
    print("  ❓ destination - FIXED")
    print()
    print()

    # NEW QUESTION IDEAS
    print("="*80)
    print("NEW QUESTION PROPOSALS (DB-VERIFIABLE)")
    print("="*80)
    print()

    # Idea 1: Accommodation brand preference
    print("--- IDEA 1: Accommodation Brand/Chain Preference ---")
    print("Field: accommodations.NAME")
    print()

    # Check for chain hotels in NAME
    chains = ["marriott", "hilton", "hyatt", "sheraton", "westin", "holiday inn",
              "comfort inn", "best western", "ramada", "days inn", "motel 6"]

    chain_count = 0
    independent_count = 0

    names_lower = df_acc["NAME"].astype(str).str.lower()
    for idx, name in names_lower.items():
        is_chain = any(chain in name for chain in chains)
        if is_chain:
            chain_count += 1
        else:
            independent_count += 1

    chain_pct = (chain_count / len(df_acc)) * 100
    indep_pct = (independent_count / len(df_acc)) * 100

    print(f"Chain hotels:       {chain_count:5,} / {len(df_acc):,} ({chain_pct:5.1f}%)")
    print(f"Independent hotels: {independent_count:5,} / {len(df_acc):,} ({indep_pct:5.1f}%)")
    print()

    if chain_pct >= 10:
        print("✅ VIABLE: Good balance for brand preference question")
        print("   Question: 'Do you prefer hotel chains or independent properties?'")
        print("   Validation: Match NAME against chain keywords")
    else:
        print("❌ NOT VIABLE: Too few chain hotels")
    print()
    print()

    # Idea 2: Price consistency vs variability
    print("--- IDEA 2: Price Consistency Preference ---")
    print("Field: accommodations.price + restaurants.Average Cost (variability)")
    print()

    # Sample trip: 5 days, select 5 accommodations
    sample_acc_prices = df_acc["price"].dropna().head(5).tolist()
    if sample_acc_prices:
        avg_price = sum(sample_acc_prices) / len(sample_acc_prices)
        std_dev = (sum((p - avg_price)**2 for p in sample_acc_prices) / len(sample_acc_prices))**0.5
        cv = (std_dev / avg_price) * 100  # Coefficient of variation

        print(f"Sample accommodation prices: {sample_acc_prices}")
        print(f"Average: ${avg_price:.0f}")
        print(f"Std dev: ${std_dev:.0f}")
        print(f"Coefficient of variation: {cv:.1f}%")
        print()
        print("✅ VIABLE: Can measure price consistency preference")
        print("   Question: 'Do you prefer consistent pricing or are you okay with varying costs?'")
        print("   Validation: Calculate CV of selected accommodation/restaurant prices")
        print("   - Low CV (< 20%) = consistent pricing")
        print("   - High CV (> 40%) = variable pricing")
    else:
        print("❌ NO DATA")
    print()
    print()

    # Idea 3: Meal frequency
    print("--- IDEA 3: Dining Frequency ---")
    print("Field: Number of restaurants selected per day")
    print()
    print("Question: 'How many meals per day do you plan to eat out?'")
    print("Options: 1 meal/day, 2 meals/day, 3 meals/day, Flexible")
    print()
    print("Validation: Count selected restaurants / trip days")
    print("  - If preference = '3 meals/day', check: restaurants_count >= days * 3")
    print("  - If preference = '1 meal/day', check: restaurants_count <= days * 1")
    print()
    print("✅ VIABLE: This is a PLANNER CHOICE (how many restaurants to include)")
    print()
    print()

    # Idea 4: Layover duration preference
    print("--- IDEA 4: Minimum Layover Duration ---")
    print("Field: Derived from flights.ArrTime and next flights.DepTime")
    print()

    # Sample calculation
    sample_flights = df_flights[["DepTime", "ArrTime", "OriginCityName", "DestCityName"]].head(10)
    print("Sample flights:")
    print(sample_flights.to_string(index=False))
    print()

    print("Question: 'What is your minimum acceptable layover time between connecting flights?'")
    print("Options: < 1 hour (risky), 1-2 hours, 2-3 hours, 3+ hours (safe)")
    print()
    print("Validation:")
    print("  - For connecting flights (same date, ArrCity = next DepCity)")
    print("  - Calculate: next_DepTime - current_ArrTime")
    print("  - Check if layover >= preference threshold")
    print()
    print("⚠️  MARGINAL: Requires identifying connecting flights (complex)")
    print("   Current DB doesn't explicitly link connecting flights")
    print()
    print()

    # Idea 5: Early check-in / late checkout importance
    print("--- IDEA 5: Accommodation Check-in/out Flexibility ---")
    print("Field: NOT IN DB - No check-in/checkout time data")
    print("❌ NOT VIABLE: No relevant DB field")
    print()
    print()

    # Idea 6: Restaurant proximity clustering
    print("--- IDEA 6: Restaurant Clustering Preference ---")
    print("Field: restaurants.City (granularity issue - no neighborhood data)")
    print()
    print("Question: 'Do you prefer restaurants clustered in one area or spread across the city?'")
    print("⚠️  LIMITED: DB only has city-level data, no coordinates/neighborhoods")
    print("   Could only verify 'all in same city' vs 'multiple cities'")
    print("   But cities are FIXED in initial_info → not a real constraint")
    print()
    print("❌ NOT VIABLE: Insufficient granularity")
    print()
    print()

    # Idea 7: Flight time of day preference (more granular)
    print("--- IDEA 7: Flight Duration Preference (per segment) ---")
    print("Field: flights.ActualElapsedTime")
    print()

    def parse_duration_minutes(s):
        if pd.isna(s):
            return None
        hours = 0
        minutes = 0
        h_match = re.search(r'(\d+)\s*hours?', str(s))
        m_match = re.search(r'(\d+)\s*minutes?', str(s))
        if h_match:
            hours = int(h_match.group(1))
        if m_match:
            minutes = int(m_match.group(1))
        return hours * 60 + minutes

    durations = [parse_duration_minutes(d) for d in df_flights["ActualElapsedTime"].dropna().head(100)]
    durations = [d for d in durations if d is not None]

    if durations:
        print(f"Sample flight durations: {min(durations)} - {max(durations)} minutes")
        print(f"Mean: {sum(durations)/len(durations):.0f} min")
        print()
        print("Question: 'What is your maximum acceptable flight duration per segment?'")
        print("Options: < 2 hours, < 4 hours, < 6 hours, Any duration")
        print()
        print("Validation: Check max(ActualElapsedTime) of selected flights")
        print()
        print("✅ VIABLE: Different from NEW-3 (total duration)")
        print("   NEW-3 = sum of all flights")
        print("   This = max of individual segments")
        print("   Both are useful constraints!")
    print()
    print()

    # Idea 8: Review count/popularity
    print("--- IDEA 8: Review Count / Popularity ---")
    print("Field: NOT IN DB - No review count field")
    print()
    print("Available review fields:")
    print("  - accommodations: review_rate_number (1-5 rating)")
    print("  - restaurants: Aggregate Rating (0-4.9 rating)")
    print("  - No 'number of reviews' or 'popularity score'")
    print()
    print("❌ NOT VIABLE: No review count data")
    print()
    print()


def summarize_recommendations():
    """Final recommendations for new questions."""
    print("="*80)
    print("FINAL NEW QUESTION RECOMMENDATIONS")
    print("="*80)
    print()

    viable_questions = [
        {
            "id": "NEW-9",
            "title": "Price Consistency Preference",
            "question": "Do you prefer consistent pricing across all bookings or are you comfortable with price variations?",
            "options": ["Consistent (similar prices)", "Moderate variation", "High variation OK (best value)"],
            "db_field": "price variance of accommodations/restaurants",
            "validation": "Calculate coefficient of variation (CV) of selected prices",
            "viable": True,
            "reason": "✅ Planner choice, DB-verifiable",
        },
        {
            "id": "NEW-10",
            "title": "Dining Frequency",
            "question": "How many meals per day do you plan to eat out?",
            "options": ["1 meal/day", "2 meals/day", "3 meals/day", "Flexible"],
            "db_field": "Number of restaurants / trip days",
            "validation": "Count selected restaurants, divide by days",
            "viable": True,
            "reason": "✅ Planner choice (how many restaurants to book)",
        },
        {
            "id": "NEW-11",
            "title": "Maximum Flight Duration (per segment)",
            "question": "What is your maximum acceptable flight duration for a single segment?",
            "options": ["< 2 hours", "< 4 hours", "< 6 hours", "Any duration"],
            "db_field": "flights.ActualElapsedTime",
            "validation": "Check max(ActualElapsedTime) of selected flights",
            "viable": True,
            "reason": "✅ Different from NEW-3 (total); this is max per segment",
        },
        {
            "id": "NEW-12",
            "title": "Accommodation Brand Preference",
            "question": "Do you prefer hotel chains or independent properties?",
            "options": ["Prefer chains", "Prefer independent", "No preference"],
            "db_field": "accommodations.NAME (keyword matching)",
            "validation": "Match NAME against chain keywords",
            "viable": False,
            "reason": "⚠️  Low coverage (~10% chains), marginal value",
        },
    ]

    print("Viable new questions: 3/4\n")

    for q in viable_questions:
        status = "✅ RECOMMEND" if q["viable"] else "❌ SKIP"
        print(f"{status} | {q['id']}: {q['title']}")
        print(f"  Question: {q['question']}")
        print(f"  DB Field: {q['db_field']}")
        print(f"  Validation: {q['validation']}")
        print(f"  Reason: {q['reason']}")
        print()

    print("="*80)
    print("UPDATED FINAL COUNT")
    print("="*80)
    print()
    print("Current: 18 questions")
    print("Add: 3 new viable questions")
    print("= 21 questions total")
    print()
    print("Section breakdown:")
    print("  - Section 1: Accommodations (7)")
    print("  - Section 2: Restaurants (5)")
    print("  - Section 3: Flights (5)")
    print("  - Section 4: Inter-city Travel (1)")
    print("  - Section 5: New Constraints (3 existing + 3 new = 6)")
    print()
    print("Note: Still under 31 questions, good for cost/token efficiency!")
    print("="*80)


if __name__ == "__main__":
    analyze_unused_fields()
    summarize_recommendations()
