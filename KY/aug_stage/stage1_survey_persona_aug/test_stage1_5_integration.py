#!/usr/bin/env python3
"""Integration test for Stage 1.5 alpha survey pipeline.

Usage:
    python3 test_stage1_5_integration.py
"""

from pathlib import Path
import json
from stage1_5 import build_alpha_survey_prompt, AlphaSurveyOutput


def test_schema_generation():
    """Test structured output schema generation."""
    from stage1_5.structured_output import get_alpha_survey_schema

    schema = get_alpha_survey_schema()
    assert "properties" in schema
    assert "persona_id" in schema["properties"]
    assert "accommodations" in schema["properties"]

    print("✅ Schema generation test passed")


def test_prompt_building():
    """Test prompt builder with sample persona."""
    sample_persona = {
        "ref_id": "stravl_test_123",
        "age_range": "20-39",
        "travel_frequency": "2-3 times per year",
        "budget_tier": "Budget",
        "season_preference": "Summer",
        "activity_level": "Very Active",
        "safety_conscious": "Very Safety Conscious",
        "destination_popularity": "Popular destinations",
        "travel_experiences": ["Beach", "Adventure", "Nature"],
        "scenery_preferences": ["Sea", "Mountain"],
    }

    trip_context = {
        "people_number": 4,
        "days": 5,
        "budget_anchor": 2000,
        "org": "New York",
        "dest": "Miami",
    }

    prompt = build_alpha_survey_prompt(sample_persona, trip_context)

    # Check key elements in prompt
    assert "stravl_test_123" in prompt
    assert "Budget" in prompt
    assert "Very Active" in prompt
    assert "4 people" in prompt
    assert "$2000" in prompt
    assert "ALL 31 fields" in prompt or "31 questions" in prompt

    print("✅ Prompt building test passed")
    print(f"   Prompt length: {len(prompt)} chars")


def test_pydantic_validation():
    """Test Pydantic model validation."""
    sample_output = {
        "persona_id": "stravl_test",
        "accommodations": {
            "price": {"value": "budget", "importance_score": 8, "reason": "Budget traveler"},
            "rating": {"value": 3.5, "importance_score": 5, "reason": "Moderate"},
            "room_type": {"value": "double", "importance_score": 2, "reason": "Flexible"},
            "house_rule": {"value": "no_smoking", "importance_score": 7, "reason": "Safety"},
            "minimum_nights": {"value": 1, "importance_score": 1, "reason": "Flexible"},
            "maximum_occupancy": {"value": 4, "importance_score": 8, "reason": "Group of 4"},
            "review_score": {"value": 7.0, "importance_score": 6, "reason": "Important"},
            "breakfast_included": {"value": True, "importance_score": 4, "reason": "Nice to have"},
        },
        "restaurants": {
            "price": {"value": "budget", "importance_score": 8, "reason": "Budget"},
            "rating": {"value": 4.0, "importance_score": 5, "reason": "Quality"},
            "cuisine_type": {"value": ["local", "seafood"], "importance_score": 6, "reason": "Beach destination"},
            "dietary_restrictions": {"value": None, "importance_score": 0, "reason": "None"},
            "ambiance": {"value": "casual", "importance_score": 3, "reason": "Relaxed"},
            "location_proximity": {"value": "walking_distance", "importance_score": 7, "reason": "Convenience"},
        },
        "flights": {
            "price": {"value": "economy", "importance_score": 9, "reason": "Budget priority"},
            "stops": {"value": "any", "importance_score": 2, "reason": "Flexible"},
            "class": {"value": "economy", "importance_score": 1, "reason": "Budget"},
            "departure_time": {"value": "flexible", "importance_score": 2, "reason": "No preference"},
            "arrival_time": {"value": "flexible", "importance_score": 2, "reason": "No preference"},
            "airline_preference": {"value": None, "importance_score": 0, "reason": "No preference"},
            "baggage_allowance": {"value": "checked_bag", "importance_score": 5, "reason": "Beach gear"},
        },
        "attractions": {
            "rating": {"value": 4.0, "importance_score": 6, "reason": "Quality experiences"},
            "popularity": {"value": "popular", "importance_score": 4, "reason": "Balanced"},
            "entry_fee": {"value": "low", "importance_score": 7, "reason": "Budget conscious"},
            "activity_type": {"value": ["beach", "nature", "adventure"], "importance_score": 8, "reason": "Active traveler"},
            "accessibility": {"value": "standard", "importance_score": 3, "reason": "No special needs"},
        },
        "intercity_travel": {
            "mode": {"value": "bus", "importance_score": 6, "reason": "Budget option"},
            "price": {"value": "budget", "importance_score": 8, "reason": "Budget"},
            "duration": {"value": "flexible", "importance_score": 3, "reason": "Not critical"},
            "comfort": {"value": "standard", "importance_score": 4, "reason": "Basic ok"},
            "scenic_route": {"value": True, "importance_score": 6, "reason": "Nature lover"},
        },
    }

    # Validate
    validated = AlphaSurveyOutput(**sample_output)

    # Test model_dump
    dumped = validated.model_dump()
    assert dumped["persona_id"] == "stravl_test"
    assert dumped["accommodations"]["price"]["importance_score"] == 8

    # Test alias handling for flight.class
    dumped_alias = validated.model_dump(by_alias=True)
    assert "class" in str(dumped_alias["flights"])

    print("✅ Pydantic validation test passed")


def test_stage1_7_compatibility():
    """Test Stage 1.7 compatibility with new format."""
    from stage1_7 import extract_alpha_values, check_field_conflict

    sample_persona = {
        "ref_id": "stravl_1",
        "alpha_survey": {
            "accommodations": {
                "price": {"value": "budget", "importance_score": 8, "reason": "test"},
                "rating": {"value": 3.5, "importance_score": 5, "reason": "test"},
            },
            "flights": {
                "price": {"value": "economy", "importance_score": 9, "reason": "test"},
            },
        },
    }

    # Test extraction
    alphas = extract_alpha_values(sample_persona)
    assert "accommodations.price" in alphas
    assert alphas["accommodations.price"] == 8
    assert alphas["flights.price"] == 9

    print("✅ Stage 1.7 compatibility test passed")


def main():
    print("=" * 60)
    print("Stage 1.5 Integration Tests")
    print("=" * 60)
    print()

    test_schema_generation()
    test_prompt_building()
    test_pydantic_validation()
    test_stage1_7_compatibility()

    print()
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run small-scale test: python3 run_stage1_5.py --max_records 2")
    print("2. Verify output format in outputs/stage1_5/test/data/")
    print("3. Check cost: ~$0.02 per persona × 40 personas = ~$0.80 per trip")


if __name__ == "__main__":
    main()
