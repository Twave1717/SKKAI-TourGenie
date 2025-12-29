"""Stage 1.5: LLM-based Alpha Survey Module.

This module handles LLM-based alpha value assignment for persona preferences.

Alpha values (0-10) indicate importance:
- 9-10: Must have (hard constraint)
- 7-8: Should have
- 4-6: Could have
- 0-3: Indifferent (soft constraint)
"""

from .prompt_builder import build_alpha_survey_prompt
from .llm_caller import call_llm_for_alpha, call_llm_for_alpha_anthropic
from .structured_output import (
    AlphaSurveyOutput,
    get_alpha_survey_schema,
    AlphaField,
    TypedAlphaField,
    AccommodationsAlpha,
    RestaurantsAlpha,
    FlightsAlpha,
    IntercityTravelAlpha,
    # Literal types for enum-able fields
    RoomTypeValue,
    PriceConsistencyValue,
    DiningAmbianceValue,
    LocationProximityValue,
    DiningFrequencyValue,
    FlightStopsValue,
    TimeWindowValue,
    ArrivalTimeValue,
    BaggageAllowanceValue,
    IntercityDurationValue,
)

# Cost optimization modules
from .batch_api import run_batch_pipeline, check_batch_status, download_batch_results, parse_batch_results
from .async_caller import run_async_pipeline, call_llm_for_alpha_async

# Legacy compatibility (if needed)
try:
    from .alpha_parser import parse_alpha_response, extract_alpha_values, categorize_by_importance
    _HAS_LEGACY_PARSER = True
except ImportError:
    _HAS_LEGACY_PARSER = False

__all__ = [
    "build_alpha_survey_prompt",
    "call_llm_for_alpha",
    "call_llm_for_alpha_anthropic",
    "AlphaSurveyOutput",
    "get_alpha_survey_schema",
    "AlphaField",
    "TypedAlphaField",
    "AccommodationsAlpha",
    "RestaurantsAlpha",
    "FlightsAlpha",
    "IntercityTravelAlpha",
    # Literal types
    "RoomTypeValue",
    "PriceConsistencyValue",
    "DiningAmbianceValue",
    "LocationProximityValue",
    "DiningFrequencyValue",
    "FlightStopsValue",
    "TimeWindowValue",
    "ArrivalTimeValue",
    "BaggageAllowanceValue",
    "IntercityDurationValue",
    # Cost optimization
    "run_batch_pipeline",
    "check_batch_status",
    "download_batch_results",
    "parse_batch_results",
    "run_async_pipeline",
    "call_llm_for_alpha_async",
]

# Add legacy exports if available
if _HAS_LEGACY_PARSER:
    __all__.extend([
        "parse_alpha_response",
        "extract_alpha_values",
        "categorize_by_importance",
    ])
