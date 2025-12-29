from __future__ import annotations

"""Stage 1.5 Alpha Survey Structured Output Schema (Pydantic v2).

This module defines the structured output schema for the LLM-based alpha survey.
Alpha values (0-10) indicate preference importance across 20 database fields.

OpenAI Structured Outputs require JSON Schema objects with `additionalProperties: false`.
In Pydantic v2, set `model_config = ConfigDict(extra="forbid")` for every object model.
"""

from typing import Any, List, Union, Literal, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Type variable for generic AlphaField
T = TypeVar('T')


# ---------------------------
# Alpha Survey Field Models
# ---------------------------

# Define Literal types for enum-able fields
RoomTypeValue = Literal[
    "Entire home/apt (Privacy critical)",
    "Private room (Standard)",
    "Shared room (Budget saver)",
    "Any / Flexible"
]

PriceConsistencyValue = Literal[
    "Consistent (similar prices for all accommodations)",
    "Moderate variation (some ups and downs acceptable)",
    "High variation OK (I prioritize best value, even if prices vary)",
    "No preference"
]

DiningAmbianceValue = Literal[
    "Casual / Street Food",
    "Family-friendly / Quiet",
    "Lively / Bar-style",
    "Romantic / Fine Dining",
    "No preference"
]

LocationProximityValue = Literal[
    "Must be walking distance (<10 min)",
    "Short drive/transit ok (<20 min)",
    "Willing to travel for good food",
    "No preference"
]

DiningFrequencyValue = Literal[
    "1 meal/day (mostly self-catering or hotel breakfast)",
    "2 meals/day (typical dining out)",
    "3 meals/day (all meals at restaurants)",
    "Flexible (depends on the day)"
]

FlightStopsValue = Literal[
    "Direct flights ONLY",
    "Direct preferred, 1 stop acceptable",
    "Multiple stops acceptable (for savings)",
    "No preference"
]

TimeWindowValue = Literal[
    "Morning (06:00 - 12:00)",
    "Afternoon (12:00 - 18:00)",
    "Evening/Night (18:00+)",
    "Flexible"
]

ArrivalTimeValue = Literal[
    "Morning (Arrive early to play)",
    "Afternoon (Check-in time)",
    "Evening/Night (Sleep immediately)",
    "Flexible"
]

BaggageAllowanceValue = Literal[
    "Must include checked bag",
    "Carry-on only is fine",
    "No preference"
]

IntercityDurationValue = Literal[
    "Under 2 hours",
    "2 - 4 hours",
    "4+ hours is fine",
    "No preference"
]


class AlphaField(BaseModel):
    """Individual field with alpha importance score (generic)."""

    model_config = ConfigDict(extra="forbid")

    value: Union[str, int, float, bool, List[str], None] = Field(
        ..., description="Preferred value for this field"
    )
    importance_score: int = Field(
        ..., ge=0, le=10, description="Alpha value (0-10): 9-10=MUST, 7-8=SHOULD, 4-6=COULD, 0-3=INDIFFERENT"
    )
    reason: str = Field(..., description="Brief explanation for this preference and importance level")


class TypedAlphaField(BaseModel, Generic[T]):
    """Typed field with alpha importance score."""

    model_config = ConfigDict(extra="forbid")

    value: T = Field(..., description="Preferred value for this field")
    importance_score: int = Field(
        ..., ge=0, le=10, description="Alpha value (0-10): 9-10=MUST, 7-8=SHOULD, 4-6=COULD, 0-3=INDIFFERENT"
    )
    reason: str = Field(..., description="Brief explanation for this preference and importance level")


# ---------------------------
# Category-specific Models
# ---------------------------


class AccommodationsAlpha(BaseModel):
    """Accommodations preferences (8 fields)."""

    model_config = ConfigDict(extra="forbid")

    price: AlphaField = Field(..., description="Q1-1: Hotel price tier per night preference")
    rating: AlphaField = Field(..., description="Q1-2: Minimum acceptable hotel star rating (0-5)")
    room_type: TypedAlphaField[RoomTypeValue] = Field(..., description="Q1-3: Room type requirement (entire/private/shared/flexible)")
    house_rules: AlphaField = Field(
        ..., description="Q1-4: Specific house rules requirements (smoking/pets/parties/children/visitors)"
    )
    minimum_nights: AlphaField = Field(..., description="Q1-5: Flexibility about minimum nights policies")
    maximum_occupancy: AlphaField = Field(..., description="Q1-6: Minimum room capacity needed for group")
    review_score: AlphaField = Field(..., description="Q1-7: Minimum acceptable review score (out of 5.0)")
    price_consistency: TypedAlphaField[PriceConsistencyValue] = Field(
        ..., description="Q1-8: Preference for consistent pricing vs price variations (consistent/moderate/high_variation/none)"
    )


class RestaurantsAlpha(BaseModel):
    """Restaurant preferences (6 fields)."""

    model_config = ConfigDict(extra="forbid")

    price: AlphaField = Field(..., description="Q2-1: Dining budget tier per person per meal")
    rating: AlphaField = Field(..., description="Q2-2: Minimum acceptable restaurant rating (out of 5.0)")
    cuisine_type: AlphaField = Field(..., description="Q2-3: Mandatory or banned cuisines")
    ambiance: TypedAlphaField[DiningAmbianceValue] = Field(
        ..., description="Q2-4: Dining atmosphere preference (casual/family/lively/romantic/none)"
    )
    location_proximity: TypedAlphaField[LocationProximityValue] = Field(
        ..., description="Q2-5: Location convenience importance (walking/short_drive/willing_to_travel/none)"
    )
    dining_frequency: TypedAlphaField[DiningFrequencyValue] = Field(
        ..., description="Q2-6: Planned meals per day at restaurants (1/2/3/flexible)"
    )


class FlightsAlpha(BaseModel):
    """Flight preferences (5 fields)."""

    model_config = ConfigDict(extra="forbid")

    price: AlphaField = Field(..., description="Q3-1: Ticket price tier preference")
    stops: TypedAlphaField[FlightStopsValue] = Field(..., description="Q3-2: Tolerance for layovers/stops (direct/one_stop/multiple/flexible)")
    departure_time: TypedAlphaField[TimeWindowValue] = Field(
        ..., description="Q3-3: Preferred departure time window (morning/afternoon/evening/flexible)"
    )
    arrival_time: TypedAlphaField[ArrivalTimeValue] = Field(
        ..., description="Q3-4: Preferred arrival time window (morning/afternoon/evening/flexible)"
    )
    baggage_allowance: TypedAlphaField[BaggageAllowanceValue] = Field(
        ..., description="Q3-5: Baggage allowance importance (checked_bag/carry_on/none)"
    )


class IntercityTravelAlpha(BaseModel):
    """Inter-city travel preferences (1 field)."""

    model_config = ConfigDict(extra="forbid")

    duration: TypedAlphaField[IntercityDurationValue] = Field(..., description="Q4-1: Maximum travel duration tolerance between cities")


# ---------------------------
# Top-level Output Model
# ---------------------------


class AlphaSurveyOutput(BaseModel):
    """Complete alpha survey output for a single persona.

    Contains importance scores (0-10) for all 20 database fields across 4 categories.
    """

    model_config = ConfigDict(extra="forbid")

    persona_id: str = Field(..., description="Reference ID of the persona being surveyed (e.g., stravl_1234)")

    accommodations: AccommodationsAlpha
    restaurants: RestaurantsAlpha
    flights: FlightsAlpha
    intercity_travel: IntercityTravelAlpha


# ---------------------------
# Helper function
# ---------------------------


def get_alpha_survey_schema() -> dict[str, Any]:
    """Return the JSON schema for OpenAI structured outputs.

    Usage:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[...],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "alpha_survey",
                    "schema": get_alpha_survey_schema(),
                    "strict": True
                }
            }
        )
    """
    return AlphaSurveyOutput.model_json_schema()
