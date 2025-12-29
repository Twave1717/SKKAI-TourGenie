from __future__ import annotations

"""Stage 2 structured outputs (personal constraint seeds).

We ask the model to convert each persona's free-form preferences into
DB-groundable constraints *within* TravelPlanner's sandbox schema.

OpenAI Structured Outputs require JSON Schemas with `additionalProperties: false`
on object types. In Pydantic v2 this is achieved by:

    model_config = ConfigDict(extra='forbid')

on every model.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------
# TravelPlanner enums (from the TravelPlanner paper)
# ---------------------------

Cuisine = Literal[
    "Chinese",
    "American",
    "Italian",
    "Mexican",
    "Indian",
    "Mediterranean",
    "French",
]

RoomRule = Literal[
    "No parties",
    "No smoking",
    "No children under 10",
    "No pets",
    "No visitors",
]

RoomType = Literal[
    "Entire Room",
    "Private Room",
    "Shared Room",
    "No Shared Room",
]

TransportMode = Literal[
    "flight",
    "self-driving",
    "taxi",
]


# ---------------------------
# Constraint schema
# ---------------------------

Operator = Literal[
    "<=",
    "<",
    ">=",
    ">",
    "==",
    "!=",
    "in",
    "not_in",
    "contains",
    "not_contains",
    "contains_any",
    "contains_all",
    "between",
]

# Keep fields compact but explicit: the prefix implies the DB table.
FieldName = Literal[
    # Restaurants
    "restaurant.cuisine",
    "restaurant.avg_cost",
    "restaurant.rating",
    "restaurant.name_keyword",
    "restaurant.count",

    # Accommodations
    "accommodation.room_type",
    "accommodation.house_rule",
    "accommodation.price",
    "accommodation.review",
    "accommodation.minimum_nights",
    "accommodation.maximum_occupancy",
    "accommodation.name_keyword",
    "accommodation.count",

    # Attractions
    "attraction.name_keyword",
    "attraction.address_keyword",
    "attraction.count",

    # Flights
    "flight.price",
    "flight.duration_minutes",
    "flight.depart_time",
    "flight.arrive_time",
    "flight.distance",
    "flight.count",

    # Ground transport (self-driving / taxi)
    "ground.mode",
    "ground.cost",
    "ground.duration_minutes",
    "ground.distance",
]

ConstraintValue = Union[str, int, float, List[str], List[int], List[float]]


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: FieldName
    op: Operator
    value: ConstraintValue

    # Negotiation / evaluation hooks
    hard: bool = Field(..., description="If true, this constraint is non-negotiable (Hard).")
    description: str = Field(..., min_length=5, max_length=240, description="Natural-language requirement statement (used later in discussion/consensus).")

    alpha: float = Field(..., ge=0.0, le=1.0, description="Assertiveness / importance (0..1)")
    beta: float = Field(..., ge=0.0, le=1.0, description="Flexibility (0..1). Recommend beta = 1 - alpha.")


class StructuredRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_constraints: List[Constraint] = Field(default_factory=list, min_length=1, max_length=6)
    soft_constraints: List[Constraint] = Field(default_factory=list, min_length=2, max_length=10)


class PersonaRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_index: int = Field(..., ge=0)
    structured_requirement: StructuredRequirement


class Stage2LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persona_requirements: List[PersonaRequirement] = Field(..., min_length=1)
