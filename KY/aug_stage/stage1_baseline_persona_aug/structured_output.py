from __future__ import annotations

"""Stage 1 structured output schema (Pydantic v2).

This Stage 1 produces *group personas* from a single TravelPlanner record.

Key design changes (refactor):
- Removes TripCraft-residual persona enums (traveler_type / purpose_of_travel / spending_preference / location_preference).
- Uses Stravl-style survey dimensions for persona profiles (age / budget tier / experiences / scenery / etc.).
- Keeps Stage 1 free of ground-truth DB/tool calls; DB-groundable constraints are handled in later stages.

OpenAI Structured Outputs require JSON Schema objects with `additionalProperties: false`.
In Pydantic v2, set `model_config = ConfigDict(extra="forbid")` for every object model.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------
# Stravl-style enums (labels)
# ---------------------------

AgeRange = Literal["0-19", "20-39", "40-59", "60+"]

# NOTE: Relative spending tiers (not literal dollar ranges).
BudgetTier = Literal["Frugal", "Budget", "Comfort", "Luxury"]

Season = Literal["Winter", "Spring", "Summer", "Fall"]

Experience = Literal[
    "Beach",
    "Adventure",
    "Nature",
    "Culture",
    "Nightlife",
    "History",
    "Shopping",
    "Cuisine",
]

Scenery = Literal[
    "Urban",
    "Rural",
    "Sea",
    "Mountain",
    "Lake",
    "Desert",
    "Plains",
    "Jungle",
]

ActivityLevel = Literal["Chill & Relaxed", "Balanced", "Active"]

SafetyLevel = Literal["Very Safety Conscious", "Balanced", "Ready for Anything"]

Popularity = Literal["Off the Beaten Path", "Classic Spot", "Mainstream & Trendy"]


# ---------------------------
# Persona archetypes (discussion roles)
# ---------------------------

Archetype = Literal[
    "Budget Guardian",
    "Foodie Planner",
    "Comfort Seeker",
    "Culture & Museums",
    "Outdoor & Nature",
    "Nightlife & Entertainment",
    "Logistics Optimizer",
    "Family-Oriented",
    "Photographer",
    "Spontaneous Explorer",
]


# ---------------------------
# Input/Output models
# ---------------------------


class LocalConstraintAnchor(BaseModel):
    """TravelPlanner local constraints (original schema, kept as anchors only).

    This is NOT a persona enum; it's the scenario-level anchor from the dataset.
    """
    model_config = ConfigDict(extra="forbid")

    # Keep TravelPlanner keys via aliases.
    house_rule: Optional[str] = Field(default=None, alias="house rule")
    cuisine: Optional[str] = None
    room_type: Optional[str] = Field(default=None, alias="room type")
    transportation: Optional[str] = None


class InitialInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    org: Optional[str] = None
    dest: Optional[str] = None
    days: int = Field(..., ge=1)
    visiting_city_number: int = Field(..., ge=1)
    date: List[str] = Field(default_factory=list)
    people_number: int = Field(..., ge=1)
    query: str = ""

    # Augmentation-only anchors
    budget_anchor: Optional[int] = Field(default=None, ge=0)
    local_constraint_anchor: LocalConstraintAnchor = Field(default_factory=LocalConstraintAnchor)

    level: Optional[str] = None


class PersonaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_text: str = Field(..., description="Exactly 1 sentence.")

    # Stravl-style structured dimensions
    age_range: AgeRange
    budget_tier: BudgetTier
    season: Season

    experiences: List[Experience] = Field(default_factory=list, min_length=1, max_length=4)
    scenery: List[Scenery] = Field(default_factory=list, min_length=1, max_length=4)

    activity_level: ActivityLevel
    safety_conscious: SafetyLevel
    destination_popularity: Popularity


class BudgetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Budget preference as a relative ratio to the scenario's budget_anchor.
    # Stage 2/consensus can later resolve conflicts using these multipliers.
    max_budget_multiplier: float = Field(..., ge=0.7, le=2.0)


class Stage1Persona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    archetype: Archetype

    profile: PersonaProfile
    budget_profile: BudgetProfile

    # Debug-only anchors, later reused by discussion prompts.
    grounding_anchors: List[str] = Field(default_factory=list, min_length=2, max_length=4)

    # 3~5 short natural-language preferences (soft), for Stage2 conversion / negotiation.
    seed_preferences: List[str] = Field(default_factory=list, min_length=3, max_length=5)


class Stage1Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    initial_info: InitialInfo
    group_personas: List[Stage1Persona]
