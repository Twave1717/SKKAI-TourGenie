"""Pydantic schemas for structured output parsing of TripCraft personas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal

from pydantic import BaseModel, Field


class InitialInfo(BaseModel):
    org: str
    dest: str
    days: int
    visiting_city_number: int
    date: List[str]
    people_number: int
    query: Optional[str] = ""


# Hard constraint enumerations (reference only; not enforced by Pydantic).
HARD_CONSTRAINT_ENUMS: Dict[str, List[str]] = {
    "Room Rule": [
        "No parties",
        "No smoking",
        "No children under 10",
        "No pets",
        "No visitors",
    ],
    "Room Type": [
        "Entire Room",
        "Private Room",
        "Shared Room",
        "No Shared Room",
    ],
    "Cuisine": [
        "Chinese",
        "American",
        "Italian",
        "Mexican",
        "Indian",
        "Mediterranean",
        "French",
    ],
    "Transportation Constraint": [
        "No flight",
        "No self-driving",
    ],
    "Event Types": [
        "Sports",
        "Arts & Theatre",
        "Music",
        "Film",
    ],
    "Attraction Categories": [
        "Boat Tours & Water Sports",
        "Casinos & Gambling",
        "Classes & Workshops",
        "Concerts & Shows",
        "Food & Drink",
        "Fun & Games",
        "Museums",
        "Nature & Parks",
        "Nightlife",
        "Outdoor Activities",
        "Shopping",
        "Sights & Landmarks",
        "Spas & Wellness",
        "Water & Amusement Parks",
        "Zoos & Aquariums",
    ],
}


class Profile(BaseModel):
    values: List[str] = Field(..., description="List of 3 key values or concerns (nouns).")
    focus_dimensions: List[str] = Field(..., description="List of 3 focal dimensions they care about.")
    profile_text: str = Field(..., description="One-sentence background or angle.")
    traveler_type: Optional[str] = Field(None, description="Enum aligned with persona schema.")
    purpose_of_travel: Optional[str] = Field(None, description="Enum aligned with persona schema.")
    spending_preference: Optional[str] = Field(None, description="Enum aligned with persona schema.")
    location_preference: Optional[str] = Field(None, description="Enum aligned with persona schema.")


class GroupPersona(BaseModel):
    name: str
    role: str
    profile: Profile


class GroupPersonaOutput(BaseModel):
    initial_info: InitialInfo
    group_personas: List[GroupPersona]

###########################################################################

class BudgetRequirement(BaseModel):
    value: Optional[int] = Field(None, description="Max budget for the trip in dollars.")
    alpha: float = Field(..., description="Assertiveness (0-1). Closer to 1 => hard requirement.")
    beta: float = Field(..., description="Flexibility (0-1). Typically 1 - alpha.")


class RoomRuleRequirement(BaseModel):
    value: Optional[Literal["No parties", "No smoking", "No children under 10", "No pets", "No visitors"]] = None
    alpha: float = Field(..., description="Assertiveness (0-1). Closer to 1 => hard requirement.")
    beta: float = Field(..., description="Flexibility (0-1). Typically 1 - alpha.")


class RoomTypeRequirement(BaseModel):
    value: Optional[Literal["Entire Room", "Private Room", "Shared Room", "No Shared Room"]] = None
    alpha: float = Field(..., description="Assertiveness (0-1). Closer to 1 => hard requirement.")
    beta: float = Field(..., description="Flexibility (0-1). Typically 1 - alpha.")


class CuisineRequirement(BaseModel):
    value: Optional[Literal["Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"]] = None
    alpha: float = Field(..., description="Assertiveness (0-1). Closer to 1 => hard requirement.")
    beta: float = Field(..., description="Flexibility (0-1). Typically 1 - alpha.")


class TransportationRequirement(BaseModel):
    value: Optional[Literal["No flight", "No self-driving"]] = None
    alpha: float = Field(..., description="Assertiveness (0-1). Closer to 1 => hard requirement.")
    beta: float = Field(..., description="Flexibility (0-1). Typically 1 - alpha.")


class EventRequirement(BaseModel):
    value: Optional[Literal["Sports", "Arts & Theatre", "Music", "Film"]] = None
    alpha: float = Field(..., description="Assertiveness (0-1). Closer to 1 => hard requirement.")
    beta: float = Field(..., description="Flexibility (0-1). Typically 1 - alpha.")


class AttractionRequirement(BaseModel):
    value: Optional[
        Literal[
            "Boat Tours & Water Sports",
            "Casinos & Gambling",
            "Classes & Workshops",
            "Concerts & Shows",
            "Food & Drink",
            "Fun & Games",
            "Museums",
            "Nature & Parks",
            "Nightlife",
            "Outdoor Activities",
            "Shopping",
            "Sights & Landmarks",
            "Spas & Wellness",
            "Water & Amusement Parks",
            "Zoos & Aquariums",
        ]
    ] = None
    alpha: float = Field(..., description="Assertiveness (0-1). Closer to 1 => hard requirement.")
    beta: float = Field(..., description="Flexibility (0-1). Typically 1 - alpha.")


class OptionalRequirement(BaseModel):
    preference_name: str = Field(..., description="Stable key name for this optional preference.")
    value: Optional[
        Union[
            str,
            float,
            int,
            bool,
            List[Union[str, float, int, bool, None]],
            None,
        ]
    ] = Field(..., description="Optional preference value.")
    alpha: float = Field(..., description="Assertiveness (0-1). Closer to 1 => hard requirement.")
    beta: float = Field(..., description="Flexibility (0-1). Typically 1 - alpha.")
    description: str = Field(
        ...,
        description="~50 chars; explain the preference kindly to travel companions in natural language; mention what the value means (enum or units).",
    )


class RequiredConstraints(BaseModel):
    max_budget: BudgetRequirement
    house_rule: RoomRuleRequirement
    room_type: RoomTypeRequirement
    cuisine: CuisineRequirement
    transportation: TransportationRequirement
    event: EventRequirement
    attraction: AttractionRequirement


class StructuredRequirement(BaseModel):
    required_constraints: RequiredConstraints
    optional_preferences: List[OptionalRequirement] = Field(
        default_factory=list,
        description="Optional preferences with value/alpha/beta/description.",
    )


class StructuredGroupPersona(BaseModel):
    name: str
    role: str
    profile: Profile
    structured_requirement: StructuredRequirement


class OursInputOutput(BaseModel):
    initial_info: InitialInfo
    group_personas: List[StructuredGroupPersona]
