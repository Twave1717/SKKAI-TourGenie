from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class Example(BaseModel):
    id: str = Field(..., description="Unique identifier for the example")
    query: str = Field(..., description="User query / prompt for TravelPlanner")
    expected: Optional[str] = Field(
        default=None,
        description="Serialized reference plan/answer when available (e.g., annotated_plan)",
    )
    reference_information: Optional[Any] = Field(
        default=None,
        description="Auxiliary documents provided with the query",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional fields from the dataset"
    )

    model_config = ConfigDict(extra="allow")


class Prediction(BaseModel):
    id: str
    query: str
    prediction: str
    raw_prediction: Optional[str] = None
    expected: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
