"""
구성 객체: 페르소나/회의/솔버 설정.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class PersonaConfig:
    name: str
    role: str
    focus_dimensions: Sequence[str]
    values: Sequence[str]
    profile: str = ""
    language: str = "ko"
    max_issues: int = 5


@dataclass
class MeetingConfig:
    personas: Sequence[PersonaConfig]
    moderator: str = "Moderator"
    top_constraints: int = 10
    aggregation_threshold: float = 0.0
    language: str = "ko"


@dataclass
class BenchmarkConfig:
    name: str
    schema_hint: str  # 엔지니어에게 제공할 스키마 설명
    plan_loader: str  # 파서 함수 경로 문자열 예: "tripcraft.load_plan"
    default_plan_path: Optional[str] = None
