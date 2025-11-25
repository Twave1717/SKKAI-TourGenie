"""
데이터 구조 정의: Issue, 자연어 제약, DSL 제약, 회의 결과 등.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class Issue:
    issue_id: str
    persona: str
    dimension: str
    severity: int
    description: str
    proposed_constraint_nl: str
    plan_reference: Optional[str] = None


@dataclass
class NaturalConstraint:
    constraint_id: str
    constraint_nl: str
    dimension: str
    support_personas: Sequence[str] = field(default_factory=list)
    avg_severity: Optional[float] = None
    importance_score: Optional[float] = None  # 투표 기반 가중치


@dataclass
class FormalConstraint:
    constraint_id: str
    type: str  # "hard" or "soft"
    dsl: str
    weight: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PlanInput:
    user_query: str
    predefined_constraints: Sequence[str]
    plan_path: str  # P0 경로(예: tripcraft_5day.csv)
    plan_payload: Optional[Dict] = None  # 파서가 넣는 실제 구조화 계획
    sample_id: Optional[str] = None  # 로그 저장 시 식별자


@dataclass
class MeetingOutcome:
    issues: List[Issue]
    natural_constraints: List[NaturalConstraint]
    formal_constraints: List[FormalConstraint]
    plan_variant_path: Optional[str] = None
    logs: Dict[str, str] = field(default_factory=dict)


@dataclass
class PersonaVote:
    persona: str
    scores: Dict[str, float]  # constraint_id -> 점수(예: 0~100 예산 분배)
