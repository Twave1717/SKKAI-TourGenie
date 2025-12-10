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
class AtomicConstraint:
    """
    Per-person constraint extracted from a persona issue.
    - priority: 1~5 (severity 기반), hard: True/False
    - operator: <=, >=, ==, in, not_in
    """

    persona_id: str
    dimension: str
    attr: str
    constraint_type: str  # e.g., upper_bound, lower_bound, enum_inclusion, enum_exclusion, boolean, soft_preference
    operator: str  # <=, >=, ==, in, not_in
    value: object
    priority: int
    hard: bool
    nl_summary: str
    plan_reference: Optional[str] = None

    def __post_init__(self) -> None:
        allowed_ops = {"<=", ">=", "==", "in", "not_in"}
        if self.operator not in allowed_ops:
            raise ValueError(f"Unsupported operator: {self.operator}")
        if not 1 <= self.priority <= 5:
            raise ValueError(f"priority must be 1~5, got {self.priority}")
        if not self.dimension:
            raise ValueError("dimension is required")
        if not self.attr:
            raise ValueError("attr is required")
        if not self.constraint_type:
            raise ValueError("constraint_type is required")
        if not self.persona_id:
            raise ValueError("persona_id is required")
        if not self.nl_summary:
            raise ValueError("nl_summary is required")


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
    atomic_constraints: Optional[List[AtomicConstraint]] = None
    aggregated_constraints_rule: Optional[List[NaturalConstraint]] = None
    logs: Dict[str, str] = field(default_factory=dict)


@dataclass
class PersonaVote:
    persona: str
    scores: Dict[str, float]  # constraint_id -> 점수(예: 0~100 예산 분배)


@dataclass
class ConstraintParam:
    name: str
    type: str  # int|float|str|enum|bool
    enum_values: Optional[List[str]] = None
    description: str = ""


@dataclass
class ConstraintSchema:
    id: str
    dimension: str
    description: str
    dsl_template: str
    params: List[ConstraintParam]
    is_hard_default: bool = True
    tags: List[str] = field(default_factory=list)
