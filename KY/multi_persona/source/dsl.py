"""
최소 실험용 DSL 래퍼.
- NL 제약을 그대로 FormalConstraint에 담아 반환한다.
- 별도 LLM 호출 없음.
"""

from typing import List

from .structures import FormalConstraint, NaturalConstraint


def nl_to_dsl(constraints: List[NaturalConstraint], schema_hint: str, llm, default_type: str = "hard") -> List[FormalConstraint]:
    formal: List[FormalConstraint] = []
    for c in constraints:
        formal.append(
            FormalConstraint(
                constraint_id=c.constraint_id,
                type=default_type,
                dsl=c.constraint_nl,  # NL을 그대로 DSL 필드에 저장
                weight=c.importance_score,
                metadata={"note": "passthrough: NL kept as DSL"},
            )
        )
    return formal
