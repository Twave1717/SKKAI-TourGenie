"""
Rule-based aggregation of AtomicConstraint to NaturalConstraint.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .structures import AtomicConstraint, NaturalConstraint


@dataclass
class AggregationPolicy:
    persona_weights: Dict[str, float] = field(default_factory=dict)
    numeric_mode: str = "min"  # min|max|median
    enum_mode: str = "intersection_first"  # intersection_first|union
    conflict_threshold: float = 0.3  # fraction of minority to mark conflict


def _weighted_avg(values: Sequence[float], weights: Sequence[float]) -> float:
    total_w = sum(weights)
    if not total_w:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def aggregate_atomic_constraints(constraints: List[AtomicConstraint], policy: AggregationPolicy | None = None) -> Tuple[List[NaturalConstraint], str]:
    """
    Group atomic constraints by (dimension, attr, constraint_type) and merge to NaturalConstraint.
    Returns (constraints, debug_json).
    """
    policy = policy or AggregationPolicy()
    groups: Dict[tuple, List[AtomicConstraint]] = defaultdict(list)
    for c in constraints:
        key = (c.dimension, c.attr, c.constraint_type)
        groups[key].append(c)

    natural: List[NaturalConstraint] = []
    debug_payload = []
    for key, items in groups.items():
        dimension, attr, constraint_type = key
        personas = [c.persona_id for c in items]
        weights = [policy.persona_weights.get(pid, 1.0) for pid in personas]
        severities = [c.priority for c in items]
        avg_severity = _weighted_avg(severities, weights) if severities else None
        operator = items[0].operator
        conflict = False

        agg_value = None
        if operator in {"<=", ">=", "=="}:
            numeric_vals = [c.value for c in items if isinstance(c.value, (int, float))]
            if not numeric_vals:
                continue
            if operator == "<=":
                agg_value = min(numeric_vals)
            elif operator == ">=":
                agg_value = max(numeric_vals)
            else:
                # equality는 가중 평균으로 정규화
                agg_value = _weighted_avg(numeric_vals, weights[: len(numeric_vals)])
        elif operator in {"in", "not_in"}:
            sets = []
            for c in items:
                if isinstance(c.value, list):
                    sets.append(set(c.value))
                elif isinstance(c.value, str):
                    sets.append({c.value})
            if not sets:
                continue
            if operator == "in":
                inter = set.intersection(*sets)
                if inter:
                    agg_value = sorted(inter)
                else:
                    agg_value = sorted(set.union(*sets))
                    conflict = True
            else:
                agg_value = sorted(set.union(*sets))
        else:
            # boolean/soft 등 기타는 최빈값 기준
            bools = []
            for c in items:
                val = c.value
                if isinstance(val, bool):
                    bools.append(val)
                elif isinstance(val, str):
                    bools.append(val.lower() in {"1", "true", "yes", "y"})
            if bools:
                true_ratio = sum(bools) / len(bools)
                agg_value = True if true_ratio >= 0.5 else False
                if 0.5 <= true_ratio <= 0.5 + policy.conflict_threshold:
                    conflict = True
            else:
                agg_value = items[0].value

        cid = f"{attr}-{constraint_type}"
        summary_parts = [f"{attr} {operator} {agg_value}"]
        if conflict:
            summary_parts.append("[conflict: union/majority used]")
        constraint_nl = " / ".join(summary_parts)
        natural.append(
            NaturalConstraint(
                constraint_id=cid,
                constraint_nl=constraint_nl,
                dimension=dimension,
                support_personas=list(dict.fromkeys(personas)),
                avg_severity=avg_severity,
                importance_score=avg_severity,
            )
        )
        debug_payload.append(
            {
                "key": {"dimension": dimension, "attr": attr, "constraint_type": constraint_type},
                "operator": operator,
                "personas": personas,
                "values": [c.value for c in items],
                "agg_value": agg_value,
                "avg_severity": avg_severity,
                "conflict": conflict,
            }
        )

    return natural, json_dumps(debug_payload)


def json_dumps(payload) -> str:
    try:
        import json

        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return str(payload)
