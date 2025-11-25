"""
멀티 페르소나 회의 오케스트레이션 (단순 불만/제약 생성용, 솔버 없음).
"""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel, Field

from .config import BenchmarkConfig, MeetingConfig, PersonaConfig
from .dsl import nl_to_dsl
from .llm import LLMClient
from .logger import persist_outcome
from .prompts import AGGREGATOR_TEMPLATE, render_moderator_prompt, render_persona_prompt
from .structures import FormalConstraint, Issue, MeetingOutcome, NaturalConstraint, PlanInput


class IssueModel(BaseModel):
    issue_id: str
    dimension: str
    severity: int = Field(ge=0, le=5)
    description: str
    proposed_constraint_nl: str
    plan_reference: str | None = None


class IssueList(BaseModel):
    items: List[IssueModel]


class ConstraintModel(BaseModel):
    constraint_id: str
    constraint_nl: str
    dimension: str
    support_personas: List[str] = Field(default_factory=list)
    avg_severity: float | None = None
    importance_score: float | None = None


class ConstraintList(BaseModel):
    items: List[ConstraintModel]


def _plan_to_text(plan_payload) -> str:
    if isinstance(plan_payload, (dict, list)):
        try:
            return json.dumps(plan_payload, ensure_ascii=False, indent=2)
        except Exception:
            return str(plan_payload)
    return str(plan_payload)


def collect_persona_issues(persona: PersonaConfig, plan_text: str, llm: LLMClient) -> Tuple[List[Issue], str]:
    prompt = render_persona_prompt(
        name=persona.name,
        role=persona.role,
        values=persona.values,
        focus_dimensions=persona.focus_dimensions,
        profile=persona.profile,
        max_issues=persona.max_issues,
    )
    parsed = llm.generate_structured(f"{prompt}\n\n계획:\n{plan_text}", schema=IssueList)
    issues: List[Issue] = []
    seen = set()
    for idx, item in enumerate(parsed.items):
        iid = item.issue_id or f"{persona.name}-{idx}"
        if iid in seen:
            iid = f"{iid}-{idx}"
        seen.add(iid)
        issues.append(
            Issue(
                issue_id=iid,
                persona=persona.name,
                dimension=item.dimension,
                severity=item.severity,
                description=item.description,
                proposed_constraint_nl=item.proposed_constraint_nl,
                plan_reference=item.plan_reference,
            )
        )
    if not issues:
        raise RuntimeError("LLM 응답에서 issue를 찾지 못했습니다.")
    return issues, json.dumps(parsed.model_dump(), ensure_ascii=False)


def aggregate_issues(issues: Sequence[Issue], llm: LLMClient) -> Tuple[List[NaturalConstraint], str]:
    parsed = llm.generate_structured(
        f"{AGGREGATOR_TEMPLATE}\n\n이슈들:\n{json.dumps([i.__dict__ for i in issues], ensure_ascii=False)}",
        schema=ConstraintList,
    )
    constraints: List[NaturalConstraint] = []
    seen = set()
    for idx, item in enumerate(parsed.items):
        cid = item.constraint_id or f"C-{idx}"
        if cid in seen:
            cid = f"{cid}-{idx}"
        seen.add(cid)
        constraints.append(
            NaturalConstraint(
                constraint_id=cid,
                constraint_nl=item.constraint_nl,
                dimension=item.dimension,
                support_personas=item.support_personas,
                avg_severity=item.avg_severity,
                importance_score=item.importance_score,
            )
        )
    if not constraints:
        raise RuntimeError("LLM 응답에서 constraint를 찾지 못했습니다.")
    return constraints, json.dumps(parsed.model_dump(), ensure_ascii=False)


def moderate_constraints(nl_constraints: Sequence[NaturalConstraint], llm: LLMClient) -> Tuple[List[NaturalConstraint], str]:
    payload = [c.__dict__ for c in nl_constraints]
    parsed = llm.generate_structured(
        f"{render_moderator_prompt()}\n\n초안 constraint들:\n{json.dumps(payload, ensure_ascii=False)}",
        schema=ConstraintList,
    )
    moderated: List[NaturalConstraint] = []
    seen = set()
    for idx, item in enumerate(parsed.items):
        cid = item.constraint_id or f"C-{idx}"
        if cid in seen:
            cid = f"{cid}-{idx}"
        seen.add(cid)
        moderated.append(
            NaturalConstraint(
                constraint_id=cid,
                constraint_nl=item.constraint_nl,
                dimension=item.dimension,
                support_personas=item.support_personas,
                avg_severity=item.avg_severity,
                importance_score=item.importance_score,
            )
        )
    if not moderated:
        raise RuntimeError("LLM 응답에서 moderator 결과를 찾지 못했습니다.")
    return moderated, json.dumps(parsed.model_dump(), ensure_ascii=False)


class MeetingRunner:
    def __init__(self, meeting_cfg: MeetingConfig, bench_cfg: BenchmarkConfig, llm: LLMClient):
        self.meeting_cfg = meeting_cfg
        self.bench_cfg = bench_cfg
        self.llm = llm

    def run_m2(self, plan_input: PlanInput, persona: PersonaConfig) -> MeetingOutcome:
        plan_text = _plan_to_text(plan_input.plan_payload or plan_input.plan_path)
        issues, raw_persona = collect_persona_issues(persona, plan_text, self.llm)
        nl_constraints, raw_agg = aggregate_issues(issues, self.llm)
        formal = nl_to_dsl(nl_constraints, schema_hint=self.bench_cfg.schema_hint, llm=self.llm)
        outcome = MeetingOutcome(
            issues=issues,
            natural_constraints=nl_constraints,
            formal_constraints=formal,
            plan_variant_path=None,
            logs={
                "plan": json.dumps({"plan_payload": plan_input.plan_payload, "solver": "none"}, ensure_ascii=False),
                "persona_raw": json.dumps({persona.name: raw_persona}, ensure_ascii=False),
                "aggregator_raw": raw_agg,
            },
        )
        # 로그 저장
        sample_id = plan_input.sample_id or Path(plan_input.plan_path).stem
        saved = persist_outcome(outcome, sample_id=sample_id)
        outcome.logs["saved_files"] = json.dumps(saved, ensure_ascii=False)
        return outcome

    def run_m3(self, plan_input: PlanInput, participant_count: int = None) -> MeetingOutcome:
        plan_text = _plan_to_text(plan_input.plan_payload or plan_input.plan_path)
        all_issues: List[Issue] = []
        persona_raw = {}
        personas = self._select_personas(participant_count)
        for persona in personas:
            issues, raw = collect_persona_issues(persona, plan_text, self.llm)
            all_issues.extend(issues)
            persona_raw[persona.name] = raw
        nl_constraints, raw_agg = aggregate_issues(all_issues, self.llm)
        formal = nl_to_dsl(nl_constraints, schema_hint=self.bench_cfg.schema_hint, llm=self.llm)
        outcome = MeetingOutcome(
            issues=all_issues,
            natural_constraints=nl_constraints,
            formal_constraints=formal,
            plan_variant_path=None,
            logs={
                "plan": json.dumps({"plan_payload": plan_input.plan_payload, "solver": "none"}, ensure_ascii=False),
                "persona_raw": json.dumps(persona_raw, ensure_ascii=False),
                "aggregator_raw": raw_agg,
            },
        )
        sample_id = plan_input.sample_id or Path(plan_input.plan_path).stem
        saved = persist_outcome(outcome, sample_id=sample_id)
        outcome.logs["saved_files"] = json.dumps(saved, ensure_ascii=False)
        return outcome

    def run_m4(self, plan_input: PlanInput, participant_count: int = None) -> MeetingOutcome:
        plan_text = _plan_to_text(plan_input.plan_payload or plan_input.plan_path)
        issues: List[Issue] = []
        persona_raw = {}
        personas = self._select_personas(participant_count)
        for persona in personas:
            collected, raw = collect_persona_issues(persona, plan_text, self.llm)
            issues.extend(collected)
            persona_raw[persona.name] = raw
        aggregated, raw_agg = aggregate_issues(issues, self.llm)
        moderated, raw_mod = moderate_constraints(aggregated, self.llm)
        formal = nl_to_dsl(moderated, schema_hint=self.bench_cfg.schema_hint, llm=self.llm)
        outcome = MeetingOutcome(
            issues=issues,
            natural_constraints=moderated,
            formal_constraints=formal,
            plan_variant_path=None,
            logs={
                "plan": json.dumps({"plan_payload": plan_input.plan_payload, "solver": "none"}, ensure_ascii=False),
                "persona_raw": json.dumps(persona_raw, ensure_ascii=False),
                "aggregator_raw": raw_agg,
                "moderator_raw": raw_mod,
            },
        )
        sample_id = plan_input.sample_id or Path(plan_input.plan_path).stem
        saved = persist_outcome(outcome, sample_id=sample_id)
        outcome.logs["saved_files"] = json.dumps(saved, ensure_ascii=False)
        return outcome
    def _select_personas(self, participant_count: int = None) -> List[PersonaConfig]:
        if participant_count is None:
            return list(self.meeting_cfg.personas)
        return list(self.meeting_cfg.personas[:participant_count])
