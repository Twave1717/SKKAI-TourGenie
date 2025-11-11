#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator



from benchmarks.travelplanner.eval_runner import (
    ensure_results_dir,
    example_to_query_record,
    load_examples,
    prepare_official_submission,
    sanitize_model_slug,
    write_jsonl,
)
from benchmarks.travelplanner.leaderboard import (
    DEFAULT_METRIC_KEYS,
    build_header,
    update_leaderboard,
)
from benchmarks.travelplanner.official.evaluation import eval_score as official_eval_score


app = typer.Typer(help="Evaluate offline TravelPlanner submissions and update the leaderboard.")

DEFAULT_DATASET = Path("data/travelplanner/validation.jsonl")
DEFAULT_LEADERBOARD = Path("leaderboards/TravelPlanner/main.md")
MINI_LEADERBOARD = Path("leaderboards/TravelPlanner/mini.md")
MINI_HEADER = build_header(DEFAULT_METRIC_KEYS, title="# TravelPlanner Mini Leaderboard")


class DayPlan(BaseModel):
    days: int
    current_city: str
    transportation: str
    breakfast: str
    lunch: str
    dinner: str
    attraction: str
    accommodation: str

    model_config = ConfigDict(extra="allow")


class SubmissionPrediction(BaseModel):
    id: str
    plan: List[DayPlan] = Field(..., description="Parsed plan ready for official evaluator")
    raw_response: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("plan")
    @classmethod
    def _check_plan(cls, value: List[DayPlan]) -> List[DayPlan]:
        if not value:
            raise ValueError("plan must contain at least one day entry")
        return value


class SubmissionPayload(BaseModel):
    team: str
    workflow: str
    provider: str
    model: str
    split: str = "validation"
    dataset: Optional[str] = None
    result_label: Optional[str] = None
    notes: Optional[str] = None
    predictions: List[SubmissionPrediction]
    
    model_config = ConfigDict(extra="allow")
    
    @field_validator("team", "workflow", "provider", "model", "split", mode="before")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("predictions")
    @classmethod
    def _ensure_predictions(cls, value: List[SubmissionPrediction]) -> List[SubmissionPrediction]:
        if not value:
            raise ValueError("predictions cannot be empty")
        return value


def _slug(value: str) -> str:
    slug = sanitize_model_slug(value or "")
    return slug or "submission"


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@app.command()
def main(
    submission: Path = typer.Argument(..., exists=True, readable=True, help="팀이 생성한 JSON 결과 파일"),
    dataset: Optional[Path] = typer.Option(
        None,
        "--dataset",
        "-d",
        exists=False,
        help="평가에 사용할 JSONL 데이터셋 경로 (기본값: submission 내부 dataset 또는 validation split)",
    ),
    result_label: Optional[str] = typer.Option(
        None,
        "--result-label",
        help="리더보드 표시에 사용할 추가 라벨 (기본값: submission.result_label 또는 split 값)",
    ),
    allow_partial: bool = typer.Option(
        False,
        "--allow-partial/--require-complete",
        help="제출에 포함되지 않은 샘플이 있어도 평가를 계속 진행할지 여부 (기본 False)",
    ),
    skip_leaderboard: bool = typer.Option(
        False,
        "--skip-leaderboard/--update-leaderboard",
        help="평가만 수행하고 리더보드 업데이트는 건너뜁니다.",
    ),
) -> None:
    """Validate a submission JSON, run the official evaluator, and update the leaderboard."""

    try:
        payload = SubmissionPayload.model_validate_json(submission.read_text(encoding="utf-8"))
    except ValidationError as exc:
        typer.echo(f"[error] 제출 포맷이 올바르지 않습니다: {exc}")
        raise typer.Exit(code=1) from exc

    dataset_path = Path(payload.dataset or dataset or DEFAULT_DATASET).resolve()
    if not dataset_path.exists():
        typer.echo(f"[error] Dataset not found: {dataset_path}")
        raise typer.Exit(code=1)

    examples = load_examples(dataset_path)
    if not examples:
        typer.echo(f"[error] Dataset {dataset_path} has no examples.")
        raise typer.Exit(code=1)

    dataset_ids = {example.id for example in examples}
    prediction_map: Dict[str, SubmissionPrediction] = {
        pred.id: pred for pred in payload.predictions
    }

    missing_ids = [example.id for example in examples if example.id not in prediction_map]
    extra_ids = [pred_id for pred_id in prediction_map.keys() if pred_id not in dataset_ids]

    if extra_ids:
        typer.echo(f"[warn] {len(extra_ids)} predictions do not match the dataset and will be ignored.")

    if missing_ids and not allow_partial:
        typer.echo(
            f"[error] Submission is missing {len(missing_ids)} examples "
            f"(e.g., {missing_ids[:3]} ...). Pass --allow-partial to evaluate anyway."
        )
        raise typer.Exit(code=1)

    selected_examples = [example for example in examples if example.id in prediction_map]
    if not selected_examples:
        typer.echo("[error] No matching predictions were found for the provided dataset.")
        raise typer.Exit(code=1)

    parsed_plans: List[List[Dict[str, Any]]] = []
    for example in selected_examples:
        entry = prediction_map[example.id]
        parsed_plans.append([day.model_dump(mode="python") for day in entry.plan])

    model_slug = sanitize_model_slug(f"{payload.model}-{payload.workflow}")
    variant_slug = _slug(f"{payload.team}-submission")
    run_dir = ensure_results_dir(payload.provider, model_slug, variant=variant_slug)

    run_dir.mkdir(parents=True, exist_ok=True)
    submission_copy = run_dir / "submission.json"
    submission_copy.write_text(submission.read_text(encoding="utf-8"), encoding="utf-8")

    # Prepare official evaluator inputs
    submission_records = prepare_official_submission(selected_examples, parsed_plans)
    submission_path = run_dir / "official_submission.jsonl"
    write_jsonl(submission_path, submission_records)

    query_records = [example_to_query_record(example) for example in selected_examples]

    scores, detailed_scores = official_eval_score(
        payload.split,
        submission_path,
        query_data_list=query_records,
    )
    metrics = {key: scores.get(key) for key in DEFAULT_METRIC_KEYS}

    _dump_json(run_dir / "metrics.json", metrics)
    _dump_json(run_dir / "official_metrics.json", {"scores": scores, "details": detailed_scores})

    typer.echo("[official-eval] 주요 지표:")
    typer.echo(json.dumps(scores, indent=2, ensure_ascii=False))

    effective_label = result_label or payload.result_label or payload.split
    can_update_leaderboard = (
        not skip_leaderboard and not allow_partial and not missing_ids and bool(effective_label)
    )

    if can_update_leaderboard:
        leaderboard_path = DEFAULT_LEADERBOARD
        header_lines = None
        if effective_label.strip().lower() == "test-mini":
            leaderboard_path = MINI_LEADERBOARD
            header_lines = MINI_HEADER

        update_leaderboard(
            payload.provider,
            f"{payload.model} ({payload.team}/{payload.workflow})",
            run_dir,
            metrics,
            leaderboard_path=leaderboard_path,
            header_lines=header_lines,
            metric_keys=DEFAULT_METRIC_KEYS,
            result_label=effective_label,
        )
        typer.echo(f"[leaderboard] Updated {leaderboard_path} (label={effective_label})")
    else:
        reason_parts: List[str] = []
        if skip_leaderboard:
            reason_parts.append("skip request")
        if allow_partial or missing_ids:
            reason_parts.append("incomplete coverage")
        if not effective_label:
            reason_parts.append("empty result label")
        if reason_parts:
            typer.echo(f"[leaderboard] Skipped update ({', '.join(reason_parts)})")

    typer.echo(f"[done] Results saved to {run_dir}")


if __name__ == "__main__":
    app()
