#!/usr/bin/env python

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from eval_runner import (
    BACKBONE_INSTANCE_COUNT,
    DEFAULT_METRIC_KEYS,
    Example,
    GraphState,
    MINI_LEADERBOARD_HEADER,
    MINI_LEADERBOARD_PATH,
    Prediction,
    Provider,
    TEST_MINI_TARGET,
    build_llm,
    dump_predictions,
    ensure_results_dir,
    example_to_query_record,
    infer_official_split,
    load_examples,
    normalize_provider_name,
    prepare_official_submission,
    sanitize_model_slug,
    select_test_mini_examples,
    update_leaderboard,
    write_jsonl,
)


from benchmarks.travelplanner.postprocess.grounding import (
    _flights,
    _accommodations,
    _restaurants,
    _attractions,
)

from benchmarks.travelplanner.postprocess.constraint_extraction import (
    extract_constraints_from_example,
    validate_constraints,
)

from benchmarks.travelplanner.solvers import TravelRouteSolver


SYSTEM_PROMPT = """You are a travel planning assistant. Your role is to understand the user's travel requirements and help validate their preferences.

When given a travel query, analyze and confirm:
1. Budget constraints
2. Accommodation preferences (room type, house rules)
3. Dining preferences (cuisine types, dietary restrictions)
4. Transportation needs
5. Activity interests

Respond naturally to acknowledge the requirements. You don't need to generate the detailed itinerary - that will be handled by the optimization system."""


def _normalize_city_name(text: str) -> str:
    """데이터베이스 쿼리를 위해 도시명 정규화"""
    import re
    return re.sub(r"\s+", " ", text.strip())


def _query_database_candidates(
    origin: str,
    destination: str,
    date: str,
    trip_days: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    TravelPlanner 데이터베이스에서 모든 후보 항목 조회
    
    Args:
        origin: 출발 도시
        destination: 도착 도시
        date: 출발 날짜 (또는 날짜 목록)
        trip_days: 여행 일수
    
    Returns:
        항공편, 호텔, 레스토랑, 관광지가 포함된 딕셔너리
    """
    import pandas as pd
    
    candidates = {
        'flights': [],
        'hotels': [],
        'restaurants': [],
        'attractions': [],
    }
    
    # 도시명 정규화
    origin_norm = _normalize_city_name(origin)
    dest_norm = _normalize_city_name(destination)
    
    # 날짜가 리스트인 경우 파싱
    if isinstance(date, list) and len(date) > 0:
        date = date[0]
    date_str = str(date) if date else ""
    
    try:
        # 항공편 조회
        flights_api = _flights()
        if flights_api.data is not None:
            # 출발 항공편
            outbound = flights_api.data[
                (flights_api.data["OriginCityName"].astype(str).str.lower() == origin_norm.lower()) &
                (flights_api.data["DestCityName"].astype(str).str.lower() == dest_norm.lower())
            ]
            if date_str:
                outbound = outbound[outbound["FlightDate"] == date_str]
            
            for _, row in outbound.iterrows():
                candidates['flights'].append({
                    'id': str(row.get('Flight Number', 'UNK')),
                    'price': float(row.get('Price', 0)),
                    'is_outbound': True,
                })
            
            # 귀국 항공편
            inbound = flights_api.data[
                (flights_api.data["OriginCityName"].astype(str).str.lower() == dest_norm.lower()) &
                (flights_api.data["DestCityName"].astype(str).str.lower() == origin_norm.lower())
            ]
            if date_str:
                inbound = inbound[inbound["FlightDate"] == date_str]
            
            for _, row in inbound.iterrows():
                candidates['flights'].append({
                    'id': str(row.get('Flight Number', 'UNK')),
                    'price': float(row.get('Price', 0)),
                    'is_outbound': False,
                })
    except Exception as e:
        print(f"[경고] 항공편 조회 실패: {e}")
    
    try:
        # 호텔 조회
        hotels_api = _accommodations()
        if hotels_api.data is not None:
            hotels_df = hotels_api.data[
                hotels_api.data["city"].astype(str).str.lower() == dest_norm.lower()
            ]
            
            for _, row in hotels_df.iterrows():
                candidates['hotels'].append({
                    'id': str(row.get('NAME', 'Unknown Hotel')),
                    'price_per_night': float(row.get('price', 100)),
                    'category': str(row.get('house_rules', 'Hotel')),
                    'house_rules': str(row.get('house_rules', '')),
                    'room_type': str(row.get('room type', '')),
                })
    except Exception as e:
        print(f"[경고] 호텔 조회 실패: {e}")
    
    try:
        # 레스토랑 조회
        restaurants_api = _restaurants()
        if restaurants_api.data is not None:
            restaurants_df = restaurants_api.data[
                restaurants_api.data["City"].astype(str).str.lower() == dest_norm.lower()
            ]
            
            for _, row in restaurants_df.iterrows():
                avg_cost = float(row.get('Average Cost', 20))
                time_cost = 45 + int(avg_cost / 5)  # 소요 시간 추정
                
                candidates['restaurants'].append({
                    'id': str(row.get('Name', 'Unknown Restaurant')),
                    'price': avg_cost,
                    'time_cost': min(time_cost, 90),
                    'category': str(row.get('Cuisines', 'General')),
                })
    except Exception as e:
        print(f"[경고] 레스토랑 조회 실패: {e}")
    
    try:
        # 관광지 조회
        attractions_api = _attractions()
        if attractions_api.data is not None:
            attractions_df = attractions_api.data[
                attractions_api.data["City"].astype(str).str.lower() == dest_norm.lower()
            ]
            
            for _, row in attractions_df.iterrows():
                candidates['attractions'].append({
                    'id': str(row.get('Name', 'Unknown Attraction')),
                    'price': 15.0,  # 기본값
                    'time_cost': 90,  # 기본값
                    'category': 'Attraction',
                })
    except Exception as e:
        print(f"[경고] 관광지 조회 실패: {e}")
    
    return candidates


def _format_solver_result_as_plan(
    solver_result: Dict[str, Any],
    example: Example
) -> str:
    """
    OR-Tools 솔버 결과를 TravelPlanner 계획 형식으로 변환
    평가기 호환성을 위해 grounding.py 형식과 일치
    
    Args:
        solver_result: TravelRouteSolver의 최적화된 여행 일정
        example: 컨텍스트를 위한 원본 예제
    
    Returns:
        여행 계획을 나타내는 JSON 문자열
    """
    if not solver_result:
        return "[]"
    
    daily_plans = solver_result.get('daily_plans', [])
    flights = solver_result.get('flights', {})
    hotel = solver_result.get('hotel', {})
    
    metadata = example.metadata or {}
    origin = _normalize_city_name(str(metadata.get('org', 'Origin')))
    destination = _normalize_city_name(str(metadata.get('dest', 'Destination')))
    
    plan = []
    
    for day_idx, day_plan in enumerate(daily_plans):
        day_num = day_idx + 1
        
        # 현재 도시 레이블
        if day_idx == 0:
            current_city = f"from {origin} to {destination}"
        else:
            current_city = destination
        
        day_entry = {
            'days': day_num,
            'current_city': current_city,
            'transportation': '-',
            'breakfast': '-',
            'lunch': '-',
            'dinner': '-',
            'attraction': '-',
            'accommodation': '-',
        }
        
        # 교통수단
        if day_idx == 0 and flights.get('outbound'):
            flight = flights['outbound']
            day_entry['transportation'] = f"Flight Number: {flight.get('id', 'F001')}, from {origin} to {destination}, Price: {int(flight.get('price', 0))}"
        elif day_idx == len(daily_plans) - 1 and flights.get('return'):
            flight = flights['return']
            day_entry['transportation'] = f"Flight Number: {flight.get('id', 'F002')}, from {destination} to {origin}, Price: {int(flight.get('price', 0))}"
            day_entry['current_city'] = f"from {destination} to {origin}"
        else:
            day_entry['transportation'] = f"Taxi in {destination}; Cost: 50"
        
        # 식사
        meals = day_plan.get('meals', {})
        if meals.get('breakfast'):
            b = meals['breakfast']
            day_entry['breakfast'] = f"{b.get('id', 'Restaurant')}, {destination}; Cost: {int(b.get('price', 0))}"
        if meals.get('lunch'):
            l = meals['lunch']
            day_entry['lunch'] = f"{l.get('id', 'Restaurant')}, {destination}; Cost: {int(l.get('price', 0))}"
        if meals.get('dinner'):
            d = meals['dinner']
            day_entry['dinner'] = f"{d.get('id', 'Restaurant')}, {destination}; Cost: {int(d.get('price', 0))}"
        
        # 관광지
        attractions = day_plan.get('attractions', [])
        if attractions:
            attr_names = [f"{a.get('id', 'Attraction')}, {destination}" for a in attractions]
            day_entry['attraction'] = "; ".join(attr_names) + ";"
        
        # 숙소 (마지막 날 제외)
        if day_idx < len(daily_plans) - 1 and hotel:
            price_per_night = hotel.get('price_per_night', 0)
            total_cost = price_per_night * (len(daily_plans) - 1)
            day_entry['accommodation'] = f"{hotel.get('id', 'Hotel')}, {destination}; Cost: {int(price_per_night)} per night (~{int(total_cost)})"
        
        plan.append(day_entry)
    
    return json.dumps(plan, ensure_ascii=False)


def build_hybrid_graph(llm, system_prompt: Optional[str] = None):
    """
    하이브리드 접근법을 위한 LangGraph 워크플로우 구축
    
    eval_runner.py와 유사하지만 솔버 통합 추가
    """
    graph = StateGraph(GraphState)
    
    def call_llm(state: GraphState) -> GraphState:
        """LLM 노드: 요구사항 이해 및 검증"""
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        user_content = state["query"]
        context_sections: List[str] = []
        
        # 메타데이터 요약 추가
        metadata = state.get("metadata", {}) or {}
        if metadata:
            summary_keys = ["org", "dest", "date", "days", "people_number", "budget", "local_constraint"]
            summary_lines = []
            for key in summary_keys:
                if key in metadata and metadata[key] not in (None, ""):
                    value = metadata[key]
                    if isinstance(value, list):
                        value = ", ".join(map(str, value))
                    summary_lines.append(f"{key}: {value}")
            if summary_lines:
                context_sections.append("여행 정보:\n" + "\n".join(summary_lines))
        
        # 참조 정보 추가
        reference_information = state.get("reference_information")
        if reference_information:
            formatted_refs: List[str] = []
            if isinstance(reference_information, list):
                for item in reference_information:
                    if isinstance(item, dict):
                        desc = item.get("Description") or item.get("description") or "참고자료"
                        content = item.get("Content") or item.get("content") or ""
                        formatted_refs.append(f"{desc}:\n{content}")
                    else:
                        formatted_refs.append(str(item))
            else:
                formatted_refs.append(str(reference_information))
            
            if formatted_refs:
                context_sections.append("참고자료:\n" + "\n\n".join(formatted_refs))
        
        if context_sections:
            user_content = f"{user_content}\n\n" + "\n\n".join(context_sections)
        
        messages.append(HumanMessage(content=user_content))
        
        response: AIMessage = llm.invoke(messages)
        
        return {
            "query": state["query"],
            "reference_information": reference_information,
            "metadata": metadata,
            "messages": messages + [response],
            "prediction": response.content,
        }
    
    graph.add_node("call_llm", call_llm)
    graph.add_edge(START, "call_llm")
    graph.add_edge("call_llm", END)
    
    return graph.compile()


def _process_example_hybrid(
    app,
    example: Example,
    idx: int,
    provider: Provider,
    model: Optional[str],
    save_messages: bool,
    run_dir: Path,
    solver_timeout: int = 30,
) -> Prediction:
    """
    하이브리드 LLM + OR-Tools 접근법으로 예제 처리
    
    eval_runner.py 구조를 따르지만 솔버 통합 추가
    """
    # 단계 1: LLM 호출 (LangGraph를 통해)
    result = app.invoke(
        {
            "query": example.query,
            "reference_information": example.reference_information,
            "metadata": example.metadata,
        },
        config={
            "configurable": {"run_name": f"hybrid-{provider}-{model or 'default'}"},
            "metadata": {"example_id": example.id},
        },
    )
    
    llm_response = result["prediction"]
    
    # 단계 2: 제약 조건 추출
    constraints = extract_constraints_from_example(example)
    
    # 제약 조건 검증
    warnings = validate_constraints(constraints)
    if warnings:
        for warning in warnings:
            print(f"[경고] {example.id}: {warning}")
    
    # 단계 3: 후보 항목 데이터베이스 조회
    metadata = example.metadata or {}
    origin = metadata.get('org', '')
    destination = metadata.get('dest', '')
    date = metadata.get('date', '')
    trip_days = constraints['trip_days']
    
    candidates = _query_database_candidates(origin, destination, date, trip_days)
    
    # 단계 4: OR-Tools 솔버 실행
    solver = TravelRouteSolver(candidates, constraints)
    solver_result = solver.solve(time_limit_seconds=solver_timeout)
    
    # 단계 5: 결과 포맷팅
    if solver_result:
        prediction_json = _format_solver_result_as_plan(solver_result, example)
        raw_prediction = json.dumps(solver_result, indent=2, ensure_ascii=False)
    else:
        # 실행 불가능 - 빈 계획 반환
        prediction_json = "[]"
        raw_prediction = "실행 불가능: 모든 제약 조건을 만족할 수 없음"
    
    # 단계 6: 예측 메타데이터 생성
    metadata_payload = dict(example.metadata or {})
    metadata_payload['_hybrid'] = {
        'llm_provider': provider,
        'llm_model': model or 'default',
        'solver_type': 'ortools_cp_sat',
        'constraints': constraints,
        'candidates_count': {k: len(v) for k, v in candidates.items()},
        'status': solver_result.get('status', 'infeasible') if solver_result else 'infeasible',
        'total_cost': solver_result.get('total_cost', 0) if solver_result else 0,
        'constraint_warnings': warnings,
    }
    
    prediction = Prediction(
        id=example.id,
        query=example.query,
        prediction=prediction_json,
        raw_prediction=raw_prediction,
        expected=example.expected,
        metadata=metadata_payload,
    )
    
    # 요청 시 메시지 저장
    if save_messages:
        message_path = run_dir / f"{example.id}_messages.json"
        message_path.write_text(
            json.dumps([m.dict() for m in result["messages"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    
    return prediction


def main(
    provider: Provider = typer.Option("openai", help="LLM provider"),
    model: Optional[str] = typer.Option(None, help="LLM model name"),
    dataset: Path = typer.Option(
        Path("data/travelplanner/validation.jsonl"),
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to JSONL dataset",
    ),
    temperature: float = typer.Option(0.2, help="LLM sampling temperature"),
    test_mini: bool = typer.Option(False, help="Use 60-example test-mini subset"),
    solver_timeout: int = typer.Option(30, help="OR-Tools solver timeout (seconds)"),
    save_messages: bool = typer.Option(False, help="Save LLM messages to disk"),
    run_official_eval: bool = typer.Option(True, help="Run official evaluator"),
    official_set_type: Optional[str] = typer.Option("validation", help="Dataset split"),
    limit: Optional[int] = typer.Option(None, help="Max examples to evaluate"),
) -> None:
    """하이브리드 LLM + OR-Tools 접근법으로 TravelPlanner 평가"""
    
    load_dotenv(override=False)
    provider = normalize_provider_name(provider)
    
    # 예제 로드
    examples = load_examples(dataset)
    if not examples:
        typer.echo("예제를 불러올 수 없습니다. 종료합니다.")
        raise typer.Exit(code=1)
    
    if limit is not None:
        if limit <= 0:
            raise typer.Exit("--limit는 양수여야 합니다")
        examples = examples[:limit]
    
    variant = None
    if test_mini:
        examples = select_test_mini_examples(examples, TEST_MINI_TARGET)
        variant = "test-mini"
    
    total_examples = len(examples)
    typer.echo(f"하이브리드 LLM + OR-Tools로 {total_examples}개 예제 평가 중...")
    
    # LLM 인스턴스 초기화 (병렬 처리용)
    llm_instances = []
    for _ in range(BACKBONE_INSTANCE_COUNT):
        llm = build_llm(provider, model, temperature)
        app = build_hybrid_graph(llm, system_prompt=SYSTEM_PROMPT)
        llm_instances.append(app)
    
    # 결과 디렉토리 생성
    hybrid_provider = f"{provider}+ortools"
    run_dir = ensure_results_dir(hybrid_provider, model, variant=variant)
    
    # 예제 병렬 처리
    predictions: List[Optional[Prediction]] = [None] * total_examples
    
    with ThreadPoolExecutor(max_workers=len(llm_instances)) as executor:
        with typer.progressbar(
            length=total_examples,
            label=f"TravelPlanner {'test-mini' if test_mini else 'validation'} · Hybrid ({provider}+OR-Tools)"
        ) as progress:
            futures = {}
            for idx, example in enumerate(examples):
                app_instance = llm_instances[idx % len(llm_instances)]
                future = executor.submit(
                    _process_example_hybrid,
                    app_instance,
                    example,
                    idx,
                    provider,
                    model,
                    save_messages,
                    run_dir,
                    solver_timeout,
                )
                futures[future] = idx
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    prediction = future.result()
                    predictions[idx] = prediction
                except Exception as e:
                    typer.echo(f"\n[오류] 예제 {idx} 실패: {e}")
                    predictions[idx] = Prediction(
                        id=examples[idx].id,
                        query=examples[idx].query,
                        prediction="[]",
                        raw_prediction=f"오류: {str(e)}",
                        expected=examples[idx].expected,
                        metadata=examples[idx].metadata,
                    )
                finally:
                    progress.update(1)
    
    predictions = [p for p in predictions if p is not None]
    
    dump_predictions(run_dir / "predictions.jsonl", predictions)
    

    metrics: Dict[str, Any] = {key: 0.0 for key in DEFAULT_METRIC_KEYS}
    
    if run_official_eval:
        from benchmarks.travelplanner.official.pipeline import (
            load_parsed_plans,
            write_generated_plan_files,
        )
        from datasets import load_dataset
        
        effective_set_type = official_set_type or infer_official_split(dataset)
        if not effective_set_type:
            typer.echo("[경고] 데이터셋 분할을 추론할 수 없어 공식 평가를 건너뜁니다")
        else:
            dataset_records = load_dataset('osunlp/TravelPlanner', effective_set_type)[effective_set_type]
            dataset_records_list = [dict(record) for record in dataset_records]
            
            source_indices = []
            for example in examples:
                idx = example.metadata.get('_source_idx', 0)
                source_indices.append(idx if isinstance(idx, int) else 0)
            
            query_records = [dataset_records_list[idx] for idx in source_indices]
            
            official_output_dir = run_dir / "official_output"
            model_tag = f"{provider}-hybrid"
            write_generated_plan_files(
                official_output_dir,
                effective_set_type,
                model_tag,
                predictions,
            )
            
            parsed_plans = load_parsed_plans(
                official_output_dir,
                effective_set_type,
                model_tag,
                total_examples=total_examples,
            )
            
            submission_path = run_dir / "official_submission.jsonl"
            submission_records = prepare_official_submission(examples, parsed_plans)
            write_jsonl(submission_path, submission_records)
            
            try:
                from benchmarks.travelplanner.official.evaluation.eval import eval_score as official_eval_score
                
                typer.echo(f"[공식 평가] 평가기 실행 중...")
                scores, detailed_scores = official_eval_score(
                    effective_set_type,
                    submission_path,
                    query_records,
                )
                
                metrics = {key: scores.get(key) for key in DEFAULT_METRIC_KEYS}
                
                (run_dir / "official_metrics.json").write_text(
                    json.dumps({"scores": scores, "details": detailed_scores}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                
                typer.echo("[공식 평가] 점수:")
                typer.echo(json.dumps(scores, indent=2, ensure_ascii=False))
                
            except Exception as e:
                typer.echo(f"[오류] 공식 평가 실패: {e}")
                import traceback
                traceback.print_exc()
    
    # 메트릭 저장
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    
    # 리더보드 업데이트
    display_model = f"{model or 'default'} (Hybrid)"
    if test_mini:
        update_leaderboard(
            hybrid_provider,
            display_model,
            run_dir,
            metrics,
            leaderboard_path=MINI_LEADERBOARD_PATH,
            header_lines=MINI_LEADERBOARD_HEADER,
            metric_keys=DEFAULT_METRIC_KEYS,
            result_label="test-mini",
        )
    
    typer.echo(f"\n✅ 하이브리드 평가 완료!")
    typer.echo(f"결과 저장 위치: {run_dir}")
    typer.echo(f"\n메트릭:")
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    typer.run(main)
