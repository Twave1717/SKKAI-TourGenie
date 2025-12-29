# Multi Persona 실행 구조 (불만/제약 생성 전용)

이 문서는 `KY/multi_persona` 패키지의 실제 실행 흐름과 구성 요소를 상세히 설명한다. 현재 버전은 **불편 포인트 → 자연어 제약 → DSL 필드 기록**까지만 수행하며, 솔버/재계획 단계는 포함되지 않는다.

## 엔트리포인트와 실행 모드
- 실행 진입: `python -m run_experiment` (파일: `run_experiment.py`).
- 입력: TripCraft CSV 한 행(`--files`, `--row-index`), 사용할 페르소나 수(`--participants`), 모드(`--mode`: `single`/`aggregate`/`consensus`).
- LLM 선택: 기본은 스텁(`StubClient`)이며 `--use-openai`를 지정하면 `OpenAIClient(gpt-4.1)`을 사용한다. 스텁은 모든 호출에서 오류를 던지므로 실제 실행은 `--use-openai`가 필수다.
- 샘플 식별자: `{csv_stem}_row{idx}_p{participants}_{mode}` 로 생성해 `runs/` 하위에 결과를 저장한다.

## 처리 파이프라인 개요
1) **입력 로드**  
   - `load_tripcraft_plan(path, row_index)`가 TripCraft CSV 한 행을 파싱해 `plan_payload`(meta, itinerary, references, raw_row)를 만든다.
2) **페르소나 이슈 생성(Round0)**  
   - `collect_persona_issues()`가 페르소나 프롬프트(`prompt_templates/persona_v1.txt`)를 채워 넣고 LLM 구조화 출력을 `IssueModel` 리스트로 받는다.  
   - 결과는 `Issue`로 정규화되며 issue_id 중복을 방지해 재명명한다.
3) **이슈 집계 → 자연어 제약(Round1)**  
   - `aggregate_issues()`가 모든 `Issue`를 JSON으로 넘기고 `aggregator_v1.txt` 템플릿을 사용해 `ConstraintList`를 받는다.  
   - 각 항목을 `NaturalConstraint`로 변환하며 constraint_id 중복 시 접미사를 추가한다.
4) **토론/모더레이션(Round2, consensus 모드만)**  
   - `moderate_constraints()`가 중복/일반화를 목표로 `moderator_v1.txt` 템플릿으로 재가공해 `importance_score` 등을 채운다.
5) **NL → DSL 필드 패스스루(Round3)**  
   - `dsl.nl_to_dsl()`은 실제 변환 없이 `NaturalConstraint.constraint_nl`을 그대로 `FormalConstraint.dsl`에 담아 type=hard, weight=importance_score로 반환한다. 현재는 LLM 호출 없음.
6) **로그/결과 저장**  
   - `logger.persist_outcome()`이 `runs/<sample_id>/`에 issues.json, natural_constraints.json, formal_constraints.json 및 raw 로그(plan/persona/aggregator/moderator)를 기록하고 저장 경로를 outcome.logs["saved_files"]로 남긴다.

## 모드별 동작 (`MeetingRunner`, `source/meeting.py`)
- 공통: `plan_payload`를 문자열로 직렬화해 프롬프트에 붙이며, 모든 LLM 호출은 `response_format={"type": "json_object"}` 로 요청하고 Pydantic 모델(`IssueModel`, `ConstraintModel`)로 검증한다.
- `run_m2` (`single`): 단일 페르소나만 Round0→Round1→Round3 수행. `persona_raw`는 한 명의 결과만 포함.
- `run_m3` (`aggregate`): 지정한 N명(기본 4명) 페르소나를 `_select_personas`로 선택 후 각각 Round0 실행, 모든 이슈를 합쳐 Round1→Round3 수행. 모더레이터 없음.
- `run_m4` (`consensus`): `run_m3` 흐름에 Round2(모더레이터)를 추가해 최종 `natural_constraints`에 반영한다.

## 주요 구성 요소
- `source/config.py`: 페르소나/회의/벤치 설정 데이터클래스. MeetingConfig는 참가자 목록, moderator 이름, top_constraints 등 실행 파라미터를 담는다.
- `source/structures.py`: Issue/NaturalConstraint/FormalConstraint/PlanInput/MeetingOutcome 등 핵심 데이터 모델. `PlanInput`은 plan_path와 plan_payload를 모두 보관해 로그에 남긴다.
- `source/prompts.py` + `source/prompt_templates/*`: 프롬프트 로더 및 렌더러. 파일이 없거나 읽기 실패 시 stderr 경고 후 `sys.exit(1)`로 종료한다.
- `source/llm.py`:  
  - `OpenAIClient`는 chat.completions에 `json_object` 응답을 요구하고 Pydantic으로 파싱한다. 파싱 실패 시 `_recover_json`으로 마지막 중괄호까지 잘라 재시도한다. RateLimit/API 오류는 지수 백오프하며 `LLM_DEBUG=1`이면 요청/응답 프리뷰를 stderr에 남긴다.  
  - `StubClient`는 호출 시 바로 예외를 던져 의도치 않은 스텁 사용을 방지한다.
- `source/tripcraft_parser.py`: TripCraft CSV 파서. `annotation_plan`/`reference_information` 컬럼을 `ast.literal_eval`로 복원하고 일자별 일정 요약을 만들어준다(`format_trip`). 기본 데이터 경로는 `<repo>/benchmarks/TripCraft/tripcraft`.
- `source/logger.py`: Outcome을 파일로 직렬화하는 헬퍼. 폴더를 생성하고 텍스트/JSON 파일을 기록하며 경로 dict를 반환한다.

## 페르소나 프리셋
- 파일: `source/prompt_templates/persona_preset.jsonl`.
- 20명(예산, 이동 피로도, 식사/로컬 경험, 가족 배려, 현지 감성, 시간 효율, 액티비티, 럭셔리, 야간 체험, 접근성, 사진, 역사, 웰니스, 친환경, 쇼핑, 펫 동반, 슬로우 트래블, 스케줄 최적화, 어린이, 가성비 미식, 백팩커 등) 정의. `run_experiment.build_runner()`에서 무작위 샘플링해 MeetingConfig.personas를 구성한다. `--participants`는 1~8 범위이며, 생략 시 single=1, 나머지=4가 기본.

## 결과물 구조 (`runs/<sample_id>/`)
- `issues.json`: `Issue` 배열.
- `atomic_constraints.json`: (aggregation=rule) per-person 정규화 제약.
- `aggregated_constraints_rule.json`: (aggregation=rule) 규칙 기반 집계 결과.
- `natural_constraints.json`: 모드/집계 방식에 따른 최종 자연어 제약.
- `formal_constraints.json`: NL 패스스루 DSL.
- `plan.txt`: `{plan_payload, solver="none"}` 직렬화본.
- `persona_raw.txt`: 페르소나별 LLM 원문(JSON string).
- `atomic_raw.txt`: (aggregation=rule) AtomicConstraint 추출 LLM 원문(JSON string).
- `aggregator_raw.txt`, `moderator_raw.txt`: 각 단계의 LLM 구조화 응답 문자열.

## 주의/제약 사항
- Solver는 연결되어 있지 않으므로 `plan_variant_path`는 항상 None, 생성된 제약만 저장된다.
- 프롬프트 템플릿 파일이 없거나 깨지면 즉시 종료한다. `prompt_templates` 폴더를 반드시 유지해야 한다.
- LLM 스텁은 실행을 중단시키므로 실제 실험은 `--use-openai`와 `.env`의 `OPENAI_API_KEY` 설정이 필요하다.
- CSV 데이터는 기본 `benchmarks/TripCraft/tripcraft/*.csv` 를 가정한다. 다른 위치를 쓸 경우 `--files`에 절대/상대 경로를 전달한다.

## Planless 준비 사항 (Step2 스케치)
- 프롬프트: `source/prompt_templates/persona_planless_v1.txt`, `aggregator_planless_v1.txt`, `moderator_planless_v1.txt` 추가(계획 없이 요구/금기 생성 및 집계/모더레이션용).
- 렌더 함수: `prompts.py`에 planless persona/aggregator/moderator 렌더러 포함.
- 예정 결과/로그: `planless_persona_raw.txt`, `planless_aggregator_raw.txt`, `planless_moderator_raw.txt`, `planless_constraints_raw.json`, `constraint_mapping.json`, `new_schema_proposals.json`.
- 모드/러너 추가(`--mode planless_multi`)와 Constraint Library/Mapper/Generator는 다음 단계에서 구현 예정.
