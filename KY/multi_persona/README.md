# Multi Persona (회의/불만 생성 전용, 솔버 없음)

- 목표: TripCraft CSV에서 단일 행을 읽어 1~4인 페르소나가 불만/추가 제약을 생성하고 병합/토론하는 과정만 실행.
- 솔버/재계획은 수행하지 않음. 결과는 제약/RAW 로그로만 남김.

## 환경
- `conda env create -f KY/multi_persona/environment.yml`
- `conda activate multi-persona`
- `.env`에 `OPENAI_API_KEY` (gpt-4.1 사용 시), 없으면 스텁 LLM으로 동작.

## 실행 (단일 엔트리: run_experiment.py)
```
python -m run_experiment \
  --files tripcraft_3day.csv \
  --mode consensus \
  --participants 2 \
  --row-index 0 \
  [--use-openai] \
  [--aggregation llm|rule]
```
- mode: `single`(1인), `aggregate`(병합), `consensus`(토론)
- participants: 1~8명 사용 (preset 20명 내 무작위 샘플)
- aggregation: `llm`(기존 LLM 집계) | `rule`(AtomicConstraint 기반 규칙 집계, 현재 프롬프트/LLM 추출 필요)
- row-index: CSV에서 사용할 행(기본 0, 한 건만 사용)
- files: 생략 시 tripcraft 디렉토리 모든 CSV
- 결과: `runs/<stem>_row<idx>_p<participants>_<mode>/`
  - issues.json, natural_constraints.json, formal_constraints.json
  - plan.txt(입력 plan_payload, solver="none"), persona_raw.txt, aggregator_raw.txt, moderator_raw.txt(토론 모드)
  - (rule) atomic_constraints.json, aggregated_constraints_rule.json

## 구성 요소
- `source/meeting.py`: 페르소나 이슈 생성 → 병합 →(토론 시) 모더레이션 → NL→DSL 패스스루 → 로그 저장
- `source/tripcraft_parser.py`: TripCraft CSV 파서(단일 행 로드), 요약 포맷 함수 포함
- `source/prompts.py` + `source/prompt_templates/*.txt`: 프롬프트 템플릿 (없으면 경고 후 종료)
- `source/llm.py`: OpenAIClient(구조화 출력) 및 스텁
- `source/dsl.py`: NL을 그대로 FormalConstraint.dsl에 담는 패스스루
- `source/logger.py`: runs/ 하위에 산출물 저장

## 프롬프트/Structured Output
- 모든 LLM 호출에 `response_format={"type": "json_object"}` 사용, 파서는 `items` 배열을 우선 사용.
- 템플릿 요약:
  - persona_v1.txt: issue_id/dimension/severity/description/proposed_constraint_nl/plan_reference 포함한 items 생성
  - aggregator_v1.txt: 유사 이슈 묶어 constraint_id/constraint_nl/dimension/support_personas/avg_severity 생성
  - moderator_v1.txt: 제약 병합/중요도(importance_score) 부여
  - engineer_v1.txt: NL을 DSL 필드에 매핑하는 형태로 items 생성(현재 패스스루)

## 동작 시나리오
1. `load_tripcraft_plan`으로 CSV 한 행을 plan_payload로 로드.
2. 선택한 페르소나 수(1~4)가 issue(items) 생성.
3. aggregator가 NL constraint로 일반화, consensus 모드면 moderator가 병합/중요도 정리.
4. dsl 패스스루로 NL→FormalConstraint 변환.
5. runs 디렉터리에 RAW/제약/DSL/입력 plan_payload 저장(솔버 없음).

## 남은 과제
- 실제 솔버 연동(재계획), DSL 검증/파싱 강화가 필요하면 추후 추가.
