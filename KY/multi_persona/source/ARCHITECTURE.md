# Multi-Persona Meeting System (설계 개요)

## 목표
- 불합리 포인트를 멀티 페르소나 회의로 발굴 → 자연어 제약 → Solver DSL 제약 → 재계획(P_MP) → 기존 벤치 및 인간/LLM 관점 품질 개선 여부 검증.
- P0은 `benchmarks/TripCraft/tripcraft` 등 기본 솔버 계획(예: `tripcraft_5day.csv`)을 사용.

## 현재 코드가 다루는 베이스라인
- **M0**: Solver-only (기본 제약만).
- **M1**: 단일 LLM 직접 리라이팅(제약 생성 없음).
- **M2**: Single Persona가 issue→NL constraint→DSL 변환 후 재계획.
  - 현재 구현은 M2까지만 제공하며, 멀티 페르소나(M3/M4)는 제거된 상태.

## 라운드 기반 회의 (M2)
- **Round0**: 단일 페르소나가 issue 최대 K개 JSON 기록(심각도/차원/제약 초안 포함).
- **Round1**: 단일 페르소나 이슈를 기반으로 NL constraint 리스트 생성(aggregator 프롬프트).
- **Round2**: Engineer가 NL→Solver DSL(hard/soft) 변환.

## 컴포넌트
- **config.py**: 페르소나/회의/솔버 설정 정의.
- **structures.py**: Issue/NL 제약/DSL 제약/투표/결과 데이터 구조.
- **prompts.py**: 페르소나·어그리게이터·모더레이터·엔지니어 프롬프트 템플릿.
- **llm.py**: LLM 호출 인터페이스(교체 가능).
- **dsl.py**: NL 제약 → Solver DSL 변환기 스텁.
- **solver.py**: DSL 제약을 받아 P1/P_MP를 생성하는 솔버 어댑터 스텁.
- **meeting.py**: M0~M4 실행 오케스트레이션, 라운드 상태 전이.
- **runner.py**: CLI/스크립트 진입점 예시.
- (제거됨) metrics.py: 현재 실험에는 불필요하여 삭제.

## 데이터 경로/입력
- P0 경로 예시: `benchmarks/TripCraft/tripcraft/tripcraft_5day.csv`.
- 입력: 사용자 쿼리, 벤치 기본 제약, baseline 계획 P0.
- 출력: 추가 제약 C_new, 재계획 P_MP 및 부가 로그(issues/투표/검증 코멘트).

## 실행 흐름 개요
1) `MeetingRunner.run_mX()` 호출 (M0~M4 선택).
2) (M2~M4) 페르소나 프롬프트 생성 → LLM 호출 → Issue 리스트 수집.
3) (M3/M4) 어그리게이터/모더레이터로 병합·일반화 → NL 제약.
4) 엔지니어 프롬프트로 NL→DSL 변환.
5) `SolverAdapter.solve()`로 재계획 생성.
6) (옵션) Round4 체크 및 평가 훅 실행.
