# 3-Stage Persona Generation Pipeline

TravelPlanner 벤치마크를 위한 **갈등 기반 페르소나 생성 파이프라인**입니다.

## 🎯 핵심 개념

**"Compatible Enemies" 전략**:
- **호환 가능 (Compatible)**: 예산/시즌이 맞아 함께 여행 가능
- **갈등 (Enemies)**: 선호도가 달라 협상이 필요한 여행 계획

이 접근법은 다중 에이전트 협상 시나리오 평가를 위한 **통제된 갈등**을 제공합니다.

---

## 📁 폴더 구조

```
stage1_survey_persona_aug/
├── run_stage1.py                # Stage 1: MMR 기반 k×10 페르소나 검색
├── run_stage1_5.py              # Stage 1.5: Batch API 모드 (기본)
├── run_stage1_5_1_7_iterative.py # Stage 1.5+1.7: Iterative 모드
├── run_stage1_7.py              # Stage 1.7: Solvable Conflict 선택
│
├── core/                        # 공통 유틸리티
│   ├── retrieval.py             # Conflict-aware MMR 엔진
│   ├── vectorization.py         # 22차원 페르소나 벡터화
│   ├── stravl_loader.py         # Stravl 데이터 로더
│   └── ...
│
├── stage1_5/                    # LLM 알파 서베이 모듈
│   ├── prompt_builder.py        # 프롬프트 생성
│   ├── batch_api.py             # Batch API (50% 할인)
│   ├── async_caller.py          # Async API (Iterative용)
│   └── ...
│
├── stage1_7/                    # Solvable Conflict 선택 모듈
│   ├── solvability_check.py     # Hard/Soft 충돌 체크
│   └── combination_selector.py  # 최적 조합 선택
│
└── outputs/                     # 모든 출력 통합
    ├── stage1/test/data/        # Stage 1 결과
    ├── stage1_5/test/data/      # Stage 1.5 결과 (Batch)
    ├── stage1_5_iterative/      # Stage 1.5 결과 (Iterative)
    ├── stage1_7/test/data/      # Stage 1.7 결과 (최종)
    └── stage1_7_iterative/      # Stage 1.7 결과 (Iterative)
```

---

## 🚀 빠른 시작

> **⚡️ 1분 안에 시작하기**: [QUICKSTART.md](QUICKSTART.md)를 참조하세요.
>
> ```bash
> ./run.sh --max_records 10  # 테스트 모드
> ```
>
> **참고**: API 키는 `KY/aug_stage/.env` 파일에서 자동으로 로드됩니다.

### 전제조건

```bash
# 의존성 설치
pip install -r requirements.txt
```

> **💡 API 키**: `KY/aug_stage/.env` 파일에 `OPENAI_API_KEY`가 설정되어 있으며,
> run.sh가 자동으로 로드합니다. 별도 설정이 필요 없습니다.

### End-to-End 실행 (run.sh)

```bash
# 전체 파이프라인 실행 (1000개 레코드)
./run.sh

# 테스트 모드 (10개 레코드)
./run.sh --max_records 10

# Iterative 모드
./run.sh --mode iterative --max_records 10

# 도움말
./run.sh --help
```

**자세한 사용법**: [QUICKSTART.md](QUICKSTART.md) 참조

### 기본 워크플로우 (개별 실행)

```bash
# 1. Stage 1: k×10 페르소나 검색
python3 run_stage1.py --split test --max_records 10 --write_meta

# 2. Stage 1.5: LLM 알파 서베이 (Batch API, 63% 비용 절감)
python3 run_stage1_5.py --max_records 10

# 3. Stage 1.7: Solvable Conflict 선택
python3 run_stage1_7.py --max_records 10 --write_meta
```

### Iterative 모드 (비용 최적화)

```bash
# 1. Stage 1: 동일
python3 run_stage1.py --split test --max_records 10

# 2+3. Stage 1.5+1.7 통합 (Iterative, 최대 92% 비용 절감)
python3 run_stage1_5_1_7_iterative.py --max_records 10
```

---

## 📊 파이프라인 아키텍처

```
TravelPlanner Test Split (1,000개)
         ↓
    [Stage 1: MMR Retrieval]
    Conflict-aware MMR
         ↓
    k×10 Personas (20 or 40)
         ↓
    [Stage 1.5: Alpha Survey]
    LLM + Batch API or Iterative
         ↓
    k×10 Personas + Alpha Values
         ↓
    [Stage 1.7: Solvable Conflict Selection]
    최적 조합 선택
         ↓
    Final N Personas (2 or 4)
```

---

## 🔧 Stage 1: MMR 기반 페르소나 검색

### 목적

Conflict-aware MMR로 k×10개 페르소나 검색 (예: 2명 여행 → 20개 페르소나)

### 주요 기능

- **Conflict-aware MMR**: 적합성과 갈등 균형
- **Auto strategy**: budget_war | pace_war | taste_war 자동 선택
- **People upsampling**: 1인 여행을 2/4인으로 확장
- **Multi-threading**: 병렬 처리 (--workers)

### 사용법

```bash
# 기본 사용 (테스트용 2개)
python3 run_stage1.py --split test --max_records 2 --write_meta

# 전체 test set (1,000개)
python3 run_stage1.py --split test --workers 8 --write_meta

# k 배수 조정 (기본 10)
python3 run_stage1.py --split test --k_multiplier 5 --max_records 10
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--split` | - | TravelPlanner split (test만 지원) |
| `--max_records` | `0` | 처리할 레코드 수 (0 = 전체) |
| `--k_multiplier` | `10` | k×10 배수 |
| `--workers` | `1` | 병렬 작업자 수 |
| `--conflict_strategy` | `auto` | 갈등 전략 (auto/budget_war/pace_war/taste_war) |
| `--lambda_param` | `0.6` | MMR 균형 (0=최대갈등, 1=최대적합성) |
| `--write_meta` | - | 메타데이터 저장 |

### 출력 형식

**데이터**: `outputs/stage1/test/data/{source_id}.json`

```json
{
  "source_id": "0_1",
  "initial_info": {
    "people_number": 2,
    "days": 5,
    "budget_anchor": 1800,
    "org": "Charlotte",
    "dest": ["Asheville", "Roanoke"]
  },
  "personas": [
    // 20개 Stravl 페르소나
    {
      "ref_id": "stravl_6015",
      "budget_tier": "Mid-range",
      "activity_level": "Very Active",
      ...
    }
  ],
  "target_final_count": 2
}
```

---

## 💰 Stage 1.5: LLM 알파 서베이

### 목적

각 페르소나의 선호도를 20개 여행 필드에 대해 **0-10 알파값**으로 변환

### 알파값 의미

- **9-10**: MUST HAVE (hard constraint) - 절대 타협 불가
- **7-8**: SHOULD HAVE - 강한 선호
- **4-6**: COULD HAVE - 약한 선호
- **0-3**: INDIFFERENT (soft constraint) - 무관심

### 20개 필드

- **Accommodations** (5): price, rating, room_type, house_rule, parking
- **Restaurants** (4): price, rating, cuisine_type, dietary_restrictions
- **Flights** (4): price, stops, class, departure_time
- **Attractions** (4): rating, popularity, entry_fee, activity_type
- **Inter-city Travel** (3): mode, price, duration

### 모드 선택

#### Mode 1: Batch API (기본, 권장)

**장점**:
- 50% 비용 할인
- Reason축약으로 추가 26% 절감
- **총 63% 비용 절감** ($143.37 → $53.29)

**단점**:
- 0-24시간 대기 시간

**사용법**:

```bash
# 기본 사용 (제출 후 대기)
python run_stage1_5.py

# 제출만 (나중에 확인)
python run_stage1_5.py --no_wait

# 나중에 재개
python run_stage1_5.py --resume_batch_id $(cat outputs/stage1_5_batch/alpha_survey_batch_id.txt)

# GPT-4.1-mini 사용 (85% 절감, 품질 낮음)
python run_stage1_5.py --model gpt-4.1-mini
```

**비용 (9,200 페르소나)**:
- GPT-4.1: **$53.29** (63% 절감)
- GPT-4.1-mini: **$21.31** (85% 절감)

#### Mode 2: Iterative (최적화)

**장점**:
- Solvable 조합 발견 시 즉시 중단
- **최대 92% 비용 절감** (Round 1 성공 시)
- 실시간 처리 (1-3시간)

**단점**:
- Async API 사용 (batch 할인 없음)
- 평균적으로 Batch보다 비쌀 수 있음

**사용법**:

```bash
# 기본 사용 (1 persona per round)
python run_stage1_5_1_7_iterative.py

# 커스텀 설정
python run_stage1_5_1_7_iterative.py \
  --personas_per_round 2 \
  --max_rounds 10 \
  --max_records 10

# GPT-4.1-mini 사용
python run_stage1_5_1_7_iterative.py --model gpt-4.1-mini
```

**비용 비교 (9,200 페르소나)**:

| 시나리오 | Personas 처리 | Batch 비용 | Iterative 비용 | 승자 |
|---------|--------------|------------|----------------|------|
| **Best Case (Round 1)** | 1,000 (10%) | $5.33 | **$11.58** | Batch 저렴 |
| **Average (Round 3)** | 3,000 (30%) | $15.99 | **$34.75** | Batch 저렴 |
| **Worst Case** | 10,000 (100%) | $53.29 | $115.84 | Batch 저렴 |

**추천**:
- **Production**: Batch API (안정적, 저렴)
- **Exploration**: Iterative (유연성, 디버깅)

### 파라미터

#### Batch API 모드

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--stage1_dir` | `outputs/stage1/test/data` | Stage 1 입력 디렉토리 |
| `--out_dir` | `outputs/stage1_5/test/data` | 출력 디렉토리 |
| `--model` | `gpt-4.1` | LLM 모델 |
| `--temperature` | `0.1` | 샘플링 온도 |
| `--batch_dir` | `outputs/stage1_5_batch` | Batch 파일 디렉토리 |
| `--no_wait` | - | 제출만 하고 대기 안함 |
| `--resume_batch_id` | - | 기존 batch ID 재개 |

#### Iterative 모드

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--personas_per_round` | `1` | 라운드당 처리 페르소나 수 |
| `--max_rounds` | `10` | 최대 라운드 수 |
| `--max_concurrent` | `10` | 동시 API 호출 수 |
| `--max_combinations` | `1000` | 조합 평가 최대 수 |
| `--write_meta` | - | 메타데이터 저장 |

### 출력 형식

**Batch 모드**: `outputs/stage1_5/test/data/{source_id}.json`

```json
{
  "source_id": "0_1",
  "initial_info": {...},
  "personas": [
    {
      "ref_id": "stravl_6015",
      "alpha_survey": {
        "accommodations": {
          "price": {
            "value": "budget",
            "importance_score": 8,
            "reason": "Traveling on a tight budget"
          },
          "rating": {
            "value": "4+ stars",
            "importance_score": 3,
            "reason": "Not very concerned about ratings"
          }
        },
        "restaurants": {...},
        ...
      }
    }
  ]
}
```

**Iterative 모드**: `outputs/stage1_5_iterative/test/data/{source_id}.json` + `outputs/stage1_7_iterative/test/data/{source_id}.json`

---

## ✅ Stage 1.7: Solvable Conflict 선택

### 목적

k×10 페르소나 중 **solvable conflict**를 만족하는 최적 N명 조합 선택

### Solvable Conflict 정의

1. **Hard constraints (α≥9) 충돌 없음** → 실행 가능
2. **Soft constraints (4≤α<9) 충돌 있음** → 협상 필요
3. **최소 2개 차원에서 충돌** → 흥미로운 문제

### 선택 알고리즘

```python
# 1. 모든 조합 생성
combinations = C(20, 2)  # 2명 여행: 190개 조합

# 2. 각 조합에 대해 solvable 체크
for combo in combinations:
    # Hard constraint 충돌 체크 (α≥9)
    if has_hard_conflict(combo):
        continue  # 스킵

    # Soft constraint 충돌 개수 (4≤α<9)
    soft_conflicts = count_soft_conflicts(combo)

    if soft_conflicts >= 2:
        # 점수 계산
        score = soft_conflicts + category_diversity + alpha_variance

# 3. 최고 점수 조합 선택
best_combo = max(scored_combinations, key=lambda x: x.score)
```

### 사용법

```bash
# 기본 사용
python3 run_stage1_7.py --write_meta

# 테스트 (10개만)
python3 run_stage1_7.py --max_records 10 --write_meta

# Iterative 모드는 run_stage1_5_1_7_iterative.py에 통합됨
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--stage1_5_dir` | `outputs/stage1_5/test/data` | Stage 1.5 입력 디렉토리 |
| `--out_dir` | `outputs/stage1_7/test/data` | 출력 디렉토리 |
| `--max_combinations` | `1000` | 평가할 최대 조합 수 |
| `--write_meta` | - | 메타데이터 저장 |

### 출력 형식

**데이터**: `outputs/stage1_7/test/data/{source_id}.json`

```json
{
  "source_id": "0_1",
  "initial_info": {...},
  "personas": [
    // 최종 2명만
    {
      "ref_id": "stravl_6015",
      "alpha_survey": {...}
    },
    {
      "ref_id": "stravl_7234",
      "alpha_survey": {...}
    }
  ]
}
```

**메타데이터**: `outputs/stage1_7/test/meta/{source_id}.json`

```json
{
  "source_id": "0_1",
  "conflict_analysis": {
    "is_solvable": true,
    "soft_conflicts": [
      {
        "field": "accommodations.price",
        "personas": [
          {"persona_id": "stravl_6015", "value": "budget", "alpha": 8},
          {"persona_id": "stravl_7234", "value": "luxury", "alpha": 7}
        ]
      }
    ],
    "conflict_count": 5,
    "score": 12.34
  },
  "selected_ref_ids": ["stravl_6015", "stravl_7234"]
}
```

---

## 💵 전체 비용 분석

### Stage 1: 무료 (오프라인 검색)

### Stage 1.5: LLM 비용

#### 비용 계산 (9,200 페르소나 기준)

**토큰 사용량 (페르소나당)**:
- Input: 2,192 tokens (프롬프트 + reason축약)
- Output: 900 tokens (20 필드 × 45 tokens)
- Total: 3,092 tokens

**Batch API (GPT-4.1, 50% 할인)**:
```python
input_cost = (9200 × 2192 / 1e6) × $1.00 = $20.17  # 50% 할인
output_cost = (9200 × 900 / 1e6) × $4.00 = $33.12   # 50% 할인
total = $53.29
```

**Async API (GPT-4.1, 할인 없음)**:
```python
input_cost = (9200 × 2192 / 1e6) × $2.00 = $40.34
output_cost = (9200 × 900 / 1e6) × $8.00 = $66.24
total = $106.58
```

**GPT-4.1-mini (Batch API)**:
```python
input_cost = (9200 × 2192 / 1e6) × $0.40 = $8.07   # 50% 할인
output_cost = (9200 × 900 / 1e6) × $1.60 = $13.25  # 50% 할인
total = $21.31
```

#### 비용 비교표

| 모드 | 처리량 | 비용 | 절감률 | 시간 | 권장 |
|------|--------|------|--------|------|------|
| **Batch API (GPT-4.1)** | 100% | **$53.29** | 63% | 0-24h | ✅ Production |
| Batch API (GPT-4.1-mini) | 100% | $21.31 | 85% | 0-24h | 저품질 OK |
| Iterative (Best) | 10% | $11.58 | 92% | 1-3h | 탐색용 |
| Iterative (Avg) | 30% | $34.75 | 76% | 1-3h | - |
| Iterative (Worst) | 100% | $106.58 | 26% | 1-3h | ❌ |

### Stage 1.7: 무료 (조합 검색)

### 전체 파이프라인 비용 (1,000 trips)

- **Batch 모드**: $53.29 (Stage 1.5만 유료)
- **Iterative 모드 (평균)**: $34.75 (Stage 1.5+1.7 통합)

---

## 📖 추가 문서

### Iterative 모드 알고리즘

```python
For each Stage 1 record:
  enriched_personas = []

  For round = 1 to 10:
    # 1. K개 페르소나 처리 (Async API)
    batch = next K personas from Stage 1
    enriched_batch = alpha_survey(batch)  # LLM 호출
    enriched_personas += enriched_batch

    # 2. Solvability 체크
    if len(enriched_personas) >= target_count:
      result = find_solvable_combination(enriched_personas)

      if result is not None:
        # Solvable 조합 발견!
        save_outputs(result)
        return SUCCESS  # 이 레코드 처리 중단

    # 3. 아직 solvable 아님, 다음 라운드로

  # 최대 라운드 도달, solvable 조합 없음
  return NO_SOLVABLE
```

### 데이터 규모

- **TravelPlanner test set**: 1,000개 여행 시나리오
- **Stravl 페르소나**: 80,301개 설문 응답자
- **Stage 1 출력**: trip당 20~40개 페르소나 (people × 10)
- **Stage 1.7 출력**: trip당 2~4명 최종 페르소나

---

## 🐛 문제 해결

### Stravl CSV 없음

```bash
export STRAVL_CSV_PATH=/path/to/Stravl_Travel_Preference_Data.csv
```

또는 자동 다운로드 (첫 실행 시)

### LLM API 오류

```bash
# API 키 확인
echo $OPENAI_API_KEY

# Claude 사용 시
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Batch API 상태 확인

```bash
# Batch ID 확인
cat outputs/stage1_5_batch/alpha_survey_batch_id.txt

# OpenAI CLI로 상태 확인
openai api batches.retrieve -i batch_xxxxx
```

### Memory 부족

```bash
# 배치 처리
python3 run_stage1.py --max_records 100
python3 run_stage1_5.py --max_records 100
python3 run_stage1_7.py --max_records 100
```

---

## 🎓 학술적 정당성

### 왜 LLM 제거? (Stage 1)

1. **Data Provenance**: Stravl 데이터셋 직접 인용 가능
2. **Reproducibility**: 완전한 재현성 (seed 고정)
3. **Transparency**: 명확한 알고리즘 (블랙박스 없음)
4. **Cost**: Stage 1은 API 비용 $0

### 왜 LLM 사용? (Stage 1.5)

1. **Structured Mapping**: 설문 응답 → DB 필드 매핑
2. **Importance Scoring**: 0-10 알파값으로 제약 강도 표현
3. **Reasoning**: "이 사람은 왜 이 필드를 중요하게 여기는가" 설명

### 왜 Solvable Conflict?

1. **Controlled Evaluation**: 특정 갈등 차원 테스트 가능
2. **Realistic**: Hard constraint 충돌하면 실행 불가능
3. **Interesting**: Soft constraint 충돌로 협상 유도

---

## 📚 참고 문헌

- **Stravl Dataset**: Li et al. (2024)
- **TravelPlanner**: Xie et al. (2024)
- **MMR**: Carbonell & Goldstein (1998)

---

## 📄 라이센스

MIT (Stravl 데이터는 각 라이센스 참조)
