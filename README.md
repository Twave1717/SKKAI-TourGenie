# Agentic AI Workflow (TravelPlanner Baseline)

LangGraph 기반의 에이전틱 워크플로를 TravelPlanner 벤치마크에 적용하기 위한 최소 골격입니다.      
팀원별 실험은 개인 폴더에서 진행하고, 공용으로는 데이터 다운로드 스크립트와 평가 러너를 공유합니다.

## Quick Start

```bash
# 1) Python 3.11 및 의존성 설치
pyenv install 3.11
pyenv local 3.11
poetry shell # 또는 poetry env activate
poetry install

# 2) 환경 변수 템플릿 복사 후 API 키/경로 입력
cp .env.example .env
# 필요 시 모든 쉘에서 공유되도록 source
set -a && source .env && set +a

# 3) TravelPlanner DB(항공/숙소/식당 등) 수동 다운로드
# Google Drive: https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view (database.zip)
# 프로젝트 루트(또는 하위 폴더 어디든)에 database.zip만 두면 4번 스크립트가 자동으로 찾아서 압축 해제해 줍니다.

# 4) TravelPlanner 데이터 내려받기
poetry run python scripts/fetch_travelplanner.py --split validation --dest data/travelplanner --force-db --download-db

# 5) test-mini(60개 고정) 베이스라인 평가
poetry run python benchmarks/travelplanner/eval_runner.py --provider openai --model gpt-4.1 --test-mini
# 전체 validation/test를 전부 돌리려면 --test-mini 플래그를 제거하세요.
# Tool-Calling TravelPlanner 에이전트 병렬 실행 예시(기본 4스레드, TRAVELPLANNER_WORKERS로 조절)
# TRAVELPLANNER_WORKERS=4 poetry run python benchmarks/travelplanner/eval_runner.py --provider "travelplanner(workflow)" --model gpt-4.1 --test-mini
# (별도 지정이 없으면 항상 validation split을 사용)

# 6) 팀 제출 JSON 자동 평가 & 리더보드 갱신
# 샘플 포맷 검증용: --allow-partial 플래그를 주면 예시 파일(2건)로도 실행됩니다.
poetry run python scripts/eval_submission.py --submission examples/team_submission_example.json --allow-partial
```

위 명령으로 최소 구성 요소를 확인한 뒤, 세부 옵션은 아래 문서를 참고해 확장하세요.  
**이 문서의 모든 설명은 기본적으로 test-mini(60개 고정 셋) 실행을 기준으로 하며, 전체 validation/test 평가가 필요하면 `--test-mini` 플래그를 제거한 동일 명령을 사용하면 됩니다.**

## Evaluation / Leaderboard

팀(예: HM/KY/TJ/YJ)이 자체 워크플로로 TravelPlanner 출력을 생성하면, 지정된 JSON 포맷으로 저장한 뒤 `scripts/eval_submission.py`만 호출하면 공식 evaluator 및 리더보드 업데이트가 한 번에 수행됩니다.

### 1. 제출 JSON 포맷

`examples/team_submission_example.json`을 참고해 다음 구조를 맞춰 주세요.

```json
{
  "team": "HM",                  // 필수, 리더보드 표시에 사용
  "workflow": "hm-baseline",     // 필수, 동일 팀 내 실험 구분
  "provider": "openai",          // 필수, 리더보드 Provider 컬럼
  "model": "gpt-4.1",            // 필수, 리더보드 Model 컬럼
  "split": "validation",         // 기본값 validation, test/test-mini 가능
  "result_label": "test-mini",   // (선택) 리더보드 링크 꼬리표. 없으면 split 사용
  "predictions": [
    {
      "id": "validation_00000",  // dataset JSONL의 id와 일치
      "plan": [                  // 공식 evaluator가 기대하는 Plan JSON
        {
          "days": 1,
          "current_city": "Washington → Myrtle Beach",
          "transportation": "...",
          "breakfast": "...",
          "lunch": "...",
          "dinner": "...",
          "attraction": "...",
          "accommodation": "..."
        }
      ],
      "raw_response": "...",     // (선택) 원본 문자열
      "notes": "..."             // (선택)
    }
  ]
}
```

- `plan` 배열은 최소 하루 이상이어야 하며, 위 8개 키를 모두 포함해야 합니다(추가 필드는 허용).
- 리더보드 기록을 남기려면 평가 대상 데이터셋의 **모든 예제**가 포함되어야 합니다. 예시 파일은 포맷 참고용으로 2건만 포함되어 있으므로 `--allow-partial` 옵션을 주지 않으면 실패하게 되어 있습니다.

### 2. 평가 스크립트

```bash
# test-mini 제출 → mini 리더보드 자동 갱신(권장 기본)
poetry run python scripts/eval_submission.py \
  --submission KY/runs/20251105_mini.json \
  --result-label test-mini \
  --dataset data/travelplanner/validation.jsonl

# 전체 validation(180건) 제출 → 메인 리더보드
poetry run python scripts/eval_submission.py \
  --submission HM/runs/20251104_hm_baseline.json \
  --dataset data/travelplanner/validation.jsonl
```

| Option | Default | 설명 |
| --- | --- | --- |
| `--dataset` | 제출 파일의 `dataset` 또는 `data/travelplanner/validation.jsonl` | 평가 기준 데이터셋. test-mini도 validation 원본에서 추출하므로 동일 경로를 사용합니다. |
| `--result-label` | `submission.result_label` → `split` | 리더보드에 표기할 꼬리표. test-mini 결과는 `test-mini`를 명시해 mini 보드에 기록하세요. |
| `--allow-partial` | `False` | 제출에 일부 예제만 있어도 평가를 계속합니다. 리더보드 업데이트는 비활성화됩니다. |
| `--skip-leaderboard` | `False` | 점수만 계산하고 리더보드는 수정하지 않습니다. |

스크립트는 아래 파일을 자동으로 생성합니다.

- `results/travelplanner/<provider>/<model-workflow>-<team>-submission/<timestamp>/`
  - `submission.json`: 원본 제출 사본
  - `official_submission.jsonl`: 공식 evaluator 입력
  - `metrics.json`, `official_metrics.json`: 요약 및 세부 점수

### 3. 리더보드 규칙

1. `--allow-partial` 없이 전체 데이터셋을 평가한 경우에만 리더보드가 갱신됩니다.
2. 동일 `(provider, model, result_label)` 조합은 최신 실행 결과로 대체됩니다.
3. Mini 세트 결과는 `leaderboards/TravelPlanner/mini.md`에 `(...test-mini)` 꼬리표로 표시되고, full/validation/test 결과는 `leaderboards/TravelPlanner/main.md`에 기록됩니다. 두 파일은 항상 `leaderboards/TravelPlanner/README.md`에 통합되어 노출되며, 테이블은 Final Pass Rate → Hard/Commonsense Pass Rate 순으로 정렬해 `Rank` 컬럼을 자동 부여합니다.

### TravelPlanner Leaderboards

`leaderboards/TravelPlanner/README.md`에 Main/Mini 리더보드가 함께 정리되어 있으며, 새 결과를 업데이트하면 해당 파일이 자동으로 재생성됩니다. 각 행의 `Results` 링크는 커밋된 `results/` 디렉터리의 실행 로그를 가리키므로, 실험별 산출물을 깃 히스토리에서 직접 확인할 수 있습니다.

## 환경 준비

Python 3.11이 필요합니다. `pyenv`와 Poetry를 사용하는 것을 전제로 합니다.

```bash
# Python 3.11 설치 및 활성화
pyenv install 3.11
pyenv local 3.11

# Poetry 가상환경을 pyenv 파이썬으로 생성
poetry env use $(pyenv which python)

# 의존성 설치
poetry install

# 환경 변수 템플릿 복사 후 키 설정
cp .env.example .env
```

`.env` 파일에는 사용하는 LLM 제공업체의 키를 채웁니다 (`OPENAI_API_KEY`, `UPSTAGE_API_KEY`, `GOOGLE_API_KEY`). LangSmith 추적을 사용하려면 `.env.example`에 있는 `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`, `LANGCHAIN_API_KEY` 항목도 함께 설정하세요.

## 데이터 준비

- 공식 데이터셋: <https://huggingface.co/datasets/osunlp/TravelPlanner>
- 리더보드 참고: <https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard>

TravelPlanner를 비롯한 대용량 데이터는 깃헙에 업로드하지 않고,    
최초 설치 시 별도 스크립트로 다운받습니다.

```bash
poetry run python scripts/fetch_travelplanner.py --split test --dest data/travelplanner
```

기본적으로 JSONL 파일이 `data/travelplanner/` 아래에 생성됩니다. 다른 벤치마크(예: TripCraft)를 추가할 때는 `--dataset` 옵션을 활용해 Hugging Face 경로를 변경하면 됩니다.

공식 evaluator를 사용하려면 TravelPlanner에서 제공하는 참조 DB(csv/json)를 추가로 내려받아야 합니다. **Google Drive(<https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view>)에서 `database.zip`을 직접 내려받은 뒤** 아래와 같이 압축을 풀어 주세요.

```bash
# 수동으로 내려받은 database.zip을 지정해 압축 해제
poetry run python scripts/fetch_travelplanner.py \
  --download-db \
  --db-zip /path/to/database.zip
```

DB는 기본적으로 `benchmarks/travelplanner/official/database/`에 풀리며, 기존 DB를 덮어쓰려면 `--force-db`를 함께 지정하세요. 필수 파일(`flights/clean_Flights_2022.csv`, `restaurants/clean_restaurant_2022.csv`, `accommodations/clean_accommodations_2022.csv`, `googleDistanceMatrix/distance.csv`)이 모두 존재하는지 스크립트가 검증합니다. 필요 시 `--download-db` 옵션이 Google Drive에서 직접 내려받을 수도 있지만, 환경에 따라 SSL 인증서 문제로 실패할 수 있으므로 수동 다운로드를 권장합니다.

TravelPlanner는 split마다 제공되는 필드가 다릅니다.
- `train`: 정답으로 활용 가능한 `annotated_plan`이 포함되어 있습니다.
- `validation` / `test`: 참고 자료(`reference_information`)만 제공되며 정답 플랜은 없습니다.

스크립트는 각 예시에 `id`를 자동 부여하므로 저장된 JSONL에는 `test_00000` 같은 고유 ID가 포함됩니다.

## 베이스라인 평가

LangGraph로 구성한 단순 백본 LLM 평가기를 실행하려면 다음 명령을 사용합니다.

```bash
poetry run python benchmarks/travelplanner/eval_runner.py --provider openai
```

주요 옵션 (test-mini 실행을 기본으로 가정)

| Option | Default | 설명 |
| --- | --- | --- |
| `--provider {openai,google,upstage,"travelplanner(workflow)"}` | `openai` | `travelplanner(workflow)`는 논문과 동일한 Tool-Calling 워크플로를 실행합니다. 괄호 때문에 반드시 따옴표로 감싸 주세요. |
| `--model` | 제공업체 기본 | 사용할 모델명을 지정합니다. 예: `gpt-4.1`, `gpt-4-1106-preview`. |
| `--dataset` | `data/travelplanner/validation.jsonl` | 평가에 사용할 JSONL 경로. test-mini도 validation 원본에서 추출합니다. |
| `--system-prompt` | TravelPlanner 전용 기본 프롬프트 | JSON 플랜 스키마 강제 문장을 수정할 때 사용합니다. |
| `--temperature` | `0.2` | LLM 샘플링 온도. |
| `--save-messages` | `False` | LangGraph 메시지 히스토리를 JSON으로 저장합니다. |
| `--test-mini` | `False`(전체 split) → **권장: `True`** | 60개 고정 셋을 사용해 빠르게 회귀 테스트합니다. 본 문서는 항상 이 옵션을 켠 상태를 기준으로 설명합니다. |
| `--run-official-eval` | `True` | TravelPlanner 공식 지표 계산 여부. |
| `--official-set-type` | 데이터 경로에서 자동 추론 (주로 `validation`) | Hugging Face split을 명시적으로 지정합니다. |
| `--use-official-parser` | `False` | OpenAI 파서를 호출해 자연어 플랜을 JSON으로 정제합니다. test-mini에서도 파서 결과를 권장합니다. |
| `--parser-model` | 평가 모델과 동일 (미지정 시 `gpt-4-1106-preview`) | 공식 파서에 사용할 OpenAI 모델을 바꿉니다. |

실행 결과는 `results/travelplanner/<provider>/<model>/<timestamp>/`에 저장되며, `metrics.json`(요약 지표), `predictions.jsonl`(예측값), `*_messages.json`(옵션, 메시지 로그)로 구성됩니다. `LANGCHAIN_TRACING_V2`를 활성화하면 LangSmith에서 각 예제의 추적을 확인할 수 있습니다.

데이터셋의 `reference_information`과 메타데이터(출발지, 목적지, 여행일 등)는 자동으로 프롬프트에 병합되어 백본 모델도 지도 정보를 활용할 수 있습니다.

예측 이후에는 공식 데이터베이스(식당·숙소·관광지·항공편)를 조회해 LLM 출력이 만든 자유 형식 문자열을 표준 JSON 스키마로 보정합니다. 정규화된 플랜은 `prediction` 필드에 저장되고, 원본 응답은 `raw_prediction` 필드에 함께 남으므로 필요하면 두 값을 비교할 수 있습니다.

### Tool-Calling 에이전트 모드

논문 속 TravelPlanner Tool Calling 파이프라인을 그대로 사용하려면 `--provider "travelplanner(workflow)"`를 지정하세요. 내부적으로 `benchmarks/travelplanner/official/agents/tool_agents.py`의 React+Planner 에이전트를 호출하며, 각 예제에 대해 도구 호출 로그(`results/.../travelplanner_agent_logs/`)를 남깁니다.

```bash
# TRAVELPLANNER_WORKERS는 병렬로 띄울 LangGraph backbone 수 (기본 4)
# 예제: 8개 워커로 60개 mini 세트를 평가
TRAVELPLANNER_WORKERS=8 \
poetry run python benchmarks/travelplanner/eval_runner.py \
  --provider "travelplanner(workflow)" \
  --model gpt-4-1106-preview \
  --test-mini

# 중단된 실행 재개 (동일 run_dir)
poetry run python benchmarks/travelplanner/eval_runner.py \
  --provider "travelplanner(workflow)" \
  --model gpt-4.1 \
  --test-mini \
  --resume-run "results/travelplanner/travelplanner(workflow)/gpt-4.1-test-mini/20251104-203629"
```

- OpenAI·Google API 키가 모두 필요합니다 (`OPENAI_API_KEY`, `GOOGLE_API_KEY`).
- `TRAVELPLANNER_WORKERS` 환경 변수를 조정해 병렬 에이전트 수를 변경할 수 있습니다. 워커 수 × 도구 호출 횟수만큼 OpenAI/Google 쿼리가 발생하므로 요금과 레이트리밋을 감안해 설정하세요.
- 실행 시간과 비용이 크므로 `--test-mini` 같은 소규모 셋업으로 먼저 검증한 뒤 full validation/test로 확장하는 것을 권장합니다.
- 생성된 플랜은 동일한 공식 evaluator를 거치므로 `official_metrics.json`에서 논문 수치와 비교할 수 있습니다.
- 각 예제의 중간 산출물은 `travelplanner_agent_logs/validation_xxxxx_{actions.json,scratchpad.txt,log}`에 저장되고, 실행 도중 중단되더라도 `travelplanner_agent_predictions.jsonl`에 완료된 예측이 실시간으로 append 됩니다. 또 루트에는 실행 설정을 담은 `run_config.json`이 생성됩니다. `--resume-run` 옵션으로 동일 디렉터리를 지정하면 이 메타데이터를 검증한 뒤, 성공한 예제는 자동으로 건너뛰고 나머지만 이어서 추론합니다 (config mismatch 시에는 안전하게 중단).

### 공식 평가 연동

`benchmarks/travelplanner/official/` 디렉터리에 TravelPlanner 저장소의 evaluator, tools, database, postprocess 모듈을 그대로 포팅했습니다. 실행 시 `official_output/<split>/generated_plan_*.json`에 공식 포맷으로 변환된 플랜이 저장되며, `--use-official-parser`를 지정하면 레퍼런스 파서(`postprocess/openai_request.py`, `element_extraction.py`)를 그대로 호출해 JSON을 정제합니다.

```bash
poetry run python benchmarks/travelplanner/eval_runner.py \
  --provider openai \
  --dataset data/travelplanner/validation.jsonl \
  --run-official-eval \
  --official-set-type validation
```

| Option | Default | 설명 |
| --- | --- | --- |
| `--run-official-eval` | `True` | 공식 evaluator 호출 여부 |
| `--official-set-type` | 데이터 경로에서 추론 (`validation`) | Hugging Face split(`train`/`validation`/`test`)을 명시적으로 지정 |
| `--use-official-parser` | `False` | GPT 파서를 호출해 자연어 플랜을 JSON으로 변환 |

공식 evaluator는 `gradio`, `pandas`, `numpy` 등을 요구하며, 모든 지표는 `official_metrics.json`과 `metrics.json`에 동일하게 기록됩니다.

## 리더보드 자동 갱신

`eval_runner.py` 실행이 끝나면 최신 공식 메트릭이 `leaderboards/TravelPlanner/main.md`에 자동 반영됩니다. `--test-mini` 결과는 `leaderboards/TravelPlanner/mini.md`에 모델명 그대로 기록되며, `Results` 링크 끝의 `(test-mini)` 표기로 정식 기록과 구분됩니다. 두 파일의 내용은 항상 `leaderboards/TravelPlanner/README.md`에 통합되어 노출됩니다.

## TripCraft Benchmark

TripCraft는 TripPlanner보다 시간·공간 제약이 더 강한 3/5/7일 일정 벤치마크입니다. 원본 저장소 전체를 `benchmarks/TripCraft`에 포함했으며, 아래 스텝만 추가로 수행하면 동일한 워크플로 안에서 실행할 수 있습니다. TripCraft를 실행하기 전에 `set -a && source .env && set +a` 명령으로 공용 환경 변수를 불러오면 `OPENAI_API_KEY`, `TRIPCRAFT_DB_ROOT`, `TRIPCRAFT_OUTPUT_DIR` 값을 따로 export 하지 않아도 됩니다.

### 1. 데이터베이스 준비

TripCraft는 30GB가 넘는 오프라인 DB(항공편/숙소/식당/관광지/이벤트)를 요구합니다. 공식 릴리스에서 `TripCraft_database.zip` 을 내려받은 뒤 다음 스크립트로 압축을 풀어 두세요.

```bash
# TripCraft_database.zip 이 프로젝트 루트 또는 benchmarks/TripCraft 밑에 있으면 자동 탐색됩니다.
poetry run python scripts/fetch_tripcraft.py \
  --archive /path/to/TripCraft_database.zip \
  --dest benchmarks/TripCraft/TripCraft_database
```

압축 해제 위치를 바꾸고 싶다면 `--dest` 경로를 조정하거나, 실행 시 `TRIPCRAFT_DB_ROOT` 환경변수를 덮어쓰면 됩니다.

### 2. TripCraft 플래너 실행

`benchmarks/tripcraft/eval_runner.py` 는 TravelPlanner 러너와 비슷한 CLI를 제공하며, 내부적으로 `benchmarks/TripCraft/run.sh` 를 호출합니다. 최소 실행 예시는 다음과 같습니다.

```bash
export OPENAI_API_KEY=sk-...
poetry run python benchmarks/tripcraft/eval_runner.py run \
  --model-name gpt-4.1-mini \
  --day 3day \
  --set-type 3day_gpt4o_orig \
  --strategy direct_og \
  --csv-file benchmarks/TripCraft/tripcraft/tripcraft_3day.csv \
  --output-dir benchmarks/TripCraft/output
```

- `--day` 는 3day/5day/7day 중 하나를 선택합니다. 별도 지정이 없으면 `tripcraft/tripcraft_{day}.csv` 를 자동으로 사용합니다.
- `--set-type` 은 TripCraft 원본 스크립트에서 폴더명을 구분하는 용도로 사용됩니다. (예: `3day_gpt4o_orig`)
- `--strategy` 는 `agents/prompts.py` 에 정의된 프롬프트를 선택합니다. (`direct_og`, `direct_param` 등)
- TripCraft 실행 시 `OPENAI_API_KEY` 가 반드시 설정되어 있어야 하며, 추가 도구는 필요 없습니다.

### 3. Postprocess & 평가

TripCraft는 자연어 플랜을 JSON으로 변환한 뒤 평가합니다. 전 과정은 TripCraft 전용 conda 환경에서 해결할 수 있도록 `scripts/tripcraft_postprocess.py` 와 `scripts/tripcraft_eval.py` 를 제공합니다.

```bash
# TripCraft conda env (e.g., `conda activate tripcraft`)에서 실행
python scripts/tripcraft_postprocess.py \
  --input-root results/tripcraft/gpt41mini_test_mini \
  --output-jsonl results/tripcraft/gpt41mini_test_mini.jsonl \
  --csv-3day benchmarks/TripCraft/tripcraft/tripcraft_3day.csv \
  --csv-5day benchmarks/TripCraft/tripcraft/tripcraft_5day.csv \
  --csv-7day benchmarks/TripCraft/tripcraft/tripcraft_7day.csv \
  --openai-model gpt-4.1-mini
```

위 스크립트가 TripCraft 자연어 출력을 `postprocess/sample_evaluation_format.jsonl` 스키마로 변환합니다. 이어서 TripCraft 전용 평가/리더보드 스크립트를 실행하세요.

```bash
python scripts/tripcraft_eval.py \
  --submission results/tripcraft/gpt41mini_test_mini.jsonl \
  --provider openai \
  --model gpt-4.1-mini \
  --workflow test-mini \
  --result-label gpt41mini_test_mini
```

- `tripcraft_eval.py` 는 TripCraft evaluator를 호출해 Delivery Rate/Commonsense/Hard/Final Pass Rate 등을 계산하고, 결과를 `results/tripcraft/<result-label>/metrics.json` 에 저장한 뒤 `leaderboards/TripCraft/main.md` 를 업데이트합니다.
- TravelPlanner용 Poetry 환경이 필요하지 않으며 TripCraft conda 환경 하나만으로 “생성 → 후처리 → 평가 → 리더보드” 전체 파이프라인을 반복 실행할 수 있습니다.

## 다음 단계

- 개인 폴더에서 LangGraph 노드/도구를 확장한 뒤, 공용 평가 러너에 연결하여 성능을 비교하세요.
- TripCraft 등 다른 벤치마크도 동일한 구조로 `benchmarks/` 아래에 추가하여 재사용성을 확보하세요.
