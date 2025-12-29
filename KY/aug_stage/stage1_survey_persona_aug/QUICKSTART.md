# Quick Start Guide

1분 안에 파이프라인 실행하기

---

## 📋 전제 조건

```bash
# 1. API 키 설정 (KY/aug_stage/.env 파일에 이미 있음)
# run.sh가 자동으로 로드합니다

# 2. 의존성 설치 (처음 1회만)
pip install -r requirements.txt
```

> **💡 API 키**: `KY/aug_stage/.env` 파일에 `OPENAI_API_KEY`가 설정되어 있습니다.
> run.sh가 자동으로 로드하므로 별도 설정이 필요 없습니다.

---

## 🚀 기본 사용법

### 전체 파이프라인 실행 (1000개 레코드)

```bash
./run.sh
```

**예상 시간**: 2-3시간 (Stage 1) + 0-24시간 (Stage 1.5 Batch API) + 10분 (Stage 1.7)
**예상 비용**: $53.29

---

## 🧪 테스트 모드 (10개 레코드)

### Batch 모드 (권장)

```bash
./run.sh --max_records 10
```

**예상 시간**: 5분 (Stage 1) + 10-30분 (Stage 1.5) + 1분 (Stage 1.7)
**예상 비용**: $0.58

### Iterative 모드 (실시간)

```bash
./run.sh --mode iterative --max_records 10
```

**예상 시간**: 5분 (Stage 1) + 15-30분 (Stage 1.5+1.7 통합)
**예상 비용**: $0.13 - $0.38 (best case)

---

## 📊 출력 확인

```bash
# Stage 1 출력 (k×10 페르소나)
ls outputs/stage1/test/data/*.json

# Stage 1.5 출력 (alpha 값 포함)
ls outputs/stage1_5/test/data/*.json

# Stage 1.7 출력 (최종 N명)
ls outputs/stage1_7/test/data/*.json

# 샘플 출력 보기
cat outputs/stage1_7/test/data/0_1.json | jq
```

---

## ⚙️ 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode <batch\|iterative>` | 파이프라인 모드 | `batch` |
| `--max_records <N>` | 처리할 레코드 수 (0 = 전체) | `0` |
| `--model <model>` | LLM 모델 | `gpt-4.1` |
| `--workers <N>` | Stage 1 병렬 작업자 수 | `8` |
| `--skip_stage1` | Stage 1 스킵 (이미 완료한 경우) | - |
| `--help` | 도움말 출력 | - |

---

## 📝 사용 예시

### 1. Stage 1만 다시 실행

```bash
# Stage 1만 실행
python3 run_stage1.py --split test --max_records 10 --write_meta
```

### 2. Stage 1.5+1.7만 실행 (Stage 1 스킵)

```bash
./run.sh --skip_stage1 --max_records 10
```

### 3. GPT-4.1-mini 사용 (저렴한 옵션)

```bash
./run.sh --model gpt-4.1-mini --max_records 10
```

**예상 비용**: $0.23 (85% 절감, 품질 낮음)

### 4. 병렬 처리 늘리기

```bash
./run.sh --workers 16 --max_records 100
```

---

## 🐛 문제 해결

### API 키 오류

```bash
Error: OPENAI_API_KEY environment variable is not set
```

**해결**:
```bash
export OPENAI_API_KEY="sk-..."
```

### Stage 1.5 Batch API 대기 중

Batch API는 0-24시간 소요됩니다. 나중에 재개하려면:

```bash
# 1. Batch ID 확인
cat outputs/stage1_5_batch/alpha_survey_batch_id.txt

# 2. 나중에 재개
python3 run_stage1_5.py --resume_batch_id <batch_id>
```

### 메모리 부족

```bash
# 배치 처리 (100개씩)
./run.sh --max_records 100
```

---

## 📚 더 알아보기

- 전체 문서: [README.md](README.md)
- 논문 작성: [PAPER.md](PAPER.md)
- 비용 최적화: README.md의 "전체 비용 분석" 섹션

---

## 🎯 권장 워크플로우

### 개발/테스트

```bash
# 1. 소규모 테스트 (10개)
./run.sh --mode iterative --max_records 10

# 2. 중규모 테스트 (100개)
./run.sh --max_records 100

# 3. 결과 검증
python validate_outputs.py
```

### 프로덕션

```bash
# 전체 실행 (1000개)
./run.sh
```

**예상 소요 시간**: 1-2일 (Batch API 대기 시간 포함)
**예상 비용**: $53.29
