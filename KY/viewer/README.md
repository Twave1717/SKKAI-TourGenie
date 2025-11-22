# TripCraft 라벨링 뷰어 (Streamlit)

간단히 CSV 여행계획을 열람하고, 한글 번역본 확인 및 라벨링(휴먼/LLM)을 남길 수 있는 도구입니다.

## 준비 (conda)
1. 프로젝트 루트로 이동: `cd /Users/ky/AgenticAI/Challenge`
2. conda 환경 생성: `conda env create -f KY/viewer/environment.yml`
3. 활성화: `conda activate tripcraft-viewer`
<!-- 4. `.env`에 `UPSTAGE_API_KEY`, `OPENAI_API_KEY`가 들어있는지 확인합니다. -->

## 실행
```
streamlit run KY/viewer/app.py
```
- 좌측: 영어 원본 + 한글 번역, 일정 테이블/참고정보
- 우측: 휴먼 라벨(불완전 체크 + 서술), GPT-4.1 검토, 이전/다음 이동
- CSV 선택: 사이드바에서 `benchmarks/TripCraft/tripcraft` 내 파일 선택

## 번역/LLM 캐시 미리 생성(선택)
Upstage 번역 + GPT-4.1-mini 리뷰를 각각 8개 스레드 클라이언트로 병렬 실행하며, 진행상황은 `tqdm`으로 표시됩니다.
```
python KY/viewer/preprocess.py               # 모든 CSV
python KY/viewer/preprocess.py --files tripcraft_3day_mini.csv
```
- 캐시 파일: `KY/viewer/cache/translations.json`
- LLM 리뷰 캐시: `KY/viewer/cache/llm_reviews.json`

## 라벨 저장 위치
- `KY/viewer/labels/<csv>_labels.json`
- 예시 파일: `KY/viewer/labels/tripcraft_3day_mini.csv_labels.example.json`

## 스크립트
- `KY/viewer/data_loader.py`: CSV 로더/파서 유틸
- `KY/viewer/preprocess.py`: 번역 캐시 생성 스크립트 (Upstage `solar-pro-2`)
