import json, os, glob, datetime

# ① 최신 결과 폴더 자동 찾기
src_dir = max(glob.glob(r"results\\travelplanner\\openai\\gpt-4.1-test-mini\\*"), key=os.path.getmtime)
src = os.path.join(src_dir, "official_submission.jsonl")

# ② jsonl → json 배열로 변환
preds = [json.loads(line) for line in open(src, encoding="utf-8")]

# ③ 저장 위치 (HM/runs/시간_mini.json)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
dst = f"HM/runs/{ts}_mini.json"
os.makedirs(os.path.dirname(dst), exist_ok=True)

# ④ 팀 제출 포맷
submission = {
    "team": "HM",
    "workflow": "hm-baseline",
    "provider": "openai",
    "model": "gpt-4.1",
    "split": "validation",
    "result_label": "test-mini",
    "predictions": preds
}

json.dump(submission, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OK] 제출 파일 생성: {dst}")
