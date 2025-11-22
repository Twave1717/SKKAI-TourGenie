import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Support both package import and direct script run
try:
    from data_loader import (
        DATA_DIR,
        ROOT_DIR,
        build_plan_text,
        list_csv_files,
        load_rows,
    )
except ImportError:  # pragma: no cover - streamlit run path
    from .data_loader import (  # type: ignore
        DATA_DIR,
        ROOT_DIR,
        build_plan_text,
        list_csv_files,
        load_rows,
    )


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
TRANSLATION_CACHE_PATH = CACHE_DIR / "translations.json"
LLM_REVIEW_CACHE_PATH = CACHE_DIR / "llm_reviews.json"
FORMATTED_CACHE_PATH = CACHE_DIR / "formatted_plans.json"
FORMATTED_META_PATH = CACHE_DIR / "formatted_meta.json"
FORMATTED_POI_PATH = CACHE_DIR / "formatted_poi_tables.json"
TRANSLATED_CSV_DIR = CACHE_DIR / "translated_csv"
FORMATTED_TRANS_PATH = CACHE_DIR / "formatted_plans_translated.json"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT_DIR / ".env", override=False)
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
# Default to solar-pro2 (works with structured outputs); override via UPSTAGE_MODEL if needed.
UPSTAGE_MODEL = os.getenv("UPSTAGE_MODEL", "solar-pro2")

upstage_client = None
if UPSTAGE_API_KEY:
    upstage_client = OpenAI(api_key=UPSTAGE_API_KEY, base_url=UPSTAGE_BASE_URL)

openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Thread pools
PARALLEL_CLIENTS = 8
upstage_clients: List[OpenAI] = []
openai_clients: List[OpenAI] = []
if UPSTAGE_API_KEY:
    upstage_clients = [OpenAI(api_key=UPSTAGE_API_KEY, base_url=UPSTAGE_BASE_URL) for _ in range(PARALLEL_CLIENTS)]
if OPENAI_API_KEY:
    openai_clients = [OpenAI(api_key=OPENAI_API_KEY) for _ in range(PARALLEL_CLIENTS)]

# JSON schema for structured translation output
TRANSLATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "org": {"type": "string", "description": "출발지"},
                "dest": {"type": "string", "description": "도착지"},
                "days": {"type": "string", "description": "총 일수"},
                "visiting_city_number": {"type": "string", "description": "방문 도시 수"},
                "date": {"type": "string", "description": "날짜 리스트/범위"},
                "people_number": {"type": "string", "description": "인원수"},
                "local_constraint": {"type": "string", "description": "현지 제약"},
                "budget": {"type": "string", "description": "예산"},
                "query": {"type": "string", "description": "요청/쿼리"},
                "level": {"type": "string", "description": "난이도"},
                "persona": {"type": "string", "description": "페르소나"},
                "annotation_plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "string", "description": "일차 번호"},
                            "current_city": {"type": "string", "description": "현재 도시/구간"},
                            "transportation": {
                                "type": "string",
                                "description": "이동 수단/노선/출도착 시각 포함(시간 그대로 보존, 정보 손실/요약 금지)",
                            },
                            "breakfast": {"type": "string", "description": "아침"},
                            "attraction": {"type": "string", "description": "관광지 목록"},
                            "lunch": {"type": "string", "description": "점심"},
                            "dinner": {"type": "string", "description": "저녁"},
                            "accommodation": {"type": "string", "description": "숙소"},
                            "event": {"type": "string", "description": "이벤트/공연"},
                            "point_of_interest_list": {
                                "type": "string",
                                "description": "POI 이름, 시간(원문 그대로), 교통/거리 정보 포함. 세미콜론 등 구분 기호/내용을 절대 삭제하거나 '-'로 대체하지 말 것.",
                            },
                        },
                        "required": [
                            "days",
                            "current_city",
                            "transportation",
                            "breakfast",
                            "attraction",
                            "lunch",
                            "dinner",
                            "accommodation",
                            "event",
                            "point_of_interest_list",
                        ],
                    },
                    "description": "일정 리스트(1일차~n일차) 배열; 각 아이템은 day별 필드 포함",
                },
                "reference_information": {
                    "type": "string",
                    "description": "참고 정보 전체를 한국어로 번역해 작성 (원문 정보 누락 금지).",
                },
                "preserved_times": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "원문에서 추출한 HH:MM 형식 시간 목록",
                },
            },
            "required": [
                "org",
                "dest",
                "days",
                "visiting_city_number",
                "date",
                "people_number",
                "local_constraint",
                "budget",
                "query",
                "level",
                "persona",
                "annotation_plan",
                "reference_information",
            ],
        },
    },
}


def load_translation_cache() -> Dict[str, str]:
    if TRANSLATION_CACHE_PATH.exists():
        try:
            return json.loads(TRANSLATION_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_translation_cache(cache: Dict[str, str]) -> None:
    TRANSLATION_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def load_review_cache() -> Dict[str, str]:
    if LLM_REVIEW_CACHE_PATH.exists():
        try:
            return json.loads(LLM_REVIEW_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_review_cache(cache: Dict[str, str]) -> None:
    LLM_REVIEW_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def save_formatted_cache(cache: Dict[str, str]) -> None:
    FORMATTED_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def save_formatted_meta(meta: Dict[str, str]) -> None:
    FORMATTED_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def save_formatted_poi(poi_tables: Dict[str, str]) -> None:
    FORMATTED_POI_PATH.write_text(json.dumps(poi_tables, ensure_ascii=False, indent=2))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def poi_md_table_from_str(poi_str: str) -> str:
    entries = [seg.strip() for seg in (poi_str or "").split(";") if seg.strip()]
    if not entries:
        return ""
    lines = ["| POI |", "|---|"]
    for seg in entries:
        lines.append(f"| {seg} |")
    return "\n".join(lines)

def contains_korean(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


def needs_translation(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned or contains_korean(cleaned):
        return False
    return len(cleaned) >= 40


def translate_to_korean(
    text: str,
    cache: Dict[str, str],
    *,
    save_on_update: bool = True,
    seen_text_cache: Dict[str, str] | None = None,
) -> str:
    """Translate text via Upstage model with caching; no-op for short/Ko text."""
    if not needs_translation(text):
        return text
    if text in cache:
        return cache[text]
    if seen_text_cache is not None and text in seen_text_cache:
        return seen_text_cache[text]
    if not upstage_client:
        return "UPSTAGE_API_KEY가 설정되어 있지 않습니다."

    def extract_times(s: str) -> list[str]:
        # HH:MM or H:MM patterns
        return re.findall(r"\b\d{1,2}:\d{2}\b", s)

    times = extract_times(text)
    time_hint = ""
    if times:
        time_hint = (
            "Times (HH:MM) must be preserved exactly as in the source. Do not drop or reformat them. "
            f"Times in the text: {', '.join(times)}. "
        )

    prompt = (
        "Translate the following travel plan content into Korean in a natural tone. "
        "Do NOT omit any detail: keep all numbers, times, currencies, place names, and list items exactly. "
        "정보의 손실이 없도록 모든 항목을 그대로 번역하세요. 세미콜론/구분자/POI/시간/거리/교통 정보 삭제·요약·재구성 금지. "
        "Preserve structure so it is easy to read, but do not invent or remove fields. "
        "Return JSON per the provided schema with each CSV column translated (org, dest, days, visiting_city_number, "
        "date, people_number, local_constraint, budget, query, level, persona, annotation_plan, reference_information). "
        "Critical rules: "
        "- days, visiting_city_number, people_number, budget: 원본 문자열 그대로, 단위/라벨 추가 금지. "
        "- annotation_plan.days: 원본 day 값만(예: 1, 2, 3), 경로/문구 붙이지 말 것. 이동 구간은 current_city에만 기재. "
        "- annotation_plan.point_of_interest_list: 세미콜론으로 연결된 원문 POI/시간/거리/교통 정보를 삭제하거나 요약하지 말고 그대로 보존(필요 시 한글 병기 가능). "
        "- reference_information: 전체를 한국어로 번역하며, 원문 정보 누락 없이 모두 포함. "
        "- 모든 고유명사(도시/레스토랑 등)는 한국어로 번역 후 괄호 안에 영어 원문을 함께 표기(예: 로스앤젤레스(Los Angeles)). "
        f"{time_hint}\n\n"
        f"{text}"
    )
    try:
        completion = upstage_client.chat.completions.create(
            model=UPSTAGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional Korean translator for travel plans. "
                        "정보의 손실 없이 숫자/시간/장소/목록/구분자(세미콜론 포함)를 그대로 살려 번역하세요. "
                        "단위/라벨을 임의 추가하지 말고, 원본 값과 구조를 그대로 유지하세요. "
                        "모든 고유명사는 한국어로 번역한 뒤 괄호 안에 영어 원문을 표기하세요 (예: 로스앤젤레스(Los Angeles))."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2000,
            response_format=TRANSLATION_RESPONSE_FORMAT,
        )
        content = completion.choices[0].message.content
        translation = content
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                translation = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
        translation = str(translation).strip()
        if translation:
            if save_on_update:
                cache[text] = translation  # store under raw text only (legacy format)
                save_translation_cache(cache)
            if seen_text_cache is not None:
                seen_text_cache[text] = translation
            return translation
        return "번역 오류: 응답이 비어 있습니다."
    except Exception as exc:
        return f"번역 오류: {exc}"


def llm_review_key(file_name: str, row_index: int) -> str:
    return f"{file_name}:{row_index}"


def run_llm_review(row: Dict[str, str]) -> str:
    """Run GPT-4.1 QA review for a trip row."""
    if not openai_client:
        return "OPENAI_API_KEY가 설정되어 있지 않습니다."

    plan_text = build_plan_text(row)
    prompt = (
        "You are a critical travel QA reviewer. Identify risks or incomplete areas for a human traveler. "
        "Call out unrealistic timing, missing logistics (tickets, transfers), safety/seasonal issues, "
        "budget mismatches, dietary constraints, or weak attractions. Always surface minor cautions even if the plan "
        "looks solid. After the general cautions, add 3 persona-based bullets (e.g., older couple needing slower pace, "
        "family with kids, budget solo traveler) noting why each might dislike or struggle with the plan. "
        "답변은 한국어로 말 하세요.\n\n"
        f"Trip data:\n{plan_text[:4500]}"
    )
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Travel QA assistant that points out weaknesses candidly. 답변은 한국어로 말하세요. 불릿 포맷을 유지하세요.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2000,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else "LLM 검토 오류: 응답이 비어 있습니다."
    except Exception as exc:
        return f"LLM 검토 오류: {exc}"


def run_llm_review_with_client(row: Dict[str, str], client: OpenAI) -> str:
    """Run GPT-4.1 review with specific client (for threading)."""
    if not client:
        return "OPENAI_API_KEY가 설정되어 있지 않습니다."
    plan_text = build_plan_text(row)
    prompt = (
        "You are a critical travel QA reviewer. Identify risks or incomplete areas for a human traveler. "
        "Call out unrealistic timing, missing logistics (tickets, transfers), safety/seasonal issues, "
        "budget mismatches, dietary constraints, or weak attractions. Always surface minor cautions even if the plan "
        "looks solid. After the general cautions, add 3 persona-based bullets (e.g., older couple needing slower pace, "
        "family with kids, budget solo traveler) noting why each might dislike or struggle with the plan. 답변은 한국어로 말 하세요.\n\n"
        f"Trip data:\n{plan_text[:4500]}"
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Travel QA assistant that points out weaknesses candidly. 답변은 한국어로 말하세요. 불릿 포맷을 유지하세요.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2000,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else "LLM 검토 오류: 응답이 비어 있습니다."
    except Exception as exc:
        return f"LLM 검토 오류: {exc}"


def precompute_translations(paths: Iterable[Path], limit: int | None = None) -> int:
    """Bulk-translate long fields for given CSV paths; returns count of new translations."""
    cache = load_translation_cache()
    cache_lock = Lock()
    stop_event = Event()
    tasks: List[str] = []
    for path in paths:
        rows = load_rows(path)
        for row in rows:
            text = build_plan_text(row)
            if needs_translation(text):
                tasks.append(text)

    if limit:
        tasks = tasks[:limit]

    if not tasks or not upstage_clients:
        return 0

    start_len = len(cache)
    error_log: List[str] = []
    skipped = 0
    added_count = 0
    seen_text_cache: Dict[str, str] = {}

    def worker(text: str, client: OpenAI) -> bool:
        if stop_event.is_set():
            return False
        nonlocal skipped, added_count
        with cache_lock:
            if text in cache:
                skipped += 1
                return False
        # temporary swap client
        global upstage_client
        prev_client = upstage_client
        upstage_client = client
        translation = translate_to_korean(text, cache, save_on_update=False, seen_text_cache=seen_text_cache)
        upstage_client = prev_client
        if translation and not str(translation).startswith("번역 오류"):
            with cache_lock:
                if text not in cache:
                    cache[text] = translation
                    save_translation_cache(cache)  # flush every addition
                    added_count += 1
                    return True
        else:
            with cache_lock:
                error_log.append(f"trans_error: {translation}")
            stop_event.set()
            raise RuntimeError(f"Translation failed: {translation}")

    with ThreadPoolExecutor(max_workers=PARALLEL_CLIENTS) as executor:
        futures = []
        for idx, text in enumerate(tasks):
            client = upstage_clients[idx % len(upstage_clients)]
            futures.append(executor.submit(worker, text, client))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Solar translations", ncols=80):
            try:
                future.result()
            except Exception:
                stop_event.set()
                for f in futures:
                    f.cancel()
                with cache_lock:
                    save_translation_cache(cache)
                raise
    added = added_count
    if added:
        save_translation_cache(cache)
    elif cache and not TRANSLATION_CACHE_PATH.exists():
        save_translation_cache(cache)

    end_len = len(cache)
    if error_log:
        print(f"[WARN] Translation errors: {len(error_log)} (showing first 3)")
        for msg in error_log[:3]:
            print(" -", msg[:400])
    print(f"[Stats] Translations: added={added} (cache delta {end_len - start_len}), skipped={skipped}, cache_size={end_len}")
    return added


def translate_csv_core(paths: Iterable[Path], limit: int | None = None) -> None:
    """Translate only core CSV columns and write translated CSVs to cache/translated_csv."""
    if not upstage_client:
        print("UPSTAGE_API_KEY가 없어 CSV 번역을 건너뜁니다.")
        return
    ensure_dir(TRANSLATED_CSV_DIR)
    if not upstage_clients:
        # fallback: make a pool of clients
        upstage_clients.extend([OpenAI(api_key=UPSTAGE_API_KEY, base_url=UPSTAGE_BASE_URL) for _ in range(PARALLEL_CLIENTS)])

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "csv_translation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "org": {"type": "string"},
                    "dest": {"type": "string"},
                    "days": {"type": "string"},
                    "visiting_city_number": {"type": "string"},
                    "date": {"type": "string"},
                    "people_number": {"type": "string"},
                    "local_constraint": {"type": "string"},
                    "budget": {"type": "string"},
                    "query": {"type": "string"},
                    "level": {"type": "string"},
                    "persona": {"type": "string"},
                    "annotation_plan": {"type": "array", "items": {"type": "object"}},
                },
                "required": [
                    "org",
                    "dest",
                    "days",
                    "visiting_city_number",
                    "date",
                    "people_number",
                    "local_constraint",
                    "budget",
                    "query",
                    "level",
                    "persona",
                    "annotation_plan",
                ],
            },
        },
    }

    def worker(row: Dict[str, str], client: OpenAI) -> Dict[str, str] | None:
        fieldnames = [
            "org",
            "dest",
            "days",
            "visiting_city_number",
            "date",
            "people_number",
            "local_constraint",
            "budget",
            "query",
            "level",
            "persona",
            "annotation_plan",
        ]
        src = {k: row.get(k, "") for k in fieldnames}
        prompt = (
            "Translate the following travel plan fields into Korean, preserving ALL details (times, POI text, numbers). "
            "Return JSON with the same field names. annotation_plan must remain a list (keep POI times/distances intact). "
            "고유명사는 한글 번역 후 괄호에 영어를 병기하세요.\n\n"
            f"{json.dumps(src, ensure_ascii=False, indent=2)}"
        )
        last_exc = None
        content = ""
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=UPSTAGE_MODEL,
                    messages=[
                        {"role": "system", "content": "Professional Korean translator. 정보 손실 없이 그대로 번역하세요."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=schema,
                    temperature=0,
                )
                content = completion.choices[0].message.content
                try:
                    translated = json.loads(content)
                except Exception as e:
                    raise RuntimeError(f"JSON decode error: {e} | content: {content}")
                translated["annotation_plan"] = json.dumps(translated.get("annotation_plan", ""), ensure_ascii=False)
                return translated
            except Exception as e:
                last_exc = e
                if attempt == 2:
                    raise RuntimeError(f"{e} | content: {content}")
                # retry
                continue
        if last_exc:
            raise last_exc
        return None

    for path in paths:
        error_log_path = TRANSLATED_CSV_DIR / f"errors_{path.name}.log"
        error_log_path.write_text("")  # reset
        rows = load_rows(path)
        if limit:
            rows = rows[:limit]
        out_path = TRANSLATED_CSV_DIR / f"translated_{path.name}"
        fieldnames = [
            "org",
            "dest",
            "days",
            "visiting_city_number",
            "date",
            "people_number",
            "local_constraint",
            "budget",
            "query",
            "level",
            "persona",
            "annotation_plan",
        ]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            lock = Lock()
            with ThreadPoolExecutor(max_workers=PARALLEL_CLIENTS) as executor:
                futures = []
                for idx, row in enumerate(rows):
                    client = upstage_clients[idx % len(upstage_clients)]
                    futures.append(executor.submit(worker, row, client))
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"translate_csv {path.name}", ncols=80):
                    try:
                        result = future.result()
                        if result:
                            with lock:
                                writer.writerow(result)
                                f.flush()
                    except Exception as e:
                        msg = f"[WARN] translation failed for {path.name}: {e}\n"
                        print(msg.strip())
                        with lock:
                            with error_log_path.open("a") as ef:
                                ef.write(msg)
        print(f"Translated CSV saved: {out_path}")
        if error_log_path.exists() and error_log_path.stat().st_size > 0:
            print(f"Error log: {error_log_path}")


def precompute_llm_reviews(paths: Iterable[Path], limit: int | None = None) -> int:
    """Run GPT-4.1 reviews ahead of time; returns count of new reviews cached."""
    cache = load_review_cache()
    cache_lock = Lock()
    stop_event = Event()
    if not openai_clients:
        return 0

    tasks: List[Tuple[str, Dict[str, str]]] = []
    for path in paths:
        rows = load_rows(path)
        for row in rows:
            key = llm_review_key(path.name, row.get("__index__", 0))
            if key in cache:
                continue
            tasks.append((key, row))

    if limit:
        tasks = tasks[:limit]

    if not tasks:
        return 0

    added = 0
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=PARALLEL_CLIENTS) as executor:
        future_map = {}
        for idx, (key, row) in enumerate(tasks):
            client = openai_clients[idx % len(openai_clients)]
            future_map[executor.submit(run_llm_review_with_client, row, client)] = key
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="GPT-4.1 reviews", ncols=80):
            try:
                review = future.result()
                key = future_map[future]
                with cache_lock:
                    if key not in cache:
                        cache[key] = review
                        added += 1
                        save_review_cache(cache)  # flush every addition
                    if review.startswith("LLM 검토 오류"):
                        errors.append(f"{key}: {review}")
                        stop_event.set()
                        raise RuntimeError(review)
            except Exception:
                stop_event.set()
                for f in future_map:
                    f.cancel()
                with cache_lock:
                    save_review_cache(cache)
                raise
    if added:
        save_review_cache(cache)
    elif cache and not LLM_REVIEW_CACHE_PATH.exists():
        save_review_cache(cache)
    if errors:
        print(f"[WARN] LLM review errors: {len(errors)} (showing first 3)")
        for msg in errors[:3]:
            print(" -", msg[:200])
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Korean translations for TripCraft CSVs.")
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific CSV filenames to process (default: all in data dir)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "translations", "reviews", "format", "poi", "translate_csv", "format_translated"],
        default="all",
        help="Select what to run: translations only, reviews only, both, format preview, POI md tables, or translate CSV columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tasks (useful for quick dry runs).",
    )
    args = parser.parse_args()

    if args.files:
        targets = [DATA_DIR / name for name in args.files]
    else:
        targets = list_csv_files()

    missing = [p for p in targets if not p.exists()]
    if missing:
        print("Missing files:", ", ".join(str(p) for p in missing))
        return

    if args.mode == "format_translated":
        ensure_dir(TRANSLATED_CSV_DIR)
        formatted: Dict[str, str] = {}
        for path in TRANSLATED_CSV_DIR.glob("translated_*.csv"):
            with path.open() as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    ap_raw = row.get("annotation_plan", "")
                    try:
                        ap = json.loads(ap_raw)
                    except Exception:
                        ap = []
                    key = f"{path.name}:{idx}"
                    parts = []
                    parts.append(f"## 여행 {idx + 1}: {row.get('org','')} → {row.get('dest','')}")
                    parts.append(
                        f"- 날짜: {row.get('date','')}  \n- 인원: {row.get('people_number','')}명  \n- 예산: {row.get('budget','')}"
                    )
                    for day in ap or []:
                        parts.append(f"\n### Day {day.get('days')}")
                        parts.append(f"- 구간: {day.get('current_city')}")
                        parts.append(f"- 교통: {day.get('transportation')}")
                        parts.append(f"- 관광: {day.get('attraction')}")
                        parts.append(
                            f"- 식사: 아침 {day.get('breakfast')} / 점심 {day.get('lunch')} / 저녁 {day.get('dinner')}"
                        )
                        parts.append(f"- 숙박: {day.get('accommodation')}")
                        poi_md = poi_md_table_from_str(day.get("point_of_interest_list", ""))
                        if poi_md:
                            parts.append("\nPOI\n" + poi_md)
                    formatted[key] = "\n".join(parts)
        if formatted:
            FORMATTED_TRANS_PATH.write_text(json.dumps(formatted, ensure_ascii=False, indent=2))
            print(f"Formatted translated plans saved: {FORMATTED_TRANS_PATH}")
        else:
            print("No translated CSVs found under cache/translated_csv")
        return

    if args.mode == "translate_csv":
        translate_csv_core(targets, limit=args.limit)
        return

    # Simple rule-based formatting preview
    if args.mode == "format":
        def poi_md_table(poi_str: str) -> str:
            entries = [seg.strip() for seg in (poi_str or "").split(";") if seg.strip()]
            if not entries:
                return ""
            lines = ["| POI |", "|---|"]
            for seg in entries:
                lines.append(f"| {seg} |")
            return "\n".join(lines)

        formatted: Dict[str, str] = {}
        formatted_meta: Dict[str, str] = {}
        for path in targets:
            rows = load_rows(path)
            for row in rows:
                ap = row.get("annotation_plan_parsed")
                if not ap:
                    continue
                key = f"{path.name}:{row['__index__']}"
                parts = []
                parts.append(f"## 여행 {row['__index__'] + 1}: {row.get('org')} → {row.get('dest')}")
                parts.append(f"- 날짜: {row.get('date')}  \n- 인원: {row.get('people_number')}명  \n- 예산: {row.get('budget')}")
                if row.get("reference_information"):
                    formatted_meta[key] = row.get("reference_information")
                for day in ap:
                    parts.append(f"\n### Day {day.get('days')}")
                    parts.append(f"- 구간: {day.get('current_city')}")
                    parts.append(f"- 교통: {day.get('transportation')}")
                    parts.append(f"- 관광: {day.get('attraction')}")
                    parts.append(f"- 식사: 아침 {day.get('breakfast')} / 점심 {day.get('lunch')} / 저녁 {day.get('dinner')}")
                    parts.append(f"- 숙박: {day.get('accommodation')}")
                    poi_md = poi_md_table(day.get("point_of_interest_list"))
                    if poi_md:
                        parts.append("\nPOI\n" + poi_md)
                formatted[key] = "\n".join(parts)
        if formatted:
            save_formatted_cache(formatted)
            print(f"Formatted plans saved: {FORMATTED_CACHE_PATH}")
        if formatted_meta:
            save_formatted_meta(formatted_meta)
            print(f"Formatted meta saved: {FORMATTED_META_PATH}")
        return

    added = added_reviews = 0
    if args.mode in ("all", "translations"):
        added = precompute_translations(targets, limit=args.limit)
        print(f"Translations added: {added}")
    else:
        print("Translations skipped (mode=reviews)")

    if args.mode in ("all", "reviews"):
        added_reviews = precompute_llm_reviews(targets, limit=args.limit)
        print(f"LLM reviews added: {added_reviews}")
    else:
        print("LLM reviews skipped (mode=translations)")

    print(f"Translation cache: {TRANSLATION_CACHE_PATH}")
    print(f"LLM review cache: {LLM_REVIEW_CACHE_PATH}")


if __name__ == "__main__":
    main()
