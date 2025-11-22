import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

# Allow imports when running via `streamlit run KY/viewer/app.py`
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

# Cache paths for formatted Markdown
CACHE_DIR = APP_DIR / "cache"
FORMATTED_CACHE_PATH = CACHE_DIR / "formatted_plans.json"
FORMATTED_TRANS_PATH = CACHE_DIR / "formatted_plans_translated.json"
FORMATTED_META_PATH = CACHE_DIR / "formatted_meta.json"

try:
    from data_loader import (
        DATA_DIR,
        build_plan_text,
        itinerary_table,
        list_csv_files,
        load_rows,
    )
    from preprocess import (
        load_translation_cache,
        translate_to_korean,
        load_review_cache,
    )
except Exception:  # pragma: no cover - fallback for package imports
    from .data_loader import (
        DATA_DIR,
        build_plan_text,
        itinerary_table,
        list_csv_files,
        load_rows,
    )
    from .preprocess import (
        load_translation_cache,
        translate_to_korean,
        load_review_cache,
    )


LABEL_DIR = APP_DIR / "labels"

LABEL_DIR.mkdir(parents=True, exist_ok=True)


# Env
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_json_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


# Streamlit caches to speed up "다음" 전환
@st.cache_data(show_spinner=False)
def cached_rows(path: Path) -> List[Dict[str, Any]]:
    return load_rows(path)


@st.cache_data(show_spinner=False)
def cached_json(path: Path) -> Dict[str, Any]:
    return load_json_cache(path)


@st.cache_data(show_spinner=False)
def cached_translation_cache() -> Dict[str, str]:
    return load_translation_cache()


@st.cache_data(show_spinner=False)
def cached_review_cache() -> Dict[str, str]:
    return load_review_cache()
def load_labels(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_labels(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def run_llm_review(row: Dict[str, Any]) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY가 설정되어 있지 않습니다."

    plan_text = build_plan_text(row)
    prompt = (
        "You are a critical travel QA reviewer. Identify risks or incomplete areas for a human traveler. "
        "Call out unrealistic timing, missing logistics (tickets, transfers), safety/seasonal issues, "
        "budget mismatches, dietary constraints, or weak attractions. Always surface minor cautions even if the plan "
        "is mostly solid. After the general cautions, add 3 persona-based bullets (e.g., older couple needing slower "
        "pace, family with kids, budget solo traveler) on why each might dislike or struggle with the plan. "
        "답변은 한국어로 말 하세요.\n\n"
        f"Trip data:\n{plan_text[:4500]}"
    )
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "system",
                "content": "Travel QA assistant that points out weaknesses candidly. 답변은 한국어로 말하세요. 불릿 포맷을 유지하세요.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.5,
        "max_tokens": 2000,
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=40,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"LLM 검토 오류: {exc}"


def label_key(file_name: str, row_index: int) -> str:
    return f"{file_name}:{row_index}"


def main() -> None:
    st.set_page_config(page_title="TripCraft Labeling Viewer", layout="wide")
    st.title("TripCraft 여행 계획 라벨링 뷰어")

    csv_files = list_csv_files()
    if not csv_files:
        st.error(f"CSV를 찾을 수 없습니다: {DATA_DIR}")
        return

    file_names = [p.name for p in csv_files]
    default_idx = 0
    if "active_file" in st.session_state and st.session_state["active_file"] in file_names:
        default_idx = file_names.index(st.session_state["active_file"])

    selected_name = st.sidebar.selectbox("CSV 파일 선택", file_names, index=default_idx)
    selected_path = DATA_DIR / selected_name
    trips = cached_rows(selected_path)
    st.sidebar.write(f"총 {len(trips)}개 여행 계획")

    if st.session_state.get("active_file") != selected_name:
        st.session_state["row_index"] = 0
        st.session_state["active_file"] = selected_name
        st.session_state["llm_review"] = {}

    if "row_index" not in st.session_state:
        st.session_state["row_index"] = 0
    row_index = st.session_state["row_index"]
    row_index = max(0, min(row_index, max(len(trips) - 1, 0)))
    st.session_state["row_index"] = row_index

    row = trips[row_index]
    plan_text = build_plan_text(row)
    review_cache = cached_review_cache()

    formatted_en_cache = cached_json(FORMATTED_CACHE_PATH)
    formatted_meta_cache = cached_json(FORMATTED_META_PATH)
    formatted_ko_cache = cached_json(FORMATTED_TRANS_PATH)
    cache_key = label_key(selected_name, row_index)
    # 영문: 원본 포맷 캐시
    formatted_en_text = formatted_en_cache.get(cache_key, plan_text)
    # 한글: 번역 포맷 캐시 (translated_ 접두어 키 우선)
    cache_key_ko = f"translated_{cache_key}"
    formatted_ko_text = formatted_ko_cache.get(cache_key_ko) or formatted_ko_cache.get(cache_key, plan_text)
    meta_text = formatted_meta_cache.get(cache_key, "")

    labels_path = LABEL_DIR / f"{selected_name}_labels.json"
    label_store = load_labels(labels_path)
    row_label_key = label_key(selected_name, row_index)
    saved_label = label_store.get(row_label_key, {})

    left, right = st.columns([6, 4])

    with left:
        st.markdown(
            f"**여행 {row_index + 1} / {len(trips)}** · 파일: `{selected_name}`<br>"
            f"- {row.get('org', '')} → {row.get('dest', '')} · 날짜: {row.get('date', '')} · 인원: {row.get('people_number', '')}명 · 예산: {row.get('budget', '')}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"- 도시수: {row.get('visiting_city_number', '')} · 난이도: {row.get('level', '')} · 페르소나: {row.get('persona', '')}"
        )

        tab_kr, tab_en = st.tabs(["한글 번역", "English"])
        with tab_kr:
            st.markdown(formatted_ko_text)
        with tab_en:
            st.markdown(formatted_en_text)
        if meta_text:
            with st.expander("참고 정보 (meta)", expanded=False):
                st.markdown(f"```\n{meta_text}\n```")

    with right:
        st.subheader("휴먼 라벨링")
        default_incomplete = bool(saved_label.get("incomplete", False))
        default_desc = saved_label.get("description", "")

        incomplete = st.checkbox(
            "불완전 체크", value=default_incomplete, key=f"incomplete_{row_label_key}"
        )
        description = st.text_area(
            "불완전 설명 (정성 검토)",
            value=default_desc,
            height=220,
            key=f"description_{row_label_key}",
        )
        if st.button("라벨 저장", key=f"save_{row_label_key}"):
            existing_llm = label_store.get(row_label_key, {}).get("llm_review") or review_cache.get(row_label_key)
            label_store[row_label_key] = {
                "file": selected_name,
                "row_index": row_index,
                "org": row.get("org", ""),
                "dest": row.get("dest", ""),
                "incomplete": incomplete,
                "description": description,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "llm_review": existing_llm,
            }
            save_labels(labels_path, label_store)
            st.success("라벨이 저장되었습니다.")

        if saved_label.get("updated_at"):
            st.caption(f"마지막 저장: {saved_label.get('updated_at')}")

        st.divider()
        st.subheader("LLM 검토 (자동 생성, 한국어)")
        review_store: Dict[str, str] = st.session_state.get("llm_review", {})
        existing_review = (
            review_store.get(row_label_key)
            or saved_label.get("llm_review")
            or review_cache.get(row_label_key)
        )
        review_style = (
            "padding:10px;border:1px solid #ddd;border-radius:6px;"
            "white-space:pre-wrap;font-size:14px;line-height:1.5;"
            "background:transparent;color:inherit;"
        )
        if existing_review:
            st.markdown(f"<div style='{review_style}'>{existing_review}</div>", unsafe_allow_html=True)
        else:
            review_text = run_llm_review(row)
            review_store[row_label_key] = review_text
            st.session_state["llm_review"] = review_store
            label_store[row_label_key] = {
                "file": selected_name,
                "row_index": row_index,
                "org": row.get("org", ""),
                "dest": row.get("dest", ""),
                "incomplete": incomplete,
                "description": description,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "llm_review": review_text,
            }
            save_labels(labels_path, label_store)
            st.markdown(f"<div style='{review_style}'>{review_text}</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("좌측 메인: EN 원본 / KR 번역")
        st.markdown("우측: 휴먼 라벨 + GPT-4.1 검토")
        st.markdown("라벨은 `KY/viewer/labels` 폴더에 저장됩니다.")
        st.markdown("---")
        st.markdown("**이동**")
        if st.button("◀ 이전", disabled=row_index == 0, use_container_width=True):
            st.session_state["row_index"] = max(0, row_index - 1)
            st.rerun()
        if st.button("다음 ▶", disabled=row_index >= len(trips) - 1, use_container_width=True):
            st.session_state["row_index"] = min(len(trips) - 1, row_index + 1)
            st.rerun()


if __name__ == "__main__":
    main()
