from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATASET_NAME = "osunlp/TravelPlanner"


def _try_load_dataset(split: str, hf_cache_dir: Path) -> Iterable[Dict[str, Any]]:
    """Robust HF loader (handles datasets that expose per-split configs)."""
    from datasets import get_dataset_config_names, load_dataset  # type: ignore

    # 1) Some datasets support: load_dataset(path, split=...)
    try:
        return load_dataset(DATASET_NAME, split=split, cache_dir=str(hf_cache_dir))
    except Exception:
        pass

    # 2) Some datasets expose config names like ["train","validation","test"].
    cfgs = []
    try:
        cfgs = list(get_dataset_config_names(DATASET_NAME))
    except Exception:
        cfgs = []

    if split in cfgs:
        return load_dataset(DATASET_NAME, split, split=split, cache_dir=str(hf_cache_dir))

    # 3) Fallback: first config
    if cfgs:
        return load_dataset(DATASET_NAME, cfgs[0], split=split, cache_dir=str(hf_cache_dir))

    # 4) Final fallback
    return load_dataset(DATASET_NAME, split=split, cache_dir=str(hf_cache_dir))


def load_travelplanner_split(
    split: str,
    *,
    cache_min_path: Path,
    cache_db_summary_path: Optional[Path],
    hf_cache_dir: Path,
    keep_reference_information: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load TravelPlanner split with caching.

    Returns:
      rows_min: list of minimal dict rows (json-serializable)
      db_summaries: optional dict {source_id -> summary} if cache_db_summary_path provided
    """
    if cache_min_path.exists() and (cache_db_summary_path is None or cache_db_summary_path.exists()):
        rows = [json.loads(l) for l in cache_min_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        db_summaries = None
        if cache_db_summary_path is not None:
            db_summaries = json.loads(cache_db_summary_path.read_text(encoding="utf-8"))
        return rows, db_summaries

    cache_min_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_db_summary_path is not None:
        cache_db_summary_path.parent.mkdir(parents=True, exist_ok=True)

    ds = _try_load_dataset(split, hf_cache_dir)

    keep = [
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
        "id",
    ]
    if keep_reference_information:
        keep.append("reference_information")

    rows_min: List[Dict[str, Any]] = []
    db_summaries: Dict[str, Any] = {}

    # Optional db summary
    if cache_db_summary_path is not None:
        from travelplanner_loader import stable_source_id
        from db_summary import extract_db_summary

    with cache_min_path.open("w", encoding="utf-8") as f:
        for r in ds:
            obj = {k: r.get(k) for k in keep}
            rows_min.append(obj)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if cache_db_summary_path is not None:
                sid = stable_source_id(obj)
                ref = r.get("reference_information")
                db_summaries[sid] = extract_db_summary(ref)

    if cache_db_summary_path is not None:
        cache_db_summary_path.write_text(json.dumps(db_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        return rows_min, db_summaries
    return rows_min, None
