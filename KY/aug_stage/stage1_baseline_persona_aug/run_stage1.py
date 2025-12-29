#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from hf_dataset_cache import load_travelplanner_split
from persona_reference_retrieval import (
    load_stravl_library,
    persona_reference_block,
    select_persona_references,
    trip_features_from_initial_info,
)
from structured_output import Stage1Output
from tp_quantiles import load_tp_pppn_quantiles
from travelplanner_loader import grounding_anchors, normalize_initial_info, stable_source_id


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _ensure_stravl_csv(data_dir: Path) -> Optional[Path]:
    """Return csv_path if exists; else try to download via persona_seed_bank URL (optional).

    We keep this minimal; if you already have the CSV, set STRAVL_CSV_PATH.
    """
    override = os.getenv("STRAVL_CSV_PATH")
    if override:
        p = Path(override).expanduser().resolve()
        return p if p.exists() else None

    # default location
    p = (data_dir / "stravl" / "Stravl_Travel_Preference_Data.csv").resolve()
    if p.exists():
        return p

    # Optional: auto-download from GitHub raw (same as prior pipeline).
    try:
        from urllib.request import urlopen
        import shutil

        url = "https://raw.githubusercontent.com/Stravl/Stravl-Data/main/Stravl_Travel_Preference_Data.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with urlopen(url, timeout=60) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(p)
        return p
    except Exception:
        return None


def _generate_stage1(
    client: OpenAI,
    *,
    sys_prompt: str,
    user_tmpl: str,
    initial_info: Dict[str, Any],
    source_id: str,
    persona_refs_json: str,
    temperature: float,
    max_retries: int,
) -> Dict[str, Any]:
    # IMPORTANT: do not use `.format()` because the prompt contains many `{}`.
    user_prompt = user_tmpl
    user_prompt = user_prompt.replace("{initial_info_json}", json.dumps(initial_info, ensure_ascii=False, indent=2))
    user_prompt = user_prompt.replace("{persona_references_json}", persona_refs_json)

    need_n = int(initial_info.get("people_number") or 1)
    last_err = ""

    for _ in range(max_retries + 1):
        user_prompt_retry = user_prompt if not last_err else (user_prompt + "\n\n[VALIDATION ERROR]\n" + last_err)

        resp = client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt_retry},
            ],
            temperature=temperature,
            text_format=Stage1Output,
        )

        out: Stage1Output = resp.output_parsed
        obj = out.model_dump(by_alias=True)

        # Force stable id + exact initial_info copy.
        obj["source_id"] = source_id
        obj["initial_info"] = initial_info

        gps = obj.get("group_personas") or []
        if len(gps) != need_n:
            last_err = f"Expected exactly {need_n} personas, got {len(gps)}."
            continue

        # Overwrite grounding_anchors with *only* augmented initial_info-based anchors.
        anchors = grounding_anchors(initial_info)
        for p in gps:
            p["grounding_anchors"] = anchors

        return obj

    raise RuntimeError(f"Stage1 generation failed. Last error: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="Only `test` is recommended for this refactor.")
    ap.add_argument("--max_records", type=int, default=0, help="0 = all")
    ap.add_argument("--people_choices", default="2,4", help="used when people_number < 2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_retries", type=int, default=2)

    ap.add_argument("--data_dir", default="data", help="where to cache downloaded inputs")
    ap.add_argument("--out_dir", default="outputs_stage1", help="where to write stage1 outputs")
    ap.add_argument("--no_resume", action="store_true", help="do not skip existing outputs")

    # Persona reference retrieval (RAG-lite, rule + diversity)
    ap.add_argument("--no_persona_refs", action="store_true", help="disable persona reference injection")
    ap.add_argument("--ref_multiplier", type=int, default=10, help="K = people_number * ref_multiplier")
    ap.add_argument("--ref_preselect", type=int, default=500, help="pool size for MMR reranking")
    ap.add_argument("--ref_lambda", type=float, default=0.75, help="MMR lambda (relevance vs diversity)")
    ap.add_argument("--stravl_max_rows", type=int, default=0, help="0 = all rows, else truncate for speed/debug")

    # Meta logging
    ap.add_argument("--store_meta", action="store_true", help="store meta/debug info in output json")
    args = ap.parse_args()

    # Refactor target: TravelPlanner test split only
    if args.split != "test":
        raise SystemExit("This refactor is intended for TravelPlanner `test` only. Use --split test.")

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Put it in .env or env vars.")

    here = Path(__file__).resolve().parent
    data_dir = (here / args.data_dir).resolve()
    out_dir = (here / args.out_dir / args.split).resolve()
    data_out_dir = out_dir / "data"
    meta_out_dir = out_dir / "meta"
    data_out_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)

    hf_cache_dir = data_dir / "hf_cache"
    split_cache = data_dir / "travelplanner_min" / f"{args.split}.jsonl"
    dbsum_cache = data_dir / "travelplanner_db_summary" / f"{args.split}.json"

    # Load TravelPlanner with caching (+ optional DB summary cache)
    rows, db_summaries = load_travelplanner_split(
        args.split,
        cache_min_path=split_cache,
        cache_db_summary_path=(dbsum_cache if args.store_meta else None),
        hf_cache_dir=hf_cache_dir,
        keep_reference_information=False,
    )

    if args.max_records and args.max_records > 0:
        rows = rows[: args.max_records]

    people_choices = [int(x) for x in args.people_choices.split(",") if x.strip()]
    rng = random.Random(args.seed)

    sys_prompt = _read_text(here / "prompts" / "system_prompt.txt")
    user_tmpl = _read_text(here / "prompts" / "user_prompt.txt")

    # TravelPlanner pppn quantiles (for mapping pppn -> Stravl budget tier)
    tp_quantiles = load_tp_pppn_quantiles(here / "artifacts")

    # Stravl library (optional)
    stravl_csv = _ensure_stravl_csv(data_dir)
    stravl_lib: List[Dict[str, Any]] = []
    if (not args.no_persona_refs) and stravl_csv is not None:
        stravl_lib = load_stravl_library(stravl_csv, max_rows=args.stravl_max_rows)

    client = OpenAI()

    # Pre-normalize in a single thread to keep randomness deterministic.
    jobs: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for r in rows:
        sid = stable_source_id(r)
        info, norm_meta = normalize_initial_info(r, people_choices=people_choices, rng=rng)
        # minimal raw row for meta/debug
        raw_min = {k: r.get(k) for k in ("org","dest","days","visiting_city_number","date","people_number","local_constraint","budget","query","level","id")}
        jobs.append((sid, info, norm_meta, raw_min))

    def _job(sid: str, initial_info: Dict[str, Any], norm_meta: Dict[str, Any], raw_min: Dict[str, Any]) -> str:
        data_path = data_out_dir / f"{sid}.json"
        meta_path = meta_out_dir / f"{sid}.json" if args.store_meta else None
        if (not args.no_resume) and data_path.exists() and (not args.store_meta or (meta_path and meta_path.exists())):
            return "skipped"

        # Deterministic per-record RNG seed
        sid_seed = int(hashlib.sha1(sid.encode("utf-8")).hexdigest()[:8], 16) ^ args.seed

        persona_refs_json = "[]"
        refs: List[Dict[str, Any]] = []
        if stravl_lib and (not args.no_persona_refs):
            trip = trip_features_from_initial_info(initial_info, tp_quantiles=tp_quantiles)
            k = int(initial_info.get("people_number") or 1) * int(args.ref_multiplier)
            refs = select_persona_references(
                trip,
                stravl_lib,
                k=k,
                base_seed=sid_seed,
                preselect=args.ref_preselect,
                mmr_lambda=float(args.ref_lambda),
            )
            persona_refs_json = persona_reference_block(refs)

        obj = _generate_stage1(
            client,
            sys_prompt=sys_prompt,
            user_tmpl=user_tmpl,
            initial_info=initial_info,
            source_id=sid,
            persona_refs_json=persona_refs_json,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )

        if args.store_meta:
            trip = trip_features_from_initial_info(initial_info, tp_quantiles=tp_quantiles)
            meta: Dict[str, Any] = {
                "raw_row_min": raw_min,
                "normalization": norm_meta,
                "tp_pppn_quantiles_used": tp_quantiles,
                "trip_features": {
                    "pppn": trip.pppn,
                    "season_code": trip.season_code,
                    "mapped_budget_code": trip.mapped_budget_code,
                    "budget_anchor": trip.budget_anchor,
                    "people_number": trip.people_number,
                    "days": trip.days,
                },
                "db_summary": (db_summaries.get(sid) if isinstance(db_summaries, dict) else None),
                "persona_reference_injection": {
                    "enabled": (not args.no_persona_refs),
                    "k": len(refs),
                    "multiplier": args.ref_multiplier,
                    "preselect": args.ref_preselect,
                    "mmr_lambda": args.ref_lambda,
                    "stravl_csv_path": str(stravl_csv) if stravl_csv else None,
                    "stravl_rows_loaded": len(stravl_lib),
                    "selected_refs": [
                        {
                            "ref_id": r.get("ref_id"),
                            "age_ranges": r.get("age_ranges"),
                            "budget_pppn": r.get("budget_pppn"),
                            "season": r.get("season"),
                            "experiences": r.get("experiences"),
                            "scenery": r.get("scenery"),
                        }
                        for r in refs
                    ][:50],  # cap to keep meta compact
                },
                "model": {"name": "gpt-4.1", "temperature": args.temperature},
            }
            if meta_path:
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        data_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return "ok"

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_job, sid, info, norm_meta, raw_min) for (sid, info, norm_meta, raw_min) in jobs]
        for f in tqdm(cf.as_completed(futures), total=len(futures), desc=f"Stage1 ({args.split})"):
            f.result()


if __name__ == "__main__":
    main()
