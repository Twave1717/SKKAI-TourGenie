from __future__ import annotations

"""Persona reference retrieval (rule-based + diversity reranking).

Goal
----
Given a TravelPlanner `initial_info` (org/dest/days/date/people/budget_anchor),
retrieve K = people_number * multiplier persona *references* from a grounded pool
(Stravl Travel Preference Data). These references are then injected into the Stage1
prompt to help the model generate more realistic, diverse group personas while
staying within a closed-world setting.

Why MMR-like diversity?
-----------------------
If you simply take top-K by a single score (e.g., budget match), you'll get many
near-duplicates. A simple Maximum Marginal Relevance (MMR) reranking improves
coverage/diversity among exemplars while keeping relevance.
"""

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from stravl_codec import (
    ACTIVITY,
    BUDGET_PPPN,
    EXPERIENCE,
    POPULARITY,
    SAFETY,
    SCENERY,
    SEASON,
    decode_form_fields,
    get_ci,
    norm_key,
    parse_int_list,
)


@dataclass(frozen=True)
class TripFeatures:
    people_number: int
    days: int
    budget_anchor: Optional[int]
    pppn: Optional[float]  # per-person-per-night
    season_code: Optional[int]  # 0..3 (Stravl)
    mapped_budget_code: Optional[int]  # 0..3 (Stravl bucket, from TP quantiles)
    dest: str
    org: str


def _parse_date_ymd(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def infer_season_code(dates: List[str]) -> Optional[int]:
    """Map month -> Stravl season code."""
    if not dates:
        return None
    dt = _parse_date_ymd(dates[0]) or _parse_date_ymd(dates[-1])
    if not dt:
        return None
    m = dt.month
    # Northern hemisphere convention (matches Stravl labels)
    if m in (12, 1, 2):
        return 0  # Winter
    if m in (3, 4, 5):
        return 1  # Spring
    if m in (6, 7, 8):
        return 2  # Summer
    return 3  # Fall


def compute_pppn(budget_anchor: Optional[int], people: int, days: int) -> Optional[float]:
    if budget_anchor is None or people <= 0 or days <= 0:
        return None
    try:
        return float(budget_anchor) / float(people * days)
    except Exception:
        return None


def map_pppn_to_stravl_budget_code(pppn: Optional[float], tp_quantiles: Optional[Dict[str, float]]) -> Optional[int]:
    """Quantile-based mapping from TravelPlanner pppn -> Stravl budget bucket (0..3).

    tp_quantiles expects keys like: q25, q50, q75
    If tp_quantiles is None, falls back to the literal Stravl thresholds.
    """
    if pppn is None:
        return None

    if tp_quantiles and all(k in tp_quantiles for k in ("q25", "q50", "q75")):
        q25, q50, q75 = float(tp_quantiles["q25"]), float(tp_quantiles["q50"]), float(tp_quantiles["q75"])
        if pppn <= q25:
            return 0
        if pppn <= q50:
            return 1
        if pppn <= q75:
            return 2
        return 3

    # Fallback: literal Stravl bins
    if pppn < 50:
        return 0
    if pppn < 100:
        return 1
    if pppn < 250:
        return 2
    return 3


def trip_features_from_initial_info(initial_info: Dict[str, Any], tp_quantiles: Optional[Dict[str, float]] = None) -> TripFeatures:
    people = int(initial_info.get("people_number") or 1)
    days = int(initial_info.get("days") or 1)
    budget_anchor = initial_info.get("budget_anchor")
    try:
        budget_anchor = int(budget_anchor) if budget_anchor is not None else None
    except Exception:
        budget_anchor = None

    dates = initial_info.get("date") or []
    dates = [str(x) for x in dates] if isinstance(dates, list) else []
    season_code = infer_season_code(dates)
    pppn = compute_pppn(budget_anchor, people, days)
    mapped_budget_code = map_pppn_to_stravl_budget_code(pppn, tp_quantiles)

    return TripFeatures(
        people_number=people,
        days=days,
        budget_anchor=budget_anchor,
        pppn=pppn,
        season_code=season_code,
        mapped_budget_code=mapped_budget_code,
        dest=str(initial_info.get("dest") or ""),
        org=str(initial_info.get("org") or ""),
    )


def _stable_seed(seed_id: str, base_seed: int) -> random.Random:
    h = hashlib.sha1(seed_id.encode("utf-8")).hexdigest()[:8]
    return random.Random(int(h, 16) ^ base_seed)


def load_stravl_library(csv_path: Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    """Load and decode Stravl preferences into a compact in-memory library."""
    lib: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            decoded = decode_form_fields(row)
            # keep both decoded + coded buckets for scoring/diversity
            b_list = parse_int_list(get_ci(row, "FORM_B"))
            c_list = parse_int_list(get_ci(row, "FORM_C"))
            b_code = b_list[0] if b_list else None
            c_code = c_list[0] if c_list else None
            lib.append(
                {
                    "ref_id": f"stravl_{i}",
                    "source": "stravl",
                    "budget_code": b_code,
                    "season_code": c_code,
                    **decoded,
                }
            )
    return lib


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _feature_sim(x: Dict[str, Any], y: Dict[str, Any]) -> float:
    # Similarity over experiences + scenery (dominant), plus some exact matches
    sim = 0.0
    sim += 0.6 * _jaccard(x.get("experiences") or [], y.get("experiences") or [])
    sim += 0.4 * _jaccard(x.get("scenery") or [], y.get("scenery") or [])
    if x.get("budget_code") is not None and x.get("budget_code") == y.get("budget_code"):
        sim += 0.2
    if x.get("season_code") is not None and x.get("season_code") == y.get("season_code"):
        sim += 0.1
    return min(sim, 1.0)


def _rule_score(trip: TripFeatures, ref: Dict[str, Any]) -> float:
    score = 0.0
    # Budget bucket match (using quantile-mapped bucket)
    tb = trip.mapped_budget_code
    rb = ref.get("budget_code")
    if tb is not None and rb is not None:
        if tb == rb:
            score += 2.0
        elif abs(tb - rb) == 1:
            score += 1.0

    # Season match
    ts = trip.season_code
    rs = ref.get("season_code")
    if ts is not None and rs is not None and ts == rs:
        score += 1.0

    # Small preference for richer seed rows (more multi-select signals)
    score += 0.05 * min(len(ref.get("experiences") or []), 4)
    score += 0.03 * min(len(ref.get("scenery") or []), 4)
    return score


def select_persona_references(
    trip: TripFeatures,
    library: List[Dict[str, Any]],
    k: int,
    *,
    base_seed: int = 42,
    preselect: int = 500,
    mmr_lambda: float = 0.75,
) -> List[Dict[str, Any]]:
    """Select K persona references using a rule-score + MMR-style diversity reranking.

    Steps:
    1) score all refs by rule-score
    2) take top `preselect`
    3) greedy MMR rerank to pick K diverse references
    """
    if k <= 0 or not library:
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = [( _rule_score(trip, r), r) for r in library]
    scored.sort(key=lambda x: x[0], reverse=True)
    pool = [r for _, r in scored[: max(preselect, k)]]

    # Deterministic tie-breaking via per-trip RNG.
    rng = _stable_seed(f"{trip.org}->{trip.dest}:{trip.people_number}:{trip.days}:{trip.budget_anchor}", base_seed)

    # If pool is too uniform (often happens when tb==3 for most), shuffle within score-ties.
    # We'll do a light shuffle to avoid always choosing the same ids.
    rng.shuffle(pool)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    while len(selected) < k and pool:
        best_idx = -1
        best_val = -1e9
        for idx, cand in enumerate(pool):
            cid = cand.get("ref_id")
            if cid in selected_ids:
                continue
            rel = _rule_score(trip, cand)
            if not selected:
                mmr = rel
            else:
                max_sim = max(_feature_sim(cand, s) for s in selected)
                mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
            # small noise for tie-break
            mmr += rng.random() * 1e-6
            if mmr > best_val:
                best_val = mmr
                best_idx = idx

        if best_idx < 0:
            break
        best = pool.pop(best_idx)
        selected.append(best)
        if isinstance(best.get("ref_id"), str):
            selected_ids.add(best["ref_id"])

    return selected


_BUDGET_TIER_STYLE = {
    0: "Frugal",
    1: "Budget",
    2: "Comfort",
    3: "Luxury",
}


def _budget_tier_from_code(code: Any) -> Optional[str]:
    try:
        return _BUDGET_TIER_STYLE.get(int(code))
    except Exception:
        return None


def persona_reference_block(refs: List[Dict[str, Any]], *, max_items: int = 200) -> str:
    """Render a compact JSON block to inject into prompts."""
    slim: List[Dict[str, Any]] = []
    for r in refs[:max_items]:
        slim.append(
            {
                "ref_id": r.get("ref_id"),
                "budget_tier": _budget_tier_from_code(r.get("budget_code")),
                "season": r.get("season"),
                "experiences": r.get("experiences") or [],
                "scenery": r.get("scenery") or [],
                "activity_level": r.get("activity_level"),
                "safety_conscious": r.get("safety_conscious"),
                "destination_popularity": r.get("destination_popularity"),
                "age_ranges": r.get("age_ranges") or [],
            }
        )
    return json.dumps(slim, ensure_ascii=False, indent=2)
