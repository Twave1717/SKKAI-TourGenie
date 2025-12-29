from __future__ import annotations

"""Conflict-aware Persona Retrieval for Multi-Agent Travel Planning.

This module implements an improved MMR-based retrieval system that:
1. Selects an "anchor" persona that best fits the trip context
2. Selects "opponents" that create productive conflicts with the anchor
3. Balances conflict potential with feasibility (compatible enemies)

Key design principles:
- "Compatible Enemies": Personas can travel together (budget/season match)
  but will argue about what to do (activity/interest conflict)
- Strategic diversity: Not random, but targeted conflict dimensions
- Context-aware: Trip characteristics influence conflict weights
"""

import hashlib
import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .vectorization import (
    ConflictWeights,
    VectorDimensions,
    calculate_conflict_score,
    cosine_distance,
    extract_subvector,
    jaccard_similarity_multihot,
    vectorize_persona,
    weighted_euclidean_distance,
)


@dataclass(frozen=True)
class TripContext:
    """Normalized trip context for relevance scoring."""

    people_number: int
    days: int
    budget_anchor: Optional[int]
    pppn: Optional[float]  # per-person-per-night
    season_code: Optional[int]  # 0..3
    mapped_budget_code: Optional[int]  # 0..3
    dest: str
    org: str
    visiting_city_number: int

    # Derived features for matching
    is_short_trip: bool  # < 4 days
    is_budget_constrained: bool  # pppn < 100
    is_large_group: bool  # >= 4 people
    is_multi_city: bool  # visiting_city_number > 1


def create_trip_context(
    initial_info: Dict[str, Any],
    *,
    pppn: Optional[float] = None,
    season_code: Optional[int] = None,
    mapped_budget_code: Optional[int] = None,
) -> TripContext:
    """Create a TripContext from TravelPlanner initial_info.

    Args:
        initial_info: TravelPlanner initial_info dict
        pppn: Pre-computed per-person-per-night budget
        season_code: Pre-computed season code (0-3)
        mapped_budget_code: Pre-computed budget tier (0-3)

    Returns:
        TripContext with derived features
    """
    people = int(initial_info.get("people_number") or 1)
    days = int(initial_info.get("days") or 1)
    budget = initial_info.get("budget_anchor")
    budget = int(budget) if budget is not None else None
    visiting = int(initial_info.get("visiting_city_number") or 1)

    return TripContext(
        people_number=people,
        days=days,
        budget_anchor=budget,
        pppn=pppn,
        season_code=season_code,
        mapped_budget_code=mapped_budget_code,
        dest=str(initial_info.get("dest") or ""),
        org=str(initial_info.get("org") or ""),
        visiting_city_number=visiting,
        is_short_trip=(days < 4),
        is_budget_constrained=(pppn is not None and pppn < 100),
        is_large_group=(people >= 4),
        is_multi_city=(visiting > 1),
    )


# ---------------------------
# Relevance scoring (Context fit)
# ---------------------------

def calculate_relevance_score_budget_flexible(
    persona_vec: np.ndarray,
    persona_raw: Dict[str, Any],
    trip_context: TripContext,
) -> float:
    """Score relevance with budget diversity allowed (for budget_war strategy).

    This version ignores budget differences to allow diverse budget preferences
    in the candidate pool, which enables budget conflicts in the final selection.

    Args:
        persona_vec: Vectorized persona features
        persona_raw: Original persona dict
        trip_context: Trip context object

    Returns:
        Relevance score in [0, ~3] range (without budget scoring)
    """
    score = 0.0

    # --- Budget: DO NOT filter (allow diversity) ---
    # This is the key difference from calculate_relevance_score()

    # --- Season compatibility (important) ---
    persona_season = persona_raw.get("season_code")
    trip_season = trip_context.season_code
    if persona_season is not None and trip_season is not None:
        if persona_season == trip_season:
            score += 1.5  # Season match
        else:
            score += 0.3  # At least flexible

    # --- Activity level fit ---
    activity_level = persona_vec[VectorDimensions.ACTIVITY_LEVEL]
    if trip_context.is_short_trip:
        # Short trips → prefer active travelers
        score += activity_level * 0.8
    else:
        # Longer trips → balanced or relaxed OK
        score += (1.0 - abs(activity_level - 0.5)) * 0.5

    # --- Multi-city trips → prefer adventurous/flexible ---
    if trip_context.is_multi_city:
        safety_level = persona_vec[VectorDimensions.SAFETY_LEVEL]
        score += safety_level * 0.5

    # --- Experience/Scenery diversity bonus ---
    exp_vec = extract_subvector(persona_vec, "experiences")
    scenery_vec = extract_subvector(persona_vec, "scenery")
    num_interests = int(np.sum(exp_vec) + np.sum(scenery_vec))
    score += min(num_interests * 0.15, 1.0)

    return float(score)


def calculate_relevance_score(
    persona_vec: np.ndarray,
    persona_raw: Dict[str, Any],
    trip_context: TripContext,
) -> float:
    """Score how well a persona fits the trip context.

    Higher score = better fit (this persona is a plausible trip participant).

    Args:
        persona_vec: Vectorized persona features
        persona_raw: Original persona dict (for categorical checks)
        trip_context: Trip context object

    Returns:
        Relevance score in [0, ~5] range (higher = better fit)
    """
    score = 0.0

    # --- Budget compatibility (critical) ---
    persona_budget = persona_raw.get("budget_code")
    trip_budget = trip_context.mapped_budget_code
    if persona_budget is not None and trip_budget is not None:
        budget_diff = abs(persona_budget - trip_budget)
        if budget_diff == 0:
            score += 2.5  # Perfect match
        elif budget_diff == 1:
            score += 1.5  # Adjacent tier (acceptable)
        elif budget_diff == 2:
            score += 0.5  # Stretching it
        # diff == 3: no bonus (Frugal vs Luxury won't work)

    # --- Season compatibility (important) ---
    persona_season = persona_raw.get("season_code")
    trip_season = trip_context.season_code
    if persona_season is not None and trip_season is not None:
        if persona_season == trip_season:
            score += 1.5  # Season match
        else:
            score += 0.3  # At least flexible

    # --- Activity level fit ---
    activity_level = persona_vec[VectorDimensions.ACTIVITY_LEVEL]
    if trip_context.is_short_trip:
        # Short trips → prefer active travelers
        score += activity_level * 0.8
    else:
        # Longer trips → balanced or relaxed OK
        score += (1.0 - abs(activity_level - 0.5)) * 0.5

    # --- Multi-city trips → prefer adventurous/flexible ---
    if trip_context.is_multi_city:
        safety_level = persona_vec[VectorDimensions.SAFETY_LEVEL]  # 0=safe, 1=adventurous
        score += safety_level * 0.5

    # --- Experience/Scenery diversity bonus (prefer non-empty profiles) ---
    exp_vec = extract_subvector(persona_vec, "experiences")
    scenery_vec = extract_subvector(persona_vec, "scenery")
    num_interests = int(np.sum(exp_vec) + np.sum(scenery_vec))
    score += min(num_interests * 0.15, 1.0)  # Cap at 1.0

    return float(score)


# ---------------------------
# Anchor selection
# ---------------------------

def select_anchor_persona(
    pool: List[Tuple[Dict[str, Any], np.ndarray]],
    trip_context: TripContext,
    *,
    top_k: int = 10,
    seed: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray, int]:
    """Select the anchor persona (trip leader / best context fit).

    Strategy:
    1. Score all personas by relevance
    2. Take top-K candidates
    3. Pick one with slight randomization (avoid always choosing the same)

    Args:
        pool: List of (persona_dict, persona_vector) tuples
        trip_context: Trip context
        top_k: Number of top candidates to consider
        seed: Random seed for tie-breaking

    Returns:
        (anchor_persona_dict, anchor_vector, anchor_index_in_pool)
    """
    scored: List[Tuple[float, int, Dict[str, Any], np.ndarray]] = []

    for idx, (persona, vec) in enumerate(pool):
        rel_score = calculate_relevance_score(vec, persona, trip_context)
        scored.append((rel_score, idx, persona, vec))

    # Sort by relevance (descending)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top-K and pick with slight randomization
    candidates = scored[:top_k]
    rng = random.Random(seed)
    # Weighted random choice: higher score = higher probability
    weights = [x[0] + 0.1 for x in candidates]  # +0.1 to avoid zero weights
    chosen = rng.choices(candidates, weights=weights, k=1)[0]

    _, idx, persona, vec = chosen
    return persona, vec, idx


# ---------------------------
# Conflict-aware MMR
# ---------------------------

def select_opponents_mmr(
    anchor_vec: np.ndarray,
    anchor_idx: int,
    pool: List[Tuple[Dict[str, Any], np.ndarray]],
    trip_context: TripContext,
    *,
    k: int = 3,
    lambda_param: float = 0.6,
    conflict_strategy: str = "adaptive",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select opponent personas using conflict-aware MMR.

    Algorithm:
    1. Compute conflict weights based on strategy
    2. Iteratively select personas that:
       - Are relevant to the trip (lambda * relevance)
       - Maximize conflict with already-selected personas ((1 - lambda) * conflict)
    3. Use Minimax conflict: new persona should differ from at least one existing member

    Args:
        anchor_vec: Anchor persona vector
        anchor_idx: Index of anchor in pool (to skip)
        pool: Full persona pool
        trip_context: Trip context
        k: Number of opponents to select
        lambda_param: MMR balance (0=max conflict, 1=max relevance)
        conflict_strategy: One of ['adaptive', 'budget_war', 'pace_war', 'taste_war']
        seed: Random seed

    Returns:
        List of k opponent persona dicts
    """
    # Determine conflict weights
    if conflict_strategy == "adaptive":
        weights = ConflictWeights.adaptive(trip_context.__dict__)
    elif conflict_strategy == "budget_war":
        weights = ConflictWeights.budget_war()
    elif conflict_strategy == "pace_war":
        weights = ConflictWeights.pace_war()
    elif conflict_strategy == "taste_war":
        weights = ConflictWeights.taste_war()
    else:
        weights = ConflictWeights.default()

    rng = random.Random(seed)

    selected: List[Dict[str, Any]] = []
    selected_vecs: List[np.ndarray] = [anchor_vec]  # Start with anchor
    selected_indices: set[int] = {anchor_idx}

    candidates = [(p, v, i) for i, (p, v) in enumerate(pool) if i != anchor_idx]

    # Pre-shuffle to reduce bias in tie-breaking
    rng.shuffle(candidates)

    for _ in range(k):
        if not candidates:
            break

        best_mmr = -1e9
        best_candidate = None
        best_idx_in_candidates = -1

        for cand_idx, (persona, vec, pool_idx) in enumerate(candidates):
            if pool_idx in selected_indices:
                continue

            # --- Relevance component ---
            relevance = calculate_relevance_score(vec, persona, trip_context)

            # --- Conflict component (Minimax strategy) ---
            # Measure conflict against all selected personas
            conflicts = [
                calculate_conflict_score(vec, sel_vec, weights, normalize=True)
                for sel_vec in selected_vecs
            ]
            # Use MINIMUM conflict (ensures new persona differs from at least one)
            # This prevents all opponents from being clones
            min_conflict = min(conflicts) if conflicts else 0.0

            # --- MMR score ---
            mmr_score = lambda_param * relevance + (1.0 - lambda_param) * min_conflict

            # Tiny random noise for tie-breaking
            mmr_score += rng.random() * 1e-6

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_candidate = (persona, vec, pool_idx)
                best_idx_in_candidates = cand_idx

        if best_candidate is None:
            break

        persona, vec, pool_idx = best_candidate
        selected.append(persona)
        selected_vecs.append(vec)
        selected_indices.add(pool_idx)

        # Remove from candidates (optional optimization)
        if best_idx_in_candidates >= 0:
            candidates.pop(best_idx_in_candidates)

    return selected


# ---------------------------
# Main retrieval function
# ---------------------------

def retrieve_conflicting_group(
    pool: List[Dict[str, Any]],
    trip_context: TripContext,
    *,
    k: int = 4,
    lambda_param: float = 0.6,
    conflict_strategy: str = "adaptive",
    prefilter_size: int = 0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Retrieve a k-persona group with controlled conflict.

    Pipeline:
    1. Vectorize all personas
    2. (Optional) Pre-filter to top-N by budget/season compatibility
    3. Select anchor (best context fit)
    4. Select k-1 opponents using conflict-aware MMR

    Args:
        pool: List of Stravl persona dicts
        trip_context: Trip context object
        k: Total group size (including anchor)
        lambda_param: MMR balance parameter
        conflict_strategy: Conflict weight strategy
        prefilter_size: Pre-filter pool size (0 = auto: k*10, -1 = no filtering)
        seed: Random seed

    Returns:
        List of k persona dicts (anchor first, then opponents)
    """
    if k <= 0 or not pool:
        return []

    # --- Step 1: Vectorize ---
    vectorized_pool: List[Tuple[Dict[str, Any], np.ndarray]] = []
    for persona in pool:
        vec = vectorize_persona(persona)
        vectorized_pool.append((persona, vec))

    # --- Determine pre-filter size ---
    if prefilter_size == 0:
        # Auto mode: k * 10 (e.g., 2명 → 20명, 4명 → 40명)
        actual_prefilter_size = k * 10
    elif prefilter_size == -1:
        # No filtering
        actual_prefilter_size = 0
    else:
        # Use specified size
        actual_prefilter_size = prefilter_size

    # --- Step 2: Pre-filter (optional) ---
    if actual_prefilter_size > 0 and len(vectorized_pool) > actual_prefilter_size:
        # Strategy-aware pre-filtering
        if conflict_strategy == "budget_war":
            # For budget_war: allow budget diversity, filter only by season
            scored = [
                (calculate_relevance_score_budget_flexible(vec, p, trip_context), p, vec)
                for p, vec in vectorized_pool
            ]
        else:
            # Default: filter by relevance (budget + season compatibility)
            scored = [
                (calculate_relevance_score(vec, p, trip_context), p, vec)
                for p, vec in vectorized_pool
            ]
        scored.sort(key=lambda x: x[0], reverse=True)
        vectorized_pool = [(p, v) for _, p, v in scored[:actual_prefilter_size]]

    if k == 1:
        # Special case: only need anchor
        anchor, _, _ = select_anchor_persona(vectorized_pool, trip_context, seed=seed)
        return [anchor]

    # --- Step 3: Select anchor ---
    anchor_persona, anchor_vec, anchor_idx = select_anchor_persona(
        vectorized_pool, trip_context, seed=seed
    )

    # --- Step 4: Select opponents ---
    opponents = select_opponents_mmr(
        anchor_vec,
        anchor_idx,
        vectorized_pool,
        trip_context,
        k=k - 1,
        lambda_param=lambda_param,
        conflict_strategy=conflict_strategy,
        seed=seed,
    )

    # Return anchor first, then opponents
    return [anchor_persona] + opponents


# ---------------------------
# Helper: Stable seed generation
# ---------------------------

def stable_retrieval_seed(trip_context: TripContext, base_seed: int = 42) -> int:
    """Generate a deterministic seed from trip context.

    Args:
        trip_context: Trip context
        base_seed: Base random seed

    Returns:
        Integer seed for retrieval
    """
    key = f"{trip_context.org}→{trip_context.dest}:{trip_context.people_number}:{trip_context.days}:{trip_context.budget_anchor}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) ^ base_seed
