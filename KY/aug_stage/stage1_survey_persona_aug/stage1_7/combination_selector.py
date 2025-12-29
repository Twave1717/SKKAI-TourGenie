"""Combination selector for finding best N-persona groups."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

from .solvability_check import analyze_conflict_dimensions


def score_combination(personas: List[Dict[str, Any]]) -> float:
    """Score a persona combination based on conflict quality.

    Higher score = better combination.

    Scoring factors:
    - More soft conflicts = better (more interesting)
    - No hard conflicts = required
    - Diversity in conflict types = better

    Args:
        personas: List of enriched personas

    Returns:
        Score (higher is better), or -1 if not solvable
    """
    analysis = analyze_conflict_dimensions(personas)

    if not analysis["is_solvable"]:
        return -1.0  # Not solvable

    # Base score: number of soft conflicts
    score = float(analysis["conflict_count"])

    # Bonus: diversity in conflict categories
    conflict_fields = [c["field"] for c in analysis["soft_conflicts"]]
    conflict_categories = set(f.split(".")[0] for f in conflict_fields)
    category_diversity_bonus = len(conflict_categories) * 2.0

    # Bonus: balance of conflict intensity
    # Prefer combinations where conflicts are spread across 4-9 range
    soft_conflicts = analysis["soft_conflicts"]
    if soft_conflicts:
        alphas = []
        for conflict in soft_conflicts:
            for p in conflict["personas"]:
                alphas.append(p["alpha"])

        if alphas:
            alpha_std = (
                sum((a - sum(alphas) / len(alphas)) ** 2 for a in alphas) / len(alphas)
            ) ** 0.5
            diversity_bonus = alpha_std * 0.5  # Reward variance in alpha values
        else:
            diversity_bonus = 0.0
    else:
        diversity_bonus = 0.0

    total_score = score + category_diversity_bonus + diversity_bonus

    return total_score


def select_best_combination(
    personas: List[Dict[str, Any]],
    target_count: int,
    max_combinations: int = 1000,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Select the best N-persona combination from a pool.

    Args:
        personas: Pool of enriched personas (e.g., 20 or 40)
        target_count: Target number of personas (e.g., 2 or 4)
        max_combinations: Maximum combinations to evaluate

    Returns:
        Tuple of (best_personas, analysis) or None if no solvable combination found
    """
    if len(personas) < target_count:
        return None

    # Generate all possible combinations
    all_combinations = list(itertools.combinations(personas, target_count))

    # Limit combinations to avoid explosion
    if len(all_combinations) > max_combinations:
        # Sample combinations deterministically
        import random

        rng = random.Random(42)
        all_combinations = rng.sample(all_combinations, max_combinations)

    # Score each combination
    scored_combinations = []

    for combo in all_combinations:
        score = score_combination(list(combo))
        if score > 0:  # Only keep solvable combinations
            scored_combinations.append((score, list(combo)))

    if not scored_combinations:
        return None  # No solvable combinations found

    # Sort by score (descending)
    scored_combinations.sort(key=lambda x: x[0], reverse=True)

    # Get best combination
    best_score, best_personas = scored_combinations[0]

    # Analyze best combination
    analysis = analyze_conflict_dimensions(best_personas)
    analysis["score"] = best_score
    analysis["total_combinations_evaluated"] = len(all_combinations)
    analysis["solvable_combinations_found"] = len(scored_combinations)

    return best_personas, analysis
