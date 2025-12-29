from __future__ import annotations

"""Persona Vectorization for Conflict-aware MMR.

Converts Stravl survey data into numerical vectors suitable for:
- Relevance scoring (trip context matching)
- Conflict scoring (inter-persona friction)
- Weighted distance calculations

Design principles:
- Normalize all features to [0, 1] range
- Separate hard constraints (budget, season) from soft preferences
- Support multi-hot encoding for multi-select fields (experiences, scenery)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# Feature dimension indices
# ---------------------------

class VectorDimensions:
    """Named indices for persona vector components."""

    # Hard constraints (0-2)
    BUDGET_LEVEL = 0  # 0-3 normalized to [0, 1]
    SEASON_CODE = 1   # 0-3 normalized to [0, 1]
    AGE_MIN = 2       # Minimum age bucket (0-3 normalized)

    # Behavioral preferences (3-5)
    ACTIVITY_LEVEL = 3     # 0=Chill, 1=Balanced, 2=Active → [0, 1]
    SAFETY_LEVEL = 4       # 0=Safe, 1=Balanced, 2=Adventurous → [0, 1]
    POPULARITY_LEVEL = 5   # 0=Hidden, 1=Classic, 2=Trendy → [0, 1]

    # Experiences (6-13): 8-dim multi-hot
    # Beach, Adventure, Nature, Culture, Nightlife, History, Shopping, Cuisine
    EXP_START = 6
    EXP_END = 14  # exclusive

    # Scenery (14-21): 8-dim multi-hot
    # Urban, Rural, Sea, Mountain, Lake, Desert, Plains, Jungle
    SCENERY_START = 14
    SCENERY_END = 22  # exclusive

    TOTAL_DIM = 22


# ---------------------------
# Vectorization functions
# ---------------------------

def vectorize_persona(persona: Dict[str, Any]) -> np.ndarray:
    """Convert a Stravl persona dict to a normalized feature vector.

    Args:
        persona: Decoded Stravl row with keys like:
            - budget_code (int 0-3)
            - season_code (int 0-3)
            - age_ranges (List[str])
            - activity_level (str)
            - safety_conscious (str)
            - destination_popularity (str)
            - experiences (List[str])
            - scenery (List[str])

    Returns:
        22-dim numpy array with all features normalized to [0, 1]
    """
    vec = np.zeros(VectorDimensions.TOTAL_DIM, dtype=np.float32)

    # --- Hard constraints ---
    budget_code = persona.get("budget_code")
    if isinstance(budget_code, int) and 0 <= budget_code <= 3:
        vec[VectorDimensions.BUDGET_LEVEL] = budget_code / 3.0

    season_code = persona.get("season_code")
    if isinstance(season_code, int) and 0 <= season_code <= 3:
        vec[VectorDimensions.SEASON_CODE] = season_code / 3.0

    # Age: use minimum age bucket as a proxy (for simplicity)
    age_ranges = persona.get("age_ranges") or []
    age_codes = []
    for ar in age_ranges:
        if ar == "0-19":
            age_codes.append(0)
        elif ar == "20-39":
            age_codes.append(1)
        elif ar == "40-59":
            age_codes.append(2)
        elif ar == "60+":
            age_codes.append(3)
    if age_codes:
        vec[VectorDimensions.AGE_MIN] = min(age_codes) / 3.0

    # --- Behavioral preferences ---
    activity = persona.get("activity_level") or ""
    if "Chill" in activity or "Relaxed" in activity:
        vec[VectorDimensions.ACTIVITY_LEVEL] = 0.0
    elif "Balanced" in activity:
        vec[VectorDimensions.ACTIVITY_LEVEL] = 0.5
    elif "Active" in activity:
        vec[VectorDimensions.ACTIVITY_LEVEL] = 1.0

    safety = persona.get("safety_conscious") or ""
    if "Very Safety" in safety:
        vec[VectorDimensions.SAFETY_LEVEL] = 0.0
    elif "Balanced" in safety:
        vec[VectorDimensions.SAFETY_LEVEL] = 0.5
    elif "Ready for Anything" in safety:
        vec[VectorDimensions.SAFETY_LEVEL] = 1.0

    popularity = persona.get("destination_popularity") or ""
    if "Off the Beaten" in popularity:
        vec[VectorDimensions.POPULARITY_LEVEL] = 0.0
    elif "Classic" in popularity:
        vec[VectorDimensions.POPULARITY_LEVEL] = 0.5
    elif "Mainstream" in popularity or "Trendy" in popularity:
        vec[VectorDimensions.POPULARITY_LEVEL] = 1.0

    # --- Multi-hot encodings ---
    # Experiences: map to indices 0-7
    exp_map = {
        "Beach": 0, "Adventure": 1, "Nature": 2, "Culture": 3,
        "Nightlife": 4, "History": 5, "Shopping": 6, "Cuisine": 7,
    }
    for exp in (persona.get("experiences") or []):
        idx = exp_map.get(exp)
        if idx is not None:
            vec[VectorDimensions.EXP_START + idx] = 1.0

    # Scenery: map to indices 0-7
    scenery_map = {
        "Urban": 0, "Rural": 1, "Sea": 2, "Mountain": 3,
        "Lake": 4, "Desert": 5, "Plains": 6, "Jungle": 7,
    }
    for sc in (persona.get("scenery") or []):
        idx = scenery_map.get(sc)
        if idx is not None:
            vec[VectorDimensions.SCENERY_START + idx] = 1.0

    return vec


def extract_subvector(vec: np.ndarray, category: str) -> np.ndarray:
    """Extract a specific category of features from the full vector.

    Args:
        vec: 22-dim persona vector
        category: One of ['hard', 'behavioral', 'experiences', 'scenery', 'all']

    Returns:
        Sub-vector for the requested category
    """
    if category == "hard":
        return vec[0:3]  # budget, season, age
    elif category == "behavioral":
        return vec[3:6]  # activity, safety, popularity
    elif category == "experiences":
        return vec[VectorDimensions.EXP_START:VectorDimensions.EXP_END]
    elif category == "scenery":
        return vec[VectorDimensions.SCENERY_START:VectorDimensions.SCENERY_END]
    elif category == "all":
        return vec
    else:
        raise ValueError(f"Unknown category: {category}")


# ---------------------------
# Distance/Similarity metrics
# ---------------------------

def weighted_euclidean_distance(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute weighted Euclidean distance between two persona vectors.

    Args:
        vec_a, vec_b: 22-dim persona vectors
        weights: Optional 22-dim weight vector (default: all 1.0)

    Returns:
        Weighted L2 distance
    """
    if weights is None:
        weights = np.ones_like(vec_a)
    diff = vec_a - vec_b
    weighted_diff = diff * weights
    return float(np.linalg.norm(weighted_diff))


def jaccard_similarity_multihot(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Jaccard similarity for multi-hot encoded vectors.

    Args:
        vec_a, vec_b: Binary vectors (0 or 1)

    Returns:
        Jaccard index in [0, 1]
    """
    intersection = np.sum(np.minimum(vec_a, vec_b))
    union = np.sum(np.maximum(vec_a, vec_b))
    if union == 0:
        return 0.0
    return float(intersection / union)


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine similarity).

    Args:
        vec_a, vec_b: Feature vectors

    Returns:
        Cosine distance in [0, 2] (0=identical, 2=opposite)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0  # neutral distance
    cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return float(1.0 - cos_sim)


# ---------------------------
# Conflict scoring
# ---------------------------

class ConflictWeights:
    """Predefined weight profiles for different conflict strategies."""

    @staticmethod
    def default() -> np.ndarray:
        """Balanced weights: all features equally important."""
        return np.ones(VectorDimensions.TOTAL_DIM, dtype=np.float32)

    @staticmethod
    def budget_war() -> np.ndarray:
        """Emphasize budget differences while keeping compatibility."""
        w = np.ones(VectorDimensions.TOTAL_DIM, dtype=np.float32)
        w[VectorDimensions.BUDGET_LEVEL] = 3.0  # High budget conflict
        w[VectorDimensions.SEASON_CODE] = 0.5   # Season must match (low weight)
        w[VectorDimensions.ACTIVITY_LEVEL] = 1.5
        return w

    @staticmethod
    def pace_war() -> np.ndarray:
        """Emphasize activity level conflicts (fast vs slow travelers)."""
        w = np.ones(VectorDimensions.TOTAL_DIM, dtype=np.float32)
        w[VectorDimensions.ACTIVITY_LEVEL] = 4.0  # Very high conflict on pace
        w[VectorDimensions.POPULARITY_LEVEL] = 2.0  # Tourist vs hidden gems
        w[VectorDimensions.BUDGET_LEVEL] = 0.8  # Budget should be compatible
        return w

    @staticmethod
    def taste_war() -> np.ndarray:
        """Emphasize experience/interest conflicts (what to do)."""
        w = np.ones(VectorDimensions.TOTAL_DIM, dtype=np.float32)
        w[VectorDimensions.BUDGET_LEVEL] = 0.7  # Budget compatible
        w[VectorDimensions.SEASON_CODE] = 0.5   # Season compatible
        # Heavy weights on experiences (what activities to do)
        w[VectorDimensions.EXP_START:VectorDimensions.EXP_END] = 3.0
        return w

    @staticmethod
    def adaptive(trip_context: Dict[str, Any]) -> np.ndarray:
        """Context-aware weights based on trip characteristics.

        Args:
            trip_context: Dict with keys like 'budget_anchor', 'days', 'people_number'

        Returns:
            Adaptive weight vector
        """
        w = np.ones(VectorDimensions.TOTAL_DIM, dtype=np.float32)

        # If budget is tight, emphasize budget conflicts
        budget = trip_context.get("budget_anchor", 0)
        people = trip_context.get("people_number", 1)
        pppn = budget / max(1, people * trip_context.get("days", 1))
        if pppn < 100:  # Low budget trip
            w[VectorDimensions.BUDGET_LEVEL] = 3.0

        # If short trip (< 4 days), emphasize pace/activity
        days = trip_context.get("days", 7)
        if days < 4:
            w[VectorDimensions.ACTIVITY_LEVEL] = 3.5
            w[VectorDimensions.EXP_START:VectorDimensions.EXP_END] = 2.5

        # If many people (>= 4), emphasize interest diversity
        if people >= 4:
            w[VectorDimensions.EXP_START:VectorDimensions.EXP_END] = 2.0

        return w


def calculate_conflict_score(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    weights: np.ndarray,
    *,
    normalize: bool = True,
) -> float:
    """Calculate conflict score between two personas.

    Higher score = more conflict potential (good for diverse groups).

    Args:
        vec_a, vec_b: Persona vectors
        weights: Dimension-wise conflict weights
        normalize: Whether to normalize by vector dimension

    Returns:
        Conflict score (higher = more conflict)
    """
    # For multi-hot fields (experiences, scenery), use inverse Jaccard
    exp_a = extract_subvector(vec_a, "experiences")
    exp_b = extract_subvector(vec_b, "experiences")
    exp_conflict = 1.0 - jaccard_similarity_multihot(exp_a, exp_b)

    scenery_a = extract_subvector(vec_a, "scenery")
    scenery_b = extract_subvector(vec_b, "scenery")
    scenery_conflict = 1.0 - jaccard_similarity_multihot(scenery_a, scenery_b)

    # For continuous/ordinal fields (budget, activity, etc.), use weighted distance
    other_mask = np.ones(VectorDimensions.TOTAL_DIM, dtype=bool)
    other_mask[VectorDimensions.EXP_START:VectorDimensions.EXP_END] = False
    other_mask[VectorDimensions.SCENERY_START:VectorDimensions.SCENERY_END] = False

    other_vec_a = vec_a[other_mask]
    other_vec_b = vec_b[other_mask]
    other_weights = weights[other_mask]
    other_conflict = weighted_euclidean_distance(other_vec_a, other_vec_b, other_weights)

    # Combine: weighted sum of different conflict types
    exp_weight_avg = np.mean(weights[VectorDimensions.EXP_START:VectorDimensions.EXP_END])
    scenery_weight_avg = np.mean(weights[VectorDimensions.SCENERY_START:VectorDimensions.SCENERY_END])

    total_conflict = (
        other_conflict +
        exp_conflict * exp_weight_avg * 8.0 +  # Scale by number of experience dims
        scenery_conflict * scenery_weight_avg * 8.0  # Scale by number of scenery dims
    )

    if normalize:
        # Normalize by theoretical max conflict
        max_conflict = weighted_euclidean_distance(
            np.zeros(VectorDimensions.TOTAL_DIM),
            np.ones(VectorDimensions.TOTAL_DIM),
            weights,
        )
        total_conflict = total_conflict / max(1e-6, max_conflict)

    return float(total_conflict)
