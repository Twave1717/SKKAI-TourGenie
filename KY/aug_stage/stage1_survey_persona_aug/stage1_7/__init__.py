"""Stage 1.7: Solvable Conflict Selection Module.

This module handles "solvable conflict" checking and final persona selection.

Solvable Conflict Definition:
- Hard constraints (α≥9) do NOT conflict between personas
- Soft constraints (4≤α<9) DO conflict between personas
- At least 2 dimensions must have conflicts
"""

from .solvability_check import (
    check_solvable_conflict,
    analyze_conflict_dimensions,
    extract_alpha_values,
    check_field_conflict,
)
from .combination_selector import select_best_combination, score_combination

__all__ = [
    "check_solvable_conflict",
    "analyze_conflict_dimensions",
    "extract_alpha_values",
    "check_field_conflict",
    "select_best_combination",
    "score_combination",
]
