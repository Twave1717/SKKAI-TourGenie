from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

from . import TRIPCRAFT_ROOT


@contextmanager
def _temp_cwd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


@lru_cache(maxsize=1)
def _load_tripcraft_eval_module():
    eval_path = TRIPCRAFT_ROOT / "evaluation" / "eval.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"TripCraft evaluator not found at {eval_path}")

    spec = importlib.util.spec_from_file_location("tripcraft_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for TripCraft evaluator at {eval_path}")

    module = importlib.util.module_from_spec(spec)
    with _temp_cwd(eval_path.parent):
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def run_tripcraft_eval(set_type: str, evaluation_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    module = _load_tripcraft_eval_module()
    scores, details = module.eval_score(set_type, file_path=str(evaluation_file))  # type: ignore[attr-defined]
    return scores, details
