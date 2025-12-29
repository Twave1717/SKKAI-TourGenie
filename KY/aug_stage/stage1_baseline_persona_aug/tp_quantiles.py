from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def load_tp_pppn_quantiles(artifacts_dir: Path) -> Optional[Dict[str, float]]:
    """Load TravelPlanner pppn quantiles from precomputed artifacts.

    We prefer validation stats if present; else train stats; else None.
    Expected file names (from your pipeline):
      - travelplanner_stats_validation__validation.json
      - travelplanner_stats_train__train.json
    """
    candidates = [
        artifacts_dir / "travelplanner_stats_validation__validation.json",
        artifacts_dir / "travelplanner_stats_train__train.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            q = (obj.get("per_person_per_night") or obj.get("budget_per_person_per_night") or {}).get("quantiles") or {}
            # your artifacts use keys like "0.25", "0.5", "0.75"
            if all(k in q and q[k] is not None for k in ("0.25", "0.5", "0.75")):
                return {"q25": float(q["0.25"]), "q50": float(q["0.5"]), "q75": float(q["0.75"])}
        except Exception:
            continue
    return None
