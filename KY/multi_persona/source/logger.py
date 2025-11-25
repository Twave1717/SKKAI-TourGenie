"""
실행 산출물을 파일로 저장하는 헬퍼.
"""

import json
from pathlib import Path
from typing import Dict

from .structures import MeetingOutcome


def persist_outcome(outcome: MeetingOutcome, sample_id: str, base_dir: str = "runs") -> Dict[str, str]:
    """
    MeetingOutcome의 raw 로그와 이슈/제약/계획을 파일로 저장한다.
    반환: 저장된 파일 경로 dict.
    """
    base = Path(base_dir) / sample_id
    base.mkdir(parents=True, exist_ok=True)
    paths = {}
    # issues
    issues_path = base / "issues.json"
    issues_path.write_text(json.dumps([i.__dict__ for i in outcome.issues], ensure_ascii=False, indent=2))
    paths["issues"] = str(issues_path)
    # natural constraints
    nc_path = base / "natural_constraints.json"
    nc_path.write_text(json.dumps([c.__dict__ for c in outcome.natural_constraints], ensure_ascii=False, indent=2))
    paths["natural_constraints"] = str(nc_path)
    # formal constraints
    fc_path = base / "formal_constraints.json"
    fc_path.write_text(json.dumps([c.__dict__ for c in outcome.formal_constraints], ensure_ascii=False, indent=2))
    paths["formal_constraints"] = str(fc_path)
    # raw logs
    for key, value in outcome.logs.items():
        p = base / f"{key}.txt"
        p.write_text(str(value))
        paths[key] = str(p)
    return paths
