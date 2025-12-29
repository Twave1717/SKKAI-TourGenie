#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _categorize(err: str) -> str:
    s = (err or "").lower()
    if "parse_error" in s:
        return "parse_error"
    if "invalid field" in s:
        return "invalid_field"
    if "invalid op" in s:
        return "invalid_op"
    if "invalid cuisine" in s:
        return "invalid_cuisine"
    if "invalid room type" in s:
        return "invalid_room_type"
    if "invalid room rule" in s:
        return "invalid_room_rule"
    if "invalid ground mode" in s:
        return "invalid_ground_mode"
    if "persona_index" in s:
        return "persona_index_mismatch"
    if "must be a list" in s or "must be objects" in s:
        return "shape_error"
    if "no cuisines in snapshot" in s:
        return "snapshot_missing_cuisines"
    if "no room types in snapshot" in s:
        return "snapshot_missing_room_types"
    if "no room rules in snapshot" in s:
        return "snapshot_missing_room_rules"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", required=True, help="meta/ directory produced by run_stage2.py --write_meta")
    ap.add_argument("--out", default="", help="write stats json")
    ap.add_argument("--md", default="", help="write markdown summary")
    args = ap.parse_args()

    meta_dir = Path(args.meta_dir).expanduser().resolve()
    files = sorted([p for p in meta_dir.glob("*.json") if p.is_file()])
    if not files:
        raise SystemExit(f"No meta json files found in {meta_dir}")

    attempts: List[int] = []
    first_pass_ok = 0
    error_types = Counter()
    attempt_hist = Counter()

    missing_validation = 0

    for p in files:
        m = _load_json(p)
        v = m.get("validation")
        if not isinstance(v, dict):
            missing_validation += 1
            continue

        a = v.get("attempts")
        if a is None and isinstance(v.get("attempt_logs"), list):
            a = len(v["attempt_logs"])
        try:
            a_i = int(a)
        except Exception:
            a_i = None

        if a_i is not None:
            attempts.append(a_i)
            attempt_hist[str(a_i)] += 1

        if v.get("first_pass_valid") is True:
            first_pass_ok += 1

        # collect errors from invalid attempts
        logs = v.get("attempt_logs") or []
        if isinstance(logs, list):
            for step in logs:
                if not isinstance(step, dict):
                    continue
                if step.get("valid") is True:
                    continue
                errs = step.get("errors") or []
                if isinstance(errs, list):
                    for e in errs:
                        error_types[_categorize(str(e))] += 1

    n = len(files) - missing_validation if (len(files) - missing_validation) > 0 else len(files)
    stats = {
        "n_meta_files": len(files),
        "n_with_validation": len(files) - missing_validation,
        "missing_validation": missing_validation,
        "one_pass_valid_rate": (first_pass_ok / n) if n else 0.0,
        "avg_attempts": float(statistics.mean(attempts)) if attempts else None,
        "avg_retries": float(statistics.mean([a - 1 for a in attempts])) if attempts else None,
        "attempt_hist": dict(attempt_hist),
        "top_error_types": error_types.most_common(20),
    }

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.out:
        out_p = Path(args.out).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.md:
        md_p = Path(args.md).expanduser().resolve()
        md_p.parent.mkdir(parents=True, exist_ok=True)
        md = []
        md.append("# B. Validator loop efficiency")
        md.append("")
        md.append(f"- Meta files: **{stats['n_meta_files']}**")
        md.append(f"- With validation info: **{stats['n_with_validation']}**")
        md.append(f"- 1-pass valid rate: **{stats['one_pass_valid_rate']:.3f}**")
        if stats["avg_attempts"] is not None:
            md.append(f"- Avg attempts: **{stats['avg_attempts']:.2f}** (avg retries: {stats['avg_retries']:.2f})")
        md.append("")
        md.append("## Attempt histogram")
        md.append("")
        for k, v in sorted(stats["attempt_hist"].items(), key=lambda x: int(x[0])):
            md.append(f"- attempts={k}: {v}")
        md.append("")
        md.append("## Top error types")
        md.append("")
        for et, c in stats["top_error_types"]:
            md.append(f"- {et}: {c}")
        md_p.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
