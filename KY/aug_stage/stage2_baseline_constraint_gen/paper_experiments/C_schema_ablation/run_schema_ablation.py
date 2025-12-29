#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List


SCHEMA_MODES = ["none", "schema_only", "schema_alias", "schema_alias_snapshot"]


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--db_root", required=True, help="TravelPlanner official DB root")
    ap.add_argument("--stage1_dir", required=True, help="Stage1 outputs (json files)")
    ap.add_argument("--out_dir", default="outputs_stage2", help="base output dir for stage2 runs")
    ap.add_argument("--artifacts_dir", default="artifacts", help="where to write summary json/md")
    ap.add_argument("--max_records", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--strict_values", action="store_true", help="strict categorical checks against snapshot (recommended)")
    args = ap.parse_args()

    here = Path(__file__).resolve()
    project = here.parents[2]  # stage2_constraint_gen/
    run_stage2 = project / "run_stage2.py"
    a_script = project / "paper_experiments" / "A_diversity" / "compute_diversity_stats.py"
    b_script = project / "paper_experiments" / "B_validator" / "compute_validator_efficiency.py"

    artifacts_dir = (project / args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"split": args.split, "modes": {}}

    for mode in SCHEMA_MODES:
        # 1) run Stage2
        cmd = [
            "python3",
            str(run_stage2),
            "--split",
            args.split,
            "--schema_mode",
            mode,
            "--db_root",
            args.db_root,
            "--stage1_dir",
            args.stage1_dir,
            "--out_dir",
            args.out_dir,
            "--max_records",
            str(args.max_records),
            "--workers",
            str(args.workers),
            "--temperature",
            str(args.temperature),
            "--max_retries",
            str(args.max_retries),
            "--write_meta",
        ]
        if args.strict_values:
            cmd.append("--strict_values")
        _run(cmd)

        # output directory convention: outputs_stage2/<split>/<schema_mode>/
        out_run_dir = (project / args.out_dir / args.split / mode).resolve()

        # 2) A: diversity stats
        a_out = artifacts_dir / f"A_diversity__{mode}.json"
        _run(["python3", str(a_script), "--in_dir", str(out_run_dir), "--out", str(a_out)])

        # 3) B: validator efficiency
        b_out = artifacts_dir / f"B_validator__{mode}.json"
        _run(["python3", str(b_script), "--meta_dir", str(out_run_dir / "meta"), "--out", str(b_out)])

        summary["modes"][mode] = {
            "out_dir": str(out_run_dir),
            "A_diversity": _read_json(a_out),
            "B_validator": _read_json(b_out),
        }

    out_summary = artifacts_dir / "C_schema_ablation_summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nWrote:", out_summary)


if __name__ == "__main__":
    main()
