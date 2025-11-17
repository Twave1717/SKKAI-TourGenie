from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import typer

from . import TRIPCRAFT_ROOT, default_csv_for_day

app = typer.Typer(help="TripCraft runner wrapper that mirrors the TravelPlanner CLI.")


def _run_subprocess(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    typer.echo(f"[TripCraft] Running: {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


@app.command()
def run(
    provider: str = typer.Option("openai", help="Currently informational; TripCraft uses LangChain ChatOpenAI."),
    model: str = typer.Option("gpt-4.1-mini", "--model-name", help="LLM to use for planning."),
    day: str = typer.Option("3day", help="Trip length split (3day/5day/7day)."),
    set_type: str = typer.Option(
        "3day_gpt4o_orig",
        help="Output folder name used inside TripCraft/output (matches original scripts).",
    ),
    strategy: str = typer.Option(
        "direct_og",
        help="TripCraft planning strategy. Matches agents/prompts definitions.",
    ),
    csv_file: Optional[Path] = typer.Option(
        None,
        help="Custom CSV file containing reference information. Defaults to the official TripCraft splits.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Destination directory for TripCraft outputs. Defaults to benchmarks/TripCraft/output.",
    ),
    evaluation_file: Optional[Path] = typer.Option(
        None,
        help=(
            "Optional JSONL file to evaluate after planning. "
            "Must conform to postprocess/sample_evaluation_format.jsonl."
        ),
    ),
    evaluation_split: str = typer.Option(
        "3d",
        help="Argument passed to evaluation/eval.py --set_type when evaluation_file is provided.",
    ),
    tripcraft_dir: Path = typer.Option(
        TRIPCRAFT_ROOT,
        help="Path to the upstream TripCraft repository (default: benchmarks/TripCraft).",
    ),
) -> None:
    """
    Execute the TripCraft planning pipeline via the legacy run.sh script and (optionally) run evaluation.
    """

    tripcraft_dir = tripcraft_dir.resolve()
    csv_path = csv_file or default_csv_for_day(day, base_dir=tripcraft_dir)
    out_dir = Path(output_dir) if output_dir else tripcraft_dir / "output"
    run_script = tripcraft_dir / "run.sh"
    eval_script = tripcraft_dir / "evaluation" / "eval.py"
    db_dir = tripcraft_dir / "TripCraft_database"

    if not run_script.exists():
        raise FileNotFoundError(
            f"run.sh not found at {run_script}. Did you move or initialize the TripCraft folder?"
        )

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    env = os.environ.copy()

    db_root = env.get("TRIPCRAFT_DB_ROOT")
    if db_root:
        db_root_path = Path(db_root).expanduser()
        if not db_root_path.is_absolute():
            db_root_path = (Path.cwd() / db_root_path).resolve()
        env["TRIPCRAFT_DB_ROOT"] = str(db_root_path)
    else:
        env["TRIPCRAFT_DB_ROOT"] = str(db_dir.resolve())

    env["MODEL_NAME"] = model
    env["DAY"] = day
    env["SET_TYPE"] = set_type
    env["STRATEGY"] = strategy
    env["CSV_FILE"] = str(csv_path)
    env["OUTPUT_DIR"] = str(out_dir)
    if "OPENAI_API_KEY" not in env:
        raise typer.BadParameter("OPENAI_API_KEY must be set in the environment to run TripCraft.")

    _run_subprocess(["bash", str(run_script.name)], cwd=tripcraft_dir, env=env)

    if evaluation_file is not None:
        if not eval_script.exists():
            raise FileNotFoundError(f"TripCraft evaluation script not found at {eval_script}")
        eval_cmd = [
            "python",
            str(eval_script),
            "--set_type",
            evaluation_split,
            "--evaluation_file_path",
            str(evaluation_file),
        ]
        _run_subprocess(eval_cmd, cwd=tripcraft_dir / "evaluation", env=env)

    typer.echo("[TripCraft] Completed.")


if __name__ == "__main__":
    app()
