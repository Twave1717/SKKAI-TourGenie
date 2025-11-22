from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer

from . import TRIPCRAFT_ROOT, default_csv_for_day


def _load_env_defaults() -> None:
    """Populate os.environ using values from the repo-level .env file if needed."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_defaults()

app = typer.Typer(help="TripCraft runner wrapper that mirrors the TravelPlanner CLI.")


def _run_subprocess(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    typer.echo(f"[TripCraft] Running: {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _write_mini_csv(src: Path, dest: Path, rows: int = 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", newline="", encoding="utf-8") as fin, dest.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError(f"Source CSV '{src}' is empty.")
        writer.writerow(header)
        for idx, row in enumerate(reader):
            if idx >= rows:
                break
            writer.writerow(row)


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
    test_mini: bool = typer.Option(
        False,
        "--test-mini",
        help="Run 20 samples for each of 3day/5day/7day splits sequentially and group their outputs.",
    ),
) -> None:
    """
    Execute the TripCraft planning pipeline via the legacy run.sh script and (optionally) run evaluation.
    """

    tripcraft_dir = tripcraft_dir.resolve()
    run_script = tripcraft_dir / "run.sh"
    eval_script = tripcraft_dir / "evaluation" / "eval.py"
    db_dir = tripcraft_dir / "TripCraft_database"

    if not run_script.exists():
        raise FileNotFoundError(
            f"run.sh not found at {run_script}. Did you move or initialize the TripCraft folder?"
        )

    def resolve_output_dir(target: Optional[Path]) -> Path:
        if target is not None:
            path = Path(target)
        else:
            env_output = os.environ.get("TRIPCRAFT_OUTPUT_DIR")
            path = Path(env_output) if env_output else tripcraft_dir / "output"
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    def resolve_csv_path(requested_day: str, explicit: Optional[Path]) -> Path:
        if explicit is not None:
            csv_path = explicit if explicit.is_absolute() else (Path.cwd() / explicit).resolve()
        else:
            csv_path = default_csv_for_day(requested_day, base_dir=tripcraft_dir)
            if not csv_path.is_absolute():
                csv_path = (tripcraft_dir / csv_path).resolve()
        return csv_path

    def invoke_tripcraft(
        actual_day: str,
        actual_set_type: str,
        csv_path: Path,
        out_dir: Path,
        eval_target: Optional[Path],
        tqdm_position: Optional[int] = None,
    ) -> None:
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
        env["DAY"] = actual_day
        env["SET_TYPE"] = actual_set_type
        env["STRATEGY"] = strategy
        env["CSV_FILE"] = str(csv_path)
        env["OUTPUT_DIR"] = str(out_dir)
        if tqdm_position is not None:
            env["TRIPCRAFT_TQDM_POSITION"] = str(tqdm_position)
        else:
            env.pop("TRIPCRAFT_TQDM_POSITION", None)
        if "OPENAI_API_KEY" not in env:
            raise typer.BadParameter("OPENAI_API_KEY must be set in the environment to run TripCraft.")

        _run_subprocess(["bash", str(run_script.name)], cwd=tripcraft_dir, env=env)

        if eval_target is not None:
            if not eval_script.exists():
                raise FileNotFoundError(f"TripCraft evaluation script not found at {eval_script}")
            eval_cmd = [
                "python",
                str(eval_script),
                "--set_type",
                evaluation_split,
                "--evaluation_file_path",
                str(eval_target),
            ]
            _run_subprocess(eval_cmd, cwd=tripcraft_dir / "evaluation", env=env)

    out_dir = resolve_output_dir(output_dir)
    typer.echo(f"[TripCraft] Output directory resolved to: {out_dir}")

    if test_mini:
        if evaluation_file is not None:
            raise typer.BadParameter("--test-mini currently does not support --evaluation-file.")
        typer.echo("[TripCraft] Running test-mini (20 samples each for 3/5/7-day splits).")
        with tempfile.TemporaryDirectory(prefix="tripcraft_mini_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            futures = []
            day_order = ("3day", "5day", "7day")
            with ThreadPoolExecutor(max_workers=len(day_order)) as executor:
                for pos, mini_day in enumerate(day_order):
                    base_csv = resolve_csv_path(mini_day, None)
                    mini_csv = tmpdir_path / f"{Path(base_csv).stem}_mini.csv"
                    _write_mini_csv(base_csv, mini_csv, rows=20)
                    nested_set_type = str(Path(set_type) / mini_day)
                    typer.echo(
                        f"[TripCraft] Dispatching {mini_day} mini-run -> set_type={nested_set_type}, csv={mini_csv}"
                    )
                    futures.append(
                        executor.submit(
                            invoke_tripcraft,
                            actual_day=mini_day,
                            actual_set_type=nested_set_type,
                            csv_path=mini_csv,
                            out_dir=out_dir,
                            eval_target=None,
                            tqdm_position=pos,
                        )
                    )
                for fut in as_completed(futures):
                    fut.result()
        typer.echo("[TripCraft] Completed test-mini runs.")
        return

    explicit_csv = Path(csv_file) if csv_file is not None else None
    csv_path = resolve_csv_path(day, explicit_csv)
    invoke_tripcraft(day, set_type, csv_path, out_dir, evaluation_file)
    typer.echo("[TripCraft] Completed.")


if __name__ == "__main__":
    app()
