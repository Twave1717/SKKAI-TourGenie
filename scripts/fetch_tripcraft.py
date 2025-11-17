"""Utilities to prepare TripCraft database assets inside the benchmarks folder."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import typer

DEFAULT_TRIPCRAFT_ROOT = Path("benchmarks/TripCraft")
DEFAULT_DB_DIR = DEFAULT_TRIPCRAFT_ROOT / "TripCraft_database"
DEFAULT_ARCHIVE_CANDIDATES = (
    Path("TripCraft_database.zip"),
    DEFAULT_TRIPCRAFT_ROOT / "TripCraft_database.zip",
)

app = typer.Typer(add_completion=False, help="Prepare TripCraft database assets.")


def _resolve_archive(archive: Optional[Path]) -> Path:
    if archive:
        if not archive.exists():
            raise typer.BadParameter(f"지정한 TripCraft database 아카이브가 존재하지 않습니다: {archive}")
        return archive

    for candidate in DEFAULT_ARCHIVE_CANDIDATES:
        if candidate.exists():
            return candidate

    raise typer.BadParameter(
        "TripCraft database archive를 찾을 수 없습니다. "
        "TripCraft 공식 배포본에서 TripCraft_database.zip을 내려받아 --archive 경로를 지정하세요."
    )


def _prepare_tripcraft_db(archive: Path, dest: Path, *, force: bool) -> None:
    if dest.exists():
        if force:
            typer.echo(f"Removing existing TripCraft database at {dest}")
            shutil.rmtree(dest)
        else:
            typer.echo(f"TripCraft database already exists at {dest}. Use --force to overwrite.")
            return

    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_copy = tmp_dir / "tripcraft_db.zip"
        shutil.copy2(archive, archive_copy)

        with zipfile.ZipFile(archive_copy, "r") as zf:
            zf.extractall(tmp_dir)

        # 오리지널 TripCraft 아카이브는 TripCraft_database/ 루트를 포함하고 있음
        extracted_root = tmp_dir / "TripCraft_database"
        if not extracted_root.exists():
            raise RuntimeError("Archive did not contain a 'TripCraft_database/' directory.")

        shutil.copytree(extracted_root, dest)

    typer.echo(f"TripCraft database ready at {dest}")


@app.command()
def main(
    dest: Path = typer.Option(
        DEFAULT_DB_DIR,
        help="Directory where TripCraft_database will be placed.",
    ),
    archive: Optional[Path] = typer.Option(
        None,
        help="Path to TripCraft_database.zip. If omitted, common locations are searched automatically.",
    ),
    force: bool = typer.Option(
        False,
        help="Overwrite existing TripCraft database directory.",
    ),
) -> None:
    """
    Unpack the TripCraft database archive into the benchmarks folder so that run.sh can consume it.
    """

    resolved_archive = _resolve_archive(archive)
    typer.echo(f"Using TripCraft database archive: {resolved_archive}")
    _prepare_tripcraft_db(resolved_archive, dest=dest, force=force)


if __name__ == "__main__":
    app()
