"""Utilities to download TravelPlanner datasets and optional database assets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import shutil
import tempfile
import zipfile

import requests
import typer
from datasets import load_dataset


app = typer.Typer(
    add_completion=False,
    help="Download TravelPlanner dataset splits and (optionally) the official database assets.",
)

GOOGLE_DRIVE_FILE_ID = "1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE"
GOOGLE_DRIVE_DOWNLOAD_URL = "https://docs.google.com/uc?export=download"
DOWNLOAD_CHUNK_SIZE = 32768
DATABASE_REQUIRED_FILES: tuple[tuple[str, str], ...] = (
    ("flights", "clean_Flights_2022.csv"),
    ("restaurants", "clean_restaurant_2022.csv"),
    ("accommodations", "clean_accommodations_2022.csv"),
    ("googleDistanceMatrix", "distance.csv"),
)


def _database_is_complete(dest: Path) -> bool:
    return all((dest / subdir / filename).exists() for subdir, filename in DATABASE_REQUIRED_FILES)


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _download_from_google_drive(file_id: str, destination: Path) -> None:
    session = requests.Session()
    params = {"id": file_id}
    response = session.get(GOOGLE_DRIVE_DOWNLOAD_URL, params=params, stream=True)
    response.raise_for_status()

    token = _get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(GOOGLE_DRIVE_DOWNLOAD_URL, params=params, stream=True)
        response.raise_for_status()

    with destination.open("wb") as handle:
        for chunk in response.iter_content(DOWNLOAD_CHUNK_SIZE):
            if chunk:
                handle.write(chunk)


def _prepare_database(
    dest: Path,
    *,
    force: bool,
    local_zip: Optional[Path],
) -> None:
    if dest.exists():
        if _database_is_complete(dest) and not force:
            typer.echo(f"Database already present at {dest}. Use --force-db to overwrite.")
            return
        typer.echo(f"Removing existing database directory at {dest}")
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "database.zip"

        resolved_zip: Optional[Path] = None
        if local_zip:
            resolved_zip = local_zip
        else:
            cwd_zip = Path.cwd() / "database.zip"
            if cwd_zip.exists():
                resolved_zip = cwd_zip
            else:
                for candidate in Path.cwd().rglob("database.zip"):
                    resolved_zip = candidate
                    break

        if resolved_zip:
            if not resolved_zip.exists():
                raise typer.BadParameter(f"지정한 database 아카이브가 존재하지 않습니다: {local_zip}")
            typer.echo(f"Using local database archive: {resolved_zip}")
            shutil.copy2(resolved_zip, archive_path)
        else:
            typer.echo("Downloading TravelPlanner database (Google Drive)")
            _download_from_google_drive(GOOGLE_DRIVE_FILE_ID, archive_path)

        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(tmp_dir)

        extracted_root = tmp_dir / "database"
        if not extracted_root.exists():
            raise RuntimeError("Downloaded archive did not contain a 'database/' directory.")

        shutil.copytree(extracted_root, dest)

    if not _database_is_complete(dest):
        raise RuntimeError(
            f"Database setup incomplete at {dest}. Required files: "
            f"{', '.join(str(Path(subdir) / filename) for subdir, filename in DATABASE_REQUIRED_FILES)}"
        )

    typer.echo(f"Database assets ready at {dest}")

    repo_db = Path("benchmarks/travelplanner/official/database")
    try:
        if repo_db.resolve() != dest.resolve():
            if repo_db.exists():
                shutil.rmtree(repo_db)
            shutil.copytree(dest, repo_db)
            typer.echo(f"Mirrored database to {repo_db}")
    except FileNotFoundError:
        typer.echo(f"경고: {repo_db} 경로를 생성할 수 없습니다.")


@app.command()
def main(
    split: str = typer.Option(
        "test",
        help="Dataset split to download (train/test/validation)",
    ),
    dest: Path = typer.Option(
        Path("data/travelplanner"),
        help="Destination directory for the downloaded data",
    ),
    dataset: str = typer.Option(
        "osunlp/TravelPlanner",
        help="Hugging Face dataset name",
    ),
    download_db: bool = typer.Option(
        False,
        "--download-db/--skip-db",
        help="Also download and unpack the official TravelPlanner database assets.",
    ),
    db_dest: Path = typer.Option(
        Path("benchmarks/travelplanner/official/database"),
        help="Destination directory for the TravelPlanner database assets.",
    ),
    force_db: bool = typer.Option(
        False,
        help="Force re-download and overwrite any existing database assets.",
    ),
    db_zip: Optional[Path] = typer.Option(
        None,
        help="Path to a pre-downloaded database.zip archive. If provided, the download step is skipped.",
    ),
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {dataset}:{split} -> {dest}")

    dataset_dict = load_dataset(dataset, name=split)
    if hasattr(dataset_dict, "keys"):
        selected_split = split if split in dataset_dict else next(iter(dataset_dict.keys()))
        if selected_split != split:
            typer.echo(f"요청한 split '{split}'을 찾지 못해 '{selected_split}' split을 저장합니다.")
        ds = dataset_dict[selected_split]
    else:
        selected_split = split
        ds = dataset_dict

    ds = ds.map(lambda example, idx: {"id": f"{selected_split}_{idx:05d}"}, with_indices=True)

    output_path = dest / f"{selected_split}.jsonl"
    ds.to_json(str(output_path))

    typer.echo(f"Saved {len(ds)} examples to {output_path}")

    if download_db:
        _prepare_database(db_dest, force=force_db, local_zip=db_zip)


if __name__ == "__main__":
    app()
