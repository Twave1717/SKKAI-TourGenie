from __future__ import annotations

from pathlib import Path


OFFICIAL_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = OFFICIAL_ROOT / "database"
EVALUATION_DIR = OFFICIAL_ROOT / "evaluation"
TOOLS_DIR = OFFICIAL_ROOT / "tools"
UTILS_DIR = OFFICIAL_ROOT / "utils"


def resolve_database_path(*parts: str) -> Path:
    return DATABASE_DIR.joinpath(*parts)


def resolve_tool_path(*parts: str) -> Path:
    return TOOLS_DIR.joinpath(*parts)
