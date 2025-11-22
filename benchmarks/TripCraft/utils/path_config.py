import os
from functools import lru_cache

TRIPCRAFT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(TRIPCRAFT_ROOT, "..", ".."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path

    candidate_repo = os.path.abspath(os.path.join(REPO_ROOT, path))
    if os.path.exists(candidate_repo):
        return candidate_repo

    return os.path.abspath(os.path.join(TRIPCRAFT_ROOT, path))


@lru_cache(maxsize=1)
def get_db_root() -> str:
    """
    Resolve the TripCraft database root directory.

    Priority:
    1. Environment variable TRIPCRAFT_DB_ROOT (relative paths resolved from TripCraft root).
    2. <TripCraft_root>/TripCraft_database.
    """
    env_root = os.environ.get("TRIPCRAFT_DB_ROOT")
    if env_root:
        return _resolve_path(env_root)

    return os.path.join(TRIPCRAFT_ROOT, "TripCraft_database")


def get_db_path(*relative_parts: str) -> str:
    """
    Construct an absolute path inside the TripCraft database directory.
    """
    return os.path.join(get_db_root(), *relative_parts)
