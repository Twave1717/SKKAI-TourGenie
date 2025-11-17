import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_db_root() -> str:
    """
    Resolve the TripCraft database root directory.

    Priority:
    1. Environment variable TRIPCRAFT_DB_ROOT.
    2. <repo_root>/TripCraft_database
    """
    env_root = os.environ.get("TRIPCRAFT_DB_ROOT")
    if env_root:
        return env_root

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "TripCraft_database")


def get_db_path(*relative_parts: str) -> str:
    """
    Construct an absolute path inside the TripCraft database directory.
    """
    return os.path.join(get_db_root(), *relative_parts)
