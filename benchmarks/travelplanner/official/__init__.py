"""Official TravelPlanner evaluation package (ported)."""

import sys
from pathlib import Path

from .path_utils import OFFICIAL_ROOT, DATABASE_DIR, EVALUATION_DIR, TOOLS_DIR, UTILS_DIR


def _ensure_hf_shims() -> None:
    try:
        import huggingface_hub
    except ModuleNotFoundError:
        return

    if not hasattr(huggingface_hub, "HfFolder"):
        class _CompatHfFolder:
            _token_path = Path.home() / ".huggingface" / "token"

            @classmethod
            def path_token(cls):
                return cls._token_path

            @classmethod
            def get_token(cls):
                try:
                    return cls.path_token().read_text(encoding="utf-8").strip()
                except FileNotFoundError:
                    return None

            @classmethod
            def save_token(cls, token: str) -> None:
                path = cls.path_token()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(token.strip(), encoding="utf-8")

            @classmethod
            def delete_token(cls) -> None:
                try:
                    cls.path_token().unlink()
                except FileNotFoundError:
                    pass

        huggingface_hub.HfFolder = _CompatHfFolder

    if not hasattr(huggingface_hub, "whoami"):
        def _compat_whoami(*_args, **_kwargs):
            token = huggingface_hub.HfFolder.get_token()
            return {"token": token} if token else {}

        huggingface_hub.whoami = _compat_whoami


_ensure_hf_shims()

# Ensure legacy modules that expect `utils.*` absolute imports keep working
_official_root_str = str(OFFICIAL_ROOT)
if _official_root_str not in sys.path:
    sys.path.insert(0, _official_root_str)

__all__ = [
    "OFFICIAL_ROOT",
    "DATABASE_DIR",
    "EVALUATION_DIR",
    "TOOLS_DIR",
    "UTILS_DIR",
]
