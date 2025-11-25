"""
프롬프트 템플릿 로더. 파일이 없으면 경고 후 종료한다.
"""

import importlib.resources as pkg_resources
import sys
from typing import Sequence


def load_template(name: str) -> str:
    pkg = __package__ + ".prompt_templates"
    try:
        file = pkg_resources.files(pkg).joinpath(name)
    except Exception as exc:
        sys.stderr.write(f"[WARN] template load failed ({name}): {exc}\n")
        sys.exit(1)
    if not file.exists():
        sys.stderr.write(f"[WARN] template not found: {file}\n")
        sys.exit(1)
    try:
        return file.read_text(encoding="utf-8")
    except Exception as exc:
        sys.stderr.write(f"[WARN] template read error ({file}): {exc}\n")
        sys.exit(1)


PERSONA_TEMPLATE = load_template("persona_v1.txt")
AGGREGATOR_TEMPLATE = load_template("aggregator_v1.txt")
ENGINEER_TEMPLATE = load_template("engineer_v1.txt")
MODERATOR_TEMPLATE = load_template("moderator_v1.txt")


def render_persona_prompt(
    name: str,
    role: str,
    values: Sequence[str],
    focus_dimensions: Sequence[str],
    profile: str,
    max_issues: int = 5,
) -> str:
    return PERSONA_TEMPLATE.format(
        name=name,
        role=role,
        values=", ".join(values),
        focus_dimensions=", ".join(focus_dimensions),
        profile=profile,
        max_issues=max_issues,
    )


def render_engineer_prompt(schema_hint: str, constraint_id: str, constraint_nl: str) -> str:
    return ENGINEER_TEMPLATE.format(
        schema_hint=schema_hint,
        constraint_id=constraint_id,
        constraint_nl=constraint_nl,
    )


def render_moderator_prompt() -> str:
    return MODERATOR_TEMPLATE
