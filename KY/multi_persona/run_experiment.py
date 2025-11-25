"""
실험 및 실행 스크립트(단일 엔트리 포인트).
- TripCraft CSV에서 단일 행(row_index)만 사용.
- 참여자 수(1~4)에 따라 페르소나 불만/제약 생성 및 병합/토론 실행.
- mode: single(1인), aggregate(병합), consensus(토론).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

try:
    from .source.config import BenchmarkConfig, MeetingConfig, PersonaConfig
    from .source.llm import OpenAIClient, stub_client
    from .source.meeting import MeetingRunner
    from .source.structures import PlanInput
    from .source.tripcraft_parser import load_tripcraft_plan
except ImportError:
    # python -m run_experiment 실행 시 경로 보정
    BASE = Path(__file__).resolve().parent
    sys.path.append(str(BASE.parent.parent))
    from KY.multi_persona.source.config import BenchmarkConfig, MeetingConfig, PersonaConfig  # type: ignore
    from KY.multi_persona.source.llm import OpenAIClient, stub_client  # type: ignore
    from KY.multi_persona.source.meeting import MeetingRunner  # type: ignore
    from KY.multi_persona.source.structures import PlanInput  # type: ignore
    from KY.multi_persona.source.tripcraft_parser import load_tripcraft_plan  # type: ignore


def find_project_root() -> Path:
    candidates = [Path.cwd()] + list(Path(__file__).resolve().parents)
    for base in candidates:
        if (base / "benchmarks" / "TripCraft" / "tripcraft").exists():
            return base
        if (base / "pyproject.toml").exists():
            return base
    return Path(__file__).resolve().parents[1]


ROOT_DIR = find_project_root()
DATA_DIR = ROOT_DIR / "benchmarks" / "TripCraft" / "tripcraft"


def list_csv_files(data_dir: Path = DATA_DIR) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


def load_persona_presets() -> List[PersonaConfig]:
    preset_path = ROOT_DIR / "KY" / "multi_persona" / "source" / "prompt_templates" / "persona_preset.jsonl"
    if not preset_path.exists():
        raise FileNotFoundError(f"persona preset file not found: {preset_path}")
    personas: List[PersonaConfig] = []
    with preset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            personas.append(
                PersonaConfig(
                    name=data.get("name", ""),
                    role=data.get("role", ""),
                    values=data.get("values", []),
                    focus_dimensions=data.get("focus_dimensions", []),
                    profile=data.get("profile", ""),
                    max_issues=data.get("max_issues", 5),
                )
            )
    if not personas:
        raise RuntimeError(f"persona preset file is empty: {preset_path}")
    return personas


def build_runner(plan_path: str, participant_count: int, use_openai: bool = False) -> MeetingRunner:
    llm = OpenAIClient(model="gpt-4.1") if use_openai else stub_client()
    personas_all = load_persona_presets()
    if participant_count < 1 or participant_count > len(personas_all):
        raise ValueError(f"participants는 1~{len(personas_all)} 사이여야 합니다.")
    personas = random.sample(personas_all, participant_count)
    meeting_cfg = MeetingConfig(
        personas=personas,
        moderator="Moderator",
        top_constraints=10,
        aggregation_threshold=0.0,
    )
    bench_cfg = BenchmarkConfig(
        name="TripCraft",
        schema_hint="- day: 1~N\n- segments: transport_type, duration_minutes, distance_km\n- meals: breakfast/lunch/dinner 여부 등",
        plan_loader="tripcraft.load_plan",
        default_plan_path=plan_path,
    )
    return MeetingRunner(meeting_cfg=meeting_cfg, bench_cfg=bench_cfg, llm=llm)


def run_experiment(plan_path: str, row_index: int, participants: int, mode: str, use_openai: bool = False):
    runner = build_runner(plan_path, participant_count=participants, use_openai=use_openai)
    plan_payload = load_tripcraft_plan(plan_path, row_index=row_index)
    sample_id = f"{Path(plan_path).stem}_row{row_index}_p{participants}_{mode}"
    plan_input = PlanInput(
        user_query="사용자 쿼리 예시",
        predefined_constraints=["벤치 기본 제약"],
        plan_path=plan_path,
        plan_payload=plan_payload,
        sample_id=sample_id,
    )
    if mode == "single":
        return runner.run_m2(plan_input, persona=runner.meeting_cfg.personas[0])
    if mode == "aggregate":
        return runner.run_m3(plan_input, participant_count=participants)
    return runner.run_m4(plan_input, participant_count=participants)


def main() -> None:
    parser = argparse.ArgumentParser(description="TripCraft 1건씩 페르소나 합의/불만 생성 실험")
    parser.add_argument("--files", nargs="*", help="특정 CSV 파일명 지정 (기본 tripcraft 디렉터리 전체)")
    parser.add_argument("--row-index", type=int, default=0, help="각 CSV에서 사용할 행(기본 0)")
    parser.add_argument("--participants", type=int, default=None, help="사용할 페르소나 수(1~4)")
    parser.add_argument("--mode", type=str, default="consensus", choices=["single", "aggregate", "consensus"], help="single=1인, aggregate=병합, consensus=토론")
    parser.add_argument("--use-openai", action="store_true", help="gpt-4.1 사용")
    args = parser.parse_args()

    # participants 기본값: single 모드면 1, 그 외 4
    participants = args.participants
    if participants is None:
        participants = 1 if args.mode == "single" else 4
    if participants < 1 or participants > 4:
        raise ValueError("participants는 1~4 사이여야 합니다.")

    if args.files:
        targets = [DATA_DIR / name for name in args.files]
    else:
        targets = list_csv_files()

    missing = [p for p in targets if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    if not args.use_openai:
        sys.stderr.write("[WARN] --use-openai 미사용: 스텁 LLM으로 실행됩니다.\n")
    elif not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    modes = {"single": "single", "aggregate": "aggregate", "consensus": "consensus"}
    with tqdm(total=len(targets), desc="runs") as bar:
        for path in targets:
            outcome = run_experiment(
                plan_path=str(path),
                row_index=args.row_index,
                participants=participants,
                mode=modes[args.mode],
                use_openai=args.use_openai,
            )
            bar.update(1)
            print(f"[DONE] {path.name} -> saved_files: {outcome.logs.get('saved_files')}")


if __name__ == "__main__":
    main()
