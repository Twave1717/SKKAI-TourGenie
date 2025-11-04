from __future__ import annotations

import os
import json
import contextlib
import io
import importlib
import importlib.util
import types
from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, TypedDict, Tuple

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

if __package__ in (None, ""):
    import sys

    # 스크립트를 직접 실행할 때도 프로젝트 루트에서 절대 import가 되도록 보정
    root_path = Path(__file__).resolve().parents[2]
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    from benchmarks.travelplanner.schemas import Example, Prediction  # type: ignore
    from benchmarks.travelplanner.leaderboard import (  # type: ignore
        DEFAULT_METRIC_KEYS,
        build_header,
        update_leaderboard,
    )
    from benchmarks.travelplanner.postprocess.grounding import (  # type: ignore
        ground_prediction,
    )
    from benchmarks.travelplanner.official.pipeline import (  # type: ignore
        load_parsed_plans,
        run_official_parsing,
        write_generated_plan_files,
    )
else:
    from .schemas import Example, Prediction
    from .leaderboard import DEFAULT_METRIC_KEYS, build_header, update_leaderboard
    from .postprocess.grounding import ground_prediction
    from .official.pipeline import (
        load_parsed_plans,
        run_official_parsing,
        write_generated_plan_files,
    )


Provider = Literal[
    "openai",
    "google",
    "upstage",
    "travelplanner",
    "travelplanner(workflow)",
]
TRAVELPLANNER_PROVIDER = "travelplanner(workflow)"
TRAVELPLANNER_PROVIDER_ALIASES = {TRAVELPLANNER_PROVIDER, "travelplanner"}
MINI_LEADERBOARD_PATH = Path("leaderboards/TravelPlanner/mini.md")
MINI_LEADERBOARD_HEADER = build_header(
    DEFAULT_METRIC_KEYS, title="# TravelPlanner Mini Leaderboard"
)
BACKBONE_INSTANCE_COUNT = 4
ROOT_DIR = Path(__file__).resolve().parents[2]
OFFICIAL_TRAVELPLANNER_DIR = Path(__file__).resolve().parent / "official"
TRAVELPLANNER_AGENT_DIR = (OFFICIAL_TRAVELPLANNER_DIR / "agents").resolve()
TRAVELPLANNER_AGENT_MODULE = None
TRAVELPLANNER_AGENT_WORKERS = int(os.environ.get("TRAVELPLANNER_WORKERS", "4"))
PARSER_WORKER_COUNT = int(os.environ.get("TRAVELPLANNER_PARSER_WORKERS", "4"))
RUN_CONFIG_FILENAME = "run_config.json"
PARTIAL_PREDICTIONS_FILENAME = "travelplanner_agent_predictions.jsonl"


class GraphState(TypedDict, total=False):
    query: str
    reference_information: Any
    metadata: Dict[str, Any]
    messages: List[BaseMessage]
    prediction: str


def normalize_provider_name(provider: Provider) -> Provider:
    if provider in TRAVELPLANNER_PROVIDER_ALIASES:
        return TRAVELPLANNER_PROVIDER  # type: ignore[return-value]
    return provider


def is_travelplanner_provider(provider: str) -> bool:
    return provider == TRAVELPLANNER_PROVIDER


@contextlib.contextmanager
def _change_cwd(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _silence_stdout():
    return contextlib.redirect_stdout(io.StringIO())


class _TeeStream(io.TextIOBase):
    def __init__(self, *targets: io.TextIOBase):
        self.targets = targets

    def write(self, data: str) -> int:
        for target in self.targets:
            target.write(data)
        return len(data)

    def flush(self) -> None:
        for target in self.targets:
            target.flush()


@contextlib.contextmanager
def _capture_stdout(log_path: Optional[Path] = None):
    buffer = io.StringIO()
    log_file = None
    stream: io.TextIOBase
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        stream = _TeeStream(buffer, log_file)
    else:
        stream = buffer
    with contextlib.redirect_stdout(stream):
        yield buffer
    if log_file:
        log_file.close()


def _write_run_config(run_dir: Path, payload: Dict[str, Any]) -> None:
    config_path = run_dir / RUN_CONFIG_FILENAME
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    config_path = run_dir / RUN_CONFIG_FILENAME
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_partial_predictions(path: Path) -> Dict[str, Prediction]:
    predictions: Dict[str, Prediction] = {}
    if not path.exists():
        return predictions
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                prediction = Prediction(**payload)
            except Exception:
                continue
            predictions[prediction.id] = prediction
    return predictions


def _hydrate_predictions_from_logs(
    agent_logs_dir: Path, examples: List[Example]
) -> Dict[str, Prediction]:
    predictions: Dict[str, Prediction] = {}
    for example in examples:
        actions_path = agent_logs_dir / f"{example.id}_actions.json"
        if not actions_path.exists():
            continue
        try:
            actions = json.loads(actions_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(actions, list):
            continue
        planner_observation: Optional[str] = None
        for entry in actions:
            if not isinstance(entry, dict):
                continue
            action_text = entry.get("action", "")
            state_text = (entry.get("state") or "").lower()
            if "Planner[" in action_text and "success" in state_text:
                planner_observation = entry.get("observation")
        if not planner_observation:
            continue
        log_path = agent_logs_dir / f"{example.id}.log"
        scratchpad_path = agent_logs_dir / f"{example.id}_scratchpad.txt"
        metadata_payload = dict(example.metadata)
        metadata_payload["_tool_agent_artifacts"] = {
            "log": log_path.name if log_path.exists() else None,
            "scratchpad": scratchpad_path.name if scratchpad_path.exists() else None,
            "actions": actions_path.name,
        }
        predictions[example.id] = Prediction(
            id=example.id,
            query=example.query,
            prediction=planner_observation,
            raw_prediction=planner_observation,
            expected=example.expected,
            metadata=metadata_payload,
        )
    return predictions


def _render_action_log(action_log: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in action_log or []:
        step = entry.get("step")
        prefix = f"{step}" if step is not None else ""
        thought = entry.get("thought")
        action = entry.get("action")
        observation = entry.get("observation")
        if thought:
            lines.append(f"Thought {prefix}: {thought}" if prefix else f"Thought: {thought}")
        if action:
            lines.append(f"Action {prefix}: {action}" if prefix else f"Action: {action}")
        if observation:
            lines.append(f"Observation {prefix}: {observation}" if prefix else f"Observation: {observation}")
        lines.append("")
    return "\n".join(lines).strip()


def _summarize_log_lines(text: str, limit: int = 5) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    summary: List[str] = []
    seen: Dict[str, int] = {}
    ordered: List[str] = []
    for line in lines:
        if line not in seen:
            ordered.append(line)
            seen[line] = 0
        seen[line] += 1
    for line in ordered[:limit]:
        count = seen[line]
        summary.append(f"{line} (x{count})" if count > 1 else line)
    remaining = len(ordered) - limit
    if remaining > 0:
        summary.append(f"... {remaining} more unique log lines suppressed ...")
    return summary


def _coerce_constraint_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Best-effort parsing for the stringified local_constraint column."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for parser in (json.loads, literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _has_hard_constraints(metadata: Dict[str, Any]) -> bool:
    constraint_dict = _coerce_constraint_dict(metadata.get("local_constraint"))
    if not constraint_dict:
        return False
    for value in constraint_dict.values():
        if value is None:
            continue
        if isinstance(value, str):
            if not value.strip() or value.strip() in {"-", ""}:
                continue
            return True
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, list):
            if any(item not in (None, "", "-") for item in value):
                return True
            continue
        return True
    return False


TEST_MINI_TARGET = 60
TEST_MINI_PLAN: List[Tuple[Tuple[str, str, int], int]] = [
    (("commonsense", "easy", 3), 10),
    (("commonsense", "easy", 5), 10),
    (("commonsense", "easy", 7), 10),
    (("hard", "medium", 3), 5),
    (("hard", "hard", 3), 5),
    (("hard", "medium", 5), 5),
    (("hard", "hard", 5), 5),
    (("hard", "medium", 7), 5),
    (("hard", "hard", 7), 5),
]


def select_test_mini_examples(
    examples: List[Example], target_total: int = TEST_MINI_TARGET
) -> List[Example]:
    """Always return the same curated 60-example subset for test-mini runs."""
    if target_total <= 0:
        return []
    if target_total >= len(examples):
        return examples[:target_total]

    buckets: Dict[Tuple[str, str, int], List[Example]] = defaultdict(list)
    for example in examples:
        metadata = example.metadata or {}
        level_raw = metadata.get("level") or "unknown"
        level = str(level_raw).lower()
        days_raw = metadata.get("days") or 0
        try:
            days = int(days_raw)
        except (TypeError, ValueError):
            days = 0
        difficulty = "hard" if _has_hard_constraints(metadata) else "commonsense"
        buckets[(difficulty, level, days)].append(example)

    for items in buckets.values():
        items.sort(key=lambda ex: ex.id)

    selected: List[Example] = []
    seen_ids: set[str] = set()

    for (difficulty, level, days), desired in TEST_MINI_PLAN:
        bucket = buckets.get((difficulty, level, days), [])
        take = min(desired, len(bucket))
        for example in bucket[:take]:
            if example.id in seen_ids:
                continue
            selected.append(example)
            seen_ids.add(example.id)

    if len(selected) < target_total:
        for example in sorted(examples, key=lambda ex: ex.id):
            if example.id in seen_ids:
                continue
            selected.append(example)
            seen_ids.add(example.id)
            if len(selected) >= target_total:
                break

    return selected[:target_total]


def build_llm(provider: Provider, model: Optional[str], temperature: float):
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash", temperature=temperature
        )
    if provider == "upstage":
        from langchain_upstage import ChatUpstage

        return ChatUpstage(model=model or "solar-pro", temperature=temperature)
    if is_travelplanner_provider(provider):
        raise ValueError("build_llm should not be called when provider=travelplanner(workflow)")

    raise ValueError(f"Unsupported provider: {provider}")


def create_backbone_instances(
    count: int,
    provider: Provider,
    model: Optional[str],
    temperature: float,
    system_prompt: Optional[str],
):
    if is_travelplanner_provider(provider):
        return []
    instances = []
    target = max(1, count)
    for _ in range(target):
        with _silence_stdout():
            llm = build_llm(provider, model, temperature)
            graph = build_backbone_graph(llm, system_prompt=system_prompt)
        instances.append(graph)
    return instances


def _process_example(
    app,
    example: Example,
    idx: int,
    provider: Provider,
    model: Optional[str],
    save_messages: bool,
    run_dir: Path,
):
    result = app.invoke(
        {
            "query": example.query,
            "reference_information": example.reference_information,
            "metadata": example.metadata,
        },
        config={
            "configurable": {"run_name": f"backbone-{provider}-{model or 'default'}"},
            "metadata": {"example_id": example.id},
        },
    )

    prediction_text = result["prediction"]
    try:
        grounded_text, grounding_debug = ground_prediction(
            prediction_text,
            example.metadata,
            seed=idx,
        )
    except Exception as grounding_error:  # noqa: BLE001
        grounded_text = prediction_text
        grounding_debug = {
            "error": str(grounding_error),
        }

    metadata_payload = dict(example.metadata)
    metadata_payload["_grounding"] = grounding_debug

    prediction = Prediction(
        id=example.id,
        query=example.query,
        prediction=grounded_text,
        raw_prediction=prediction_text,
        expected=example.expected,
        metadata=metadata_payload,
    )

    if save_messages:
        message_path = run_dir / f"{example.id}_messages.json"
        message_path.write_text(
            json.dumps([m.dict() for m in result["messages"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return idx, prediction


def _ensure_travelplanner_langchain_shims() -> None:
    import sys

    try:
        chat_models_mod = importlib.import_module("langchain.chat_models")
        if hasattr(chat_models_mod, "ChatOpenAI"):
            return
    except ModuleNotFoundError:
        pass
    except Exception:
        pass

    langchain_pkg = sys.modules.get("langchain")
    if langchain_pkg is None:
        langchain_pkg = types.ModuleType("langchain")
        langchain_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["langchain"] = langchain_pkg

    try:
        from langchain_openai import ChatOpenAI as ModernChatOpenAI
    except ModuleNotFoundError as exc:
        raise ImportError(
            "langchain_openai 패키지가 필요합니다. `poetry add langchain-openai` 또는 `poetry install`을 다시 실행하세요."
        ) from exc
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    chat_models_module = types.ModuleType("langchain.chat_models")
    chat_models_module.ChatOpenAI = ModernChatOpenAI
    sys.modules["langchain.chat_models"] = chat_models_module
    langchain_pkg.chat_models = chat_models_module

    prompts_module = types.ModuleType("langchain.prompts")
    prompts_module.PromptTemplate = PromptTemplate
    sys.modules["langchain.prompts"] = prompts_module
    langchain_pkg.prompts = prompts_module

    schema_module = types.ModuleType("langchain.schema")
    schema_module.AIMessage = AIMessage
    schema_module.HumanMessage = HumanMessage
    schema_module.SystemMessage = SystemMessage
    sys.modules["langchain.schema"] = schema_module
    langchain_pkg.schema = schema_module

    class _BaseLLM:
        pass

    llms_module = types.ModuleType("langchain.llms")
    base_module = types.ModuleType("langchain.llms.base")
    base_module.BaseLLM = _BaseLLM
    llms_module.base = base_module
    sys.modules["langchain.llms"] = llms_module
    sys.modules["langchain.llms.base"] = base_module
    langchain_pkg.llms = llms_module

    class _DummyCallback:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __repr__(self):
            return "OpenAI callback (dummy)"

        __str__ = __repr__

    callbacks_module = types.ModuleType("langchain.callbacks")

    def get_openai_callback():
        return _DummyCallback()

    callbacks_module.get_openai_callback = get_openai_callback
    sys.modules["langchain.callbacks"] = callbacks_module
    langchain_pkg.callbacks = callbacks_module

    _ensure_travelplanner_hf_shims()
    _ensure_travelplanner_openai_shims()


def _ensure_travelplanner_hf_shims() -> None:
    import sys

    try:
        import huggingface_hub
    except ModuleNotFoundError as exc:
        raise ImportError(
            "huggingface_hub 패키지가 필요합니다. `poetry add huggingface_hub` 이후 다시 실행하세요."
        ) from exc

    if not hasattr(huggingface_hub, "HfFolder"):
        class _CompatHfFolder:
            _token_path = Path.home() / ".huggingface" / "token"

            @classmethod
            def path_token(cls) -> Path:
                return cls._token_path

            @classmethod
            def get_token(cls) -> Optional[str]:
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
        def _compat_whoami(*_args, **_kwargs) -> Dict[str, Any]:
            token = huggingface_hub.HfFolder.get_token()
            return {"token": token} if token else {}

        huggingface_hub.whoami = _compat_whoami


def _ensure_travelplanner_openai_shims() -> None:
    import sys

    try:
        import openai
    except ModuleNotFoundError as exc:
        raise ImportError("openai 패키지가 필요합니다. `poetry add openai` 후 다시 실행하세요.") from exc

    if hasattr(openai, "error"):
        return

    class _ErrorNamespace:
        APIConnectionError = getattr(openai, "APIConnectionError", Exception)
        RateLimitError = getattr(openai, "RateLimitError", Exception)
        APIError = getattr(openai, "APIError", Exception)
        AuthenticationError = getattr(openai, "AuthenticationError", Exception)
        InvalidRequestError = getattr(openai, "InvalidRequestError", Exception)

    openai.error = _ErrorNamespace()



def _load_travelplanner_agent_module():
    global TRAVELPLANNER_AGENT_MODULE
    if TRAVELPLANNER_AGENT_MODULE is not None:
        return TRAVELPLANNER_AGENT_MODULE
    agents_file = TRAVELPLANNER_AGENT_DIR / "tool_agents.py"
    if not agents_file.exists():
        raise FileNotFoundError("TravelPlanner agent 파일을 찾을 수 없습니다. benchmarks/travelplanner/official/agents 경로를 확인하세요.")
    spec = importlib.util.spec_from_file_location("travelplanner_tool_agents", agents_file)
    module = importlib.util.module_from_spec(spec)
    _ensure_travelplanner_langchain_shims()
    import sys
    added_path = False
    agents_dir_str = str(TRAVELPLANNER_AGENT_DIR.resolve())
    if agents_dir_str not in sys.path:
        sys.path.insert(0, agents_dir_str)
        added_path = True
    with _change_cwd(TRAVELPLANNER_AGENT_DIR):
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if added_path:
        sys.path.remove(agents_dir_str)
    TRAVELPLANNER_AGENT_MODULE = module
    return module


def _run_travelplanner_tool_agent(
    examples: List[Example],
    model_name: Optional[str],
    run_dir: Path,
    progress_label: str,
    resume: bool = False,
) -> List[Prediction]:
    module = _load_travelplanner_agent_module()
    tools_list = ["notebook", "flights", "attractions", "accommodations", "restaurants", "googleDistanceMatrix", "planner", "cities"]
    effective_model = model_name or "gpt-4-1106-preview"
    agent_logs_dir = (run_dir / "travelplanner_agent_logs").resolve()
    agent_logs_dir.mkdir(parents=True, exist_ok=True)
    partial_predictions_path = (run_dir / PARTIAL_PREDICTIONS_FILENAME).resolve()

    total_examples = len(examples)
    worker_count = max(1, min(TRAVELPLANNER_AGENT_WORKERS, total_examples))
    task_queue: "queue.Queue[Tuple[int, Example]]" = queue.Queue()
    predictions_buffer: List[Optional[Prediction]] = [None] * total_examples
    progress_bar = typer.progressbar(length=total_examples, label=progress_label)
    progress_lock = threading.Lock()
    partial_lock = threading.Lock()

    existing_predictions: Dict[str, Prediction] = {}
    if resume:
        existing_predictions = _load_partial_predictions(partial_predictions_path)
        if not existing_predictions:
            existing_predictions = _hydrate_predictions_from_logs(agent_logs_dir, examples)
        if existing_predictions and not partial_predictions_path.exists():
            with partial_predictions_path.open("w", encoding="utf-8") as f:
                for example in examples:
                    pred = existing_predictions.get(example.id)
                    if not pred:
                        continue
                    f.write(json.dumps(pred.model_dump(), ensure_ascii=False))
                    f.write("\n")
    else:
        if partial_predictions_path.exists():
            partial_predictions_path.unlink()

    def _append_partial_prediction(prediction: Prediction) -> None:
        record = prediction.model_dump()
        with partial_lock:
            with partial_predictions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

    completed = 0
    for idx, example in enumerate(examples):
        cached = existing_predictions.get(example.id)
        if cached is not None:
            predictions_buffer[idx] = cached
            completed += 1
            continue
        task_queue.put((idx, example))

    def _record_prediction(idx: int, example: Example, answer: str, scratchpad: str, action_log: List[Dict[str, Any]], log_path: Path, log_text: str) -> None:
        scratchpad_path = agent_logs_dir / f"{example.id}_scratchpad.txt"
        actions_path = agent_logs_dir / f"{example.id}_actions.json"
        if log_text:
            log_path.write_text(log_text, encoding="utf-8")
        if not log_text and log_path.exists():
            log_path.unlink()
        scratchpad_path.write_text(scratchpad or "", encoding="utf-8")
        actions_path.write_text(json.dumps(action_log, ensure_ascii=False, indent=2), encoding="utf-8")
        prediction_text = answer or ""
        if not isinstance(prediction_text, str):
            prediction_text = json.dumps(prediction_text, ensure_ascii=False)
        metadata_payload = dict(example.metadata or {})
        metadata_payload["_tool_agent_artifacts"] = {
            "log": log_path.name if log_text else None,
            "scratchpad": scratchpad_path.name,
            "actions": actions_path.name,
        }
        prediction_obj = Prediction(
            id=example.id,
            query=example.query,
            prediction=prediction_text,
            raw_prediction=prediction_text,
            expected=example.expected,
            metadata=metadata_payload,
        )
        predictions_buffer[idx] = prediction_obj
        _append_partial_prediction(prediction_obj)

    def _worker(worker_id: int) -> None:
        with _change_cwd(TRAVELPLANNER_AGENT_DIR):
            agent = module.ReactAgent(
                args=None,
                tools=tools_list,
                max_steps=30,
                react_llm_name=effective_model,
                planner_llm_name=effective_model,
            )

        while True:
            try:
                idx, example = task_queue.get_nowait()
            except queue.Empty:
                break
            log_path = agent_logs_dir / f"{example.id}.log"
            try:
                answer, scratchpad, action_log = agent.run(example.query, reset=True)
                log_text = _render_action_log(action_log or [])
                _record_prediction(idx, example, answer or "", scratchpad or "", action_log or [], log_path, log_text)
            finally:
                task_queue.task_done()
                with progress_lock:
                    progress_bar.update(1)

    progress_bar.__enter__()
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_worker, wid) for wid in range(worker_count)]
            if completed:
                progress_bar.update(completed)
            for future in as_completed(futures):
                future.result()
    finally:
        progress_bar.__exit__(None, None, None)

    predictions: List[Prediction] = [prediction for prediction in predictions_buffer if prediction is not None]
    return predictions


def build_backbone_graph(llm, system_prompt: Optional[str] = None):
    graph = StateGraph(GraphState)

    def call_llm(state: GraphState) -> GraphState:
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        user_content = state["query"]
        context_sections: List[str] = []

        metadata = state.get("metadata", {}) or {}
        if metadata:
            summary_keys = [
                "org",
                "dest",
                "date",
                "days",
                "people_number",
                "budget",
                "visiting_city_number",
                "local_constraint",
            ]
            summary_lines = []
            for key in summary_keys:
                if key in metadata and metadata[key] not in (None, ""):
                    value = metadata[key]
                    if isinstance(value, list):
                        value = ", ".join(map(str, value))
                    summary_lines.append(f"{key}: {value}")
            if summary_lines:
                context_sections.append("요약 정보:\n" + "\n".join(summary_lines))

        reference_information = state.get("reference_information")
        if reference_information:
            formatted_refs: List[str] = []
            if isinstance(reference_information, list):
                for item in reference_information:
                    if isinstance(item, dict):
                        desc = (
                            item.get("Description")
                            or item.get("description")
                            or "Reference"
                        )
                        content = item.get("Content") or item.get("content") or ""
                        formatted_refs.append(f"{desc}:\n{content}")
                    else:
                        formatted_refs.append(str(item))
            else:
                formatted_refs.append(str(reference_information))

            if formatted_refs:
                context_sections.append("참고 자료:\n" + "\n\n".join(formatted_refs))

        if context_sections:
            user_content = f"{user_content}\n\n" + "\n\n".join(context_sections)

        messages.append(HumanMessage(content=user_content))

        response: AIMessage = llm.invoke(messages)

        return {
            "query": state["query"],
            "reference_information": reference_information,
            "metadata": metadata,
            "messages": messages + [response],
            "prediction": response.content,
        }

    graph.add_node("call_llm", call_llm)
    graph.add_edge(START, "call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()


def load_examples(dataset_path: Path) -> List[Example]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    examples: List[Example] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx + 1} of {dataset_path}") from exc

            if not isinstance(payload, dict):
                continue

            example_id = str(
                payload.get("id") or f"{dataset_path.stem}_{idx:05d}"
            )
            query = payload.get("query")
            if not query:
                continue

            annotated_plan = payload.get("annotated_plan")
            expected_text: Optional[str] = payload.get("expected")
            if expected_text is None and annotated_plan is not None:
                try:
                    expected_text = json.dumps(annotated_plan, ensure_ascii=False)
                except TypeError:
                    expected_text = str(annotated_plan)

            reference_information = payload.get("reference_information")
            metadata = {
                k: v
                for k, v in payload.items()
                if k
                not in {"id", "query", "annotated_plan", "reference_information", "expected"}
            }
            metadata["_source_idx"] = idx

            examples.append(
                Example(
                    id=example_id,
                    query=query,
                    expected=expected_text,
                    reference_information=reference_information,
                    metadata=metadata,
                )
            )

    return examples


def ensure_results_dir(
    provider: Provider, model: Optional[str], variant: Optional[str] = None
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = sanitize_model_slug(model)
    if variant:
        model_slug = f"{model_slug}-{variant}"
    run_dir = Path("results") / "travelplanner" / provider / model_slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sanitize_model_slug(model: Optional[str]) -> str:
    if not model:
        return "default"
    allowed = {"-", "_", "."}
    return "".join(ch if ch.isalnum() or ch in allowed else "-" for ch in model)


def dump_predictions(path: Path, predictions: Iterable[Prediction]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred.model_dump_json())
            f.write("\n")


def infer_official_split(dataset_path: Path) -> Optional[str]:
    name = dataset_path.stem.lower()
    for candidate in ("train", "validation", "test"):
        if candidate in name:
            return candidate
    for part in dataset_path.parts:
        lowered = part.lower()
        for candidate in ("train", "validation", "test"):
            if candidate in lowered:
                return candidate
    return None


def prepare_official_submission(
    examples: List[Example],
    parsed_plans: List[Optional[List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    submission: List[Dict[str, Any]] = []
    if len(parsed_plans) != len(examples):
        raise ValueError("Parsed plan count does not match example count")
    for idx, (example, plan) in enumerate(zip(examples, parsed_plans), start=1):
        submission.append(
            {
                "idx": idx,
                "query": example.query,
                "plan": plan,
            }
        )
    return submission


def example_to_query_record(example: Example) -> Dict[str, Any]:
    record: Dict[str, Any] = dict(example.metadata or {})
    record.setdefault("id", example.id)
    record["query"] = example.query
    if example.reference_information is not None:
        record["reference_information"] = example.reference_information
    else:
        record.setdefault("reference_information", None)
    return record


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def main(
    provider: Provider = typer.Option("openai", help="LLM provider to use"),
    model: Optional[str] = typer.Option(None, help="Model name for the provider"),
    dataset: Path = typer.Option(
        Path("data/travelplanner/validation.jsonl"),
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to JSONL dataset (defaults to validation split at data/travelplanner/validation.jsonl)",
    ),
    system_prompt: Optional[str] = typer.Option(
        "You are a travel-planning agent. Respond with a JSON array where each item contains keys: days, current_city, transportation, breakfast, attraction, lunch, dinner, accommodation. Follow the TravelPlanner formatting guidelines.",
        help="System prompt prepended before the user query",
    ),
    temperature: float = typer.Option(0.2, help="Sampling temperature for the LLM"),
    save_messages: bool = typer.Option(
        False, help="If true, persist full message history alongside predictions"
    ),
    run_official_eval: bool = typer.Option(
        True,
        help="Run the official TravelPlanner evaluator on the generated JSON plans.",
    ),
    official_set_type: Optional[str] = typer.Option(
        "validation",
        help="Dataset split for official eval (train/validation/test). Defaults to validation.",
    ),
    test_mini: bool = typer.Option(
        False,
        help="Evaluate on a fixed 60-example subset (test-mini) for reproducible regression tests.",
    ),
    use_official_parser: bool = typer.Option(
        False,
        help="Use the official GPT-based parser to convert natural language plans into JSON (costly).",
    ),
    parser_model: Optional[str] = typer.Option(
        None,
        help="Model name used by the official parsing prompt (defaults to the evaluation model, or gpt-4-1106-preview if unspecified).",
    ),
    limit: Optional[int] = typer.Option(
        None,
        help="Maximum number of examples to evaluate (ignored when --test-mini is set).",
    ),
    resume_run: Optional[Path] = typer.Option(
        None,
        "--resume-run",
        help="기존 TravelPlanner run 디렉터리에서 중단된 tool-agent 실행을 이어서 진행합니다.",
        resolve_path=True,
    ),
) -> None:
    load_dotenv(override=False)
    provider = normalize_provider_name(provider)

    if test_mini and limit is not None:
        raise typer.Exit("--test-mini에서는 --limit 옵션을 사용할 수 없습니다.")

    examples = load_examples(dataset)
    if not examples:
        typer.echo("No examples loaded; aborting.")
        raise typer.Exit(code=1)

    if limit is not None:
        if limit <= 0:
            raise typer.Exit("--limit must be a positive integer")
        examples = examples[:limit]

    original_total = len(examples)
    variant = None
    if test_mini:
        target_total = min(TEST_MINI_TARGET, original_total)
        examples = select_test_mini_examples(examples, target_total)
        variant = "test-mini"

    example_ids = [example.id for example in examples]
    dataset_path_str = str(dataset.resolve())
    run_dir: Path
    resume_mode = False
    if resume_run is not None:
        if not is_travelplanner_provider(provider):
            raise typer.Exit("--resume-run 옵션은 provider=travelplanner(workflow)에서만 사용할 수 있습니다.")
        run_dir = resume_run
        if not run_dir.exists():
            raise typer.Exit(f"--resume-run 경로를 찾을 수 없습니다: {run_dir}")
        resume_mode = True
    else:
        run_dir = ensure_results_dir(provider, model, variant=variant)

    run_config_payload = {
        "provider": provider,
        "model": model,
        "dataset": dataset_path_str,
        "test_mini": test_mini,
        "limit": limit,
        "variant": variant,
        "example_ids": example_ids,
    }

    if resume_mode:
        existing_config = _load_run_config(run_dir)
        if existing_config:
            if existing_config.get("provider") and existing_config["provider"] != provider:
                raise typer.Exit("resume-run 디렉터리의 provider 값과 현재 인자가 다릅니다.")
            stored_dataset = existing_config.get("dataset")
            if stored_dataset and Path(stored_dataset).resolve() != dataset.resolve():
                raise typer.Exit("resume-run 디렉터리의 dataset 경로와 현재 인자가 다릅니다.")
            stored_variant = existing_config.get("variant")
            if stored_variant != variant:
                raise typer.Exit("resume-run 디렉터리의 variant/test-mini 설정이 다릅니다.")
            stored_ids = existing_config.get("example_ids")
            if stored_ids and stored_ids != example_ids:
                raise typer.Exit("resume-run 디렉터리의 example 목록과 현재 선택된 예제가 다릅니다.")
        else:
            typer.echo(f"[resume] {RUN_CONFIG_FILENAME}가 없어 기본 검증 없이 이어서 실행합니다.")
            resume_payload = dict(run_config_payload)
            resume_payload["timestamp"] = datetime.now().isoformat()
            resume_payload["resume_created"] = True
            _write_run_config(run_dir, resume_payload)
    else:
        run_config_payload["timestamp"] = datetime.now().isoformat()
        _write_run_config(run_dir, run_config_payload)

    total_examples = len(examples)
    inferred_split = infer_official_split(dataset)
    dataset_label = official_set_type or inferred_split or dataset.stem
    if test_mini:
        dataset_label = f"{dataset_label} test-mini"
    progress_label = f"TravelPlanner {dataset_label} · {provider}:{model or 'default'}"

    if is_travelplanner_provider(provider):
        predictions = _run_travelplanner_tool_agent(
            examples,
            model,
            run_dir,
            progress_label,
            resume=resume_mode,
        )
    else:
        app_instances = create_backbone_instances(
            BACKBONE_INSTANCE_COUNT, provider, model, temperature, system_prompt
        )
        instance_count = len(app_instances)
        predictions_buffer: List[Optional[Prediction]] = [None] * total_examples

        with ThreadPoolExecutor(max_workers=instance_count) as executor:
            with typer.progressbar(length=total_examples, label=progress_label) as progress:
                futures = {}
                for idx, example in enumerate(examples, start=1):
                    app_instance = app_instances[(idx - 1) % instance_count]
                    future = executor.submit(
                        _process_example,
                        app_instance,
                        example,
                        idx,
                        provider,
                        model,
                        save_messages,
                        run_dir,
                    )
                    futures[future] = idx

                for future in as_completed(futures):
                    idx, prediction = future.result()
                    predictions_buffer[idx - 1] = prediction
                    progress.update(1)

        predictions = [prediction for prediction in predictions_buffer if prediction]

    metrics: Dict[str, Any] = {key: 0.0 for key in DEFAULT_METRIC_KEYS}

    official_output_dir = None
    parsed_plans: Optional[List[Optional[List[Dict[str, Any]]]]] = None
    official_scores: Optional[Dict[str, Any]] = None

    dump_predictions(run_dir / "predictions.jsonl", predictions)

    effective_set_type = official_set_type or inferred_split

    if effective_set_type is None:
        raise typer.Exit("공식 평가를 위해서는 데이터셋 split(validation/test 등)을 지정해야 합니다.")

    dataset_records = load_dataset('osunlp/TravelPlanner', effective_set_type)[effective_set_type]
    dataset_records_list = [dict(record) for record in dataset_records]
    if len(dataset_records_list) < total_examples:
        raise typer.Exit("데이터셋 샘플 수가 요청된 평가 샘플 수보다 적습니다.")

    source_indices: List[int] = []
    missing_index = False
    for example in examples:
        source_idx = example.metadata.get("_source_idx")
        if isinstance(source_idx, int):
            source_indices.append(source_idx)
        else:
            missing_index = True
            break

    if missing_index or not source_indices:
        source_indices = list(range(total_examples))

    if max(source_indices, default=0) >= len(dataset_records_list):
        raise typer.Exit("선택된 예제 인덱스가 데이터셋 범위를 벗어났습니다.")

    query_records = [dataset_records_list[idx] for idx in source_indices]

    official_output_dir = run_dir / "official_output"
    model_tag = model or "baseline"
    parser_model_name = parser_model or model or "gpt-4-1106-preview"
    write_generated_plan_files(
        official_output_dir,
        effective_set_type,
        model_tag,
        predictions,
    )

    if use_official_parser:
        typer.echo(f"[official-parse] GPT 기반 파서 실행 중... (model={parser_model_name})")
        run_official_parsing(
            official_output_dir,
            run_dir / "official_tmp",
            effective_set_type,
            model_tag,
            parser_model=parser_model_name,
            total_examples=total_examples,
            worker_count=PARSER_WORKER_COUNT,
        )

    parsed_plans = load_parsed_plans(
        official_output_dir,
        effective_set_type,
        model_tag,
        total_examples=total_examples,
    )

    if run_official_eval:
        if parsed_plans is None:
            raise typer.Exit("공식 평가를 위해서는 파싱된 플랜이 필요합니다.")

        submission_path = run_dir / "official_submission.jsonl"
        submission_records = prepare_official_submission(examples, parsed_plans)
        write_jsonl(submission_path, submission_records)

        try:
            from benchmarks.travelplanner.official.evaluation.eval import (
                eval_score as official_eval_score,
            )
        except ModuleNotFoundError as exc:
            typer.echo(
                "공식 평가 모듈을 불러오지 못했습니다. Poetry 의존성을 설치했는지 확인하세요."
            )
            raise typer.Exit(code=2) from exc

        typer.echo(
            f"[official-eval] evaluator 실행 중 ({effective_set_type}, file={submission_path})"
        )
        with _capture_stdout() as captured_logs:
            scores, detailed_scores = official_eval_score(
                effective_set_type,
                submission_path,
                query_records,
            )
        log_text = captured_logs.getvalue().strip()
        if log_text:
            log_path = run_dir / "official_eval.log"
            log_path.write_text(log_text + "\n", encoding="utf-8")
            summary_lines = _summarize_log_lines(log_text)
            if summary_lines:
                typer.echo("[official-eval] 로그 요약:")
                for line in summary_lines:
                    typer.echo(f"  - {line}")
                typer.echo(f"[official-eval] 세부 로그: {log_path}")
        official_scores = scores
        (run_dir / "official_metrics.json").write_text(
            json.dumps({"scores": scores, "details": detailed_scores}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metrics = {key: scores.get(key) for key in DEFAULT_METRIC_KEYS}
        typer.echo("[official-eval] 주요 지표:")
        typer.echo(json.dumps(scores, indent=2, ensure_ascii=False))

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    display_model_name = model or "default"
    if test_mini:
        update_leaderboard(
            provider,
            display_model_name,
            run_dir,
            metrics,
            leaderboard_path=MINI_LEADERBOARD_PATH,
            header_lines=MINI_LEADERBOARD_HEADER,
            metric_keys=DEFAULT_METRIC_KEYS,
            result_label="test-mini",
        )
    else:
        update_leaderboard(
            provider,
            display_model_name,
            run_dir,
            metrics,
            metric_keys=DEFAULT_METRIC_KEYS,
            result_label=None,
        )

    typer.echo("Run complete")
    typer.echo(f"Results saved to {run_dir}")
    typer.echo(json.dumps(metrics, indent=2))
    if official_scores:
        typer.echo("Official evaluator scores 저장됨 (official_metrics.json 참조)")


if __name__ == "__main__":
    typer.run(main)
