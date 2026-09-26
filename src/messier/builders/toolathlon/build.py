"""
Import Toolathlon tasks, results, and trajectories.
"""

import json
import shutil
from ..benchmarks import UPSTREAM_COMMITS
from ..dates import resolve_model_date, resolve_task_date
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ...io import RAW, TASK_FILES_PATH
from ...models import (
    ActionSpaceType,
    BuildResult,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    ScoringRule,
    SourcePlatform,
    Task,
    Trajectory,
    VerifierMode,
    VerifierType,
)

BENCHMARK = "toolathlon"
SCAFFOLD = "toolathlon"
JSON_FIELDS = (
    "task_status",
    "config",
    "tool_calls",
    "messages",
    "key_stats",
    "agent_cost",
)


def build() -> BuildResult:
    scaffold = standardize_scaffold_name(SCAFFOLD)
    task_date = resolve_task_date(BENCHMARK)
    source_ref = (
        "hf://datasets/hkust-nlp/Toolathlon-Trajectories@"
        f"{UPSTREAM_COMMITS[BENCHMARK]}"
    )
    tasks: dict[str, Task] = {}
    records: list[Record] = []

    for path, raw_model, trial, row in source_rows():
        task_id = row["task_name"]
        config = row["config"] or {}
        if task_id not in tasks and config.get("task_str"):
            mcp_servers = config.get("needed_mcp_servers") or []
            local_tools = config.get("needed_local_tools") or []
            action_types = [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT]
            if "terminal" in mcp_servers:
                action_types.insert(0, ActionSpaceType.SHELL)

            input_files = []
            workspace = RAW / "toolathlon_tasks" / "tasks" / "finalpool" / task_id / "initial_workspace"
            if workspace.is_dir():
                for source_path in sorted(workspace.rglob("*")):
                    if not source_path.is_file():
                        continue

                    relative_path = source_path.relative_to(workspace)
                    release_path = TASK_FILES_PATH / BENCHMARK / task_id / relative_path
                    release_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, release_path)
                    input_files.append(
                        {
                            "path": f"{config['agent_workspace']}/{relative_path.as_posix()}",
                            "dataset_path": f"task_files/{BENCHMARK}/{task_id}/{relative_path.as_posix()}",
                        }
                    )

            tasks[task_id] = Task(
                benchmark=BENCHMARK,
                task_id=task_id,
                environment_id=task_id,
                environment_description=(
                    "Software applications containing accounts and data, together with a shared "
                    "workspace for reading and writing files."
                ),
                environment_metadata={
                    "action_space": {
                        "types": action_types,
                        "description": (
                            "Calls to application and utility tools, text responses, and shell "
                            "commands when a terminal is available."
                        ),
                        "mcp_servers": mcp_servers,
                        "local_tools": local_tools,
                    },
                    "environment_state": {
                        "type": EnvironmentStateType.IN_MEMORY,
                        "access": EnvironmentStateAccess.READ_WRITE,
                        "ref": source_ref,
                    },
                    "url": "https://github.com/hkust-nlp/Toolathlon",
                },
                task_description=config["task_str"],
                task_metadata={
                    "max_turns": config.get("max_turns"),
                    "single_turn_mode": config.get("single_turn_mode"),
                    "initialization": config.get("initialization"),
                    "extra_context": {
                        "system_prompt": (config.get("system_prompts") or {}).get("agent"),
                        "input_files": input_files,
                    },
                    "evaluation": config.get("evaluation"),
                },
                gold_answer=None,
                task_date=task_date,
                verifiers=[
                    {
                        "type": VerifierType.SCRIPT,
                        "mode": VerifierMode.FINAL,
                    }
                ],
                scoring_rule=ScoringRule.DIRECT,
                source_platform=SourcePlatform.HUGGINGFACE,
            )

        model, reasoning_effort = standardize_model(raw_model)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        status = row["task_status"] or {}
        evaluation = status.get("evaluation")
        metadata = {
            "source_file": path.name,
            "source_status": status,
            "key_stats": row["key_stats"],
            "agent_cost": row["agent_cost"],
            "initial_run_time": row.get("initial_run_time"),
            "completion_time": row.get("completion_time"),
        }
        if status.get("preprocess") != "done":
            result = None
            metadata["error_type"] = "infrastructure_error"
        elif status.get("running") == "running":
            result = None
            metadata["error_type"] = "missing_score"
        else:
            result = int(evaluation) if isinstance(evaluation, bool) else 0

        records.append(
            Record(
                benchmark=BENCHMARK,
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=resolve_model_date(model, raw_model),
                trial=trial,
                result=result,
                metadata=metadata,
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )

    return list(tasks.values()), [], records


def trajectories():
    scaffold = standardize_scaffold_name(SCAFFOLD)
    for path, raw_model, trial, row in source_rows():
        if not isinstance(row["messages"], list):
            continue

        model, reasoning_effort = standardize_model(raw_model)
        content = json.dumps(row["messages"], ensure_ascii=False)

        yield Trajectory(
            benchmark=BENCHMARK,
            task_id=row["task_name"],
            agent_id=format_agent_id(model, scaffold, reasoning_effort),
            agent_model=model,
            model_reasoning_effort=reasoning_effort,
            agent_scaffold=scaffold,
            model_date=resolve_model_date(model, raw_model),
            trial=trial,
            content=content,
            metadata={"source_file": path.name},
            source_platform=SourcePlatform.HUGGINGFACE,
        )


def source_rows():
    for path in sorted((RAW / BENCHMARK).glob("*.jsonl")):
        raw_model, run = path.stem.rsplit("_", 1)
        for line in path.open():
            row = json.loads(line)
            for field in JSON_FIELDS:
                value = row.get(field)
                row[field] = json.loads(value) if isinstance(value, str) else value

            yield path, raw_model, int(run) - 1, row
