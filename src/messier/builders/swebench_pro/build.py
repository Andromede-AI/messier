"""
Import SWE-bench Pro tasks and results.
"""

import json
from collections import Counter
import pandas as pd
from ..benchmarks import REPO_DESCRIPTIONS
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW, load_jsonl
from ...models import (
    BuildResult,
    ActionSpaceType,
    DataProvider,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    ScoringRule,
    Task,
    Trajectory,
    VerifierMode,
    VerifierType,
)
from .constants import (
    AP_SUBJECT_RE,
    BINARY_TRAJ_SOURCES,
    INSTANCE_SUFFIX_RE,
    PER_TEST_TRAJ_SOURCES,
)



def build() -> BuildResult:
    task_rows = load_jsonl(RAW / "swebench_pro" / "tasks.jsonl")
    tasks = [
        Task(
            benchmark="swebench_pro",
            task_id=_normalize_id(task["instance_id"]),
            environment_id=task["repo"],
            environment_description=(
                f"A checkout of {task['repo']}. "
                f"{REPO_DESCRIPTIONS[task['repo']].rstrip('.')}."
            ),
            environment_metadata={
                "action_space": {
                    "types": [ActionSpaceType.SHELL],
                    "description": (
                        "Shell commands for inspecting and editing the repository, "
                        "running tests, and producing a patch."
                    ),
                },
                "environment_state": {
                    "type": EnvironmentStateType.FILESYSTEM,
                    "access": EnvironmentStateAccess.READ_WRITE,
                    "ref": f"github://{task['repo']}@{task['base_commit']}",
                    "snapshot": None,
                },
                "base_commit": task["base_commit"],
                "repo_language": task.get("repo_language"),
                "dockerhub_tag": task.get("dockerhub_tag"),
                "before_repo_set_cmd": task.get("before_repo_set_cmd"),
                "url": f"https://github.com/{task['repo']}/tree/{task['base_commit']}",
            },
            task_description=_decode_source_value(task["problem_statement"]),
            task_metadata={
                "requirements": _decode_source_value(task.get("requirements")),
                "interface": _decode_source_value(task.get("interface")),
                "issue_specificity": _decode_source_value(task.get("issue_specificity")),
                "issue_categories": _decode_source_value(task.get("issue_categories")),
                "selected_test_files_to_run": _decode_source_value(task.get("selected_test_files_to_run")),
                "test_patch": task.get("test_patch"),
            },
            gold_answer=task["patch"],
            task_date=resolve_task_date("swebench_pro"),
            verifiers=[
                {
                    "type": VerifierType.SCRIPT,
                    "mode": VerifierMode.FINAL,
                    "name": "swebench_harness",
                }
            ],
            scoring_rule=ScoringRule.DIRECT,
            source_platform=SourcePlatform.HUGGINGFACE,
        )
        for task in task_rows
    ]

    records = []
    for row in load_jsonl(RAW / "agent_psychometrics" / "swebench_pro" / "responses.jsonl"):
        subject_id = row["subject_id"]
        match = AP_SUBJECT_RE.match(subject_id)
        raw_model = match.group("model").strip() if match else subject_id
        model, reasoning_effort = standardize_model(raw_model)
        scaffold = standardize_scaffold_name("swe-agent")
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(raw_model)

        for task_id, score in row["responses"].items():
            records.append(
                Record(
                    benchmark="swebench_pro",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    result=int(score),
                    metadata={"raw_source": "agent_psychometrics"},
                    source_platform=SourcePlatform.GITHUB,
                    data_provider=DataProvider.AGENT_PSYCHOMETRICS,
                )
            )

    tasks_by_id = {task.task_id: task for task in tasks}

    root = RAW / "swebench_pro_traj"
    for local_dir, _, raw_model, raw_scaffold in PER_TEST_TRAJ_SOURCES:
        path = root / local_dir / "train.parquet"
        if not path.exists():
            continue

        frame = pd.read_parquet(path)
        model, reasoning_effort = standardize_model(raw_model)
        scaffold = standardize_scaffold_name(raw_scaffold)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, raw_model)
        for row in frame.itertuples(index=False):
            if pd.isna(row.reward):
                continue

            task_id = _normalize_id(row.instance_id)
            if task_id not in tasks_by_id:
                continue

            test_results: dict[str, int] = {}
            for output in (row.output_optional_eval, row.output_eval):
                for test in (output or {}).get("tests", []):
                    test_results[test["name"]] = int(test["status"] == "PASSED")

            records.append(
                Record(
                    benchmark="swebench_pro",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    result=int(row.reward),
                    metadata={
                        "raw_source": local_dir,
                        "raw_instance_id": row.instance_id,
                        "unit_results": test_results or None,
                    },
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )

    for local_dir, _, raw_model, raw_scaffold in BINARY_TRAJ_SOURCES:
        path = root / local_dir / "eval_results.json"
        if not path.exists():
            continue

        model, reasoning_effort = standardize_model(raw_model)
        scaffold = standardize_scaffold_name(raw_scaffold)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, raw_model)

        for raw_task_id, passed in json.loads(path.read_text()).items():
            task_id = _normalize_id(raw_task_id)
            if not isinstance(passed, bool) or task_id not in tasks_by_id:
                continue

            records.append(
                Record(
                    benchmark="swebench_pro",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    result=int(passed),
                    metadata={"raw_source": local_dir, "raw_instance_id": raw_task_id},
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )

    trial_indices = Counter()
    for record in records:
        key = (record.agent_id, record.task_id)
        record.trial = trial_indices[key]
        trial_indices[key] += 1

    return tasks, [], records


def trajectories():
    task_ids = {
        _normalize_id(row["instance_id"])
        for row in load_jsonl(RAW / "swebench_pro" / "tasks.jsonl")
    }
    trial_indices = Counter()
    response_path = RAW / "agent_psychometrics" / "swebench_pro" / "responses.jsonl"
    for row in load_jsonl(response_path):
        match = AP_SUBJECT_RE.match(row["subject_id"])
        raw_model = match.group("model").strip() if match else row["subject_id"]
        model, reasoning_effort = standardize_model(raw_model)
        scaffold = standardize_scaffold_name("swe-agent")
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        for task_id in row["responses"]:
            trial_indices[(agent_id, task_id)] += 1

    root = RAW / "swebench_pro_traj"
    for local_dir, _, raw_model, raw_scaffold in PER_TEST_TRAJ_SOURCES:
        path = root / local_dir / "train.parquet"
        if not path.exists():
            continue

        model, reasoning_effort = standardize_model(raw_model)
        scaffold = standardize_scaffold_name(raw_scaffold)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, raw_model)
        frame = pd.read_parquet(path, columns=["instance_id", "reward", "traj"])
        for row in frame.itertuples(index=False):
            task_id = _normalize_id(row.instance_id)
            if pd.isna(row.reward) or task_id not in task_ids:
                continue

            key = (agent_id, task_id)
            trial = trial_indices[key]
            trial_indices[key] += 1
            if not row.traj:
                continue

            yield Trajectory(
                benchmark="swebench_pro",
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=trial,
                content=row.traj,
                metadata={
                    "raw_source": local_dir,
                    "raw_instance_id": row.instance_id,
                    "source_file": path.name,
                },
                source_platform=SourcePlatform.HUGGINGFACE,
            )


# helpers

def _normalize_id(instance_id: str) -> str:
    normalized = instance_id.removeprefix("instance_")
    match = INSTANCE_SUFFIX_RE.search(normalized)
    return normalized[: match.start()] if match else normalized


def _decode_source_value(value):
    """
    Decode source values stored as JSON strings.

    :param value: Source value.
    :return: Decoded text, or the original value when it is already decoded.
    """
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
