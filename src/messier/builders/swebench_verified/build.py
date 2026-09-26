"""
Import SWE-bench Verified tasks and results.
"""

import json
from collections import Counter
import pandas as pd
from ..benchmarks import REPO_DESCRIPTIONS
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW, load_jsonl
from ...models import (
    ActionSpaceType,
    BuildResult,
    DataProvider,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    HumanEffort,
    Record,
    ScoringRule,
    Task,
    Trajectory,
    VerifierMode,
    VerifierType,
)
from .constants import (
    BUCKET_BANDS,
    CODERFORGE_MODEL,
    CODERFORGE_SCAFFOLD,
    LIVESWEAGENT_TRAJ_SOURCES,
    PYTEST_RESULT_RE,
    UNITTEST_RESULT_RE,
)
from .utils import devin_records, parse_subject_id

# swe-bench Verified contains 500 tasks with one test-based verifier each


def build() -> BuildResult:
    task_rows = load_jsonl(RAW / "swebench_verified" / "tasks.jsonl")

    human_bands: dict[str, tuple[float, float, float]] = {}
    for row in load_jsonl(RAW / "bridge" / "human_minutes_by_task.jsonl"):
        human_minutes = row["human_minutes"]
        if human_minutes in BUCKET_BANDS:
            human_bands[row["task_id"]] = BUCKET_BANDS[human_minutes]
        else:
            human_bands[row["task_id"]] = (human_minutes, human_minutes, human_minutes)

    tasks = []
    for task in task_rows:
        band = human_bands.get(task["instance_id"])
        low, median, high = band if band else (None, None, None)
        tasks.append(
            Task(
                benchmark="swebench_verified",
                task_id=task["instance_id"],
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
                    "environment_setup_commit": task["environment_setup_commit"],
                    "version": task["version"],
                    "url": f"https://github.com/{task['repo']}/tree/{task['base_commit']}",
                },
                task_description=task["problem_statement"],
                task_metadata={"hints_text": task["hints_text"], "test_patch": task.get("test_patch")},
                gold_answer=task["patch"],
                task_date=resolve_task_date("swebench_verified", task["created_at"]),
                verifiers=[
                    {
                        "type": VerifierType.SCRIPT,
                        "mode": VerifierMode.FINAL,
                        "name": "swebench_harness",
                    }
                ],
                human=HumanEffort(
                    minutes_median=median,
                    minutes_low=low,
                    minutes_high=high
                )
                if median is not None
                else None,
                difficulty_label=task.get("difficulty"), # if given by the benchmark authors
                scoring_rule=ScoringRule.DIRECT,
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )

    records = []
    for row in load_jsonl(RAW / "agent_psychometrics" / "swebench_verified" / "responses.jsonl"):
        info = parse_subject_id(row["subject_id"])
        raw_model = info["agent_model"]
        info["agent_model"], info["model_reasoning_effort"] = standardize_model(info["agent_model"])
        info["agent_scaffold"] = standardize_scaffold_name(info["agent_scaffold"])
        info["model_date"] = resolve_model_date(info["agent_model"], raw_model)
        agent_id = format_agent_id(
            info["agent_model"],
            info["agent_scaffold"],
            info["model_reasoning_effort"],
        )
        for task_id, score in row["responses"].items():
            records.append(
                Record(
                    benchmark="swebench_verified",
                    task_id=task_id,
                    agent_id=agent_id,
                    result=int(score),
                    metadata={"raw_source": "agent_psychometrics"},
                    source_platform=SourcePlatform.GITHUB,
                    data_provider=DataProvider.AGENT_PSYCHOMETRICS,
                    **info,
                )
            )

    task_ids = {task.task_id for task in tasks}

    root = RAW / "swebench_verified_traj"
    for (
        local_dir,
        _,
        raw_model,
        raw_scaffold,
        result_filename,
    ) in LIVESWEAGENT_TRAJ_SOURCES:
        result_path = root / local_dir / result_filename
        if not result_path.exists():
            continue

        report = json.loads(result_path.read_text())
        status_by_task = {}
        for status, field in (
            ("resolved", "resolved_ids"),
            ("unresolved", "unresolved_ids"),
            ("empty_patch", "empty_patch_ids"),
            ("error", "error_ids"),
        ):
            for task_id in report.get(field) or []:
                status_by_task.setdefault(task_id, status)

        model, reasoning_effort = standardize_model(raw_model)
        scaffold = standardize_scaffold_name(raw_scaffold)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, raw_model)
        for task_id, status in status_by_task.items():
            if task_id not in task_ids:
                continue

            result = {
                "resolved": 1,
                "unresolved": 0,
                "empty_patch": 0,
            }.get(status)

            metadata = {"raw_source": local_dir, "status": status,}
            if status == "error":
                metadata["error_type"] = "environment_error"

            records.append(
                Record(
                    benchmark="swebench_verified",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    result=result,
                    metadata=metadata,
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )

    root = RAW / "coderforge_swebench_verified"
    model, reasoning_effort = standardize_model(CODERFORGE_MODEL)
    scaffold = standardize_scaffold_name(CODERFORGE_SCAFFOLD)
    agent_id = format_agent_id(model, scaffold, reasoning_effort)
    model_date = resolve_model_date(model, CODERFORGE_MODEL)
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["trajectory_id", "run_id", "reward", "test_output", "num_steps"])
        for row in frame.itertuples(index=False):
            task_id = row.trajectory_id.rsplit("_run", 1)[0]
            if task_id not in task_ids:
                continue

            test_results: dict[str, int] = {}
            for status, name in PYTEST_RESULT_RE.findall(row.test_output or ""):
                test_results[name] = int(status == "PASSED")
            for name, cls, status in UNITTEST_RESULT_RE.findall(row.test_output or ""):
                test_results[f"{name} ({cls})"] = int(status == "ok")

            records.append(
                Record(
                    benchmark="swebench_verified",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    result=int(row.reward),
                    metadata={
                        "raw_source": "coderforge",
                        "trajectory_id": row.trajectory_id,
                        "run_id": row.run_id,
                        "num_steps": int(row.num_steps)
                        if pd.notna(row.num_steps)
                        else None,
                        "unit_results": test_results or None,
                    },
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )

    records.extend(devin_records(task_ids, "swebench_verified"))
    trial_indices = Counter()
    for record in records:
        key = (record.agent_id, record.task_id)
        record.trial = trial_indices[key]
        trial_indices[key] += 1

    return tasks, [], records


def trajectories():
    task_ids = {
        row["instance_id"]
        for row in load_jsonl(RAW / "swebench_verified" / "tasks.jsonl")
    }
    trial_indices = Counter()
    response_path = RAW / "agent_psychometrics" / "swebench_verified" / "responses.jsonl"
    for row in load_jsonl(response_path):
        info = parse_subject_id(row["subject_id"])
        model, reasoning_effort = standardize_model(info["agent_model"])
        scaffold = standardize_scaffold_name(info["agent_scaffold"])
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        for task_id in row["responses"]:
            trial_indices[(agent_id, task_id)] += 1

    root = RAW / "swebench_verified_traj"
    for local_dir, _, raw_model, raw_scaffold, result_filename in LIVESWEAGENT_TRAJ_SOURCES:
        result_path = root / local_dir / result_filename
        if not result_path.exists():
            continue

        report = json.loads(result_path.read_text())
        model, reasoning_effort = standardize_model(raw_model)
        scaffold = standardize_scaffold_name(raw_scaffold)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        seen = set()
        for field in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids"):
            for task_id in report.get(field) or []:
                if task_id in task_ids and task_id not in seen:
                    trial_indices[(agent_id, task_id)] += 1
                    seen.add(task_id)

    root = RAW / "coderforge_swebench_verified"
    model, reasoning_effort = standardize_model(CODERFORGE_MODEL)
    scaffold = standardize_scaffold_name(CODERFORGE_SCAFFOLD)
    agent_id = format_agent_id(model, scaffold, reasoning_effort)
    model_date = resolve_model_date(model, CODERFORGE_MODEL)
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["trajectory_id", "run_id", "reward", "messages"])
        for row in frame.itertuples(index=False):
            task_id = row.trajectory_id.rsplit("_run", 1)[0]
            if task_id not in task_ids:
                continue

            key = (agent_id, task_id)
            trial = trial_indices[key]
            trial_indices[key] += 1
            if not row.messages:
                continue

            yield Trajectory(
                benchmark="swebench_verified",
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=trial,
                content=row.messages,
                metadata={
                    "raw_source": "coderforge",
                    "trajectory_id": row.trajectory_id,
                    "run_id": row.run_id,
                    "source_file": path.name,
                },
                source_platform=SourcePlatform.HUGGINGFACE,
            )
