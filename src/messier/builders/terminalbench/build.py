"""
Import TerminalBench tasks and results.
"""

import re
from collections import Counter, defaultdict, deque
from math import sqrt
import pandas as pd
import pyarrow.parquet as pq
import yaml
from ..benchmarks import UPSTREAM_COMMITS
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date, to_iso_date
from ...io import RAW, load_jsonl
from ...models import (
    BuildResult,
    ActionSpaceType,
    DataProvider,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    HumanEffort,
    Record,
    Task,
    Trajectory,
    VerifierMode,
    VerifierType,
)

# terminalbench contains shell tasks graded by one test-based verifier per task

TB_SHA = UPSTREAM_COMMITS["terminalbench"]
YH_REPO = "yoonholee/terminalbench-trajectories"
YH_FILES = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
YH_PROVIDER_RE = re.compile(r"@[\w-]+$")
YH_ACCOUNT_PREFIX = re.compile(r"^accounts/[^/]+/models/")
YH_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def standardize_agent(raw_model: str, raw_scaffold: str):
    model_name = YH_PROVIDER_RE.sub("", raw_model)
    model_name = YH_ACCOUNT_PREFIX.sub("", model_name)
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]

    model, reasoning_effort = standardize_model(YH_DATE_SUFFIX_RE.sub("", model_name))
    scaffold = standardize_scaffold_name(raw_scaffold)
    agent_id = format_agent_id(model, scaffold, reasoning_effort)
    model_date = resolve_model_date(model, raw_model)
    return model, reasoning_effort, scaffold, agent_id, model_date


def build() -> BuildResult:
    psychometrics_root = RAW / "agent_psychometrics" / "terminalbench"
    upstream_tasks_root = RAW / "terminal-bench" / "original-tasks"

    def human_effort(expert_minutes, junior_minutes) -> HumanEffort | None:
        expert = float(expert_minutes) if expert_minutes else None
        junior = float(junior_minutes) if junior_minutes else None
        if expert is None and junior is None:
            return None

        median = sqrt(expert * junior) if expert and junior else expert or junior
        return HumanEffort(
            minutes_median=median,
            minutes_low=expert,
            minutes_high=junior,
        )

    tasks = []
    for task_row in load_jsonl(psychometrics_root / "tasks.jsonl"):
        task_id = task_row["task_id"]
        task_root = upstream_tasks_root / task_id
        task_yaml = task_root / "task.yaml"
        task_config = {}

        if task_yaml.exists():
            task_config = yaml.safe_load(task_yaml.read_text()) or {}

        upstream_url = (
            "https://github.com/laude-institute/terminal-bench/tree/"
            f"{TB_SHA}/original-tasks/{task_id}"
        )
        dockerfile_path = task_root / "Dockerfile"
        dockerfile = dockerfile_path.read_text() if dockerfile_path.exists() else None
        environment_metadata = {
            "action_space": {
                "types": [ActionSpaceType.SHELL],
                "description": "Shell commands executed in the Linux environment.",
                "max_agent_timeout_sec": task_config.get("max_agent_timeout_sec"),
            },
            "environment_state": {
                "type": EnvironmentStateType.FILESYSTEM,
                "access": EnvironmentStateAccess.READ_WRITE,
                "ref": upstream_url,
                "snapshot": {"dockerfile": dockerfile} if dockerfile else None,
            },
            "url": upstream_url,
        }
        if task_config.get("author_name"):
            environment_metadata["author"] = task_config["author_name"]

        tasks.append(
            Task(
                benchmark="terminalbench",
                task_id=task_id,
                environment_id=task_id,
                environment_description=(
                    "A Linux environment containing files, software, and services that can be "
                    "inspected and modified through shell commands."
                ),
                environment_metadata=environment_metadata,
                # prefer the upstream task instruction
                task_description=task_config.get("instruction")
                or task_row["problem_statement"],
                task_metadata={
                    "category": task_row.get("category"),
                    "tags": task_row.get("tags"),
                    "parser_name": task_config.get("parser_name"),
                    "max_test_timeout_sec": task_config.get("max_test_timeout_sec"),
                    "run_tests_in_same_shell": task_config.get("run_tests_in_same_shell"),
                },
                gold_answer=task_row.get("patch"),
                task_date=resolve_task_date("terminalbench"),
                verifiers=[
                    {
                        "type": VerifierType.SCRIPT,
                        "mode": VerifierMode.FINAL,
                        "name": "terminalbench_harness",
                        "script_code": task_row.get("tests"),
                    }
                ],
                difficulty_label=task_row.get("difficulty"),
                human=human_effort(
                    task_config.get("expert_time_estimate_min"),
                    task_config.get("junior_time_estimate_min"),
                ),
                source_platform=SourcePlatform.GITHUB,
                data_provider=DataProvider.AGENT_PSYCHOMETRICS,
            )
        )

    records = []
    trial_indices = Counter()

    for row in load_jsonl(psychometrics_root / "responses.jsonl"):
        model, reasoning_effort = standardize_model(row.get("model"))
        scaffold = standardize_scaffold_name(row.get("agent"))
        model_date = None
        if model != "multiple":
            model_date = resolve_model_date(model, row.get("model")) or to_iso_date(row.get("date"))
        agent_id = format_agent_id(model, scaffold, reasoning_effort)

        for task_id, result in row["responses"].items():
            key = (agent_id, task_id)
            trial = trial_indices[key]
            records.append(
                Record(
                    benchmark="terminalbench",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=trial,
                    result=int(result),
                    metadata={
                        "raw_source": "agent_psychometrics",
                        "accuracy_headline": row.get("accuracy"),
                        "rank": row.get("rank"),
                    },
                    source_platform=SourcePlatform.GITHUB,
                    data_provider=DataProvider.AGENT_PSYCHOMETRICS,
                )
            )
            trial_indices[key] += 1

    trajectory_root = RAW / "yoonholee_terminalbench"
    trajectory_paths = sorted(trajectory_root.glob("*.parquet"))

    if trajectory_paths:
        columns = [
            "task_name",
            "agent",
            "model",
            "reward",
            "duration_seconds",
            "input_tokens",
            "output_tokens",
            "cache_tokens",
            "cost_cents",
            "trial_name",
            "trial_id",
            "started_at",
        ]
        frame = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in trajectory_paths],
            ignore_index=True,
        )
        frame = frame[frame["task_name"].isin({task.task_id for task in tasks})]
        frame = frame.sort_values(
            ["task_name", "model", "agent", "started_at", "trial_id"]
        )
        for row in frame.itertuples(index=False):
            model, reasoning_effort, scaffold, agent_id, model_date = standardize_agent(row.model, row.agent)
            key = (agent_id, row.task_name)
            trial = trial_indices[key]
            records.append(
                Record(
                    benchmark="terminalbench",
                    task_id=row.task_name,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=trial,
                    result=int(row.reward),
                    metadata={
                        "raw_source": "yoonholee",
                        "raw_model": row.model,
                        "trial_name": row.trial_name,
                        "trial_id": row.trial_id,
                        "started_at": row.started_at,
                        "duration_seconds": float(row.duration_seconds)
                        if pd.notna(row.duration_seconds)
                        else None,
                        "cost_cents": float(row.cost_cents)
                        if pd.notna(row.cost_cents)
                        else None,
                        "input_tokens": int(row.input_tokens)
                        if pd.notna(row.input_tokens)
                        else None,
                        "output_tokens": int(row.output_tokens)
                        if pd.notna(row.output_tokens)
                        else None,
                        "cache_tokens": int(row.cache_tokens)
                        if pd.notna(row.cache_tokens)
                        else None,
                    },
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )
            trial_indices[key] += 1

    return tasks, [], records


def trajectories():
    psychometrics_root = RAW / "agent_psychometrics" / "terminalbench"
    task_ids = {row["task_id"] for row in load_jsonl(psychometrics_root / "tasks.jsonl")}
    trial_indices = Counter()
    for row in load_jsonl(psychometrics_root / "responses.jsonl"):
        model, reasoning_effort = standardize_model(row.get("model"))
        scaffold = standardize_scaffold_name(row.get("agent"))
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        for task_id in row["responses"]:
            trial_indices[(agent_id, task_id)] += 1

    root = RAW / "yoonholee_terminalbench"
    paths = sorted(root.glob("*.parquet"))
    columns = ["task_name", "agent", "model", "trial_id", "started_at"]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True,)
    frame = frame[frame["task_name"].isin(task_ids)]
    frame = frame.sort_values(["task_name", "model", "agent", "started_at", "trial_id"])

    trials_by_source = defaultdict(deque)
    for row in frame.itertuples(index=False):
        model, _, _, agent_id, _ = standardize_agent(row.model, row.agent)
        key = (agent_id, row.task_name)
        source_key = (
            row.task_name,
            row.agent,
            row.model,
            row.trial_id,
            row.started_at,
        )
        trials_by_source[source_key].append(trial_indices[key])
        trial_indices[key] += 1

    for path in paths:
        batches = pq.ParquetFile(path).iter_batches(columns=[*columns, "steps"], batch_size=256)
        for batch in batches:
            for row in batch.to_pylist():
                source_key = tuple(row[column] for column in columns)
                source_trials = trials_by_source.get(source_key)
                if not source_trials:
                    continue

                trial = source_trials.popleft()
                if row["steps"] in {None, "", "null", "[]"}:
                    continue

                model, reasoning_effort, scaffold, agent_id, model_date = standardize_agent(row["model"], row["agent"])
                yield Trajectory(
                    benchmark="terminalbench",
                    task_id=row["task_name"],
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=trial,
                    content=row["steps"],
                    metadata={
                        "trial_id": row["trial_id"],
                        "started_at": row["started_at"],
                        "source_file": path.name,
                    },
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
