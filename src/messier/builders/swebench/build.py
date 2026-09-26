"""
Import original SWE-bench tasks and results.

Environment failures remain unscored instead of being counted as task failures.
"""

import json
from collections import Counter
import pandas as pd
from ..benchmarks import REPO_DESCRIPTIONS
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ..swebench_verified.utils import devin_records, parse_subject_id
from ...io import RAW
from ...models import (
    ActionSpaceType,
    BuildResult,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    ScoringRule,
    Task,
    VerifierMode,
    VerifierType,
)
from .constants import BENCHMARK, VERIFIED_BENCHMARK, VERIFIED_DATASET_NAME


def build() -> BuildResult:
    root = RAW / "swebench"
    tasks_dir = root / "data" / "tasks"
    if (
        not (root / "task_manifest.csv").exists()
        or not (root / "run_manifest.csv").exists()
        or not tasks_dir.is_dir()
    ):
        return [], [], []

    manifest = pd.read_csv(root / "task_manifest.csv")
    benchmark_by_task = dict(
        zip(
            manifest["task_id"],
            [
                VERIFIED_BENCHMARK
                if dataset_name == VERIFIED_DATASET_NAME
                else BENCHMARK
                for dataset_name in manifest["dataset_name"]
            ],
        )
    )
    tasks: list[Task] = []
    for row in manifest.itertuples(index=False):
        if benchmark_by_task[row.task_id] != BENCHMARK:
            continue

        metadata = json.loads((tasks_dir / row.task_id / "metadata.json").read_text())
        problem = (tasks_dir / row.task_id / "problem_statement.md").read_text()
        test_patch = (tasks_dir / row.task_id / "test_patch.diff").read_text()
        gold_patch = (tasks_dir / row.task_id / "oracle_patch_analysis_only.diff").read_text()
        repository = row.repo
        tasks.append(
            Task(
                benchmark=BENCHMARK,
                task_id=row.task_id,
                environment_id=repository,
                environment_description=(f"A checkout of {repository}. {REPO_DESCRIPTIONS[repository].rstrip('.')}"),
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
                        "ref": f"github://{repository}@{row.base_commit}",
                        "snapshot": None,
                    },
                    "base_commit": row.base_commit,
                    "environment_setup_commit": metadata.get("environment_setup_commit"),
                    "version": metadata.get("version"),
                    "url": f"https://github.com/{repository}/tree/{row.base_commit}",
                    "source_dataset": row.dataset_name,
                },
                task_description=problem,
                task_metadata={
                    "hints_text": metadata.get("hints_text"),
                    "split": row.split,
                    "near_duplicate_group_id": (
                        None
                        if pd.isna(getattr(row, "near_duplicate_group_id", None))
                        else row.near_duplicate_group_id
                    ),
                    "test_patch": test_patch,
                },
                gold_answer=gold_patch,
                task_date=resolve_task_date(BENCHMARK, row.created_at),
                verifiers=[
                    {
                        "type": VerifierType.SCRIPT,
                        "mode": VerifierMode.FINAL,
                        "name": "swebench_harness",
                    }
                ],
                difficulty_label=metadata.get("difficulty"),
                scoring_rule=ScoringRule.DIRECT,
                source_platform=SourcePlatform.GITHUB,
            )
        )

    records: list[Record] = []
    attempts = pd.read_csv(root / "run_manifest.csv")
    public = attempts["model_id"].str.startswith("public_swebench/", na=False)
    has_artifact = (
        attempts[["trace_path", "patch_path", "stdout_path", "stderr_path"]]
        .notna()
        .any(axis=1)
    )
    attempts = attempts[public | has_artifact]
    public = attempts["model_id"].str.startswith("public_swebench/", na=False)
    repeated_subset = public & ~attempts["run_id"].str.startswith("public_swebench:test:", na=False)
    attempts = attempts[~repeated_subset & attempts["success"].notna()]
    identity_cache: dict[tuple[str, str | None], dict] = {}
    for row in attempts.itertuples(index=False):
        benchmark = benchmark_by_task.get(row.task_id)
        if benchmark is None:
            continue

        key = (row.model_id, row.scaffold_id)
        identity = identity_cache.get(key)
        if identity is None:
            if row.model_id.startswith("public_swebench/"):
                identity = parse_subject_id(row.model_id.removeprefix("public_swebench/"))
                raw_model = identity["agent_model"]
                identity["agent_model"], identity["model_reasoning_effort"] = (standardize_model(identity["agent_model"]))
                identity["agent_scaffold"] = standardize_scaffold_name(identity["agent_scaffold"])
                identity["model_date"] = resolve_model_date(identity["agent_model"], raw_model)
            else:
                model, reasoning_effort = standardize_model(row.model_id)
                identity = {
                    "agent_model": model,
                    "model_reasoning_effort": reasoning_effort,
                    "agent_scaffold": standardize_scaffold_name(row.scaffold_id)
                    if row.scaffold_id
                    else None,
                    "model_date": resolve_model_date(model, row.model_id),
                }
            identity_cache[key] = identity

        agent_id = (
            row.model_id.removeprefix("public_swebench/")
            if row.model_id.startswith("public_swebench/")
            else row.model_id
        )
        is_environment_error = row.error_type in {"environment_error", "unknown_error",}
        result = None if is_environment_error else int(bool(row.success))
        records.append(
            Record(
                benchmark=benchmark,
                task_id=row.task_id,
                agent_id=agent_id,
                agent_model=identity["agent_model"],
                model_reasoning_effort=identity["model_reasoning_effort"],
                agent_scaffold=identity["agent_scaffold"],
                model_date=identity["model_date"],
                trial=int(row.attempt_index),
                result=result,
                metadata={
                    "run_id": row.run_id,
                    "source_reward": float(row.final_reward)
                    if pd.notna(row.final_reward)
                    else None,
                    "raw_model_id": row.model_id,
                    "raw_scaffold_id": row.scaffold_id,
                    "budget_id": row.budget_id,
                    "split": row.split,
                    "seed": int(row.seed) if pd.notna(row.seed) else None,
                    "num_steps": float(row.num_steps)
                    if pd.notna(row.num_steps)
                    else None,
                    "error_type": row.error_type,
                    "total_cost": float(row.total_cost)
                    if pd.notna(row.total_cost)
                    else None,
                },
                source_platform=SourcePlatform.GITHUB,
            )
        )

    # add Devin results for tasks in the original SWE-bench split
    swebench_only_ids = {
        task_id
        for task_id, benchmark in benchmark_by_task.items()
        if benchmark == BENCHMARK
    }
    records.extend(devin_records(swebench_only_ids, BENCHMARK))
    trial_indices = Counter()
    for record in records:
        agent_id = format_agent_id(
            record.agent_model,
            record.agent_scaffold,
            record.model_reasoning_effort,
        )
        record.agent_id = agent_id
        key = (agent_id, record.task_id)
        record.trial = trial_indices[key]
        trial_indices[key] += 1

    return tasks, [], records
