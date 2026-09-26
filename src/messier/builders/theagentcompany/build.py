"""
Import TheAgentCompany tasks, trials, and verifier results.
"""

import ast
import gzip
import json
from collections import Counter
from ..benchmarks import UPSTREAM_COMMITS
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date, to_iso_date
from ...io import RAW
from ...models import (
    ActionSpaceType,
    BuildResult,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    RecordType,
    ScoringRule,
    Task,
    Trajectory,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)
from .constants import (
    ENVIRONMENT_DESCRIPTION,
    HANDLED_FILES,
    RESULT_FILE_RE,
    SUB_DATE_RE,
)
from .utils import is_gold_file, parse_checkpoints, parse_submission, read_text_capped


def build() -> BuildResult:
    source_root = RAW / "theagentcompany"
    tasks_root = source_root / "main" / "workspaces" / "tasks"
    results_root = source_root / "experiments" / "evaluation" / "1.0.0"

    def read_if_exists(path):
        return path.read_text() if path.exists() else None

    tasks_by_id: dict[str, Task] = {}
    verifier_definitions: dict[str, VerifierDefinition] = {}

    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue

        task_id = task_dir.name
        role = task_id.split("-")[0]
        description = read_if_exists(task_dir / "task.md")
        checkpoints = read_if_exists(task_dir / "checkpoints.md")
        evaluator = read_if_exists(task_dir / "evaluator.py")
        dependencies = read_if_exists(task_dir / "dependencies.yml")
        dockerfile = read_if_exists(task_dir / "Dockerfile")
        makefile = read_if_exists(task_dir / "Makefile")

        custom_scoring = False
        if evaluator:
            tree = ast.parse(evaluator)
            for node in ast.walk(tree):
                is_result_call = (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Result"
                )
                if not is_result_call:
                    continue

                has_scoring_argument = len(node.args) > 1
                if not has_scoring_argument:
                    for keyword in node.keywords:
                        if keyword.arg == "scoring_strategy":
                            has_scoring_argument = True
                            break

                if has_scoring_argument:
                    custom_scoring = True
                    break

        rubric = parse_checkpoints(checkpoints) if checkpoints else []
        gold_files: dict[str, str | None] = {}

        for path in sorted(task_dir.iterdir()):
            if not path.is_file() or path.name in HANDLED_FILES:
                continue

            if is_gold_file(path.name):
                gold_files[path.name] = read_text_capped(path)

        gold_answer: dict | list | None = None
        if rubric or gold_files:
            gold_answer = {
                "rubric": [
                    {
                        "step": cp["step"],
                        "points": cp["points"],
                        "description": cp["expected"],
                    }
                    for cp in rubric
                ]
                or None,
                "files": gold_files or None,
            }

        verifiers = [
            {
                "type": VerifierType.SCRIPT,
                "mode": VerifierMode.FINAL,
                "name": "checkpoint",
                "source_index": index,
                "source_step": checkpoint["step"],
                "points": checkpoint["points"],
                **({"script_code": evaluator} if evaluator else {}),
            }
            for index, checkpoint in enumerate(rubric)
        ]
        if not verifiers and evaluator:
            verifiers = [
                {
                    "type": VerifierType.SCRIPT,
                    "mode": VerifierMode.FINAL,
                    "name": "evaluator",
                    "script_code": evaluator,
                }
            ]

        scoring_rule = ScoringRule.THRESHOLD if custom_scoring else ScoringRule.ALL_PASS
        task = Task(
            benchmark="theagentcompany",
            task_id=task_id,
            environment_id=role,
            environment_description=ENVIRONMENT_DESCRIPTION,
            environment_metadata={
                "action_space": {
                    "types": [ActionSpaceType.SHELL, ActionSpaceType.UI],
                    "description": (
                        "Browser interactions with GitLab, Plane, ownCloud, and RocketChat, "
                        "together with shell commands in a Linux workspace."
                    ),
                },
                "environment_state": {
                    "type": EnvironmentStateType.FILESYSTEM,
                    "access": EnvironmentStateAccess.READ_WRITE,
                    "ref": f"github://TheAgentCompany/TheAgentCompany@{UPSTREAM_COMMITS['theagentcompany_main']}/workspaces/tasks/{task_id}",
                    "snapshot": {"dockerfile": dockerfile, "makefile": makefile},
                },
                "url": f"https://github.com/TheAgentCompany/TheAgentCompany/tree/{UPSTREAM_COMMITS['theagentcompany_main']}/workspaces/tasks/{task_id}",
                "dependencies": dependencies,
            },
            task_description=description,
            task_metadata={"role": role},
            gold_answer=gold_answer,
            task_date=resolve_task_date("theagentcompany"),
            verifiers=verifiers,
            scoring_rule=scoring_rule,
            source_platform=SourcePlatform.GITHUB,
        )
        tasks_by_id[task_id] = task

        for index, checkpoint in enumerate(rubric):
            verifier_id = f"{task_id}::checkpoint_{index}"
            verifier_definitions[verifier_id] = VerifierDefinition(
                benchmark="theagentcompany",
                task_id=task_id,
                verifier_id=verifier_id,
                description=checkpoint["expected"],
                verifier={
                    "type": VerifierType.SCRIPT,
                    "mode": VerifierMode.FINAL,
                    "name": "checkpoint",
                    "source_index": index,
                    "source_step": checkpoint["step"],
                    "points": checkpoint["points"],
                    **({"script_code": evaluator} if evaluator else {}),
                },
                metadata={
                    "source_index": index,
                    "source_step": checkpoint["step"],
                    "points": checkpoint["points"],
                },
                source_platform=SourcePlatform.GITHUB,
            )

        if not rubric and evaluator:
            verifier_id = f"{task_id}::verifier:0"
            verifier_definitions[verifier_id] = VerifierDefinition(
                benchmark="theagentcompany",
                task_id=task_id,
                verifier_id=verifier_id,
                verifier={
                    "type": VerifierType.SCRIPT,
                    "mode": VerifierMode.FINAL,
                    "name": "evaluator",
                    "script_code": evaluator,
                },
                source_platform=SourcePlatform.GITHUB,
            )

    records: list[Record] = []
    trial_indices = Counter()

    for submission_dir in sorted(results_root.iterdir()):
        if not submission_dir.is_dir():
            continue

        raw_scaffold, raw_model = parse_submission(submission_dir.name)
        date_match = SUB_DATE_RE.match(submission_dir.name)
        submission_date = to_iso_date(date_match.group(1)) if date_match else None
        if raw_model == "multiple":
            model = raw_model
            reasoning_effort = None
        else:
            model, reasoning_effort = standardize_model(raw_model)

        scaffold = (
            raw_scaffold
            if model == "multiple"
            else standardize_scaffold_name(raw_scaffold)
        )

        model_date = (
            None
            if model == "multiple"
            else resolve_model_date(model, raw_model)
        )

        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        for result_path in sorted((submission_dir / "results").glob("*.json")):
            name_match = RESULT_FILE_RE.match(result_path.name)
            if not name_match:
                continue

            task_id = name_match.group(1)
            task = tasks_by_id.get(task_id)
            if task is None:
                continue

            source_result = json.loads(result_path.read_text())
            final_score = source_result.get("final_score", {})
            total_points = final_score.get("total")
            earned_points = final_score.get("result")

            if not total_points or earned_points is None:
                continue

            trial_key = (agent_id, task_id)
            trial = trial_indices[trial_key]
            trial_indices[trial_key] += 1
            checkpoint_results = source_result.get("checkpoints") or []
            source_metadata = {
                "source_submission": submission_dir.name,
                "submission_date": submission_date,
            }

            records.append(
                Record(
                    benchmark="theagentcompany",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=trial,
                    result=int(earned_points == total_points),
                    metadata={
                        **source_metadata,
                        "source_result": earned_points,
                        "source_total": total_points,
                        "source_fraction": earned_points / total_points,
                    },
                    source_platform=SourcePlatform.GITHUB,
                )
            )

            if not checkpoint_results or len(checkpoint_results) != len(task.verifiers):
                continue

            checkpoint_points = [checkpoint.get("total") for checkpoint in checkpoint_results]
            verifier_points = [verifier.points for verifier in task.verifiers]
            if checkpoint_points != verifier_points:
                continue

            all_checkpoints_pass = True
            for checkpoint in checkpoint_results:
                if checkpoint.get("result") != checkpoint.get("total"):
                    all_checkpoints_pass = False
                    break

            if all_checkpoints_pass != (earned_points == total_points):
                continue

            for index, checkpoint_result in enumerate(checkpoint_results):
                result = checkpoint_result.get("result")
                total = checkpoint_result.get("total")
                if result is None or not total:
                    continue

                records.append(
                    Record(
                        benchmark="theagentcompany",
                        task_id=task_id,
                        verifier_id=f"{task_id}::checkpoint_{index}",
                        record_type=RecordType.VERIFIER_RESULT,
                        agent_id=agent_id,
                        agent_model=model,
                        model_reasoning_effort=reasoning_effort,
                        agent_scaffold=scaffold,
                        model_date=model_date,
                        trial=trial,
                        result=result / total,
                        metadata={
                            **source_metadata,
                            "source_result": result,
                            "source_total": total,
                            "source_fraction": result / total,
                        },
                        source_platform=SourcePlatform.GITHUB,
                    )
                )

    return (list(tasks_by_id.values()), list(verifier_definitions.values()), records)


def trajectories():
    source_root = RAW / "theagentcompany"
    results_root = source_root / "experiments" / "evaluation" / "1.0.0"
    trial_indices = Counter()

    for submission_dir in sorted(results_root.iterdir()):
        if not submission_dir.is_dir():
            continue

        raw_scaffold, raw_model = parse_submission(submission_dir.name)

        if raw_model == "multiple":
            model = raw_model
            reasoning_effort = None
        else:
            model, reasoning_effort = standardize_model(raw_model)

        scaffold = raw_scaffold if model == "multiple" else standardize_scaffold_name(raw_scaffold)
        model_date = None if model == "multiple" else resolve_model_date(model, raw_model)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)

        paths_by_task = {}
        trajectory_root = submission_dir / "trajectories"
        if trajectory_root.exists():
            for path in sorted(trajectory_root.iterdir()):
                if not path.is_file():
                    continue

                task_id = path.name.removesuffix(".gz")
                for suffix in (".json", ".md", ".yaml", ".yml", ".txt"):
                    task_id = task_id.removesuffix(suffix)

                task_id = task_id.removeprefix("traj_").removesuffix("-image")
                if task_id in paths_by_task:
                    raise ValueError(f"multiple trajectories for {submission_dir.name}/{task_id}")

                paths_by_task[task_id] = path

        for result_path in sorted((submission_dir / "results").glob("*.json")):
            match = RESULT_FILE_RE.match(result_path.name)
            if not match:
                continue

            task_id = match.group(1)
            source_result = json.loads(result_path.read_text())
            final_score = source_result.get("final_score", {})
            if not final_score.get("total") or final_score.get("result") is None:
                continue

            key = (agent_id, task_id)
            trial = trial_indices[key]
            trial_indices[key] += 1
            path = paths_by_task.get(task_id)
            if path is None:
                continue

            if path.suffix == ".gz":
                with gzip.open(path, "rt") as file:
                    content = file.read()
            else:
                content = path.read_text()

            yield Trajectory(
                benchmark="theagentcompany",
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=trial,
                content=content,
                metadata={"source_path": str(path.relative_to(source_root))},
                source_platform=SourcePlatform.GITHUB,
            )
