"""Add ScienceAgentBench task definitions to imported Harbor runs."""

import tomllib
from ..benchmarks import DEFAULT_SCORING_RULES
from ..dates import resolve_task_date
from ...io import RAW
from ...models import (
    ActionSpaceType,
    EnvironmentStateAccess,
    EnvironmentStateType,
    ScoringRule,
    SourcePlatform,
    Task,
    VerifierMode,
    VerifierType,
)
from .constants import ENV_DESCRIPTIONS

BENCH = "scienceagentbench"


def task_definition(
    task_id: str,
    instruction: str | None,
    task_ref: str | None,
    slug: str,
    source_task: str,
) -> Task:
    """
    Build one task from the exact package used by the Harbor run.

    :param task_id: ScienceAgentBench task ID.
    :param instruction: Instruction recovered from the Harbor trajectory.
    :param task_ref: Immutable Harbor task-package reference.
    :param slug: Harbor dataset slug.
    :param source_task: Full task name recorded by Harbor.
    :return: Task definition without the protected benchmark artifacts.
    """
    if not task_ref:
        raise ValueError(f"ScienceAgentBench task {source_task!r} has no package reference")

    package_path = (
        RAW
        / BENCH
        / "tasks"
        / source_task
        / task_ref.removeprefix("sha256:")
    )
    instruction_path = package_path / "instruction.md"
    evaluator_path = package_path / "tests" / "eval_program.py"
    task_config_path = package_path / "task.toml"
    required_paths = (instruction_path, evaluator_path, task_config_path)
    if any(not path.exists() for path in required_paths):
        raise FileNotFoundError(
            f"ScienceAgentBench package is incomplete for {source_task}@{task_ref}"
        )

    package_instruction = instruction_path.read_text().strip()
    if instruction and instruction.strip() != package_instruction:
        raise ValueError(
            f"ScienceAgentBench instruction differs from package for {source_task}"
        )

    task_config = tomllib.loads(task_config_path.read_text())
    visual_judge_path = package_path / "tests" / "visual_judge.py"

    metadata = task_config.get("metadata") or {}
    return Task(
        benchmark=BENCH,
        task_id=task_id,
        environment_id=task_id,
        environment_description=ENV_DESCRIPTIONS[BENCH],
        environment_metadata={
            "action_space": {
                "types": [ActionSpaceType.SHELL],
                "description": "Shell and file operations for inspecting scientific data, writing a Python program, installing dependencies, and running the program.",
            },
            "environment_state": {
                "type": EnvironmentStateType.FILESYSTEM,
                "access": EnvironmentStateAccess.READ_WRITE,
                "ref": task_ref,
            },
            "url": "https://github.com/OSU-NLP-Group/ScienceAgentBench",
        },
        task_description=package_instruction,
        task_metadata={
            "harbor_dataset_slug": slug,
            "harbor_task_name": source_task,
            "harbor_task_ref": task_ref,
            "category": metadata.get("category"),
            "tags": metadata.get("tags") or [],
        },
        gold_answer=None,
        task_date=resolve_task_date(BENCH),
        verifiers=[
            {
                "type": VerifierType.SCRIPT,
                "mode": VerifierMode.FINAL,
                "uses_llm_judge": visual_judge_path.is_file(),
            }
        ],
        difficulty_label=metadata.get("difficulty"),
        scoring_rule=ScoringRule(DEFAULT_SCORING_RULES[BENCH]),
        source_platform=SourcePlatform.HUGGINGFACE,
    )
