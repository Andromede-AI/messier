"""
Add the original DABStep task details to imported Harbor runs.
"""

import re
import shutil
import tomllib
from ..dates import resolve_task_date
from ...io import RAW, TASK_FILES_PATH
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
from .constants import DABSTEP_CONTEXT_FILES, ENV_DESCRIPTIONS

BENCH = "dabstep"


def prepare_files() -> tuple[list[dict[str, str]], str]:
    """
    Copy the shared task data and read the verifier source.

    :return: Agent-visible input-file references and the shared verifier code.
    """
    input_files = []
    for filename in DABSTEP_CONTEXT_FILES:
        source_path = RAW / BENCH / "data" / "context" / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"DABStep context file is missing: {filename}")

        release_path = TASK_FILES_PATH / BENCH / "context" / filename
        release_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, release_path)
        input_files.append(
            {
                "path": f"/app/data/{filename}",
                "dataset_path": f"task_files/{BENCH}/context/{filename}",
            }
        )

    scorer_paths = sorted((RAW / BENCH / "tasks" / "adyen").glob("*/*/tests/scorer.py"))
    if not scorer_paths:
        raise FileNotFoundError("DABStep scorer is missing. Run the upstream fetch first.")

    scorer = scorer_paths[0].read_bytes()
    if any(path.read_bytes() != scorer for path in scorer_paths[1:]):
        raise ValueError("DABStep task packages contain different scorer implementations")

    return input_files, scorer.decode()


def task_definition(
    task_id: str,
    instruction: str | None,
    task_ref: str | None,
    slug: str,
    source_task: str,
    input_files: list[dict[str, str]],
    script_code: str,
) -> Task:
    """
    Build one task from the exact package used by the Harbor run.

    :param task_id: DABStep task ID.
    :param instruction: Instruction recovered from the Harbor trajectory.
    :param task_ref: Immutable Harbor task-package reference.
    :param slug: Harbor dataset slug.
    :param source_task: Full task name recorded by Harbor.
    :param input_files: Shared DABStep data included in the release.
    :param script_code: Source code for the shared DABStep verifier.
    :return: Task with its expected answer, difficulty, and scorer details.
    """
    if not task_ref:
        raise ValueError(f"DABStep task {source_task!r} has no package reference")

    package_root = RAW / BENCH / "tasks" / "adyen" / task_id
    if task_ref.startswith("sha256:"):
        package_path = RAW / BENCH / "tasks" / source_task / task_ref.removeprefix("sha256:")
    else:
        packages = [path for path in package_root.iterdir() if path.is_dir()]
        if len(packages) != 1:
            raise FileNotFoundError(f"expected one DABStep package for task {task_id}, found {len(packages)}")
        package_path = packages[0]
    instruction_path = package_path / "instruction.md"
    solution_path = package_path / "solution" / "solve.sh"
    task_config_path = package_path / "task.toml"
    if any(not path.is_file() for path in (instruction_path, solution_path, task_config_path)):
        raise FileNotFoundError(f"DABStep package is incomplete for {source_task}@{task_ref}")

    package_instruction = instruction_path.read_text().strip()
    if instruction and instruction.strip() != package_instruction:
        raise ValueError(f"DABStep instruction differs from package for {source_task}")

    answer_match = re.search(
        r"^cat <<'ANSWER_EOF' > /app/answer\.txt\n(?P<answer>.*?)\nANSWER_EOF$",
        solution_path.read_text(),
        flags=re.MULTILINE | re.DOTALL,
    )
    if answer_match is None:
        raise ValueError(f"expected answer is missing for {source_task}")

    metadata = tomllib.loads(task_config_path.read_text()).get("metadata") or {}
    return Task(
        benchmark=BENCH,
        task_id=task_id,
        environment_id=task_id,
        environment_description=ENV_DESCRIPTIONS[BENCH],
        environment_metadata={
            "action_space": {
                "types": [ActionSpaceType.SHELL],
                "description": "Shell and file operations for reading the supplied data and documentation, analyzing transactions, and writing the answer.",
            },
            "environment_state": {
                "type": EnvironmentStateType.FILESYSTEM,
                "access": EnvironmentStateAccess.READ_WRITE,
                "ref": f"task_files/{BENCH}/context",
            },
            "url": "https://huggingface.co/datasets/adyen/DABstep",
        },
        task_description=package_instruction,
        task_metadata={
            "harbor_dataset_slug": slug,
            "harbor_task_name": source_task,
            "harbor_task_ref": task_ref,
            "category": metadata.get("category"),
            "tags": metadata.get("tags") or [],
            "extra_context": {"input_files": input_files},
        },
        gold_answer=answer_match.group("answer").strip(),
        task_date=resolve_task_date(BENCH),
        verifiers=[
            {
                "type": VerifierType.SCRIPT,
                "mode": VerifierMode.FINAL,
                "name": "dabstep_scorer",
                "script_code": script_code,
                "comparison": "fuzzy",
            }
        ],
        difficulty_label=metadata.get("difficulty"),
        scoring_rule=ScoringRule.DIRECT,
        source_platform=SourcePlatform.HUGGINGFACE,
    )
