import shutil
from typing import Iterator
from ..dates import resolve_task_date
from ...io import RAW, TASK_FILES_PATH
from ...models import (
    ActionSpaceType,
    ScoringRule,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    ModelReasoningEffort,
    Record,
    RecordType,
    Task,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)
from .constants import ENV_DESCRIPTIONS, HARVEYAI_JUDGE

BENCH = "harveyai-lab"


def task_definition(
    task_id: str,
    instruction: str | None,
    task_ref: str | None,
    reward_details: dict,
    slug: str,
    source_task: str,
) -> Task:
    if not task_ref:
        raise ValueError(f"Harvey LAB task {source_task!r} has no package reference")

    package_path = (
        RAW
        / "harveyai_lab"
        / "tasks"
        / source_task
        / task_ref.removeprefix("sha256:")
    )
    instruction_path = package_path / "instruction.md"
    documents_path = package_path / "environment" / "documents"
    if not instruction_path.exists() or not documents_path.is_dir():
        raise FileNotFoundError(
            f"Harvey LAB package is missing for {source_task}@{task_ref}"
        )

    package_instruction = instruction_path.read_text().strip()
    if instruction and instruction.strip() != package_instruction:
        raise ValueError(f"Harvey LAB instruction differs from package for {source_task}")

    release_path = TASK_FILES_PATH / BENCH / task_id
    shutil.copytree(documents_path, release_path, dirs_exist_ok=True)
    input_files = []
    for path in sorted(documents_path.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(documents_path).as_posix()
        input_files.append(
            {
                "path": f"/workspace/documents/{relative_path}",
                "dataset_path": f"task_files/{BENCH}/{task_id}/{relative_path}",
            }
        )

    return Task(
        benchmark=BENCH,
        task_id=task_id,
        environment_id=task_id,
        environment_description=ENV_DESCRIPTIONS[BENCH],
        environment_metadata={
            "action_space": {
                "types": [ActionSpaceType.SHELL],
                "description": "Shell and file operations for inspecting matter files, running analyses, and producing a legal deliverable.",
            },
            "environment_state": {
                "type": EnvironmentStateType.FILESYSTEM,
                "access": EnvironmentStateAccess.READ_WRITE,
                "ref": f"task_files/{BENCH}/{task_id}",
            },
        },
        task_description=package_instruction,
        gold_answer=None,
        task_metadata={
            "harbor_dataset_slug": slug,
            "harbor_task_name": source_task,
            "extra_context": {"input_files": input_files},
        },
        task_date=resolve_task_date(BENCH),
        verifiers=[
            {
                "type": VerifierType.LLM_JUDGE,
                "mode": VerifierMode.FINAL,
                "role": "rubric_item",
                "judge_model": HARVEYAI_JUDGE,
                "name": criterion["name"],
                "weight": criterion.get("weight"),
            }
            for criterion in reward_details.get("criteria", [])
        ],
        scoring_rule=ScoringRule.ALL_PASS,
        source_platform=SourcePlatform.HUGGINGFACE,
    )


def iter_verifier_records(
    task_id: str,
    reward_details: dict,
    slug: str,
    agent_id: str,
    agent_model: str,
    model_reasoning_effort: ModelReasoningEffort | None,
    agent_scaffold: str,
    model_date: str | None,
    trial_index: int,
    trial_name: str,
    run_id: str,
) -> Iterator[tuple[VerifierDefinition, Record]]:
    for criterion in reward_details.get("criteria", []):
        verifier_id = f"{task_id}::{criterion['name']}"
        verifier = VerifierDefinition(
            benchmark=BENCH,
            task_id=task_id,
            verifier_id=verifier_id,
            description=criterion.get("description"),
            verifier={
                "type": VerifierType.LLM_JUDGE,
                "mode": VerifierMode.FINAL,
                "role": "rubric_item",
                "judge_model": HARVEYAI_JUDGE,
            },
            metadata={
                "harbor_dataset_slug": slug,
                "name": criterion["name"],
                "weight": criterion.get("weight"),
            },
            source_platform=SourcePlatform.HUGGINGFACE,
        )

        value = criterion.get("value")
        if value is None:
            continue

        record = Record(
            benchmark=BENCH,
            task_id=task_id,
            verifier_id=verifier_id,
            record_type=RecordType.VERIFIER_RESULT,
            agent_id=agent_id,
            agent_model=agent_model,
            model_reasoning_effort=model_reasoning_effort,
            agent_scaffold=agent_scaffold,
            model_date=model_date,
            trial=trial_index,
            result=int(value),
            metadata={
                "verifier_name": criterion["name"],
                "raw": criterion.get("raw"),
                "weight": criterion.get("weight"),
                "reasoning": criterion.get("reasoning"),
                "trial_id": trial_name,
                "run_id": run_id,
            },
            source_platform=SourcePlatform.HUGGINGFACE,
        )

        yield verifier, record
