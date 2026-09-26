"""
Add official ReplicationBench task details to imported Harbor runs.
"""

import json
from ..benchmarks import DEFAULT_SCORING_RULES
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
from .constants import ENV_DESCRIPTIONS

BENCH = "replicationbench"
SOURCE_ROOT = RAW / "enrichment" / BENCH / "src" / "dataset"


def load_tasks() -> dict[str, dict]:
    """
    Load the official task and paper details used to enrich Harbor results.

    :return: Source details indexed by the normalized MESSIER task ID.
    """
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            "ReplicationBench source data are missing. Run the upstream fetch first."
        )

    papers = {
        path.stem.lower(): json.loads(path.read_text())
        for path in (SOURCE_ROOT / "papers").glob("*.json")
    }
    dataset_revisions = json.loads((SOURCE_ROOT.parents[1] / "dataset_revisions.json").read_text())
    tasks = {}
    for path in (SOURCE_ROOT / "tasks").glob("*/*.json"):
        task = json.loads(path.read_text())
        paper_id = path.parent.name
        task_id = f"{paper_id}__{task['task_id']}".lower()
        tasks[task_id] = {
            "paper_id": paper_id,
            "paper": papers[paper_id.lower()],
            "task": task,
            "dataset_revisions": dataset_revisions,
        }

    return tasks


def task_definition(
    task_id: str,
    instruction: str | None,
    task_ref: str | None,
    slug: str,
    source_task: str,
    source: dict,
) -> Task:
    """
    Build one ReplicationBench task using its official public definition.

    :param task_id: Normalized MESSIER task ID.
    :param instruction: Instruction recovered from the Harbor trajectory.
    :param task_ref: Immutable Harbor task-package reference.
    :param slug: Harbor dataset slug.
    :param source_task: Full task name recorded by Harbor.
    :param source: Matching official paper and task details.
    :return: Enriched ReplicationBench task.
    """
    paper_id = source["paper_id"]
    paper = source["paper"]
    task = source["task"]
    dataset = dict(paper.get("dataset") or {})
    dataset["hf_revision"] = [
        source["dataset_revisions"][repository]
        for repository in dataset.get("hf_name") or []
    ]
    release_dir = TASK_FILES_PATH / BENCH / task_id
    release_dir.mkdir(parents=True, exist_ok=True)

    paper_context = {
        "paper_id": paper_id,
        "title": paper["title"],
        "abstract": paper["abstract"],
    }
    task_context = {
        "task_id": task["task_id"],
        "paper_id": paper_id,
        "kind": task["kind"],
        "difficulty": task["difficulty"],
        "description": task["description"],
    }

    (release_dir / "paper_masked.json").write_text(json.dumps(paper_context, indent=2) + "\n")
    (release_dir / "dataset_info.json").write_text(json.dumps(task_context, indent=2) + "\n")

    dataset_path = f"task_files/{BENCH}/{task_id}"
    input_files = [
        {
            "path": "/app/resources/paper_masked.json",
            "dataset_path": f"{dataset_path}/paper_masked.json",
        },
        {
            "path": "/app/resources/dataset_info.json",
            "dataset_path": f"{dataset_path}/dataset_info.json",
        },
    ]

    return Task(
        benchmark=BENCH,
        task_id=task_id,
        environment_id=task_id,
        environment_description=ENV_DESCRIPTIONS[BENCH],
        environment_metadata={
            "action_space": {
                "types": [ActionSpaceType.SHELL],
                "description": "Shell and file operations for inspecting the paper and data, running scientific software, and writing the requested result.",
            },
            "environment_state": {
                "type": EnvironmentStateType.FILESYSTEM,
                "access": EnvironmentStateAccess.READ_WRITE,
                "ref": task_ref,
            },
        },
        task_description=instruction,
        task_metadata={
            "harbor_dataset_slug": slug,
            "harbor_task_name": source_task,
            "harbor_task_ref": task_ref,
            "paper_id": paper_id,
            "paper_url": paper.get("paper_link"),
            "code_url": paper.get("code_link"),
            "difficulty": task.get("difficulty"),
            "parents": task.get("parents") or [],
            "extra_context": {
                "input_files": input_files,
                "dataset": dataset,
            },
        },
        gold_answer=task["expected_output"],
        task_date=resolve_task_date(BENCH),
        verifiers=[
            {
                "type": VerifierType.SCRIPT,
                "mode": VerifierMode.FINAL,
                "comparison": "tolerance",
                "tolerance": task["tolerance"],
            }
        ],
        scoring_rule=ScoringRule(DEFAULT_SCORING_RULES[BENCH]),
        source_platform=SourcePlatform.HUGGINGFACE,
    )
