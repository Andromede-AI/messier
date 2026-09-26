"""
Import GDPval tasks, rubrics, and model results.

Task definitions come from GDPval and per-task results come from BRIDGE.
"""

import json
import re
import shutil
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW, TASK_FILES_PATH, load_jsonl
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
    VerifierMode,
    VerifierType,
)
from .constants import OCCUPATION_TO_SOC, SECTOR_TO_NAICS


def build() -> BuildResult:
    slug_pattern = re.compile(r"[^a-z0-9]+")
    task_rows = load_jsonl(RAW / "gdpval" / "tasks.jsonl")
    tasks = []
    for task_row in task_rows:
        task_id = task_row["task_id"]
        input_files = []
        for filename in task_row.get("reference_files") or []:
            source_path = RAW / "gdpval" / filename
            if not source_path.is_file():
                raise FileNotFoundError(f"GDPval reference file is missing: {filename}")

            release_path = TASK_FILES_PATH / "gdpval" / task_id / source_path.name
            release_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, release_path)
            input_files.append(
                {
                    "path": source_path.name,
                    "dataset_path": f"task_files/gdpval/{task_id}/{source_path.name}",
                }
            )

        rubric = task_row.get("rubric_json")
        if isinstance(rubric, str):
            rubric = json.loads(rubric)
        if not isinstance(rubric, list):
            rubric = []

        gold_answer = [
            {
                "step": index,
                "expected": item.get("criterion"),
                "target_score": item.get("score"),
                "rubric_item_id": item.get("rubric_item_id"),
                "author_type": item.get("author_type"),
                "required": item.get("required"),
                "tags": item.get("tags"),
                "read_only": item.get("read_only"),
                "form_content": item.get("form_content"),
            }
            for index, item in enumerate(rubric)
            if isinstance(item, dict)
        ] or None

        # read occupation and sector labels
        sector = task_row.get("sector")
        occupation = task_row.get("occupation")
        occupation_slug = (slug_pattern.sub("-", occupation.lower()).strip("-") if occupation else None)
        environment_id = occupation_slug or task_row["task_id"]

        tasks.append(
            Task(
                benchmark="gdpval",
                task_id=task_id,
                environment_id=environment_id,
                environment_description=(
                    "A workplace task with reference materials and requested deliverables, "
                    "including documents, spreadsheets, presentations, and code. "
                    f"The task represents {occupation} in {sector}."
                ),
                environment_metadata={
                    "action_space": {
                        "types": [ActionSpaceType.SHELL],
                        "description": (
                            "Agent receives a prompt and a set of reference files (spreadsheets, documents, "
                            "slides, code, etc.) and must produce one or more deliverable files matching the "
                            "format and content criteria specified in the prompt."
                        ),
                    },
                    "environment_state": {
                        "type": EnvironmentStateType.FILESYSTEM,
                        "access": EnvironmentStateAccess.READ_WRITE,
                        "ref": f"task_files/gdpval/{task_id}" if input_files else None,
                    },
                    "deliverable_files": task_row.get("deliverable_files"),
                    "deliverable_file_urls": task_row.get("deliverable_file_urls"),
                    "deliverable_file_hf_uris": task_row.get("deliverable_file_hf_uris"),
                    "url": "https://huggingface.co/datasets/openai/gdpval",
                },
                task_description=task_row["prompt"],
                task_date=resolve_task_date("gdpval"),
                task_metadata={
                    "sector": sector,
                    "occupation": occupation,
                    "rubric_pretty": task_row.get("rubric_pretty"),
                    "extra_context": {"input_files": input_files} if input_files else None,
                },
                soc_code=OCCUPATION_TO_SOC[occupation],
                naics_code=SECTOR_TO_NAICS[sector],
                gold_answer=gold_answer,
                verifiers=[
                    {
                        "type": VerifierType.LLM_JUDGE,
                        "mode": VerifierMode.FINAL,
                        "threshold": 4,
                    }
                ],
                scoring_rule=ScoringRule.THRESHOLD,
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )

    # we extract normalized model results from the shared BRIDGE release
    records = []
    for row in load_jsonl(RAW / "bridge" / "gdpval_normalized_results.jsonl"):
        raw_model = row["model"]
        score = row["score"]

        model, reasoning_effort = standardize_model(raw_model)
        # technically an evaluation framework, but recorded as the scaffold because
        # no more specific agent configuration is reported
        scaffold = standardize_scaffold_name(row.get("agent") or row.get("scaffold") or "inspect_ai")
        model_date = resolve_model_date(model, raw_model)
        result = int(score) if isinstance(score, (int, float)) else None

        agent_id = format_agent_id(model, scaffold, reasoning_effort)

        records.append(
            Record(
                benchmark="gdpval",
                task_id=row["task_id"],
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=0,
                result=result,
                metadata=row.get("metadata") or {},
                source_platform=SourcePlatform.GITHUB,
                data_provider=DataProvider.BRIDGE,
            )
        )

    return tasks, [], records
