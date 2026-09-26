"""
Import OnlineMind2Web tasks and human judgments.
"""

import json
from urllib.parse import urlparse
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW, load_jsonl
from ...models import (
    BuildResult,
    ActionSpaceType,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    HumanEffort,
    Record,
    ScoringRule,
    Task,
    VerifierMode,
    VerifierType,
)
from .constants import (
    ACTION_SPACE_DESCRIPTION,
    AGENTS,
    TASK_ID_SUFFIX_RE,
)


def build() -> BuildResult:
    labels_path = RAW / "onlinemind2web" / "human_label_111825.json"
    rows = json.loads(labels_path.read_text())
    primary = {
        TASK_ID_SUFFIX_RE.sub("", row["task_id"]): row
        for row in load_jsonl(RAW / "onlinemind2web" / "tasks_primary.jsonl")
    }
    fallback = {
        TASK_ID_SUFFIX_RE.sub("", row["task_id"]): row
        for row in load_jsonl(RAW / "onlinemind2web" / "tasks_fallback.jsonl")
    }

    tasks: list[Task] = []
    for row in rows:
        task_data = {}
        for source in (primary, fallback):
            candidate = source.get(row["task_id"])
            if candidate and candidate["confirmed_task"] == row["confirmed_task"]:
                task_data = candidate
                break

        website = task_data.get("website")
        domain = None

        if website:
            domain = urlparse(website).netloc.lower().removeprefix("www.") or None

        reference_length = task_data.get("reference_length")
        environment_description = (
            f"A live website at {domain} that the agent navigates and modifies "
            "through a browser."
            if domain
            else "A live public website that the agent navigates and modifies through a browser."
        )
        tasks.append(
            Task(
                benchmark="onlinemind2web",
                task_id=row["task_id"],
                environment_id=domain or row["task_id"],
                environment_description=environment_description,
                environment_metadata={
                    "action_space": {
                        "types": [ActionSpaceType.UI, ActionSpaceType.TEXT],
                        "description": ACTION_SPACE_DESCRIPTION,
                    },
                    "environment_state": {
                        "type": EnvironmentStateType.LIVE_WEB,
                        "access": EnvironmentStateAccess.READ_WRITE,
                        "ref": website,
                        "snapshot": None,
                    },
                    "website": website,
                    "url": "https://huggingface.co/datasets/osunlp/Online-Mind2Web",
                },
                task_description=row["confirmed_task"],
                task_date=resolve_task_date("onlinemind2web"),
                verifiers=[
                    {
                        "type": VerifierType.HUMAN_LABEL,
                        "mode": VerifierMode.FINAL,
                        "scale": "0=failure, 1=success",
                    }
                ],
                human=HumanEffort(steps=int(reference_length))
                if reference_length is not None
                else None,
                difficulty_label=task_data.get("level"),
                scoring_rule=ScoringRule.DIRECT,
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )

    records: list[Record] = []
    for row in rows:
        for col, (scaffold, model, override_date) in AGENTS.items():
            raw_label = row.get(f"{col}_human_label")
            if raw_label is None:
                continue

            standardized_model, reasoning_effort = standardize_model(model)
            standardized_model = standardized_model or "undisclosed"
            standardized_scaffold = standardize_scaffold_name(scaffold)
            agent_id = format_agent_id(
                standardized_model,
                standardized_scaffold,
                reasoning_effort,
            )
            model_date = resolve_model_date(standardized_model, model) or override_date

            records.append(
                Record(
                    benchmark="onlinemind2web",
                    task_id=row["task_id"],
                    agent_id=agent_id,
                    agent_model=standardized_model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=standardized_scaffold,
                    model_date=model_date,
                    result=int(int(raw_label) == 1),
                    metadata={"human_label_raw": raw_label},
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )

    return tasks, [], records
