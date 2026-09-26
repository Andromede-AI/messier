"""
Import Humanity's Last Exam tasks and results.
"""

import json
from ..benchmarks import UPSTREAM_COMMITS
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ...io import RAW
from ..dates import resolve_model_date, resolve_task_date
from ...models import (
    BuildResult,
    ActionSpaceType,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    ScoringRule,
    Task,
    VerifierMode,
    VerifierType,
)

def build() -> BuildResult:
    judged_path = RAW / "hle" / "judged_hle_pro.json"
    scaffold = standardize_scaffold_name("Sup AI")
    judgments_by_task = json.loads(judged_path.read_text())
    tasks: list[Task] = []
    records: list[Record] = []

    for task_id, task_judgments in judgments_by_task.items():
        tasks.append(
            Task(
                benchmark="hle",
                task_id=task_id,
                environment_id="hle",
                environment_description=(
                    "An expert-written question spanning mathematics, science, engineering, "
                    "medicine, the humanities, and other specialist fields, sometimes with "
                    "an image and access to live web search."
                ),
                environment_metadata={
                    "action_space": {
                        "types": [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT],
                        "description": "Web-search calls and text responses.",
                    },
                    "environment_state": {
                        "type": EnvironmentStateType.LIVE_WEB,
                        "access": EnvironmentStateAccess.READ_ONLY,
                        "ref": f"hf://datasets/cais/hle@{UPSTREAM_COMMITS['hle']}",
                        "snapshot": None,
                    },
                    "url": "https://huggingface.co/datasets/cais/hle",
                },
                task_description=None,
                gold_answer=None,
                task_date=resolve_task_date("hle"),
                verifiers=[
                    {
                        "type": VerifierType.LLM_JUDGE,
                        "mode": VerifierMode.FINAL,
                        "judge_model": "openai/gpt-5.1",
                    }
                ],
                scoring_rule=ScoringRule.DIRECT,
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )

        for model_key, judgment in task_judgments.get("judge_response", {}).items():
            if model_key == "main":
                continue

            correct = judgment.get("correct") == "yes"
            model, reasoning_effort = standardize_model(model_key)
            model_date = resolve_model_date(model, model_key)
            agent_id = format_agent_id(model, scaffold, reasoning_effort)
            records.append(
                Record(
                    benchmark="hle",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    result=int(correct),
                    metadata={"confidence": judgment.get("confidence")},
                    source_platform=SourcePlatform.GITHUB,
                )
            )

    return tasks, [], records
