"""
Import LiveCodeBench tasks and reported pass@1 summaries.

The source does not provide individual trials, so pass@1 remains a source summary.
"""

import json
from ..benchmarks import UPSTREAM_COMMITS
from ..standardize import format_agent_id, standardize_model
from ..dates import resolve_model_date, to_iso_date
from ...io import RAW, load_jsonl
from ...models import (
    ActionSpaceType,
    BuildResult,
    EnvironmentStateAccess,
    EnvironmentStateType,
    HumanEffort,
    Record,
    RecordType,
    SourcePlatform,
    Task,
    VerifierMode,
    VerifierType,
)
from .constants import (
    ACTION_SPACE_DESCRIPTION,
    ATCODER_POSITION_BANDS,
    PLATFORM_DESCRIPTIONS,
    RATING_BANDS,
)


def build() -> BuildResult:
    raw = RAW / "livecodebench"
    revision = UPSTREAM_COMMITS["livecodebench"]

    # we load the external ratings used for human-time estimates
    leetcode_path = raw / "leetcode_ratings.json"
    leetcode_ratings = {
        str(row["ID"]): float(row["Rating"])
        for row in json.loads(leetcode_path.read_text())
    }

    codeforces_ratings: dict[str, float] = {}
    codeforces_path = raw / "codeforces_problemset.json"
    codeforces_problems = json.loads(codeforces_path.read_text())["result"]["problems"]
    for problem in codeforces_problems:
        if "rating" in problem:
            task_id = f"{problem['contestId']}_{problem['index']}"
            codeforces_ratings[task_id] = float(problem["rating"])

    def estimate_human_effort(task_id: str, platform: str) -> HumanEffort | None:
        rating = None
        if platform == "leetcode":
            rating = leetcode_ratings.get(task_id)
        elif platform == "codeforces":
            rating = codeforces_ratings.get(task_id)

        if rating is not None:
            low, middle, high = next(
                (band for cutoff, band in RATING_BANDS if rating < cutoff),
                RATING_BANDS[-1][1],
            )
        elif platform == "atcoder":
            suffix = task_id.rsplit("_", 1)[-1].lower()
            if suffix not in ATCODER_POSITION_BANDS:
                return None

            low, middle, high = ATCODER_POSITION_BANDS[suffix]
        else:
            return None

        return HumanEffort(
            minutes_median=middle,
            minutes_low=low,
            minutes_high=high
        )

    # we create one task for each programming problem
    tasks: list[Task] = []
    for task_row in load_jsonl(raw / "tasks.jsonl"):
        task_id = task_row["question_id"]
        platform = task_row["platform"]
        public_test_count = len(json.loads(task_row["public_test_cases"]))

        tasks.append(
            Task(
                benchmark="livecodebench",
                task_id=task_id,
                environment_id=platform,
                environment_description=PLATFORM_DESCRIPTIONS[platform],
                environment_metadata={
                    "action_space": {
                        "types": [ActionSpaceType.TEXT],
                        "description": ACTION_SPACE_DESCRIPTION,
                    },
                    "environment_state": {
                        "type": EnvironmentStateType.NONE,
                        "access": EnvironmentStateAccess.NONE,
                        "ref": f"hf://datasets/livecodebench/code_generation_lite@{revision}",
                        "snapshot": None,
                    },
                    "platform": platform,
                },
                task_description=task_row["question_content"],
                task_metadata={
                    "question_title": task_row["question_title"],
                    "extra_context": {"starter_code": task_row["starter_code"]}
                    if task_row["starter_code"]
                    else None,
                    "n_public_tests": public_test_count,
                    "contest_id": task_row["contest_id"],
                    "contest_date": task_row["contest_date"],
                },
                task_date=to_iso_date(task_row["contest_date"]),
                verifiers=[
                    {
                        "type": VerifierType.SCRIPT,
                        "mode": VerifierMode.FINAL,
                        "name": "online_judge",
                    }
                ],
                difficulty_label=task_row["difficulty"],
                human=estimate_human_effort(task_id, platform),
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )

    # we preserve each reported pass@1 value as a source summary
    performance_data = json.loads((raw / "performances_generation.json").read_text())
    records: list[Record] = []

    for performance in performance_data["performances"]:
        task_id = performance["question_id"]
        raw_model = performance["model"]
        pass_at_1_percent = float(performance["pass@1"])
        model, reasoning_effort = standardize_model(raw_model)
        model_date = resolve_model_date(model, raw_model)
        records.append(
            Record(
                benchmark="livecodebench",
                task_id=task_id,
                agent_id=format_agent_id(model, None, reasoning_effort),
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=None,
                model_date=model_date,
                trial=None,
                record_type=RecordType.SOURCE_SUMMARY,
                result=pass_at_1_percent / 100,
                metadata={"summary": "pass@1", "raw_model": raw_model},
                source_platform=SourcePlatform.GITHUB,
            )
        )

    return tasks, [], records
