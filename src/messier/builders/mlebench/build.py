"""
Import MLE-bench competition tasks and trial results.

The source's any-medal field defines the binary trial result.
"""

from collections import Counter
import re
import yaml
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import ENRICH, RAW, load_jsonl
from ...models import (
    ActionSpaceType,
    BuildResult,
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


def build() -> BuildResult:
    id_year = re.compile(r"(?:^|[-_])(20\d{2})(?:[-_]|$)")
    description_iso = re.compile(r"\b(20\d{2})-\d{2}-\d{2}\b")
    description_month = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+(20\d{2})\b",
        re.IGNORECASE,
    )

    def launch_date(competition_id: str, description: str | None) -> str | None:
        if match := id_year.search(competition_id):
            return f"{match.group(1)}-01-01"
        if description:
            years = [int(year) for year in description_iso.findall(description)]
            years.extend(int(year) for year in description_month.findall(description))
            if years:
                return f"{min(years)}-01-01"

        return None

    tasks = []
    competition_dir = ENRICH / "mlebench" / "mlebench" / "competitions"
    for path in sorted(item for item in competition_dir.iterdir() if item.is_dir()):
        description_path = path / "description.md"
        description = (description_path.read_text() if description_path.exists() else None)
        config_path = path / "config.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        grader = (
            config.get("grader")
            if isinstance(config.get("grader"), dict)
            else {}
        )
        grader_name = grader.get("name")
        grade_path = path / "grade.py"
        grade_script = grade_path.read_text() if grade_path.exists() else None
        competition_id = config.get("id", path.name)
        competition_name = config.get("name") or path.name
        verifier = {
            "type": VerifierType.SCRIPT,
            "name": grader_name,
            "mode": VerifierMode.FINAL,
            "success_condition": "any_medal",
        }
        if grade_script:
            verifier["script_code"] = grade_script
        task_date = launch_date(
            competition_id,
            description
        ) or resolve_task_date("mlebench")

        tasks.append(
            Task(
                benchmark="mlebench",
                task_id=competition_id,
                environment_id=competition_id,
                environment_description=(
                    f"A Kaggle competition environment for {competition_name}, containing "
                    "training and test data, common machine learning libraries, and a "
                    "grading server."
                ),
                environment_metadata={
                    "action_space": {
                        "types": [ActionSpaceType.SHELL],
                        "description": (
                            "Shell commands for inspecting the data, training models, and "
                            "producing a submission file."
                        ),
                    },
                    "environment_state": {
                        "type": EnvironmentStateType.FILESYSTEM,
                        "access": EnvironmentStateAccess.READ_WRITE,
                        "ref": f"kaggle://competitions/{competition_id}",
                    },
                    "url": f"https://www.kaggle.com/competitions/{competition_id}",
                },
                task_description=description,
                task_date=task_date,
                task_metadata={
                    "name": config.get("name"),
                    "competition_type": config.get("competition_type"),
                    "awards_medals": config.get("awards_medals"),
                    "prizes": config.get("prizes"),
                    "grader_name": grader_name,
                },
                verifiers=[verifier],
                scoring_rule=ScoringRule.DIRECT,
                source_platform=SourcePlatform.GITHUB,
            )
        )

    records = []
    trial_indices = Counter()
    for row in load_jsonl(RAW / "bridge" / "mlebench_normalized_results.jsonl"):
        source_scores = row.get("score") or {}
        if (
            not isinstance(source_scores, dict)
            or source_scores.get("any_medal") is None
        ):
            continue

        task_id = row["task_id"]
        raw_model = row["model"]
        model, reasoning_effort = standardize_model(raw_model)
        scaffold = standardize_scaffold_name(row.get("agent") or row.get("scaffold"))
        model_date = resolve_model_date(model, raw_model)

        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        key = (task_id, agent_id)
        trial = trial_indices[key]
        trial_indices[key] += 1

        records.append(
            Record(
                benchmark="mlebench",
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=trial,
                result=int(source_scores["any_medal"]),
                metadata={
                    "source_results": source_scores,
                    **row.get("metadata", {})
                },
                source_platform=SourcePlatform.GITHUB,
                data_provider=DataProvider.BRIDGE,
            )
        )
    return tasks, [], records
