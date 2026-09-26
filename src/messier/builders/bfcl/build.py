"""
Import BFCL Live and Multi-Turn tasks and results.
"""

from collections import defaultdict
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW
from ...models import (
    BuildResult,
    ActionSpaceType,
    SourcePlatform,
    Record,
    Task,
    VerifierMode,
    VerifierType,
)
from .constants import CATEGORIES, DROP_SUBTESTS, ENV_DESCRIPTIONS
from .utils import extract_task_content, read_score_file


def build() -> BuildResult:
    root = RAW / "bfcl" / "2025-12-16" / "score"
    task_ids_by_subtest: dict[str, set[str]] = defaultdict(set)
    scores_by_pair: dict[tuple[str, str], tuple[dict, dict[str, dict]]] = {}
    content_by_task: dict[str, dict] = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue

        model = model_dir.name
        for category_name in CATEGORIES:
            category_dir = model_dir / category_name
            if not category_dir.is_dir():
                continue

            for score_path in sorted(category_dir.glob("*_score.json")):
                subtest = score_path.stem.removesuffix("_score").removeprefix("BFCL_v4_")
                if subtest in DROP_SUBTESTS:
                    continue

                aggregate, failures = read_score_file(score_path)
                if aggregate is None:
                    continue
                if "total_count" not in aggregate or "correct_count" not in aggregate:
                    raise ValueError(f"missing BFCL counts in {score_path}")

                scores_by_pair[(model, subtest)] = aggregate, failures
                task_ids_by_subtest[subtest].update(failures.keys())

                for task_id, failure in failures.items():
                    if task_id not in content_by_task:
                        content_by_task[task_id] = extract_task_content(failure)

    for (model, subtest), (aggregate, failures) in scores_by_pair.items():
        n_tasks = len(task_ids_by_subtest[subtest])
        if n_tasks != aggregate["total_count"]:
            raise ValueError(f"incomplete BFCL task set for {model}/{subtest}")
        if n_tasks - len(failures) != aggregate["correct_count"]:
            raise ValueError(f"inconsistent BFCL score for {model}/{subtest}")

    tasks = []
    for subtest, task_ids in task_ids_by_subtest.items():
        is_live = subtest.startswith("live_")
        benchmark = CATEGORIES["live" if is_live else "multi_turn"]

        verifier_mode = VerifierMode.FINAL if is_live else VerifierMode.SEQUENTIAL

        for task_id in sorted(task_ids):
            content = content_by_task.get(task_id, {})
            action_space = content.get("action_space") or {
                "types": [ActionSpaceType.TOOL_CALLS],
                "mode": "unknown",
            }
            user_turns = content.get("user_turns") or []
            environment_metadata = {"action_space": action_space}
            if content.get("environment_state") is not None:
                environment_metadata["environment_state"] = content["environment_state"]

            task_metadata = {
                "bfcl_subtest": subtest,
                "user_turns": user_turns,
            }
            if content.get("action_path") is not None:
                task_metadata["action_path"] = content["action_path"]

            description = content.get("task_description")
            tasks.append(
                Task(
                    benchmark=benchmark,
                    task_id=task_id,
                    environment_id=subtest,
                    environment_description=ENV_DESCRIPTIONS[subtest],
                    environment_metadata=environment_metadata,
                    task_description=description,
                    task_metadata=task_metadata,
                    gold_answer=content.get("gold_answer"),
                    task_date=resolve_task_date(benchmark),
                    verifiers=[
                        {
                            "type": VerifierType.EXACT_MATCH,
                            "mode": verifier_mode
                        }
                    ],
                    source_platform=SourcePlatform.GITHUB,
                )
            )

    records = []
    for (raw_model, subtest), (_, failures) in scores_by_pair.items():
        benchmark = CATEGORIES["live" if subtest.startswith("live_") else "multi_turn"]

        if raw_model.lower().endswith("-fc"):
            bare_model, scaffold = raw_model[:-3], "bfcl-fc"
        else:
            bare_model, scaffold = raw_model, "bfcl-prompted"

        standardized_model, reasoning_effort = standardize_model(bare_model)
        standardized_scaffold = standardize_scaffold_name(scaffold)
        agent_id = format_agent_id(
            standardized_model,
            standardized_scaffold,
            reasoning_effort,
        )
        model_date = resolve_model_date(
            standardized_model,
            bare_model,
            raw_model,
        )

        for task_id in task_ids_by_subtest[subtest]:
            passed = task_id not in failures
            records.append(
                Record(
                    benchmark=benchmark,
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=standardized_model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=standardized_scaffold,
                    model_date=model_date,
                    result=int(passed),
                    source_platform=SourcePlatform.GITHUB,
                )
            )

    return tasks, [], records
