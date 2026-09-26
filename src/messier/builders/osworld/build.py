"""
Import OSWorld tasks and trajectory results.

Numerical rewards are preserved while only full reward defines trial success.
"""

import json
import re
import shutil
import urllib.parse
from collections import Counter
from ..standardize import format_agent_id, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW, TASK_FILES_PATH
from ...models import (
    ActionSpaceType,
    BuildResult,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    ModelReasoningEffort,
    Record,
    RecordType,
    ScoringRule,
    Task,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)
from .constants import (
    ACTION_SPACE_DESCRIPTION,
    BENCHMARK,
    DOMAIN_ENV_DESCRIPTIONS,
    ZIP_SUFFIX_RE,
    ZIP_AGENT_MAP,
    ZIP_REASONING_EFFORT,
    ZIP_SUPPORTING_MODELS,
)

def build() -> BuildResult:
    root = RAW / "osworld"
    configs_root = root / "configs" / "examples"
    trajectories_root = root / "zips"
    test_listing = root / "configs" / "test_nogdrive.json"
    if not test_listing.exists():
        return [], [], []

    task_ids_by_domain = json.loads(test_listing.read_text())
    tasks_by_id: dict[str, Task] = {}

    for domain, task_ids in task_ids_by_domain.items():
        for task_id in task_ids:
            config_path = configs_root / domain / f"{task_id}.json"
            if not config_path.exists():
                continue

            config = json.loads(config_path.read_text())
            evaluator = config.get("evaluator") or {}
            evaluator_function = evaluator.get("func")
            input_files = []

            for action in config.get("config") or []:
                if action.get("type") != "download":
                    continue

                for file_config in action.get("parameters", {}).get("files", []):
                    source_path = urllib.parse.unquote(file_config["url"].split("/resolve/main/", 1)[1])
                    raw_path = root / "files" / source_path
                    if not raw_path.is_file():
                        raise FileNotFoundError(f"OSWorld input file is missing: {source_path}")

                    dataset_path = f"task_files/osworld/{source_path}"
                    release_path = TASK_FILES_PATH / "osworld" / source_path
                    release_path.parent.mkdir(parents=True, exist_ok=True)
                    if not release_path.exists():
                        shutil.copy2(raw_path, release_path)

                    input_files.append(
                        {
                            "path": file_config["path"],
                            "dataset_path": dataset_path,
                        }
                    )

            related_apps = config.get("related_apps") or []
            tasks_by_id[task_id] = Task(
                benchmark=BENCHMARK,
                task_id=task_id,
                environment_id=domain,
                environment_description=DOMAIN_ENV_DESCRIPTIONS[domain],
                environment_metadata={
                    "action_space": {
                        "types": [ActionSpaceType.UI, ActionSpaceType.SHELL],
                        "description": ACTION_SPACE_DESCRIPTION,
                    },
                    "environment_state": {
                        "type": EnvironmentStateType.FILESYSTEM,
                        "access": EnvironmentStateAccess.READ_WRITE,
                        "ref": f"osworld://{domain}/{config.get('snapshot', domain)}",
                        "snapshot": {
                            "vm_snapshot": config.get("snapshot"),
                            "related_apps": related_apps,
                        },
                    },
                },
                task_description=config["instruction"].strip() or None,
                task_metadata={
                    "snapshot": config.get("snapshot"),
                    "related_apps": related_apps,
                    "source_url": config.get("source"),
                    "evaluator_func": evaluator_function,
                    "infeasible": evaluator_function == "infeasible",
                    "possibility_of_env_change": config.get("possibility_of_env_change"),
                    "hint": config.get("hint"),
                    "initialization": config.get("config"),
                    "extra_context": {
                        "input_files": input_files,
                    }
                    if input_files
                    else None,
                },
                gold_answer=evaluator.get("expected"),
                task_date=resolve_task_date(BENCHMARK),
                verifiers=[
                    {
                        "type": VerifierType.SCRIPT,
                        "mode": VerifierMode.FINAL,
                        "name": evaluator_function,
                        "scale": "0.0-1.0",
                        "threshold": 1.0,
                    }
                ],
                scoring_rule=ScoringRule.THRESHOLD,
                source_platform=SourcePlatform.GITHUB,
            )

    records: list[Record] = []
    trial_indices = Counter()

    for aggregate_path in sorted(trajectories_root.glob("*_aggregated.json")):
        stem = aggregate_path.name[: -len("_aggregated.json")]
        archive = ZIP_SUFFIX_RE.sub("", stem)
        identity = ZIP_AGENT_MAP.get(archive)
        if identity is None:
            continue

        standardized_model, scaffold = identity
        model_date = resolve_model_date(standardized_model)
        reasoning_effort = (
            ModelReasoningEffort(ZIP_REASONING_EFFORT[archive])
            if archive in ZIP_REASONING_EFFORT
            else None
        )

        step_match = re.search(r"(\d+)_?steps?", stem)
        step_budget = int(step_match.group(1)) if step_match else None
        standardized_scaffold = standardize_scaffold_name(scaffold)
        if step_budget is not None:
            standardized_scaffold = f"{standardized_scaffold}-{step_budget}steps"

        agent_id = format_agent_id(
            standardized_model,
            standardized_scaffold,
            reasoning_effort,
        )
        scores = json.loads(aggregate_path.read_text())
        arguments_path = trajectories_root / f"{stem}_args.json"
        arguments = (
            json.loads(arguments_path.read_text())
            if arguments_path.exists()
            else {}
        )

        for domain, scores_by_task in scores.items():
            for task_id, score in scores_by_task.items():
                if task_id not in tasks_by_id:
                    continue

                reward = float(score)
                key = (agent_id, task_id)
                trial = trial_indices[key]
                trial_indices[key] += 1
                metadata = {
                    "zip_stem": stem,
                    "scaffold": scaffold,
                    "step_budget": step_budget,
                    "action_space": arguments.get("action_space"),
                    "observation_type": arguments.get("observation_type"),
                    "domain": domain,
                    "raw_model": arguments.get("model"),
                }
                if archive in ZIP_SUPPORTING_MODELS:
                    metadata["supporting_models"] = ZIP_SUPPORTING_MODELS[archive]

                records.append(
                    Record(
                        benchmark=BENCHMARK,
                        task_id=task_id,
                        agent_id=agent_id,
                        agent_model=standardized_model,
                        agent_scaffold=standardized_scaffold,
                        model_reasoning_effort=reasoning_effort,
                        model_date=model_date,
                        trial=trial,
                        result=int(reward >= 1.0),
                        metadata=metadata,
                        source_platform=SourcePlatform.HUGGINGFACE,
                    )
                )
                records.append(
                    Record(
                        benchmark=BENCHMARK,
                        task_id=task_id,
                        verifier_id=f"{task_id}::verifier:0",
                        record_type=RecordType.VERIFIER_RESULT,
                        agent_id=agent_id,
                        agent_model=standardized_model,
                        agent_scaffold=standardized_scaffold,
                        model_reasoning_effort=reasoning_effort,
                        model_date=model_date,
                        trial=trial,
                        result=reward,
                        metadata=metadata,
                        source_platform=SourcePlatform.HUGGINGFACE,
                    )
                )

    verifier_definitions = [
        VerifierDefinition(
            benchmark=BENCHMARK,
            task_id=task.task_id,
            verifier_id=f"{task.task_id}::verifier:0",
            verifier=task.verifiers[0],
            gold_answer=task.gold_answer,
            source_platform=task.source_platform,
            data_provider=task.data_provider,
        )
        for task in tasks_by_id.values()
    ]

    return list(tasks_by_id.values()), verifier_definitions, records
