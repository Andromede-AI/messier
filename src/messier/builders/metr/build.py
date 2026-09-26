"""Import HCAST, RE-Bench, and SWAA tasks and results from METR."""

import shutil
from collections import Counter
from ..benchmarks import UPSTREAM_COMMITS
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import ENRICH, RAW, TASK_FILES_PATH, load_jsonl
from ...models import (
    BuildResult,
    ActionSpaceType,
    DataProvider,
    EnvironmentStateAccess,
    EnvironmentStateType,
    HumanEffort,
    Record,
    RecordType,
    ScoringRule,
    SourcePlatform,
    Task,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)
from .constants import BENCHMARK_IDS
from .utils import load_manifests, load_readmes

REPO_URL = {
    "hcast": f"https://github.com/METR/hcast-public/tree/{UPSTREAM_COMMITS['metr_hcast']}",
    "rebench": f"https://github.com/METR/RE-Bench/tree/{UPSTREAM_COMMITS['metr_rebench']}",
}


def build() -> BuildResult:
    manifests = {
        "HCAST": load_manifests(ENRICH / "hcast"),
        "RE-Bench": load_manifests(ENRICH / "rebench"),
        "SWAA": {},
    }
    readmes = {
        "HCAST": load_readmes(ENRICH / "hcast"),
        "RE-Bench": load_readmes(ENRICH / "rebench"),
        "SWAA": {},
    }

    rows = list(load_jsonl(RAW / "metr" / "runs.jsonl"))
    threshold_tasks = {
        (BENCHMARK_IDS[row["task_source"]], row["task_id"])
        for row in rows
        if row["score_cont"] != row["score_binarized"]
    }

    tasks_by_id = {}
    verifier_definitions = []
    records = []
    trials: Counter = Counter()

    for row in rows:
        task_source = row["task_source"]
        benchmark = BENCHMARK_IDS[task_source]
        task_id = row["task_id"]
        family = row["task_family"]
        task_name = task_id.split("/", 1)[1]
        manifest = manifests[task_source].get((family, task_name), {})

        key = (benchmark, task_id)
        if key not in tasks_by_id:
            verifier_mode = (
                VerifierMode.SEQUENTIAL
                if benchmark == "rebench"
                else VerifierMode.FINAL
            )
            verifier = {
                "type": VerifierType.SCRIPT,
                "mode": verifier_mode,
            }
            if manifest.get("script_code"):
                verifier["script_code"] = manifest["script_code"]
            if benchmark != "swaa":
                verifier["framework"] = "metr_task_standard"
            verifiers = [verifier]
            if benchmark == "swaa":
                environment_url = None
                action_space = {
                    "types": [ActionSpaceType.TEXT],
                    "description": (
                        "A text response selecting an option or completing a short "
                        "software task."
                    ),
                }
                environment_state = {
                    "type": EnvironmentStateType.NONE,
                    "access": EnvironmentStateAccess.NONE,
                }
                environment_description = (
                    "A fixed software-related question requiring one multiple-choice answer "
                    "or short completion."
                )
            else:
                environment_url = f"{REPO_URL[benchmark]}/{family}"
                action_types = [ActionSpaceType.SHELL]
                action_description = (
                    "Shell commands for interacting with task files, programs, "
                    "and services."
                )
                if benchmark == "rebench":
                    action_types.append(ActionSpaceType.TOOL_CALLS)
                    action_description = (
                        "Shell commands for running experiments and a scoring tool "
                        "for testing candidate solutions."
                    )

                action_space = {
                    "types": action_types,
                    "description": action_description,
                }
                environment_state = {
                    "type": EnvironmentStateType.FILESYSTEM,
                    "access": EnvironmentStateAccess.READ_WRITE,
                    "ref": environment_url,
                    "snapshot": manifest.get("family_env"),
                }
                environment_description = (
                    "A machine learning research environment containing code, data, compute "
                    "resources, and a scoring program."
                    if benchmark == "rebench"
                    else "A software environment containing the files, programs, "
                    "services, and tools needed to complete the task."
                )

            scoring_rule = (
                ScoringRule.THRESHOLD
                if key in threshold_tasks
                else ScoringRule.DIRECT
            )

            environment_metadata: dict = {
                "action_space": action_space,
                "environment_state": environment_state,
            }
            if environment_url:
                environment_metadata["url"] = environment_url
            if manifest.get("resources"):
                environment_metadata["resources"] = manifest["resources"]

            # we use README instructions when available and manifests otherwise
            family_readme = readmes[task_source].get(family, {})
            task_readme = (family_readme.get("variants") or {}).get(task_name, {})
            task_description = (
                task_readme.get("instructions")
                or manifest.get("description")
                or task_readme.get("summary")
            )
            input_files = []
            if benchmark == "hcast" and family == "debug_small_libs":
                source_dir = ENRICH / "hcast" / family / "assets" / task_name
                release_dir = TASK_FILES_PATH / benchmark / family / task_name
                shutil.copytree(source_dir, release_dir, dirs_exist_ok=True)
                for source_path in sorted(source_dir.iterdir()):
                    if source_path.is_file():
                        input_files.append(
                            {
                                "path": f"/home/agent/app/{source_path.name}",
                                "dataset_path": (
                                    f"task_files/{benchmark}/{family}/{task_name}/"
                                    f"{source_path.name}"
                                ),
                            }
                        )

            task_metadata = {
                "task_family": family,
                "family_name": manifest.get("family_name"),
                "family_summary": family_readme.get("family_summary"),
                "family_expertise": manifest.get("family_expertise"),
                "manifest_version": manifest.get("version"),
                "task_version": row["task_version"],
            }
            if input_files:
                task_metadata["extra_context"] = {"input_files": input_files}
                environment_state["ref"] = f"task_files/{benchmark}/{family}/{task_name}"

            tasks_by_id[key] = Task(
                benchmark=benchmark,
                task_id=task_id,
                environment_id=family,
                environment_description=environment_description,
                environment_metadata=environment_metadata,
                task_description=task_description,
                task_metadata=task_metadata,
                verifiers=verifiers,
                scoring_rule=scoring_rule,
                task_date=resolve_task_date(benchmark),
                human=HumanEffort(minutes_median=row["human_minutes"]),
                source_platform=SourcePlatform.GITHUB,
                data_provider=DataProvider.METR,
            )
            verifier_id = f"{task_id}::verifier:0"
            verifier_definitions.append(
                VerifierDefinition(
                    benchmark=benchmark,
                    task_id=task_id,
                    verifier_id=verifier_id,
                    verifier=verifiers[0],
                    metadata={"scoring": manifest.get("scoring")},
                    source_platform=SourcePlatform.GITHUB,
                    data_provider=DataProvider.METR,
                )
            )

        model, reasoning_effort = standardize_model(row["alias"])
        scaffold = standardize_scaffold_name(row["scaffold"])
        model_date = resolve_model_date(model, row["model"])
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        trial_key = (benchmark, task_id, agent_id)
        trial = trials[trial_key]
        trials[trial_key] = trial + 1
        infrastructure_error = row["fatal_error_from"] == "serverOrTask"
        records.append(
            Record(
                benchmark=benchmark,
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=trial,
                result=None if infrastructure_error else row["score_binarized"],
                metadata={
                    "source_score": row["score_cont"],
                    "run_id": row["run_id"],
                    "human_source": row["human_source"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "tokens_count": row["tokens_count"],
                    "fatal_error_from": row["fatal_error_from"],
                    **(
                        {"error_type": "infrastructure_error"}
                        if infrastructure_error
                        else {}
                    ),
                    "cloned": row["cloned"],
                    "time_limit": row["time_limit"],
                },
                source_platform=SourcePlatform.GITHUB,
                data_provider=DataProvider.METR,
            )
        )
        if key in threshold_tasks and not infrastructure_error:
            records.append(
                Record(
                    benchmark=benchmark,
                    task_id=task_id,
                    verifier_id=f"{task_id}::verifier:0",
                    record_type=RecordType.VERIFIER_RESULT,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=trial,
                    result=row["score_cont"],
                    metadata={"run_id": row["run_id"]},
                    source_platform=SourcePlatform.GITHUB,
                    data_provider=DataProvider.METR,
                )
            )

    return list(tasks_by_id.values()), verifier_definitions, records
