"""
Import contributed Harbor trials and verifier results.
"""

import json
from collections import Counter
from ..benchmarks import DEFAULT_SCORING_RULES
from ..standardize import format_agent_id, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW
from ...models import (
    BuildResult,
    ActionSpaceType,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    RecordType,
    ScoringRule,
    Task,
    Trajectory,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)
from . import dabstep, harveyai, replicationbench, scienceagentbench, utils
from .constants import ENV_DESCRIPTIONS, SLUG_TO_NAME, VERIFIER_TYPES


def build() -> BuildResult:
    graded_runs = RAW / "harbor_runs" / "graded"
    if not graded_runs.exists():
        return [], [], []

    tasks_by_id: dict[tuple[str, str], Task] = {}
    verifiers_by_id: dict[tuple[str, str], VerifierDefinition] = {}
    records: list[Record] = []
    trial_indices: Counter = Counter()
    replicationbench_tasks = replicationbench.load_tasks()
    dabstep_input_files, dabstep_script_code = dabstep.prepare_files()

    for verifier_path in sorted(graded_runs.rglob("verifier")):
        trial_path = verifier_path.parent
        config_path = trial_path / "config.json"
        config = json.loads(config_path.read_text())
        source_slug = config["task"]["source"]
        source_task = utils.source_task(config)
        benchmark = SLUG_TO_NAME.get(source_slug)
        if benchmark is None:
            continue

        raw_model, model, reasoning_effort = utils.model_identity(config)
        scaffold = standardize_scaffold_name(config["agent"]["name"])
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, raw_model)
        run_id = trial_path.parent.name
        reward, reward_details = utils.read_reward(trial_path)

        if reward is None:
            continue

        result = utils.load_result(trial_path)
        task_id = utils.task_id(result, source_slug)
        task_ref = utils.task_ref(config, result)
        trial_key = (benchmark, task_id, agent_id)
        trial_index = trial_indices[trial_key]
        trial_indices[trial_key] += 1

        instruction = None
        trajectory_path = trial_path / "agent" / "trajectory.json"
        if trajectory_path.exists():
            steps = json.loads(trajectory_path.read_text()).get("steps", [])
            for step in steps:
                if step.get("source") == "user" and isinstance(step.get("message"), str):
                    instruction = step["message"]
                    break

        existing_task = tasks_by_id.get((benchmark, task_id))
        if existing_task is None or (not existing_task.task_description and instruction):
            if benchmark == harveyai.BENCH and reward_details is not None:
                tasks_by_id[(benchmark, task_id)] = harveyai.task_definition(
                    task_id=task_id,
                    instruction=instruction,
                    task_ref=task_ref,
                    reward_details=reward_details,
                    slug=source_slug,
                    source_task=source_task,
                )
            elif benchmark == replicationbench.BENCH:
                tasks_by_id[(benchmark, task_id)] = replicationbench.task_definition(
                    task_id=task_id,
                    instruction=instruction,
                    task_ref=task_ref,
                    slug=source_slug,
                    source_task=source_task,
                    source=replicationbench_tasks[task_id.lower()],
                )
            elif benchmark == scienceagentbench.BENCH:
                tasks_by_id[(benchmark, task_id)] = scienceagentbench.task_definition(
                    task_id=task_id,
                    instruction=instruction,
                    task_ref=task_ref,
                    slug=source_slug,
                    source_task=source_task,
                )
            elif benchmark == dabstep.BENCH:
                tasks_by_id[(benchmark, task_id)] = dabstep.task_definition(
                    task_id=task_id,
                    instruction=instruction,
                    task_ref=task_ref,
                    slug=source_slug,
                    source_task=source_task,
                    input_files=dabstep_input_files,
                    script_code=dabstep_script_code,
                )
            else:
                environment_ref = task_ref
                extra_context = None

                verifier = {
                    "type": VerifierType(VERIFIER_TYPES[benchmark]),
                    "mode": VerifierMode.FINAL,
                }
                if benchmark == "qcircuitbench":
                    verifier["scale"] = "0.0-1.0"
                    verifier["threshold"] = 1.0

                tasks_by_id[(benchmark, task_id)] = Task(
                    benchmark=benchmark,
                    task_id=task_id,
                    environment_id=task_id,
                    environment_description=ENV_DESCRIPTIONS[benchmark],
                    environment_metadata={
                        "action_space": {
                            "types": (
                                [ActionSpaceType.SHELL, ActionSpaceType.TOOL_CALLS]
                                if benchmark == "medagentbench"
                                else [ActionSpaceType.SHELL]
                            ),
                            "description": "Shell and file operations for inspecting task materials, running code, and producing the requested result.",
                        },
                        "environment_state": {
                            "type": EnvironmentStateType.FILESYSTEM,
                            "access": EnvironmentStateAccess.READ_WRITE,
                            "ref": environment_ref,
                        },
                    },
                    task_description=instruction,
                    gold_answer=None,
                    task_metadata={
                        "harbor_dataset_slug": source_slug,
                        "harbor_task_name": source_task,
                        "harbor_task_ref": task_ref,
                        "extra_context": extra_context,
                    },
                    task_date=resolve_task_date(benchmark),
                    verifiers=[verifier],
                    scoring_rule=ScoringRule(DEFAULT_SCORING_RULES[benchmark]),
                    source_platform=SourcePlatform.HUGGINGFACE,
                )

        agent_result = result.get("agent_result") or {}
        metadata = {
            "reward": reward,
            "trial_id": trial_path.name,
            "run_id": run_id,
            "cost_usd": agent_result.get("cost_usd"),
            "n_input_tokens": agent_result.get("n_input_tokens"),
            "n_output_tokens": agent_result.get("n_output_tokens"),
            "n_cache_tokens": agent_result.get("n_cache_tokens"),
        }
        if benchmark == "qcircuitbench":
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
                    trial=trial_index,
                    result=reward,
                    metadata=metadata,
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )
        records.append(
            Record(
                benchmark=benchmark,
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=trial_index,
                result=1 if reward >= 1.0 else 0,
                metadata=metadata,
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )

        if benchmark == harveyai.BENCH and reward_details is not None:
            for verifier, verifier_record in harveyai.iter_verifier_records(
                task_id=task_id,
                reward_details=reward_details,
                slug=source_slug,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial_index=trial_index,
                trial_name=trial_path.name,
                run_id=run_id,
            ):
                verifiers_by_id.setdefault((benchmark, verifier.verifier_id), verifier)
                records.append(verifier_record)

    return list(tasks_by_id.values()), list(verifiers_by_id.values()), records


def trajectories():
    graded_runs = RAW / "harbor_runs" / "graded"
    trial_indices = Counter()

    for verifier_path in sorted(graded_runs.rglob("verifier")):
        trial_path = verifier_path.parent
        config_path = trial_path / "config.json"
        config = json.loads(config_path.read_text())
        source_slug = config["task"]["source"]
        benchmark = SLUG_TO_NAME.get(source_slug)
        if benchmark is None or benchmark == scienceagentbench.BENCH:
            continue

        reward, _ = utils.read_reward(trial_path)
        if reward is None:
            continue

        result = utils.load_result(trial_path)
        task_id = utils.task_id(result, source_slug)
        raw_model, model, reasoning_effort = utils.model_identity(config)
        scaffold = standardize_scaffold_name(config["agent"]["name"])
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        key = (benchmark, task_id, agent_id)
        trial = trial_indices[key]
        trial_indices[key] += 1

        trajectory_path = trial_path / "agent" / "trajectory.json"
        if not trajectory_path.exists():
            continue

        yield Trajectory(
            benchmark=benchmark,
            task_id=task_id,
            agent_id=agent_id,
            agent_model=model,
            model_reasoning_effort=reasoning_effort,
            agent_scaffold=scaffold,
            model_date=resolve_model_date(model, raw_model),
            trial=trial,
            content=trajectory_path.read_text(),
            metadata={
                "trial_id": trial_path.name,
                "run_id": trial_path.parent.name,
                "source_path": str(trajectory_path.relative_to(RAW)),
            },
            source_platform=SourcePlatform.HUGGINGFACE,
        )
