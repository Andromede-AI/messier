"""
Import tau2-bench tasks, trials, and verifier results.
"""

import json
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW
from ...models import (
    ActionSpaceType,
    BuildResult,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    RecordType,
    ScoringRule,
    SourcePlatform,
    Task,
    Trajectory,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)
from .constants import (
    ACTION_DESCRIPTIONS,
    BENCHMARK,
    DOMAINS,
    ENV_DESCRIPTIONS,
    EXCLUDED_TASKS,
    SUBMISSIONS,
)

# only components that contribute to the final result are represented as verifiers

REWARD_VERIFIERS = {
    "DB": VerifierType.EXACT_MATCH,
    "ACTION": VerifierType.EXACT_MATCH,
    "COMMUNICATE": VerifierType.SCRIPT,
    "NL_ASSERTION": VerifierType.LLM_JUDGE,
    "ENV_ASSERTION": VerifierType.EXACT_MATCH,
}

REWARD_MODES = {
    "DB": VerifierMode.FINAL,
    "ACTION": VerifierMode.SEQUENTIAL,
    "COMMUNICATE": VerifierMode.SEQUENTIAL,
    "NL_ASSERTION": VerifierMode.SEQUENTIAL,
    "ENV_ASSERTION": VerifierMode.FINAL,
}


def build() -> BuildResult:
    root = RAW / "tau2_bench"
    scaffold = standardize_scaffold_name("tau2-bench")
    task_date = resolve_task_date(BENCHMARK)

    tasks_by_id: dict[str, Task] = {}
    verifiers_by_id: dict[str, VerifierDefinition] = {}
    records: list[Record] = []

    for submission_name in SUBMISSIONS:
        submission_path = root / "submissions" / submission_name / "submission.json"
        if not submission_path.exists():
            continue

        submission = json.loads(submission_path.read_text())
        raw_model = submission["model_name"]
        model, reasoning_effort = standardize_model(raw_model, submission.get("reasoning_effort"))
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, submission["model_name"])
        trajectory_files = submission.get("trajectory_files") or {}
        for domain, filename in trajectory_files.items():
            if domain not in DOMAINS:
                continue

            trajectory_path = root / "trajectories" / submission_name / filename
            if not trajectory_path.exists():
                continue

            run = json.loads(trajectory_path.read_text())
            retrieval_config = (run.get("info") or {}).get("retrieval_config")
            run_action_types = [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT]
            if retrieval_config in {"terminal_use", "AllTools"}:
                run_action_types.insert(0, ActionSpaceType.SHELL)

            # we create each task and its source reward components once
            for task_row in run.get("tasks", []):
                task_id = f"{domain}.{task_row['id']}"
                if task_id in tasks_by_id:
                    continue

                evaluation = task_row.get("evaluation_criteria") or {}
                reward_basis = evaluation.get("reward_basis") or []
                verifiers = [
                    {
                        "type": REWARD_VERIFIERS[basis],
                        "mode": REWARD_MODES[basis],
                        "name": basis.lower(),
                    }
                    for basis in reward_basis
                ]

                scenario = task_row.get("user_scenario") or {}
                instructions = scenario.get("instructions")
                description_parts = []
                if isinstance(instructions, dict):
                    reason = instructions.get("reason_for_call") or instructions.get("task_instructions")
                    if reason:
                        description_parts.append(str(reason).strip())

                    known_info = instructions.get("known_info")
                    if known_info:
                        description_parts.append(f"Known info: {known_info}")

                elif isinstance(instructions, str):
                    description_parts.append(instructions.strip())

                description = task_row.get("description") or {}
                if not description_parts and description.get("purpose"):
                    description_parts.append(str(description["purpose"]).strip())

                user_instructions = (
                    instructions.get("task_instructions")
                    if isinstance(instructions, dict)
                    else None
                )

                task = Task(
                    benchmark=BENCHMARK,
                    task_id=task_id,
                    environment_id=domain,
                    environment_description=ENV_DESCRIPTIONS[domain],
                    environment_metadata={
                        "action_space": {
                            "types": (
                                [
                                    ActionSpaceType.SHELL,
                                    ActionSpaceType.TOOL_CALLS,
                                    ActionSpaceType.TEXT,
                                ]
                                if domain == "banking_knowledge"
                                else [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT]
                            ),
                            "description": ACTION_DESCRIPTIONS[domain],
                            "configurations": (
                                {
                                    "openai_embeddings": [
                                        ActionSpaceType.TOOL_CALLS,
                                        ActionSpaceType.TEXT,
                                    ],
                                    "terminal_use": [
                                        ActionSpaceType.SHELL,
                                        ActionSpaceType.TOOL_CALLS,
                                        ActionSpaceType.TEXT,
                                    ],
                                    "AllTools": [
                                        ActionSpaceType.SHELL,
                                        ActionSpaceType.TOOL_CALLS,
                                        ActionSpaceType.TEXT,
                                    ],
                                }
                                if domain == "banking_knowledge"
                                else None
                            ),
                        },
                        "environment_state": {
                            "type": EnvironmentStateType.IN_MEMORY,
                            "access": EnvironmentStateAccess.READ_WRITE,
                            "ref": f"tau2-bench://domain/{domain}",
                            "snapshot": None,
                        },
                    },
                    task_description="\n\n".join(description_parts) or None,
                    task_metadata={
                        "domain": domain,
                        "evaluation_criteria": evaluation,
                        "purpose": description.get("purpose"),
                        "annotations": task_row.get("annotations"),
                        "user_instructions": user_instructions,
                        "user_persona": scenario.get("persona"),
                        "ticket": task_row.get("ticket"),
                        "extra_context": None,
                    },
                    gold_answer=evaluation,
                    task_date=task_date,
                    verifiers=verifiers,
                    scoring_rule=ScoringRule.ALL_PASS,
                    source_platform=SourcePlatform.GITHUB,
                )
                tasks_by_id[task_id] = task

                for basis, verifier in zip(reward_basis, task.verifiers):
                    verifier_id = f"{task_id}::{basis.lower()}"
                    verifiers_by_id[verifier_id] = VerifierDefinition(
                        benchmark=BENCHMARK,
                        task_id=task_id,
                        verifier_id=verifier_id,
                        verifier=verifier,
                        metadata={"reward_basis": basis},
                        source_platform=SourcePlatform.GITHUB,
                    )

            # we preserve each final reward and its reported component results
            for simulation in run.get("simulations", []):
                task_id = f"{domain}.{simulation['task_id']}"
                if (
                    task_id not in tasks_by_id
                    or task_id in EXCLUDED_TASKS.get(submission_name, set())
                ):
                    continue

                reward_info = simulation.get("reward_info") or {}
                if reward_info.get("reward") is None:
                    continue

                reward = float(reward_info["reward"])
                reward_basis = reward_info.get("reward_basis") or []
                reward_breakdown = reward_info.get("reward_breakdown") or {}
                shared_metadata = {
                    "source_reward": reward,
                    "domain": domain,
                    "submission": submission_name,
                    "duration_seconds": simulation.get("duration"),
                    "agent_cost": simulation.get("agent_cost"),
                    "user_cost": simulation.get("user_cost"),
                    "termination_reason": simulation.get("termination_reason"),
                    "seed": simulation.get("seed"),
                    "reward_breakdown": reward_info.get("reward_breakdown"),
                    "retrieval_config": retrieval_config,
                    "action_space_types": run_action_types,
                }

                records.append(
                    Record(
                        benchmark=BENCHMARK,
                        task_id=task_id,
                        agent_id=agent_id,
                        agent_model=model,
                        model_reasoning_effort=reasoning_effort,
                        agent_scaffold=scaffold,
                        model_date=model_date,
                        trial=simulation["trial"],
                        result=int(reward == 1.0),
                        metadata=shared_metadata,
                        source_platform=SourcePlatform.GITHUB,
                    )
                )

                if len(reward_basis) != len(tasks_by_id[task_id].verifiers):
                    continue

                for basis in reward_basis:
                    result = reward_breakdown.get(basis)
                    if result is None:
                        continue

                    verifier_id = f"{task_id}::{basis.lower()}"
                    records.append(
                        Record(
                            benchmark=BENCHMARK,
                            task_id=task_id,
                            verifier_id=verifier_id,
                            record_type=RecordType.VERIFIER_RESULT,
                            agent_id=agent_id,
                            agent_model=model,
                            model_reasoning_effort=reasoning_effort,
                            agent_scaffold=scaffold,
                            model_date=model_date,
                            trial=simulation["trial"],
                            result=int(float(result) == 1.0),
                            metadata=shared_metadata,
                            source_platform=SourcePlatform.GITHUB,
                        )
                    )

    return list(tasks_by_id.values()), list(verifiers_by_id.values()), records


def trajectories():
    root = RAW / "tau2_bench"
    scaffold = standardize_scaffold_name("tau2-bench")

    for submission_name in SUBMISSIONS:
        submission_path = root / "submissions" / submission_name / "submission.json"
        if not submission_path.exists():
            continue

        submission = json.loads(submission_path.read_text())
        raw_model = submission["model_name"]
        model, reasoning_effort = standardize_model(raw_model, submission.get("reasoning_effort"))
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, submission["model_name"])
        for domain, filename in (submission.get("trajectory_files") or {}).items():
            if domain not in DOMAINS:
                continue

            path = root / "trajectories" / submission_name / filename
            if not path.exists():
                continue

            run = json.loads(path.read_text())
            retrieval_config = (run.get("info") or {}).get("retrieval_config")
            run_action_types = [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT]
            if retrieval_config in {"terminal_use", "AllTools"}:
                run_action_types.insert(0, ActionSpaceType.SHELL)

            known_tasks = {f"{domain}.{task['id']}" for task in run.get("tasks", [])}

            for simulation in run.get("simulations", []):
                task_id = f"{domain}.{simulation['task_id']}"
                reward = (simulation.get("reward_info") or {}).get("reward")
                if (
                    task_id not in known_tasks
                    or task_id in EXCLUDED_TASKS.get(submission_name, set())
                    or reward is None
                ):
                    continue

                yield Trajectory(
                    benchmark=BENCHMARK,
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=simulation["trial"],
                    content=json.dumps(simulation, ensure_ascii=False),
                    metadata={
                        "domain": domain,
                        "submission": submission_name,
                        "source_file": filename,
                        "simulation_id": simulation.get("id"),
                        "retrieval_config": retrieval_config,
                        "action_space_types": run_action_types,
                    },
                    source_platform=SourcePlatform.GITHUB,
                )
