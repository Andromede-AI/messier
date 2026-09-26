"""
Import CyBench tasks and agent results.
"""

import json
import re
from collections import Counter
from ..benchmarks import UPSTREAM_COMMITS
from ..standardize import format_agent_id, standardize_model
from ..dates import parse_htb_readme_date, resolve_model_date, resolve_task_date
from ...io import ENRICH, RAW, load_jsonl
from ...models import (
    BuildResult,
    ActionSpaceType,
    DataProvider,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    HumanEffort,
    Record,
    Task,
    VerifierMode,
    VerifierType,
)

# cybench contains capture-the-flag tasks

def build() -> BuildResult:
    difficulty_tag = re.compile(r"^\[[^\]]+\]\s*")
    source_root = ENRICH / "cybench"
    metadata_by_task = {}

    for metadata_path in source_root.rglob("metadata.json"):
        parts = [
            part
            for part in metadata_path.relative_to(source_root).parts
            if part != "metadata"
        ][:-1]
        if len(parts) < 2:
            continue

        normalized_parts = [
            difficulty_tag.sub("", part).lower().replace(" ", "_").replace("-", "_")
            for part in parts
        ]
        readme_path = metadata_path.parent.parent / "README.md"
        metadata_by_task["_".join(normalized_parts)] = {
            "meta": json.loads(metadata_path.read_text()),
            "readme": readme_path.read_text() if readme_path.exists() else None,
            "dir_parts": parts,
        }

    # we combine both sources of human completion times
    human_minutes = {
        row["task_id"]: row["human_minutes"]
        for row in load_jsonl(RAW / "bridge" / "cybench_human_minutes_by_task.jsonl")
    }
    first_success_times = {
        row["task_id"]: row["human_minutes"]
        for row in load_jsonl(RAW / "bridge" / "cybench_fst_master.jsonl")
    }

    tasks_by_id = {}
    records = []
    trials: Counter = Counter()

    for row in load_jsonl(RAW / "bridge" / "cybench_normalized_results.jsonl"):
        task_id = row["task_id"]
        normalized_task_id = (row.get("task_id_base") or task_id).lower().replace("-", "_")
        entry = metadata_by_task.get(normalized_task_id)

        if entry is None:
            raise ValueError(f"missing CyBench metadata for {task_id}")

        metadata = entry["meta"]
        readme = entry["readme"]

        if task_id not in tasks_by_id:
            environment_parts = entry["dir_parts"]
            environment_id = "/".join(
                difficulty_tag.sub("", part).lower().replace(" ", "-")
                for part in environment_parts
            )
            environment_path = "/".join(environment_parts)
            environment_url = (
                "https://github.com/andyzorigin/cyber-bench/tree/"
                f"{UPSTREAM_COMMITS['cybench']}/benchmark/{environment_path}"
            )

            ctf_host = environment_parts[0] if len(environment_parts) >= 1 else None
            if len(environment_parts) >= 3 and environment_parts[1] == "CTF-Challenges":
                ctf_event = environment_parts[2]
            else:
                ctf_event = environment_parts[1] if len(environment_parts) >= 2 else None

            categories = metadata.get("categories") or []
            category = categories[0] if categories else None
            environment_description = (
                f"A Kali Linux environment for cybersecurity tasks from {ctf_event}, "
                "with challenge files, security tools, and, when required, a network service."
                if ctf_event
                else (
                    "A Kali Linux environment for cybersecurity tasks, with challenge files, "
                    "security tools, and, when required, a network service."
                )
            )
            network_endpoint = (metadata.get("target_host") or "").strip() or None
            environment_metadata: dict = {
                "action_space": {
                    "types": [ActionSpaceType.SHELL, ActionSpaceType.TEXT],
                    "description": "Shell commands and text responses.",
                    "tools": metadata.get("tools"),
                    "internet": metadata.get("internet"),
                    "stateful_shell": metadata.get("stateful_shell"),
                },
                "environment_state": {
                    "type": EnvironmentStateType.FILESYSTEM,
                    "access": EnvironmentStateAccess.READ_WRITE,
                    "description": "Files and directories.",
                    "ref": environment_url,
                    "snapshot": {"network_endpoint": network_endpoint}
                    if network_endpoint
                    else None,
                },
            }
            environment_metadata["url"] = environment_url
            if ctf_host:
                environment_metadata["ctf_host"] = ctf_host
            if ctf_event:
                environment_metadata["ctf_event"] = ctf_event

            subtasks = [
                subtask
                for subtask in (metadata.get("subtasks") or [])
                if isinstance(subtask, dict)
            ]
            # keep missing answers so subtask positions remain aligned
            gold_answer = [
                {
                    "step": index,
                    "description": subtask.get("question"),
                    "expected": subtask.get("answer"),
                    "answer_format": subtask.get("answer_format"),
                }
                for index, subtask in enumerate(subtasks)
            ] or None

            # available results contain one final flag check per task
            verifiers = [{"type": VerifierType.EXACT_MATCH, "mode": VerifierMode.FINAL, "name": "flag_match"}]
            source_date = parse_htb_readme_date(readme)
            task_date = resolve_task_date("cybench", source_date)
            human_minutes_value = (
                human_minutes.get(task_id)
                or first_success_times.get(task_id)
                or row.get("human_minutes")
            )
            effort = (
                HumanEffort(minutes_median=human_minutes_value)
                if human_minutes_value is not None
                else None
            )

            task_metadata = {
                "task_id_base": row.get("task_id_base") or task_id,
                "category": category,
                "categories": categories,
                "easy_prompt": metadata.get("easy_prompt"),
                "subtasks": [
                    {
                        "step": index,
                        "subtask": subtask.get("subtask"),
                        "question": subtask.get("question"),
                        "answer": subtask.get("answer"),
                        "answer_format": subtask.get("answer_format"),
                        "hints": subtask.get("hints"),
                        "commands": subtask.get("commands"),
                        "solution_files": subtask.get("solution_files"),
                        "annotator_note": subtask.get("annotator_note"),
                        "context": subtask.get("context"),
                        "tools": subtask.get("tools"),
                        "internet": subtask.get("internet"),
                        "stateful_shell": subtask.get("stateful_shell"),
                    }
                    for index, subtask in enumerate(subtasks)
                ]
                or None,
            }
            if readme is not None:
                task_metadata["readme"] = readme

            # hard_prompt contains the task without scaffold instructions
            tasks_by_id[task_id] = Task(
                benchmark="cybench",
                task_id=task_id,
                environment_id=environment_id,
                environment_metadata=environment_metadata,
                environment_description=environment_description,
                task_description=metadata.get("hard_prompt"),
                task_metadata=task_metadata,
                gold_answer=gold_answer,
                verifiers=verifiers,
                task_date=task_date,
                human=effort,
                difficulty_label=row.get("difficulty_label"),
                source_platform=SourcePlatform.GITHUB,
            )

        model, reasoning_effort = standardize_model(row.get("model"))
        scaffold = "cybench"
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, row.get("model"))
        key = (task_id, agent_id)
        trial = trials[key]
        trials[key] = trial + 1

        records.append(
            Record(
                benchmark="cybench",
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=trial,
                result=row.get("score_binarized"),
                metadata={
                    "run_id": row.get("run_id"),
                    "human_source": row.get("human_source"),
                    "eval_mode": row.get("eval_mode"),
                    "task_split": row.get("task_split"),
                },
                source_platform=SourcePlatform.GITHUB,
                data_provider=DataProvider.BRIDGE,
            )
        )

    return list(tasks_by_id.values()), [], records
