"""
Import GSO software tasks and model results.
"""

import json
from ..benchmarks import REPO_DESCRIPTIONS
from ..standardize import format_agent_id, standardize_model, standardize_scaffold_name
from ..dates import resolve_task_date, to_iso_date
from ...io import RAW, load_jsonl
from ...models import (
    BuildResult,
    ActionSpaceType,
    DataProvider,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    ScoringRule,
    Task,
    Trajectory,
    VerifierMode,
    VerifierType,
)

# each GSO task uses one test-based verifier

def build() -> BuildResult:
    task_rows = load_jsonl(RAW / "gso" / "tasks.jsonl")
    model_dates = json.loads((RAW / "agent_psychometrics" / "gso" / "agent_dates.json").read_text())
    tasks = []
    for task in task_rows:
        workspace = (task.get("repo") or "").replace("/", "__")
        task_description = (
            "<uploaded_files>\n"
            f"/workspace/{workspace}\n"
            "</uploaded_files>\n"
            f"I've uploaded a python code repository in the directory {workspace}. Consider the following test script showing an example usage of the repository:\n\n"
            "<test_script>\n"
            f"{task.get('prob_script') or ''}\n"
            "</test_script>\n\n"
            "Can you help me implement the necessary changes to the repository so that the runtime of the <test_script> is optimized?\n"
            "Basic guidelines:\n"
            "   1. Your task is to make changes to non-tests files in the /workspace directory to improve the performance of the <test_script>.\n"
            "   2. Make changes while ensuring the repository is functionally equivalent to the original.\n"
            "   3. Do not overoptimize for just the specific inputs in <test_script>. Make general performance improvements for the usage scenario shown.\n"
            "   4. You may need to rebuild the repo for your changes to take effect before testing. Some rebuilds may take time to run, so be patient with running them.\n"
            "\nFollow these steps to improve performance:\n"
            "1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n"
            "2. Create a script in the /workspace directory (e.g., /workspace/test_opt.py) to reproduce and time the example and execute it with `python /workspace/<filename.py>` using the BashTool.\n"
            "3. Edit the sourcecode of the repo to improve the performance\n"
            "4. Rebuild and rerun your script and confirm that the performance has improved!\n"
            "Your thinking should be thorough and so it's fine if it's very long.\n"
        )

        omitted_commands = (
            "git clean -xfd",
            "which python",
            "python --version",
            "uv venv",
        )
        install_commands = [
            command.replace("git clean -xfd &&", "").strip()
            for command in task.get("install_commands") or []
            if not command.startswith(omitted_commands)
        ]
        if install_commands:
            install_script = "\n".join(install_commands)
            task_description += (
                f"\nTo rebuild the repo with your changes at any point, you can use "
                f"the following in the {workspace} directory:\n"
                f"```\n{install_script}\n```\n"
            )

        script_code = "\n\n".join(task.get("tests") or [])

        tasks.append(
            Task(
                benchmark="gso",
                task_id=task["instance_id"],
                environment_id=task["repo"],
                environment_description=(f"{task['repo']} - {REPO_DESCRIPTIONS[task['repo']]}"),
                environment_metadata={
                    "action_space": {
                        "types": [ActionSpaceType.SHELL],
                        "description": "Shell commands and file edits within the repository.",
                    },
                    "environment_state": {
                        "type": EnvironmentStateType.FILESYSTEM,
                        "access": EnvironmentStateAccess.READ_WRITE,
                        "ref": f"github://{task['repo']}@{task['base_commit']}",
                        "snapshot": None,
                    },
                    "base_commit": task["base_commit"],
                    "opt_commit": task["opt_commit"],
                    "arch": task.get("arch"),
                    "instance_image_tag": task.get("instance_image_tag"),
                    "setup_commands": task.get("setup_commands"),
                    "install_commands": task.get("install_commands"),
                    "url": f"https://github.com/{task['repo']}/tree/{task['base_commit']}",
                },
                task_description=task_description,
                task_metadata={
                    "api": task.get("api"),
                    "prob_script": task.get("prob_script"),
                    "hints_text": task.get("hints_text"),
                    "gt_commit_message": task.get("gt_commit_message"),
                },
                gold_answer=task.get("gt_diff"),
                task_date=resolve_task_date("gso", task.get("created_at")),
                verifiers=[
                    {
                        "type": VerifierType.SCRIPT,
                        "mode": VerifierMode.FINAL,
                        "script_code": script_code,
                        "metric": "fraction_of_expert_runtime_improvement",
                        "threshold": 0.95,
                    }
                ],
                scoring_rule=ScoringRule.THRESHOLD,
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )

    # we assign the shared OpenHands scaffold to every submission
    scaffold = standardize_scaffold_name("openhands")
    records = []
    response_path = RAW / "agent_psychometrics" / "gso" / "responses.jsonl"
    for row in load_jsonl(response_path):
        raw_model = row["subject_id"]
        model, reasoning_effort = standardize_model(raw_model)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = to_iso_date(model_dates.get(raw_model))

        for task_id, score in row["responses"].items():
            records.append(
                Record(
                    benchmark="gso",
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    result=int(score),
                    source_platform=SourcePlatform.GITHUB,
                    data_provider=DataProvider.AGENT_PSYCHOMETRICS,
                )
            )

    return tasks, [], records


def trajectories():
    model_dates = json.loads((RAW / "agent_psychometrics" / "gso" / "agent_dates.json").read_text())
    scaffold = standardize_scaffold_name("openhands")
    record_keys = set()
    response_path = RAW / "agent_psychometrics" / "gso" / "responses.jsonl"
    for row in load_jsonl(response_path):
        model, reasoning_effort = standardize_model(row["subject_id"])
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        for task_id in row["responses"]:
            record_keys.add((task_id, agent_id))
    trajectory_root = RAW / "gso" / "trajectories"
    if not trajectory_root.exists():
        return

    for path in sorted(trajectory_root.glob("*.jsonl")):
        raw_model = path.stem
        model, reasoning_effort = standardize_model(raw_model)
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = to_iso_date(model_dates.get(raw_model))

        for line in path.open():
            if not line.strip():
                continue

            row = json.loads(line)
            if (row["instance_id"], agent_id) not in record_keys:
                continue

            yield Trajectory(
                benchmark="gso",
                task_id=row["instance_id"],
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=model_date,
                trial=0,
                content=line.rstrip("\n"),
                metadata={"source_file": path.name},
                source_platform=SourcePlatform.GITHUB,
                data_provider=DataProvider.AGENT_PSYCHOMETRICS,
            )
