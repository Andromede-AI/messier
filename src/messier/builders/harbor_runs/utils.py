import json
from pathlib import Path
from ..standardize import standardize_model


def read_reward(trial: Path) -> tuple[float | None, dict | None]:
    """
    Read the final result and any separately reported verifier results.

    :param trial: Harbor trial directory.
    :return: The final value and optional detailed results.
    """
    details = None
    details_path = trial / "verifier" / "reward-details.json"
    if details_path.exists():
        details = json.loads(details_path.read_text()).get("reward")

    reward_path = trial / "verifier" / "reward.json"
    if reward_path.exists():
        reward = json.loads(reward_path.read_text()).get("reward")
        if isinstance(reward, dict):
            reward = reward.get("score")

        return (float(reward) if reward is not None else None, details)

    reward_text_path = trial / "verifier" / "reward.txt"
    if reward_text_path.exists():
        return float(reward_text_path.read_text().strip()), details

    return None, details


def load_result(trial: Path) -> dict:
    return json.loads((trial / "result.json").read_text())


def model_identity(config: dict) -> tuple[str, str, str | None]:
    """Return the raw model name, normalized name, and reasoning setting."""
    raw_model = config["agent"]["model_name"]
    reasoning = (config["agent"].get("kwargs") or {}).get("reasoning_effort")
    model, reasoning_effort = standardize_model(raw_model.split("/", 1)[-1], reasoning)
    return raw_model, model, reasoning_effort


def source_task(config: dict) -> str:
    """Return a task name from either Harbor task-reference format."""
    task = config["task"]
    if task.get("name"):
        return task["name"]

    if task.get("path"):
        return Path(task["path"]).name

    raise ValueError("Harbor task config has neither a name nor a path")


def task_ref(config: dict, result: dict) -> str | None:
    """Return the immutable task source used by the run."""
    task = config["task"]
    identity = result.get("task_id") or {}
    if task.get("ref") or identity.get("ref"):
        return task.get("ref") or identity["ref"]

    git_url = identity.get("git_url") or task.get("git_url")
    commit = identity.get("git_commit_id") or task.get("git_commit_id")
    path = identity.get("path") or task.get("path")
    if git_url and commit and path:
        return f"{git_url.removesuffix('.git')}/tree/{commit}/{path}"

    return None


def task_id(result: dict, slug: str) -> str:
    """
    Read a task identifier and remove its organization prefix.

    :param result: Parsed Harbor result file.
    :param slug: Harbor dataset name.
    :return: Task identifier used by the dataset.
    """
    name = result["task_name"]
    organization = slug.split("/", 1)[0]
    name = name.removeprefix(f"{organization}/")
    if "/" not in slug:
        name = name.removeprefix(f"{slug}-")

    return name
