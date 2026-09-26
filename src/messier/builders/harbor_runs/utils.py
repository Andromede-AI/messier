import json
from pathlib import Path


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


def task_id(result: dict, slug: str) -> str:
    """
    Read a task identifier and remove its organization prefix.

    :param result: Parsed Harbor result file.
    :param slug: Harbor dataset name.
    :return: Task identifier used by the dataset.
    """
    name = result["task_name"]
    organization = slug.split("/", 1)[0]
    return name.removeprefix(f"{organization}/")
