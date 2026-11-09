import json
from pathlib import Path
from typing import Iterator
from ..standardize import standardize_model


PROVIDER_ERROR_MARKERS = (
    "APITimeoutError",
    "BadGatewayError",
    "LLMTimeoutError",
    "RateLimitError",
    "429 Too Many Requests",
    "502 Bad Gateway",
)
PROVIDER_ERROR_LOG_TAIL = 10_000


def iter_trials(root: Path) -> Iterator[tuple[Path, dict]]:
    """Yield trial directories and results, excluding Harbor's run summaries."""
    for result_path in sorted(root.rglob("result.json")):
        trial = result_path.parent
        if not (trial / "config.json").exists():
            continue

        result = json.loads(result_path.read_text())
        if result.get("task_name"):
            yield trial, result


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


def error_metadata(trial: Path, result: dict, reward: float | None) -> dict:
    """Return normalized error fields without discarding Harbor's source status."""
    exception = result.get("exception_info") or {}
    exception_type = exception.get("exception_type")
    if not exception_type:
        return {"error_type": "missing_score"} if reward is None else {}

    metadata = {"source_exception_type": exception_type}
    message = exception.get("exception_message") or ""
    if exception_type == "NonZeroAgentExitCodeError":
        log_path = trial / "agent" / "openhands_sdk.txt"
        if log_path.exists():
            message += log_path.read_text(errors="replace")[-PROVIDER_ERROR_LOG_TAIL:]
        if any(marker in message for marker in PROVIDER_ERROR_MARKERS):
            artifacts = artifact_metadata(trial)
            complete_output = artifacts.get("n_expected_artifacts", 0) > 0 and (
                artifacts["n_artifacts"] == artifacts["n_expected_artifacts"]
            )
            if not complete_output:
                metadata["error_type"] = "infrastructure_error"
            return metadata

    error_type = {
        "NetworkConnectionError": "infrastructure_error",
        "AgentSetupTimeoutError": "environment_error",
        "RewardFileNotFoundError": "evaluator_error",
        "VerifierTimeoutError": "evaluator_error",
        "FilesystemExecutionError": "environment_error",
        "CancelledError": "unknown_error",
    }.get(exception_type)
    if error_type:
        metadata["error_type"] = error_type
    elif reward is None:
        metadata["error_type"] = "missing_score"
    return metadata


def artifact_metadata(trial: Path) -> dict:
    """Summarize the files Harbor attempted to collect from the environment."""
    manifest_path = trial / "artifacts" / "manifest.json"
    if not manifest_path.exists():
        return {}

    manifest = json.loads(manifest_path.read_text())
    files = [entry for entry in manifest if entry.get("type") == "file"]
    return {
        "n_expected_artifacts": len(files),
        "n_artifacts": sum(entry.get("status") == "ok" for entry in files),
    }


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
