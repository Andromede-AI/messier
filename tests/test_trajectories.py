import json
from pydantic import ValidationError
from messier.io import contains_credentials
from messier.models import Trajectory


def test_trajectory_consistency(trajectories_path, task_index, record_audit):
    invalid = []
    duplicate_keys = []
    missing_tasks = []
    missing_results = []
    malformed_content = []
    exposed_credentials = []
    seen = set()

    with trajectories_path.open() as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            key = (row.get("benchmark"), row.get("task_id"), row.get("agent_id"), row.get("trial"))
            if contains_credentials(line) and len(exposed_credentials) < 3:
                exposed_credentials.append(key)
            try:
                Trajectory.model_validate(row)
            except ValidationError as error:
                if len(invalid) < 3:
                    invalid.append((key, error.errors()[:3]))

            if key in seen and len(duplicate_keys) < 3:
                duplicate_keys.append(key)
            seen.add(key)

            task_key = key[:2]
            if task_key not in task_index and len(missing_tasks) < 3:
                missing_tasks.append(key)
            if key not in record_audit["trial_keys"] and len(missing_results) < 3:
                missing_results.append(key)

            try:
                content = json.loads(row.get("content", ""))
                if not isinstance(content, (dict, list)):
                    raise ValueError
            except (json.JSONDecodeError, TypeError, ValueError):
                if len(malformed_content) < 3:
                    malformed_content.append(key)

    assert not invalid, f"invalid trajectories: {invalid}"
    assert not duplicate_keys, f"duplicate trajectories: {duplicate_keys}"
    assert not missing_tasks, f"trajectories reference missing tasks: {missing_tasks}"
    assert not missing_results, f"trajectories lack matching results: {missing_results}"
    assert not malformed_content, f"trajectory content is not JSON: {malformed_content}"
    assert not exposed_credentials, f"trajectories contain possible credentials: {exposed_credentials}"
