from ...models import ActionSpaceType, EnvironmentStateAccess, EnvironmentStateType
import json
from .constants import (
    LIVE_ACTION_DESCRIPTION,
    MULTI_TURN_ACTION_DESCRIPTION,
    MULTI_TURN_CLASS_REPO_REF,
)


def extract_user_turns(question) -> list[str]:
    """
    Extract the user message from each BFCL turn.

    :param question: Source representation of a conversation.
    :return: User messages in order.
    """
    if not isinstance(question, list):
        return []

    turns: list[str] = []
    for batch in question:
        if not isinstance(batch, list):
            continue

        user_messages = [
            message.get("content") or ""
            for message in batch
            if isinstance(message, dict) and message.get("role") == "user"
        ]

        joined = "\n\n".join(
            message
            for message in user_messages
            if message
        )
        if joined:
            turns.append(joined)

    return turns


def extract_task_content(record: dict) -> dict:
    """
    Extract a task and its available actions from a BFCL result.

    :param record: One BFCL result.
    :return: Task text, action space, initial environment state, and answer.
    """
    prompt = record.get("prompt") or {}
    functions = prompt.get("function") or []
    if functions:
        action_space = {
            "types": [ActionSpaceType.TOOL_CALLS],
            "mode": "per_task_schemas",
            "description": LIVE_ACTION_DESCRIPTION,
            "tools": [
                {"type": "function", "function": function}
                for function in functions
            ],
        }
        environment_state = {
            "type": EnvironmentStateType.NONE,
            "access": EnvironmentStateAccess.NONE,
            "ref": None,
            "snapshot": None,
        }
    else:
        action_space = {
            "types": [ActionSpaceType.TOOL_CALLS, ActionSpaceType.TEXT],
            "mode": "class_registry",
            "description": MULTI_TURN_ACTION_DESCRIPTION,
            "classes": prompt.get("involved_classes") or [],
            "excluded_methods": prompt.get("excluded_function") or [],
        }
        environment_state = {
            "type": EnvironmentStateType.IN_MEMORY,
            "access": EnvironmentStateAccess.READ_WRITE,
            "ref": MULTI_TURN_CLASS_REPO_REF,
            "snapshot": prompt.get("initial_config") or {},
        }

    user_turns = extract_user_turns(prompt.get("question"))
    if len(user_turns) <= 1:
        description = user_turns[0] if user_turns else None
    else:
        blocks = [f"Turn 1:\n{user_turns[0]}"] + [
            f"Turn {index + 1}:\n{prompt}"
            for index, prompt in enumerate(user_turns[1:], start=1)
        ]
        description = "\n\n".join(blocks)

    return {
        "task_description": description,
        "user_turns": user_turns,
        "action_space": action_space,
        "environment_state": environment_state,
        "action_path": prompt.get("path") or None,
        "gold_answer": record.get("possible_answer") or None,
    }


def read_score_file(path) -> tuple[dict | None, dict[str, dict]]:
    """
    Read a BFCL summary followed by its failed tasks.

    :param path: JSON Lines result file.
    :return: The summary and failed tasks indexed by identifier.
    """
    aggregate = None
    failed: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            record = json.loads(line)
            if aggregate is None:
                aggregate = record
            elif "id" in record:
                failed[record["id"]] = record

    return aggregate, failed
