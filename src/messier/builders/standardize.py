"""
Normalize source model and scaffold names. Relies on the centralized mappings in `agents.py`.
"""

import re
from .agents import (
    MODEL_ALIASES,
    MODEL_REASONING_CONFIGS,
    SCAFFOLD_PREFIXES,
    SCAFFOLD_ALIASES,
)
from ..models import ModelReasoningEffort

def standardize_scaffold_name(name: str | None) -> str | None:
    if not name:
        return None

    normalized = name.strip().lower()
    # the source places the active scaffold last in a comma-separated pipeline
    if "," in normalized:
        normalized = normalized.split(",")[-1].strip()

    standardized = SCAFFOLD_ALIASES.get(normalized, normalized.replace(" ", "-"))
    # some submission identifiers append model details to the scaffold
    for prefix in sorted(SCAFFOLD_PREFIXES, key=len, reverse=True):
        if standardized.startswith(prefix + "-") or standardized.startswith(prefix + "_"):
            return prefix

    return standardized


def standardize_model(
    name: str | None,
    reasoning_effort: str | ModelReasoningEffort | None = None,
) -> tuple[str | None, ModelReasoningEffort | None]:
    """
    Return the standardized model name and reported reasoning effort.

    :param name: Model name from an upstream source.
    :param reasoning_effort: Reasoning effort reported separately by the source.
    :return: Standardized model name and reasoning effort.
    """
    if not name:
        return None, None

    normalized = name.strip().lower()
    for ch in (" ", "/", "_", ".", ":"):
        normalized = normalized.replace(ch, "-")

    # separate concatenated model families and version numbers
    normalized = re.sub(
        r"\b(claude|gpt|gemini|llama|qwen|kimi|mistral|grok|deepseek|phi|doubao|swellama|codex|glm|nova)(\d)",
        r"\1-\2",
        normalized,
    )
    normalized = re.sub(
        r"(\d)(opus|sonnet|haiku|mini|nano|flash|pro|turbo)\b",
        r"\1-\2",
        normalized,
    )
    while "--" in normalized:
        normalized = normalized.replace("--", "-")

    normalized = re.sub(r"-?\(inspect\)$", "", normalized)
    normalized = re.sub(r"-inspect$", "", normalized)
    normalized = re.sub(r"-paper$", "", normalized)
    normalized = re.sub(r"-fc$", "", normalized)

    if reasoning_effort is not None:
        reasoning_effort = ModelReasoningEffort(reasoning_effort)

    if normalized in MODEL_REASONING_CONFIGS:
        normalized, effort = MODEL_REASONING_CONFIGS[normalized]
        mapped_effort = ModelReasoningEffort(effort)
        if reasoning_effort is not None and reasoning_effort != mapped_effort:
            raise ValueError(f"conflicting reasoning effort for {name!r}")
        reasoning_effort = mapped_effort

    normalized = normalized.strip("-")
    standardized = MODEL_ALIASES.get(normalized, normalized)

    return standardized, reasoning_effort


def format_agent_id(
    model: str,
    scaffold: str | None,
    reasoning_effort: ModelReasoningEffort | None = None,
) -> str:
    configured_model = model
    if reasoning_effort == ModelReasoningEffort.ENABLED:
        configured_model += " (reasoning)"
    elif reasoning_effort is not None:
        configured_model += f" ({reasoning_effort.value})"

    return f"{configured_model}+{scaffold}" if scaffold else configured_model
