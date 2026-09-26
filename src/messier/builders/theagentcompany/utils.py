"""
Read TheAgentCompany task instructions and rubrics.
"""

import fnmatch
from pathlib import Path
from .constants import (
    CHECKPOINT_RE,
    GOLD_FILE_GLOBS,
    MAX_INLINE_BYTES,
    OPENHANDS_RE,
    SUBMISSION_OVERRIDES,
    TEXT_EXTS,
)


def read_text_capped(path: Path) -> str | None:
    """
    Read a small text file.

    :param path: File to read.
    :return: Text contents, or ``None`` for binary and oversized files.
    """
    if path.suffix.lower() not in TEXT_EXTS or not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_INLINE_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None

def parse_submission(name: str) -> tuple[str, str | None]:
    if name in SUBMISSION_OVERRIDES:
        return SUBMISSION_OVERRIDES[name]

    match = OPENHANDS_RE.match(name)
    if match:
        return ("OpenHands", match.group(1))

    return (name, None)


def parse_checkpoints(doc: str) -> list[dict]:
    """
    Read verifier checkpoints from a task rubric.

    :param doc: Markdown rubric.
    :return: Ordered checkpoints with their points and expected result.
    """
    checkpoints = []
    for match in CHECKPOINT_RE.finditer(doc):
        label = match.group(1)
        step = int(label) if label and label.isdigit() else "final"
        checkpoints.append(
            (
                step,
                int(match.group(2)),
                match.start(),
                match.end()
            )
        )

    parsed = []
    for index, (step, points, _, end) in enumerate(checkpoints):
        body_end = (
            checkpoints[index + 1][2]
            if index + 1 < len(checkpoints)
            else len(doc)
        )
        parsed.append({"step": step, "points": points, "expected": doc[end:body_end].strip()})

    return parsed


def is_gold_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in GOLD_FILE_GLOBS)
