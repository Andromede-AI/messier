"""
Parse shared SWE-bench result formats.
"""

import json
from ..standardize import format_agent_id
from ...io import RAW
from ...models import SourcePlatform, Record
from .constants import AP_SUBJECT_RE, LLM_PREFIX, REST_OVERRIDES, VERSION_TOKEN


def split_scaffold_model(rest: str) -> tuple[str | None, str | None]:
    if rest in REST_OVERRIDES:
        return REST_OVERRIDES[rest]
    if "_" not in rest and "-" not in rest and LLM_PREFIX.match(rest):
        return None, rest

    for sep in ("_", "-"):
        tokens = rest.split(sep)
        if len(tokens) == 1:
            continue

        for index, token in enumerate(tokens):
            if not LLM_PREFIX.match(token):
                continue

            end = index + 1
            while end < len(tokens) and VERSION_TOKEN.match(tokens[end]):
                end += 1

            scaffold = sep.join(tokens[:index]) if index > 0 else None
            return scaffold, sep.join(tokens[index:end])

    return rest, None


def parse_subject_id(subject_id: str) -> dict:
    match = AP_SUBJECT_RE.match(subject_id)
    if not match:
        return {"agent_scaffold": subject_id, "agent_model": "undisclosed"}

    rest = match.group("rest")
    scaffold, model = split_scaffold_model(rest)
    if model is None:
        model, scaffold = "undisclosed", scaffold or rest

    return {
        "agent_scaffold": scaffold,
        "agent_model": model,
    }


def devin_records(task_ids: set[str], benchmark: str) -> list[Record]:
    path = RAW / "devin_swebench" / "devin_swe_outputs.json"
    if not path.exists():
        return []

    data = json.loads(path.read_text())
    model = "undisclosed"
    scaffold = "devin"
    records: list[Record] = []
    for item in data:
        task_id = item["instance_id"]
        if task_id not in task_ids:
            continue

        score = 1 if str(item.get("pass_or_fail", "")).lower() == "pass" else 0
        records.append(
            Record(
                benchmark=benchmark,
                task_id=task_id,
                agent_id=format_agent_id(model, scaffold),
                agent_model=model,
                agent_scaffold=scaffold,
                model_date=None,
                result=score,
                metadata={"raw_source": "devin"},
                source_platform=SourcePlatform.HUGGINGFACE,
            )
        )
    return records
