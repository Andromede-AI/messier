import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from messier.io import contains_credentials
from messier.models import Task, VerifierDefinition

CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML = re.compile(r"<[a-z!/][^>]*>|&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-f]+);", re.IGNORECASE)
PLACEHOLDER = re.compile(r"\b(?:TODO|FIXME|TBD)\b")


def test_task_schema(tasks):
    for task in tasks:
        Task.model_validate(task)


def test_verifier_schema(verifiers):
    for verifier in verifiers:
        VerifierDefinition.model_validate(verifier)


def test_no_credentials_in_tasks_or_verifiers(tasks, verifiers):
    exposed = []
    for row in [*tasks, *verifiers]:
        if contains_credentials(json.dumps(row)):
            exposed.append((row.get("benchmark"), row.get("task_id")))

    assert not exposed, f"tasks or verifiers contain possible credentials: {exposed[:3]}"


def test_verifier_references(tasks, verifiers):
    task_keys = {(task["benchmark"], task["task_id"]) for task in tasks}
    verifier_keys = Counter(
        (verifier["benchmark"], verifier["verifier_id"]) for verifier in verifiers
    )
    missing = [
        (verifier["benchmark"], verifier["task_id"], verifier["verifier_id"])
        for verifier in verifiers
        if (verifier["benchmark"], verifier["task_id"]) not in task_keys
    ]
    duplicates = [key for key, count in verifier_keys.items() if count > 1]

    assert not missing, f"verifiers reference missing tasks: {missing[:3]}"
    assert not duplicates, f"duplicate verifier IDs: {duplicates[:3]}"


def test_task_consistency(tasks, verifiers):
    definitions = defaultdict(list)
    for verifier in verifiers:
        key = (verifier["benchmark"], verifier["task_id"])
        definitions[key].append(verifier["verifier"])

    for task in tasks:
        key = (task["benchmark"], task["task_id"])
        embedded = task["verifiers"]
        represented = definitions[key]

        assert embedded, f"task has no verifier: {key}"
        assert task["n_verifiers"] == len(embedded) == len(represented), (
            f"verifier count differs for {key}"
        )
        embedded = Counter(json.dumps(verifier, sort_keys=True) for verifier in embedded)
        represented = Counter(json.dumps(verifier, sort_keys=True) for verifier in represented)
        assert embedded == represented, f"verifier definitions differ for {key}"
        assert task["benchmark_group"] is not None, f"task has no benchmark group: {key}"
        assert task["scoring_rule"] is not None, f"task has no scoring rule: {key}"
        assert task["scoring_rule"] != "direct" or task["n_verifiers"] == 1, (
            f"direct scoring requires one verifier: {key}"
        )


def test_authored_descriptions(tasks):
    missing = []
    malformed = []
    for task in tasks:
        key = (task["benchmark"], task["task_id"])
        descriptions = (
            task.get("environment_description"),
            task["environment_metadata"]["action_space"].get("description"),
        )
        if any(not description for description in descriptions):
            missing.append(key)
            continue

        for description in descriptions:
            if (
                CONTROL_CHARACTER.search(description)
                or HTML.search(description)
                or PLACEHOLDER.search(description)
            ):
                malformed.append(key)
                break

    assert not missing, f"missing authored descriptions: {missing[:3]}"
    assert not malformed, f"malformed authored descriptions: {malformed[:3]}"


def test_verifier_mode_location(tasks):
    misplaced = [
        (task["benchmark"], task["task_id"])
        for task in tasks
        if task["environment_metadata"]["action_space"].get("mode")
        in {"final", "sequential"}
    ]
    assert not misplaced, f"verifier modes stored as action-space metadata: {misplaced[:3]}"


def test_classification_coverage(tasks):
    missing = [
        (task["benchmark"], task["task_id"])
        for task in tasks
        if task.get("soc_code") is None or task.get("naics_code") is None
    ]
    assert not missing, f"tasks without SOC or NAICS labels: {missing[:3]}"


def test_action_spaces(tasks):
    invalid = []
    for task in tasks:
        action_types = task["environment_metadata"]["action_space"]["types"]
        if not action_types or len(action_types) != len(set(action_types)):
            invalid.append((task["benchmark"], task["task_id"], action_types))

    assert not invalid, f"invalid action spaces: {invalid[:3]}"


def test_environment_ids(tasks):
    descriptions = defaultdict(set)
    for task in tasks:
        if task.get("environment_id"):
            key = (task["benchmark"], task["environment_id"])
            descriptions[key].add(task.get("environment_description") or "")

    inconsistent = [key for key, values in descriptions.items() if len(values) > 1]
    assert not inconsistent, (
        f"environment IDs have multiple descriptions: {inconsistent[:3]}"
    )


def test_task_ids(tasks):
    keys = Counter((task["benchmark"], task["task_id"]) for task in tasks)
    duplicates = [key for key, count in keys.items() if count > 1]
    malformed = [
        (task["benchmark"], task["task_id"])
        for task in tasks
        if task["task_id"].strip() != task["task_id"] or "\n" in task["task_id"]
    ]

    assert not duplicates, f"duplicate task IDs: {duplicates[:3]}"
    assert not malformed, f"malformed task IDs: {malformed[:3]}"


def test_task_description_characters(tasks):
    malformed = []
    for task in tasks:
        description = task.get("task_description")
        if not description:
            continue

        if CONTROL_CHARACTER.search(description):
            malformed.append((task["benchmark"], task["task_id"]))

    assert not malformed, f"malformed task descriptions: {malformed[:3]}"


def test_extra_context(tasks, available_files):
    invalid = []
    for task in tasks:
        context = task.get("task_metadata", {}).get("extra_context")
        if context is None:
            continue

        invalid_context = (
            not isinstance(context, dict)
            or not context
            or all(value is None for value in context.values())
        )
        if invalid_context:
            invalid.append((task["benchmark"], task["task_id"]))

        for input_file in context.get("input_files", []):
            path = input_file.get("path")
            dataset_path = input_file.get("dataset_path")
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(dataset_path, str)
                or not dataset_path.startswith("task_files/")
                or ".." in Path(dataset_path).parts
                or dataset_path not in available_files
            ):
                invalid.append((task["benchmark"], task["task_id"]))
                break

    assert not invalid, f"invalid extra context: {invalid[:3]}"


def test_reference_files_and_script_code(tasks, available_files):
    invalid = []
    for task in tasks:
        key = (task["benchmark"], task["task_id"])
        gold_answer = task.get("gold_answer")
        if isinstance(gold_answer, dict):
            references = list(gold_answer.get("reference_files") or [])
            if gold_answer.get("oracle_program"):
                references.append(gold_answer["oracle_program"])

            for reference in references:
                dataset_path = reference.get("dataset_path")
                if dataset_path not in available_files:
                    invalid.append(key)
                    break

        for verifier in task["verifiers"]:
            script_code = verifier.get("script_code")
            if script_code is not None and (
                not isinstance(script_code, str) or not script_code.strip()
            ):
                invalid.append(key)
                break

    assert not invalid, f"invalid reference files or verifier code: {invalid[:3]}"


def test_task_dates(tasks):
    invalid = []
    for task in tasks:
        task_date = task.get("task_date")
        if task_date is None:
            continue

        try:
            date.fromisoformat(task_date)
        except (TypeError, ValueError):
            invalid.append((task["benchmark"], task["task_id"], task_date))

    assert not invalid, f"invalid task dates: {invalid[:3]}"
