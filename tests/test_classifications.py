from collections import Counter
import json
from messier.io import contains_credentials
from messier.taxonomy import NAICS, SOC


def test_classifications_match_tasks(tasks, classifications):
    task_labels = {
        (task["benchmark"], task["task_id"]): (task["soc_code"], task["naics_code"])
        for task in tasks
    }
    classification_labels = {
        (row.get("benchmark"), row.get("task_id")): (row.get("soc_code"), row.get("naics_code"))
        for row in classifications
    }
    classification_keys = Counter(
        (row.get("benchmark"), row.get("task_id"))
        for row in classifications
    )
    duplicate_keys = [key for key, count in classification_keys.items() if count > 1]
    missing = sorted(set(task_labels) - set(classification_keys))
    unexpected = sorted(set(classification_keys) - set(task_labels))
    mismatched = [
        key
        for key, expected in task_labels.items()
        if key in classification_labels and classification_labels[key] != expected
    ]

    assert not duplicate_keys, f"duplicate classifications: {duplicate_keys[:3]}"
    assert not missing, f"tasks without classifications: {missing[:3]}"
    assert not unexpected, f"classifications reference missing tasks: {unexpected[:3]}"
    assert not mismatched, f"task and classification labels differ: {mismatched[:3]}"


def test_classification_codes(classifications):
    invalid = [
        (row.get("benchmark"), row.get("task_id"), row.get("soc_code"), row.get("naics_code"))
        for row in classifications
        if row.get("soc_code") not in SOC or row.get("naics_code") not in NAICS
    ]

    assert not invalid, f"unknown SOC or NAICS codes: {invalid[:3]}"


def test_no_credentials_in_classifications(classifications):
    exposed = [
        (row.get("benchmark"), row.get("task_id"))
        for row in classifications
        if contains_credentials(json.dumps(row))
    ]

    assert not exposed, f"classifications contain possible credentials: {exposed[:3]}"
