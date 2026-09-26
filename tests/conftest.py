import json
import os
from pathlib import Path
import pytest
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from messier.io import HF_REPO, PROCESSED, data_path
from tests import _checks


def pytest_addoption(parser):
    parser.addoption(
        "--source",
        choices=("local", "hf"),
        default="local",
        help="Read the local build or the Hugging Face release.",
    )
    parser.addoption(
        "--revision",
        default=None,
        help="Exact Hugging Face revision to audit.",
    )


def _resolve(filename: str, source: str, revision: str | None) -> Path:
    if source == "local":
        path = data_path(filename, source="local")
        if not path.exists():
            pytest.fail(f"full-corpus audit requires {path}")

        return path

    load_dotenv()
    return Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
        )
    )


@pytest.fixture(scope="session")
def source(request) -> str:
    return request.config.getoption("--source")


@pytest.fixture(scope="session")
def revision(request) -> str | None:
    return request.config.getoption("--revision")


@pytest.fixture(scope="session")
def available_files(source, revision) -> set[str]:
    if source == "local":
        return {
            path.relative_to(PROCESSED).as_posix()
            for path in PROCESSED.rglob("*")
            if path.is_file()
        }

    load_dotenv()
    return set(
        HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=revision,
        )
    )


@pytest.fixture(scope="session")
def tasks(source, revision) -> list[dict]:
    with _resolve("tasks.jsonl", source, revision).open() as file:
        return [json.loads(line) for line in file if line.strip()]


@pytest.fixture(scope="session")
def verifiers(source, revision) -> list[dict]:
    with _resolve("verifiers.jsonl", source, revision).open() as file:
        return [json.loads(line) for line in file if line.strip()]


@pytest.fixture(scope="session")
def task_index(tasks) -> dict[tuple[str, str], dict]:
    return {(t["benchmark"], t["task_id"]): t for t in tasks if t.get("task_id")}


@pytest.fixture(scope="session")
def record_audit(source, revision, task_index, verifiers) -> dict:
    verifier_keys = {
        (verifier["benchmark"], verifier["task_id"], verifier["verifier_id"])
        for verifier in verifiers
    }

    audit = _checks.RecordAudit(task_index, verifier_keys)
    with open(_resolve("records.jsonl", source, revision)) as f:
        for line in f:
            if line.strip():
                audit.visit(json.loads(line))

    return audit.finalize()


@pytest.fixture(scope="session")
def classifications(source, revision) -> list[dict]:
    with _resolve("classifications.jsonl", source, revision).open() as file:
        return [json.loads(line) for line in file if line.strip()]


@pytest.fixture(scope="session")
def trajectories_path(source, revision) -> Path:
    return _resolve("trajectories.jsonl", source, revision)
