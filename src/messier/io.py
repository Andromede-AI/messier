"""
Local paths, dataset loaders, and Hugging Face transfers.
"""

import json
import os
import re
from pathlib import Path
from typing import Iterator
from huggingface_hub import (
    CommitInfo,
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

# local build data stays under the ignored data directory
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
ENRICH = RAW / "enrichment"
PROCESSED = DATA / "processed"
ARTIFACTS = DATA / "artifacts"
CACHE = DATA / "cache"
RAW_MANIFEST_PATH = DATA / "raw_manifest.jsonl"
CLASSIFICATIONS_PATH = ARTIFACTS / "classifications.jsonl"
TASKS_PATH = PROCESSED / "tasks.jsonl"
VERIFIERS_PATH = PROCESSED / "verifiers.jsonl"
RECORDS_PATH = PROCESSED / "records.jsonl"
TRAJECTORIES_PATH = PROCESSED / "trajectories.jsonl"
TASK_FILES_PATH = PROCESSED / "task_files"
HF_REPO = "Andromede-AI/messier"
HF_DATASET_CARD = REPO_ROOT / "docs" / "huggingface.md"
CREDENTIAL_PREFIXES = ("hf_", "sk-", "ghp_", "github_pat_", "AKIA", "AIza", "xox")
CREDENTIAL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"hf_[A-Za-z0-9]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|"
    r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{30,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}"
    r")"
)


def contains_credentials(text: str) -> bool:
    return any(prefix in text for prefix in CREDENTIAL_PREFIXES) and CREDENTIAL_PATTERN.search(text) is not None


def redact_credentials(text: str) -> str:
    return CREDENTIAL_PATTERN.sub("[REDACTED]", text) if contains_credentials(text) else text

# local and Hugging Face paths for each release file
HF_FILES = {
    "README.md": (HF_DATASET_CARD, "README.md"),
    "THIRD_PARTY_NOTICES.md": (REPO_ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    "docs/assets/logo.png": (REPO_ROOT / "docs/assets/logo.png", "docs/assets/logo.png"),
    "classifications.jsonl": (CLASSIFICATIONS_PATH, "classifications.jsonl"),
    "tasks.jsonl": (TASKS_PATH, "tasks.jsonl"),
    "verifiers.jsonl": (VERIFIERS_PATH, "verifiers.jsonl"),
    "records.jsonl": (RECORDS_PATH, "records.jsonl"),
    "trajectories.jsonl": (TRAJECTORIES_PATH, "trajectories.jsonl"),
}


def load_jsonl(path: Path) -> Iterator[dict]:
    """
    Read non-empty rows from a JSON Lines file.

    :param path: JSON Lines file to read.
    :return: An iterator over decoded rows.
    """
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def data_path(filename: str, source: str = "hf") -> Path:
    """
    Resolve a dataset file to a local path.

    :param filename: File name such as ``tasks.jsonl``.
    :param source: ``local`` for the current build or ``hf`` for the Hugging Face release.
    :return: Path to the requested file.
    """
    if filename not in HF_FILES:
        raise ValueError(f"unknown data file: {filename!r}")

    local_path, huggingface_path = HF_FILES[filename]
    if source == "local":
        return local_path
    if source == "hf":
        CACHE.mkdir(parents=True, exist_ok=True)

        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=huggingface_path,
            repo_type="dataset",
            cache_dir=str(CACHE),
            token=os.environ.get("HF_TOKEN"),
        )
        return Path(downloaded)

    raise ValueError(f"source must be 'hf' or 'local', got {source!r}")


def push_to_hf(
    commit_message: str = "Update MESSIER release",
    commit_description: str | None = None,
    revision: str | None = None,
    create_pr: bool = False,
) -> CommitInfo:
    """
    Upload release files to a Hugging Face branch or pull request.

    :param commit_message: Title of the Hugging Face commit.
    :param commit_description: Release notes attached to the commit.
    :param revision: Existing branch or pull-request revision to update.
    :param create_pr: Create a pull request instead of updating the dataset directly.
    :return: Information about the uploaded commit and pull request.
    """
    missing = [
        str(local_path)
        for local_path, _ in HF_FILES.values()
        if not local_path.exists()
    ]
    if not TASK_FILES_PATH.is_dir():
        missing.append(str(TASK_FILES_PATH))
    if missing:
        raise FileNotFoundError(f"release files are missing: {missing}")

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    api = HfApi(token=token)
    repository = api.repo_info(repo_id=HF_REPO, repo_type="dataset", revision=revision)

    local_files = {
        huggingface_path: local_path
        for local_path, huggingface_path in HF_FILES.values()
    }
    local_files.update(
        {
            path.relative_to(PROCESSED).as_posix(): path
            for path in TASK_FILES_PATH.rglob("*")
            if path.is_file()
        }
    )
    repository_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset", revision=revision)
    stale_task_files = [
        path
        for path in repository_files
        if path.startswith("task_files/") and path not in local_files
    ]
    operations = [
        CommitOperationDelete(path_in_repo=path)
        for path in stale_task_files
    ]
    operations.extend(
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=local_path)
        for path, local_path in sorted(local_files.items())
    )
    commit = api.create_commit(
        repo_id=HF_REPO,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message,
        commit_description=commit_description,
        revision=revision,
        create_pr=create_pr,
        parent_commit=repository.sha,
    )
    return commit
