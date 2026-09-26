"""
Prepare release metadata and publish the corpus to Hugging Face.
"""

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi
from .io import (
    HF_REPO,
    PROCESSED,
    REPO_ROOT,
    push_to_hf,
)


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    """
    Replace one expected piece of release text.

    :param path: Documentation file to update.
    :param pattern: Regular expression matching the existing text.
    :param replacement: Text containing the current release values.
    """
    text = path.read_text()
    updated, replacements = re.subn(pattern, replacement, text, count=1)
    if replacements != 1:
        raise RuntimeError(f"could not update release counts in {path}")
    path.write_text(updated)


def prepare_release(summary: str) -> None:
    """
    Update corpus counts and add the release to README News.

    :param summary: One-sentence description from the release notes.
    """
    benchmarks = set()
    task_count = 0
    with (PROCESSED / "tasks.jsonl").open() as file:
        for line in file:
            if not line.strip():
                continue

            task = json.loads(line)
            benchmarks.add(task["benchmark"])
            task_count += 1

    agents = set()
    record_count = 0
    with (PROCESSED / "records.jsonl").open() as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            if record.get("agent_id"):
                agents.add(record["agent_id"])
            record_count += 1

    trajectory_benchmarks = set()
    trajectory_count = 0
    with (PROCESSED / "trajectories.jsonl").open() as file:
        for line in file:
            if not line.strip():
                continue

            trajectory = json.loads(line)
            trajectory_benchmarks.add(trajectory["benchmark"])
            trajectory_count += 1

    with (PROCESSED / "verifiers.jsonl").open() as file:
        verifier_count = sum(1 for line in file if line.strip())

    counts = {
        "benchmarks": len(benchmarks),
        "agents": len(agents),
        "tasks": task_count,
        "verifiers": verifier_count,
        "records": record_count,
        "trajectories": trajectory_count,
        "trajectory_benchmarks": len(trajectory_benchmarks),
    }
    number = {key: f"{value:,}" for key, value in counts.items()}

    statistic_labels = {
        "benchmarks": "Benchmarks",
        "agents": "Agents",
        "tasks": "Tasks",
        "verifiers": "Verifiers",
        "records": "Records",
        "trajectories": "Trajectories",
    }
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "huggingface.md"):
        for key, label in statistic_labels.items():
            replace_once(
                path,
                rf"<strong>[\d,]+</strong><br>{label}",
                f"<strong>{number[key]}</strong><br>{label}",
            )
    replace_once(
        REPO_ROOT / "docs" / "release_notes.md",
        r"- \*\*Corpus\.\*\* [\d,]+ records from [\d,]+ benchmarks, [\d,]+ agents, [\d,]+ tasks, and [\d,]+ verifiers\.",
        f"- **Corpus.** {number['records']} records from {number['benchmarks']} benchmarks, {number['agents']} agents, {number['tasks']} tasks, and {number['verifiers']} verifiers.",
    )
    replace_once(
        REPO_ROOT / "docs" / "release_notes.md",
        r"- \*\*Data\.\*\* The release contains task definitions, results, verifier definitions, and [\d,]+ available trajectories\.",
        f"- **Data.** The release contains task definitions, results, verifier definitions, and {number['trajectories']} available trajectories.",
    )

    page = REPO_ROOT / "page" / "index.html"
    replace_once(page, r"The current release spans [\d,]+ benchmarks", f"The current release spans {number['benchmarks']} benchmarks")
    replace_once(page, r"Its [\d,]+ agents", f"Its {number['agents']} agents")
    replace_once(page, r"Across [\d,]+ tasks", f"Across {number['tasks']} tasks")
    replace_once(page, r"describes [\d,]+ verifiers", f"describes {number['verifiers']} verifiers")
    replace_once(
        page,
        r"includes [\d,]+ trajectories from [\d,]+ benchmarks",
        f"includes {number['trajectories']} trajectories from {number['trajectory_benchmarks']} benchmarks",
    )

    moons = ["🌕", "🌖", "🌗", "🌘", "🌑", "🌒", "🌓", "🌔"]
    heading = "## News\n"
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "huggingface.md"):
        text = path.read_text()
        if summary in text:
            continue

        news = text.split(heading, 1)[1]
        current = next((moon for line in news.splitlines() if line.startswith("- ") for moon in moons if moon in line), None)
        marker = moons[(moons.index(current) + 1) % len(moons)] if current else moons[0]
        entry = f"- **[{date.today():%d/%m/%Y}]** {marker} {summary}\n"
        path.write_text(text.replace(heading, f"{heading}{entry}", 1))


def check_huggingface() -> None:
    """
    Confirm access to the Hugging Face release repository.
    """
    huggingface_api().repo_info(repo_id=HF_REPO, repo_type="dataset")


def huggingface_api() -> HfApi:
    """
    Create an authenticated Hugging Face client.

    :return: Client with write access to the release repository.
    """
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    return HfApi(token=token)


def create_candidate(tag: str, notes: Path) -> None:
    """
    Upload the release files to a reviewable Hugging Face pull request.

    :param tag: Release tag such as ``v1.0.0``.
    :param notes: Markdown release notes attached to the candidate.
    """
    api = huggingface_api()
    references = api.list_repo_refs(HF_REPO, repo_type="dataset")
    if any(item.name == tag for item in references.tags):
        raise RuntimeError(f"{tag} has already been released on Hugging Face")

    title = f"MESSIER {tag}"
    pull_requests = [
        discussion
        for discussion in api.get_repo_discussions(
            repo_id=HF_REPO,
            repo_type="dataset",
            discussion_type="pull_request",
            discussion_status="open",
        )
        if discussion.title == title
    ]
    if len(pull_requests) > 1:
        raise RuntimeError(f"multiple open Hugging Face candidates exist for {tag}")

    revision = f"refs/pr/{pull_requests[0].num}" if pull_requests else None
    commit = push_to_hf(
        commit_message=f"MESSIER {tag}",
        commit_description=notes.read_text(),
        revision=revision,
        create_pr=revision is None,
    )
    candidate_url = commit.pr_url
    if not candidate_url and pull_requests:
        candidate_url = pull_requests[0].url
    if not candidate_url:
        raise RuntimeError(f"Hugging Face did not return a candidate URL for {tag}")
    print(candidate_url)


def promote_candidate(tag: str) -> None:
    """
    Merge the reviewed Hugging Face candidate and create its release tag.

    :param tag: Release tag such as ``v1.0.0``.
    """
    api = huggingface_api()
    references = api.list_repo_refs(HF_REPO, repo_type="dataset")
    existing_tag = next((item for item in references.tags if item.name == tag), None)
    if existing_tag:
        print(existing_tag.target_commit)
        return

    title = f"MESSIER {tag}"
    pull_requests = [
        discussion
        for discussion in api.get_repo_discussions(repo_id=HF_REPO, repo_type="dataset")
        if discussion.is_pull_request and discussion.title == title
    ]
    if len(pull_requests) != 1:
        raise RuntimeError(f"expected one Hugging Face candidate for {tag}, found {len(pull_requests)}")

    candidate = pull_requests[0]
    if candidate.status == "open":
        api.merge_pull_request(
            repo_id=HF_REPO,
            repo_type="dataset",
            discussion_num=candidate.num,
            comment=f"Published after the MESSIER {tag} release PR was merged.",
        )
    elif candidate.status != "merged":
        raise RuntimeError(f"Hugging Face candidate for {tag} is {candidate.status}")

    details = api.get_discussion_details(
        repo_id=HF_REPO,
        repo_type="dataset",
        discussion_num=candidate.num,
    )
    if not details.merge_commit_oid:
        raise RuntimeError(f"Hugging Face candidate for {tag} has no merge commit")

    api.create_tag(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=details.merge_commit_oid,
        tag=tag,
        tag_message=f"MESSIER {tag}",
        exist_ok=True,
    )
    print(details.merge_commit_oid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and publish a MESSIER release.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check Hugging Face access")

    prepare = commands.add_parser("prepare", help="update counts and README News")
    prepare.add_argument("summary", help="one-sentence release summary")

    candidate = commands.add_parser("candidate", help="upload a Hugging Face release candidate")
    candidate.add_argument("tag", help="release tag such as v1.0.0")
    candidate.add_argument("notes", type=Path, help="release-notes Markdown file")

    promote = commands.add_parser("promote", help="publish a reviewed Hugging Face candidate")
    promote.add_argument("tag", help="release tag such as v1.0.0")
    arguments = parser.parse_args()

    if arguments.command == "check":
        check_huggingface()
    elif arguments.command == "prepare":
        prepare_release(arguments.summary)
    elif arguments.command == "candidate":
        create_candidate(arguments.tag, arguments.notes)
    else:
        promote_candidate(arguments.tag)


if __name__ == "__main__":
    main()
