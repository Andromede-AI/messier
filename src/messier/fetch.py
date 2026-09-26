"""Fetch pinned source data and verify the local raw snapshot."""

import ast
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
import boto3
import remotezip
import requests
from botocore import UNSIGNED
from botocore.config import Config
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm
from .io import (
    CACHE,
    RAW,
    RAW_MANIFEST_PATH,
    REPO_ROOT,
    load_jsonl,
    redact_credentials,
)
from .builders.osworld.constants import (
    DOMAINS,
    GITHUB_RAW_BASE,
    HF_FILE_CACHE_REPO,
    HF_TRAJS_BASE,
    HF_TRAJS_REPO,
)

BRIDGE = "https://raw.githubusercontent.com/McGill-NLP/BRIDGE/main"
METR = "https://raw.githubusercontent.com/METR/eval-analysis-public/main"

BRIDGE_FILES = [
    "all_runs.jsonl",
    "swebench_normalized_results.jsonl",
    "mlebench_normalized_results.jsonl",
    "cybench_normalized_results.jsonl",
    "gdpval_normalized_results.jsonl",
    "human_minutes_by_task.jsonl",
    "combined_human_minutes.jsonl",
    "cybench_human_minutes_by_task.jsonl",
    "cybench_fst_master.jsonl",
    "all_a_pyirt.jsonl",
    "swe_a_pyirt.jsonl",
    "model_run_mapping.json",
    "cybench_time_estimations_google_gemini-3-pro-preview.jsonl",
    "cybench_time_estimations_openai_gpt-5.2-2025-12-11.jsonl",
    "swebench_time_estimations_gemini-3-pro-preview-swe-bench.jsonl",
    "swebench_time_estimations_gpt-5_2-2025-12-11-swe-bench.jsonl",
]

MATHARENA_NAMES = [
    "aime_2025_outputs",
    "aime_2026_outputs",
    "hmmt_feb_2025_outputs",
    "hmmt_feb_2026_outputs",
    "hmmt_nov_2025_outputs",
    "brumo_2025_outputs",
    "smt_2025_outputs",
    "cmimc_2025_outputs",
    "usamo_2025_outputs",
    "usamo_2026_outputs",
    "imo_2025_outputs",
    "imc_2025_outputs",
]

AP_DIRS = [
    "swebench_verified",
    "swebench_pro",
    "terminalbench",
    "gso",
    "all_benchmarks",
]
AP_EXTRAS = ["difficulty_score_histograms.png", "terminalbench_scatterplot.pdf"]


def fetch_bridge() -> None:
    for filename in BRIDGE_FILES:
        _get(
            f"{BRIDGE}/data/{filename}",
            RAW / "bridge" / filename,
        )


def fetch_agent_psychometrics() -> None:
    agent_psychometrics_dir = RAW / "agent_psychometrics"
    agent_psychometrics_dir.mkdir(parents=True, exist_ok=True)
    if all(_populated(agent_psychometrics_dir / directory) for directory in AP_DIRS):
        return

    with tempfile.TemporaryDirectory() as tmp:
        ap_tmp = Path(tmp) / "ap"
        _git(
            "clone",
            "--depth",
            "1",
            "https://github.com/dariakryvosheieva/agent-psychometrics.git",
            str(ap_tmp),
        )

        for directory in AP_DIRS:
            source = ap_tmp / "data" / directory
            if not source.exists():
                continue

            destination = agent_psychometrics_dir / directory
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination)

        for extra in AP_EXTRAS:
            source = ap_tmp / "data" / extra
            if source.exists():
                shutil.copy(source, agent_psychometrics_dir / extra)


def fetch_gso_trajectories() -> None:
    responses = RAW / "agent_psychometrics" / "gso" / "responses.jsonl"
    destination = RAW / "gso" / "trajectories"
    base = "https://storage.googleapis.com/gso-experiments"

    for row in load_jsonl(responses):
        model = row["subject_id"]
        _get(
            f"{base}/{model}/trajs/output.jsonl",
            destination / f"{model}.jsonl",
        )


def fetch_metr() -> None:
    _get(
        f"{METR}/data/external/release_dates.yaml", RAW / "metr" / "release_dates.yaml"
    )
    _get(
        f"{METR}/reports/time-horizon-1-1/data/raw/runs.jsonl",
        RAW / "metr" / "runs.jsonl",
    )


def fetch_epoch() -> None:
    _get(
        "https://epoch.ai/data/notable_ai_models.csv",
        RAW / "epoch" / "notable_ai_models.csv",
    )


def fetch_enrichment() -> None:
    from .builders.benchmarks import UPSTREAM_COMMITS

    _clone("METR/hcast-public", RAW / "enrichment" / "hcast")
    _clone("METR/RE-Bench", RAW / "enrichment" / "rebench")
    _clone("openai/mle-bench", RAW / "enrichment" / "mlebench")

    replicationbench_root = RAW / "enrichment" / "replicationbench"
    if not _populated(replicationbench_root):
        _sparse_clone(
            "Christine8888/replicationbench-release",
            replicationbench_root,
            [
                "src/dataset/papers/*.json",
                "src/dataset/tasks/*/*.json",
            ],
            revision=UPSTREAM_COMMITS["replicationbench"],
        )

    revision_path = replicationbench_root / "dataset_revisions.json"
    if not revision_path.exists():
        dataset_repositories = set()
        for paper_path in (replicationbench_root / "src" / "dataset" / "papers").glob("*.json"):
            paper = json.loads(paper_path.read_text())
            datasets = paper.get("dataset") or []
            if isinstance(datasets, dict):
                datasets = [datasets]
            for dataset in datasets:
                dataset_repositories.update(dataset.get("hf_name") or [])

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        revisions = {
            repository: api.dataset_info(repository).sha
            for repository in tqdm(
                sorted(dataset_repositories),
                desc="pinning ReplicationBench datasets",
                unit="dataset",
            )
        }
        revision_path.write_text(json.dumps(revisions, indent=2) + "\n")

    # we retain only the MLE-bench competition definitions
    mlebench_root = RAW / "enrichment" / "mlebench"
    for directory in [
        "runs",
        "experiments",
        "extras",
        "examples",
        "tests",
        "agents",
        "environment",
    ]:
        shutil.rmtree(mlebench_root / directory, ignore_errors=True)

    # we fetch CyBench task descriptions without challenge binaries
    cybench_root = RAW / "enrichment" / "cybench"
    if _populated(cybench_root):
        return

    shutil.rmtree(cybench_root, ignore_errors=True)
    cybench_root.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        bridge = Path(tmp) / "bridge"
        _git(
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/McGill-NLP/BRIDGE.git",
            str(bridge),
        )

        _git("sparse-checkout", "set", "data/cybench/benchmark", cwd=bridge)
        source = bridge / "data" / "cybench" / "benchmark"
        for item in source.iterdir():
            destination = cybench_root / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy(item, destination)

    for path in list(cybench_root.rglob("*")):
        if not path.is_file():
            continue

        filename = path.name
        keep = (
            filename.startswith("README")
            or filename == "metadata.json"
            or path.suffix == ".md"
            or filename.startswith("task_metadata")
        )

        if not keep:
            path.unlink()

    directories = sorted(
        [path for path in cybench_root.rglob("*") if path.is_dir()],
        key=lambda path: -len(path.parts),
    )
    for directory in directories:
        if not any(directory.iterdir()):
            directory.rmdir()


def fetch_terminalbench() -> None:
    terminalbench_root = RAW / "terminal-bench"
    target = terminalbench_root / "original-tasks"
    if _populated(target):
        return

    _sparse_clone(
        "laude-institute/terminal-bench",
        terminalbench_root,
        ["original-tasks/*/task.yaml", "original-tasks/*/Dockerfile"],
    )


def fetch_theagentcompany() -> None:
    _clone("TheAgentCompany/TheAgentCompany", RAW / "theagentcompany" / "main")
    experiments_root = RAW / "theagentcompany" / "experiments"
    has_trajectories = any(experiments_root.rglob("trajectories"))
    if not has_trajectories:
        _sparse_clone(
            "TheAgentCompany/experiments",
            experiments_root,
            [
                "README.md",
                "utils/*",
                "evaluation/*/*/README.md",
                "evaluation/*/*/results/*",
                "evaluation/*/*/trajectories/*",
            ],
        )


def fetch_onlinemind2web() -> None:
    _get_hf(
        "osunlp/Online_Mind2Web_Leaderboard",
        "human_label_111825.json",
        RAW / "onlinemind2web" / "human_label_111825.json",
        repo_type="space",
    )


def fetch_bfcl() -> None:
    bfcl = RAW / "bfcl"
    target = bfcl / "2025-12-16"
    if _populated(target):
        return

    # we use non-cone sparse checkout for dated source directories
    _sparse_clone("HuanzhiMao/BFCL-Result", bfcl, ["2025-12-16/*"])
    # we discard model outputs and unused categories
    shutil.rmtree(target / "result", ignore_errors=True)
    scores_root = target / "score"
    if scores_root.exists():
        for model_directory in scores_root.iterdir():
            if model_directory.is_dir():
                shutil.rmtree(model_directory / "non_live", ignore_errors=True)
                shutil.rmtree(model_directory / "agentic", ignore_errors=True)


def fetch_matharena() -> None:
    from .builders.matharena.constants import MATHARENA_COMMITS

    matharena_root = RAW / "matharena"
    matharena_root.mkdir(parents=True, exist_ok=True)
    for slug in MATHARENA_NAMES:
        revision = MATHARENA_COMMITS[slug.removesuffix("_outputs")]
        _get_hf(
            f"MathArena/{slug}",
            "data/train-00000-of-00001.parquet",
            matharena_root / f"{slug}.parquet",
            revision=revision,
        )

    # we fetch all three APEX shortlist shards
    revision = MATHARENA_COMMITS["apex_shortlist"]
    for shard in ("00000", "00001", "00002"):
        _get_hf(
            "MathArena/apex_shortlist_outputs",
            f"data/train-{shard}-of-00003.parquet",
            matharena_root / f"apex_shortlist_outputs-{shard[-2:]}.parquet",
            revision=revision,
        )


def fetch_livecodebench() -> None:
    lcb = RAW / "livecodebench"
    _get(
        "https://livecodebench.github.io/performances_generation.json",
        lcb / "performances_generation.json",
    )

    # we fetch ratings used to estimate human completion time
    _get(
        "https://raw.githubusercontent.com/zerotrac/leetcode_problem_rating/main/data.json",
        lcb / "leetcode_ratings.json",
    )
    _get(
        "https://codeforces.com/api/problemset.problems",
        lcb / "codeforces_problemset.json",
    )


def fetch_hle() -> None:
    _get(
        "https://raw.githubusercontent.com/supaihq/hle/main/judged_hle_pro.json",
        RAW / "hle" / "judged_hle_pro.json",
    )


def fetch_dabstep() -> None:
    import asyncio
    from harbor.models.task.id import PackageTaskId
    from harbor.tasks.client import TaskClient
    from .builders.benchmarks import UPSTREAM_COMMITS
    from .builders.harbor_runs.constants import DABSTEP_CONTEXT_FILES

    for filename in DABSTEP_CONTEXT_FILES:
        hf_hub_download(
            repo_id="adyen/DABstep",
            filename=f"data/context/{filename}",
            repo_type="dataset",
            revision=UPSTREAM_COMMITS["dabstep"],
            local_dir=str(RAW / "dabstep"),
            token=os.environ.get("HF_TOKEN"),
        )

    run_root = RAW / "harbor_runs" / "graded" / "adyen" / "dabstep"
    package_root = RAW / "dabstep" / "tasks"
    task_refs = {}
    for config_path in run_root.rglob("config.json"):
        task = json.loads(config_path.read_text()).get("task")
        if not task:
            continue

        name = task.get("name")
        ref = task.get("ref")
        if not name or not ref:
            continue
        if name in task_refs and task_refs[name] != ref:
            raise ValueError(f"DABStep task {name!r} has multiple source references")

        task_refs[name] = ref

    if not task_refs:
        raise ValueError("no DABStep task references found in the Harbor runs")

    task_ids = [
        PackageTaskId(org="adyen", name=name.removeprefix("adyen/"), ref=ref)
        for name, ref in sorted(task_refs.items())
    ]
    progress = tqdm(total=len(task_ids), desc="fetching DABStep", unit="task")
    try:
        asyncio.run(
            TaskClient().download_tasks(
                task_ids=task_ids,
                output_dir=package_root,
                on_task_download_complete=lambda _task, _result: progress.update(),
            )
        )
    finally:
        progress.close()


def fetch_hf_tasks() -> None:
    """
    Freeze the Hugging Face task tables used by builders.
    """
    from .builders.benchmarks import UPSTREAM_COMMITS
    from .builders.onlinemind2web.constants import FALLBACK_REVISION, PRIMARY_REVISION

    _fetch_hf_table(
        repo_id="gso-bench/gso",
        revision=UPSTREAM_COMMITS["gso"],
        split="test",
        dst=RAW / "gso" / "tasks.jsonl",
        columns=[
            "instance_id",
            "repo",
            "base_commit",
            "opt_commit",
            "api",
            "prob_script",
            "tests",
            "hints_text",
            "setup_commands",
            "install_commands",
            "created_at",
            "gt_commit_message",
            "gt_diff",
            "arch",
            "instance_image_tag",
        ],
    )
    _fetch_hf_table(
        repo_id="princeton-nlp/SWE-bench_Verified",
        revision=UPSTREAM_COMMITS["swebench_verified"],
        split="test",
        dst=RAW / "swebench_verified" / "tasks.jsonl",
        columns=[
            "instance_id",
            "repo",
            "base_commit",
            "environment_setup_commit",
            "version",
            "problem_statement",
            "hints_text",
            "test_patch",
            "patch",
            "created_at",
            "difficulty",
        ],
    )
    _fetch_hf_table(
        repo_id="ScaleAI/SWE-bench_Pro",
        revision=UPSTREAM_COMMITS["swebench_pro"],
        split="test",
        dst=RAW / "swebench_pro" / "tasks.jsonl",
        columns=[
            "instance_id",
            "repo",
            "repo_language",
            "base_commit",
            "dockerhub_tag",
            "before_repo_set_cmd",
            "problem_statement",
            "requirements",
            "interface",
            "issue_specificity",
            "issue_categories",
            "selected_test_files_to_run",
            "test_patch",
            "patch",
        ],
    )
    _fetch_hf_table(
        repo_id="openai/gdpval",
        revision=UPSTREAM_COMMITS["gdpval"],
        split="train",
        dst=RAW / "gdpval" / "tasks.jsonl",
        columns=[
            "task_id",
            "sector",
            "occupation",
            "prompt",
            "reference_files",
            "reference_file_urls",
            "reference_file_hf_uris",
            "deliverable_files",
            "deliverable_file_urls",
            "deliverable_file_hf_uris",
            "rubric_pretty",
            "rubric_json",
        ],
    )
    gdpval_tasks = list(load_jsonl(RAW / "gdpval" / "tasks.jsonl"))
    reference_files = sorted(
        {
            filename
            for task in gdpval_tasks
            for filename in task.get("reference_files") or []
        }
    )
    for filename in tqdm(reference_files, desc="fetching GDPval files", unit="file"):
        hf_hub_download(
            repo_id="openai/gdpval",
            filename=filename,
            repo_type="dataset",
            revision=UPSTREAM_COMMITS["gdpval"],
            local_dir=str(RAW / "gdpval"),
            token=os.environ.get("HF_TOKEN"),
        )
    livecodebench_path = RAW / "livecodebench" / "tasks.jsonl"
    if not livecodebench_path.exists():
        repo_id = "livecodebench/code_generation_lite"
        revision = UPSTREAM_COMMITS["livecodebench"]
        columns = [
            "question_id",
            "platform",
            "public_test_cases",
            "question_content",
            "question_title",
            "starter_code",
            "contest_id",
            "contest_date",
            "difficulty",
        ]
        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            repo_id,
            repo_type="dataset",
            revision=revision,
        )
        slices = sorted(
            path
            for path in files
            if path.startswith("test") and path.endswith(".jsonl")
        )
        livecodebench_path.parent.mkdir(parents=True, exist_ok=True)
        with livecodebench_path.open("w") as output:
            for filename in slices:
                path = hf_hub_download(
                    repo_id,
                    filename,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=str(CACHE),
                    token=os.environ.get("HF_TOKEN"),
                )
                for row in load_jsonl(Path(path)):
                    output.write(
                        json.dumps({column: row.get(column) for column in columns})
                        + "\n"
                    )

    columns = ["task_id", "confirmed_task", "website", "reference_length", "level"]
    _fetch_hf_table(
        repo_id="osunlp/Online-Mind2Web",
        revision=PRIMARY_REVISION,
        split="test",
        dst=RAW / "onlinemind2web" / "tasks_primary.jsonl",
        columns=columns,
    )
    _fetch_hf_table(
        repo_id="osunlp/Online-Mind2Web",
        revision=FALLBACK_REVISION,
        split="test",
        dst=RAW / "onlinemind2web" / "tasks_fallback.jsonl",
        columns=columns,
    )


def fetch_tau2_bench() -> None:
    from .builders.tau2_bench.constants import (
        SIERRA_S3_BASE,
        SIERRA_S3_BUCKET,
        SIERRA_S3_REGION,
        SUBMISSIONS,
    )

    root = RAW / "tau2_bench"
    bucket_base = SIERRA_S3_BASE.replace("/submissions", "")
    client = boto3.client(
        "s3", region_name=SIERRA_S3_REGION, config=Config(signature_version=UNSIGNED)
    )
    paginator = client.get_paginator("list_objects_v2")
    for submission_dir in SUBMISSIONS:
        _get(
            f"{SIERRA_S3_BASE}/{submission_dir}/submission.json",
            root / "submissions" / submission_dir / "submission.json",
        )
        for page in paginator.paginate(
            Bucket=SIERRA_S3_BUCKET,
            Prefix=f"submissions/{submission_dir}/trajectories/",
        ):
            for entry in page.get("Contents", []):
                key = entry["Key"]
                if key.endswith(".json"):
                    _get(
                        f"{bucket_base}/{urllib.parse.quote(key, safe='/')}",
                        root / "trajectories" / submission_dir / key.rsplit("/", 1)[-1],
                    )


def fetch_osworld() -> None:
    """
    Download OSWorld tasks and results without downloading complete trace archives.
    """
    from .builders.benchmarks import UPSTREAM_COMMITS

    root = RAW / "osworld"
    configs_root = root / "configs"
    archives_root = root / "zips"
    configs_root.mkdir(parents=True, exist_ok=True)
    archives_root.mkdir(parents=True, exist_ok=True)

    task_listing_path = configs_root / "test_nogdrive.json"
    task_base = f"{GITHUB_RAW_BASE}/{UPSTREAM_COMMITS['osworld']}/evaluation_examples"
    _get(
        f"{task_base}/test_nogdrive.json",
        task_listing_path,
    )
    task_ids_by_domain = json.loads(task_listing_path.read_text())
    for domain, task_ids in task_ids_by_domain.items():
        for task_id in task_ids:
            _get(
                f"{task_base}/examples/{domain}/{task_id}.json",
                configs_root / "examples" / domain / f"{task_id}.json",
            )

    input_paths = set()
    for domain, task_ids in task_ids_by_domain.items():
        for task_id in task_ids:
            config_path = configs_root / "examples" / domain / f"{task_id}.json"
            if not config_path.exists():
                continue

            config = json.loads(config_path.read_text())
            for action in config.get("config") or []:
                if action.get("type") != "download":
                    continue
                for file_config in action.get("parameters", {}).get("files", []):
                    source_path = urllib.parse.unquote(
                        file_config["url"].split("/resolve/main/", 1)[1]
                    )
                    input_paths.add(source_path)

    for source_path in tqdm(sorted(input_paths), desc="fetching OSWorld files", unit="file"):
        hf_hub_download(
            repo_id=HF_FILE_CACHE_REPO,
            filename=source_path,
            repo_type="dataset",
            revision=UPSTREAM_COMMITS["osworld_file_cache"],
            local_dir=str(root / "files"),
            token=os.environ.get("HF_TOKEN"),
        )

    session = requests.Session()
    token = os.environ.get("HF_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    listing_response = session.get(
        f"https://huggingface.co/api/datasets/{HF_TRAJS_REPO}"
    )
    archive_names = sorted(
        [
            sibling["rfilename"]
            for sibling in listing_response.json()["siblings"]
            if sibling["rfilename"].endswith(".zip")
        ]
    )

    result_path_pattern = re.compile(
        r"(?:^|/)(" + "|".join(DOMAINS) + r")/"
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/result\.txt$"
    )

    max_gap_bytes = 5_000_000
    max_batch_bytes = 20_000_000

    for archive_name in archive_names:
        stem = archive_name[:-4]
        aggregate_path = archives_root / f"{stem}_aggregated.json"
        arguments_path = archives_root / f"{stem}_args.json"
        if aggregate_path.exists() and arguments_path.exists():
            continue

        # save the OSWorld run settings and final scores from each zip file
        # if no score summary exists, read its task scores without downloading the whole file
        for attempt in range(3):
            try:
                archive_url = f"{HF_TRAJS_BASE}/{archive_name}"
                with remotezip.RemoteZip(archive_url, session=session) as archive:
                    entries = archive.infolist()
                    if not arguments_path.exists():
                        arguments_entry = next(
                            (
                                entry
                                for entry in entries
                                if entry.filename.endswith("/args.json")
                                or entry.filename == "args.json"
                            ),
                            None,
                        )
                        if arguments_entry:
                            arguments_path.write_bytes(archive.read(arguments_entry.filename))

                    if not aggregate_path.exists():
                        aggregate_entries = sorted(
                            [
                                entry
                                for entry in entries
                                if entry.filename.endswith("/all_result.json")
                                or entry.filename == "all_result.json"
                            ],
                            key=lambda entry: len(entry.filename),
                        )
                        if aggregate_entries:
                            aggregate_path.write_text(
                                json.dumps(
                                    ast.literal_eval(
                                        archive.read(
                                            aggregate_entries[0].filename
                                        ).decode()
                                    )
                                )
                            )
                        else:
                            results = []
                            for entry in entries:
                                if not entry.filename.endswith("/result.txt"):
                                    continue

                                match = result_path_pattern.search(entry.filename)
                                if match:
                                    results.append((entry, match.group(1), match.group(2)))

                            if results:
                                results.sort(key=lambda result: result[0].header_offset)
                                batches = [[results[0]]]
                                for entry in results[1:]:
                                    previous_entry = batches[-1][-1][0]
                                    previous_end = (
                                        previous_entry.header_offset
                                        + 30
                                        + len(previous_entry.filename)
                                        + len(previous_entry.extra)
                                        + previous_entry.compress_size
                                    )
                                    batch_start = batches[-1][0][0].header_offset
                                    entry_end = (
                                        entry[0].header_offset
                                        + 30
                                        + len(entry[0].filename)
                                        + len(entry[0].extra)
                                        + entry[0].compress_size
                                    )
                                    if (
                                        entry[0].header_offset - previous_end
                                        > max_gap_bytes
                                        or entry_end - batch_start > max_batch_bytes
                                    ):
                                        batches.append([entry])
                                    else:
                                        batches[-1].append(entry)

                                aggregate = {domain: {} for domain in DOMAINS}
                                for batch in batches:
                                    range_start = batch[0][0].header_offset
                                    final_entry = batch[-1][0]
                                    range_end = (
                                        final_entry.header_offset
                                        + 30
                                        + len(final_entry.filename)
                                        + len(final_entry.extra)
                                        + final_entry.compress_size
                                        + 100
                                    )
                                    buffer = session.get(
                                        archive_url,
                                        headers={"Range": f"bytes={range_start}-{range_end - 1}"},
                                    ).content
                                    for entry, domain, task_id in batch:
                                        offset = entry.header_offset - range_start
                                        if (
                                            struct.unpack(
                                                "<I",
                                                buffer[offset : offset + 4],
                                            )[0]
                                            != 0x04034B50
                                        ):
                                            continue

                                        filename_length, extra_length = struct.unpack(
                                            "<HH",
                                            buffer[offset + 26 : offset + 30],
                                        )
                                        data_start = (
                                            offset
                                            + 30
                                            + filename_length
                                            + extra_length
                                        )
                                        raw = buffer[data_start : data_start + entry.compress_size]
                                        if entry.compress_type != 0:
                                            raw = zlib.decompress(raw, -15)

                                        try:
                                            aggregate[domain][task_id] = float(raw.decode().strip())
                                        except Exception:
                                            pass

                                aggregate = {
                                    domain: results
                                    for domain, results in aggregate.items()
                                    if results
                                }
                                if aggregate:
                                    aggregate_path.write_text(json.dumps(aggregate))
                break
            except Exception as error:
                if "429" in str(error):
                    time.sleep(60 * (attempt + 1))
                    continue
                break


def fetch_yoonholee_terminalbench() -> None:
    """
    Download the TerminalBench result tables used by the builder.
    """
    from .builders.terminalbench.build import YH_FILES, YH_REPO

    root = RAW / "yoonholee_terminalbench"
    for fname in YH_FILES:
        _get_hf(
            YH_REPO,
            fname,
            root / fname.rsplit("/", 1)[-1],
        )


def fetch_swebench_verified_traj() -> None:
    """
    Download the SWE-bench Verified result files used by the builder.
    """
    from .builders.swebench_verified.constants import LIVESWEAGENT_TRAJ_SOURCES

    root = RAW / "swebench_verified_traj"
    for local_dir, repo, _, _, result_filename in LIVESWEAGENT_TRAJ_SOURCES:
        _get_hf(repo, result_filename, root / local_dir / result_filename)
        _get_hf(repo, "preds.json", root / local_dir / "preds.json")


def fetch_devin_swebench() -> None:
    """
    Download the Devin SWE-bench result file.
    """
    _get_hf(
        "OpenHandsCommunity/Devin-SWE-bench-output",
        "devin_swe_outputs.json",
        RAW / "devin_swebench" / "devin_swe_outputs.json",
    )


def fetch_coderforge_swebench_verified() -> None:
    """
    Download the CoderForge SWE-bench Verified result tables.
    """
    from .builders.swebench_verified.constants import CODERFORGE_FILES, CODERFORGE_REPO

    root = RAW / "coderforge_swebench_verified"
    for fname in CODERFORGE_FILES:
        _get_hf(
            CODERFORGE_REPO,
            fname,
            root / fname.rsplit("/", 1)[-1],
        )


def fetch_swebench_pro_traj() -> None:
    """
    Download the additional SWE-bench Pro result files used by the builder.
    """
    from .builders.swebench_pro.constants import BINARY_TRAJ_SOURCES, PER_TEST_TRAJ_SOURCES

    root = RAW / "swebench_pro_traj"
    for local_dir, repo, _, _ in PER_TEST_TRAJ_SOURCES:
        dst = root / local_dir / "train.parquet"
        _get_hf(
            repo,
            "data/train-00000-of-00001.parquet",
            dst,
        )

    for local_dir, repo, _, _ in BINARY_TRAJ_SOURCES:
        _get_hf(repo, "eval_results.json", root / local_dir / "eval_results.json")
        _get_hf(repo, "preds.json", root / local_dir / "preds.json")


def fetch_general_agentbench() -> None:
    from .builders.general_agentbench.constants import BENCHMARK_SPECS

    root = RAW / "general_agentbench"
    for file_stem in {stem for stem, _, _ in BENCHMARK_SPECS}:
        _get_hf("cx-cmu/agent_trajectories", f"{file_stem}.parquet", root / f"{file_stem}.parquet")


def fetch_harveyai_lab() -> None:
    import asyncio
    from harbor.models.task.id import PackageTaskId
    from harbor.tasks.client import TaskClient

    run_root = RAW / "harbor_runs" / "graded" / "harveyai" / "lab"
    package_root = RAW / "harveyai_lab" / "tasks"
    task_refs = {}

    for config_path in run_root.rglob("config.json"):
        config = json.loads(config_path.read_text())
        task = config.get("task")
        if not task:
            continue

        name = task.get("name")
        ref = task.get("ref")
        if not name or not ref:
            continue
        if name in task_refs and task_refs[name] != ref:
            raise ValueError(f"Harvey LAB task {name!r} has multiple source references")

        task_refs[name] = ref

    if not task_refs:
        raise ValueError("no Harvey LAB task references found in the Harbor runs")

    task_ids = [
        PackageTaskId(org="harveyai", name=name.removeprefix("harveyai/"), ref=ref)
        for name, ref in sorted(task_refs.items())
    ]
    progress = tqdm(total=len(task_ids), desc="fetching Harvey LAB", unit="task")
    try:
        asyncio.run(
            TaskClient().download_tasks(
                task_ids=task_ids,
                output_dir=package_root,
                on_task_download_complete=lambda _task, _result: progress.update(),
            )
        )
    finally:
        progress.close()


def fetch_scienceagentbench() -> None:
    import asyncio
    from harbor.models.task.id import PackageTaskId
    from harbor.tasks.client import TaskClient

    run_root = RAW / "harbor_runs" / "graded" / "scienceagentbench" / "scienceagentbench"
    package_root = RAW / "scienceagentbench" / "tasks"
    task_refs = {}

    for config_path in run_root.rglob("config.json"):
        config = json.loads(config_path.read_text())
        task = config.get("task")
        if not task:
            continue

        name = task.get("name")
        ref = task.get("ref")
        if not name or not ref:
            continue

        if name in task_refs and task_refs[name] != ref:
            raise ValueError(
                f"ScienceAgentBench task {name!r} has multiple source references"
            )

        task_refs[name] = ref

    if not task_refs:
        raise ValueError("no ScienceAgentBench task references found in the Harbor runs")

    task_ids = [
        PackageTaskId(
            org="scienceagentbench",
            name=name.removeprefix("scienceagentbench/"),
            ref=ref,
        )
        for name, ref in sorted(task_refs.items())
    ]
    progress = tqdm(total=len(task_ids), desc="fetching ScienceAgentBench", unit="task")
    try:
        asyncio.run(
            TaskClient().download_tasks(
                task_ids=task_ids,
                output_dir=package_root,
                on_task_download_complete=lambda _task, _result: progress.update(),
            )
        )
    finally:
        progress.close()


def fetch_toolathon() -> None:
    from .builders.benchmarks import UPSTREAM_COMMITS

    snapshot_download(
        repo_id="hkust-nlp/Toolathlon-Trajectories",
        repo_type="dataset",
        revision=UPSTREAM_COMMITS["toolathlon"],
        allow_patterns=["*.jsonl", "README.md"],
        local_dir=str(RAW / "toolathlon"),
        token=os.environ.get("HF_TOKEN"),
    )
    for path in (RAW / "toolathlon").glob("*.jsonl"):
        text = path.read_text()
        redacted = redact_credentials(text)
        if redacted != text:
            path.write_text(redacted)

    task_root = RAW / "toolathlon_tasks"
    if not _populated(task_root):
        _sparse_clone(
            "hkust-nlp/Toolathlon",
            task_root,
            ["tasks/finalpool/*/initial_workspace/**"],
            revision=UPSTREAM_COMMITS["toolathlon_tasks"],
        )


def _populated(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())


def _get(url: str, dst: Path) -> None:
    if dst.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as staging:
        temporary = Path(staging) / dst.name
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(dst)


def _get_hf(
    repo_id: str,
    filename: str,
    dst: Path,
    *,
    revision: str = "main",
    repo_type: str = "dataset",
) -> None:
    if dst.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as staging:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            local_dir=staging,
            token=os.environ.get("HF_TOKEN"),
        )
        Path(downloaded).replace(dst)


def _fetch_hf_table(
    repo_id: str,
    revision: str,
    split: str,
    dst: Path,
    columns: list[str]
) -> None:
    if dst.exists():
        return

    dataset = load_dataset(
        repo_id,
        revision=revision,
        split=split,
        cache_dir=str(CACHE / "datasets"),
    ).select_columns(columns)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(dst)


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
    )


def _clone(repo: str, dst: Path) -> None:
    if _populated(dst):
        return

    shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _git(
        "clone",
        "--depth",
        "1",
        f"https://github.com/{repo}.git",
        str(dst)
    )
    shutil.rmtree(dst / ".git", ignore_errors=True)


def _sparse_clone(
    repo: str,
    dst: Path,
    patterns: list[str],
    revision: str | None = None,
) -> None:
    shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _git(
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--no-checkout",
        f"https://github.com/{repo}.git",
        str(dst),
    )
    _git(
        "sparse-checkout",
        "init",
        "--no-cone",
        cwd=dst
    )
    _git(
        "sparse-checkout",
        "set",
        *patterns,
        cwd=dst
    )
    if revision:
        _git("fetch", "--depth", "1", "origin", revision, cwd=dst)
        _git("checkout", "--detach", revision, cwd=dst)
    else:
        _git("checkout", cwd=dst)
    shutil.rmtree(dst / ".git", ignore_errors=True)


UPSTREAM = [
    ("bridge",                       fetch_bridge),
    ("agent-psychometrics",          fetch_agent_psychometrics),
    ("gso-trajectories",             fetch_gso_trajectories),
    ("metr",                         fetch_metr),
    ("epoch",                        fetch_epoch),
    ("enrichment",                   fetch_enrichment),
    ("terminalbench",                fetch_terminalbench),
    ("theagentcompany",              fetch_theagentcompany),
    ("onlinemind2web",               fetch_onlinemind2web),
    ("bfcl",                         fetch_bfcl),
    ("matharena",                    fetch_matharena),
    ("livecodebench",                fetch_livecodebench),
    ("hle",                          fetch_hle),
    ("dabstep",                      fetch_dabstep),
    ("hf-tasks",                     fetch_hf_tasks),
    ("tau2-bench",                   fetch_tau2_bench),
    ("general-agentbench",           fetch_general_agentbench),
    ("harveyai-lab",                 fetch_harveyai_lab),
    ("scienceagentbench",            fetch_scienceagentbench),
    ("toolathlon",                   fetch_toolathon),
    ("osworld",                      fetch_osworld),
    ("swebench_verified_traj",       fetch_swebench_verified_traj),
    ("coderforge_swebench_verified", fetch_coderforge_swebench_verified),
    ("devin_swebench",               fetch_devin_swebench),
    ("swebench_pro_traj",            fetch_swebench_pro_traj),
    ("yoonholee_terminalbench",      fetch_yoonholee_terminalbench),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_raw_snapshot(root: Path) -> None:
    expected = {row["path"].removeprefix("raw/"): row for row in load_jsonl(RAW_MANIFEST_PATH)}
    actual = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        ignored = {".git", ".cache"} & set(relative.parts)
        if path.is_file() and not ignored and path.name not in {".DS_Store", ".manifest_sha256"}:
            actual[relative.as_posix()] = path

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"raw snapshot differs from its manifest: {len(missing)} missing and "
            f"{len(unexpected)} unexpected files"
        )

    for relative, path in tqdm(actual.items(), desc="verifying raw files", unit="file"):
        row = expected[relative]
        if path.stat().st_size != row["size"] or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"raw file does not match its manifest: {relative}")


def fetch_all(source: str = "local") -> None:
    """
    Fetch the raw inputs used by the dataset builders.

    :param source: ``local`` for the verified snapshot in ``data/raw`` or
        ``upstream`` to intentionally refresh the pinned original sources.
    """
    load_dotenv(REPO_ROOT / ".env")
    if source == "local":
        if not RAW_MANIFEST_PATH.is_file() or not RAW.is_dir() or not any(RAW.iterdir()):
            raise FileNotFoundError(
                "the local raw snapshot is missing; restore data/raw or explicitly fetch upstream"
            )

        manifest_hash = _sha256(RAW_MANIFEST_PATH)
        marker = RAW / ".manifest_sha256"
        if not marker.is_file() or marker.read_text().strip() != manifest_hash:
            _verify_raw_snapshot(RAW)
            marker.write_text(f"{manifest_hash}\n")

    elif source == "upstream":
        (RAW / ".manifest_sha256").unlink(missing_ok=True)
        pbar = tqdm(UPSTREAM, desc="fetching", unit="src")
        for name, function in pbar:
            pbar.set_description(f"fetching {name}")
            function()

        paths = []
        for path in RAW.rglob("*"):
            relative = path.relative_to(RAW)
            ignored = {".git", ".cache"} & set(relative.parts)
            if path.is_file() and not ignored and path.name not in {".DS_Store", ".manifest_sha256"}:
                paths.append(path)

        with RAW_MANIFEST_PATH.open("w") as manifest:
            for path in tqdm(sorted(paths), desc="recording raw files", unit="file"):
                row = {
                    "path": path.relative_to(RAW.parent).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                manifest.write(json.dumps(row) + "\n")

        manifest_hash = _sha256(RAW_MANIFEST_PATH)
        (RAW / ".manifest_sha256").write_text(f"{manifest_hash}\n")
    else:
        raise ValueError(f"source must be 'local' or 'upstream', got {source!r}")
