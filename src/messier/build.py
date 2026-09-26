import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich import box
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from .builders import (
    bfcl,
    cybench,
    gdpval,
    gso,
    general_agentbench,
    harbor_runs,
    hle,
    livecodebench,
    matharena,
    mlebench,
    metr,
    onlinemind2web,
    osworld,
    swebench,
    swebench_pro,
    swebench_verified,
    tau2_bench,
    terminalbench,
    theagentcompany,
    toolathlon,
)
from .builders.benchmarks import BENCHMARK_GROUPS, DEFAULT_SCORING_RULES
from .builders.dates import resolve_model_date
from .builders.standardize import format_agent_id
from .fetch import fetch_all
from .io import (
    CLASSIFICATIONS_PATH,
    PROCESSED,
    RECORDS_PATH,
    TASKS_PATH,
    TASK_FILES_PATH,
    TRAJECTORIES_PATH,
    VERIFIERS_PATH,
    redact_credentials,
)
from .models import (
    BenchmarkGroup,
    DatasetBuilderConfig,
    RecordType,
    ScoringRule,
    Task,
    VerifierDefinition,
)

BUILDERS = [
    ("swebench_verified", swebench_verified),
    ("swebench_pro", swebench_pro),
    ("terminalbench", terminalbench),
    ("gso", gso),
    ("metr", metr),
    ("mlebench", mlebench),
    ("cybench", cybench),
    ("gdpval", gdpval),
    ("theagentcompany", theagentcompany),
    ("onlinemind2web", onlinemind2web),
    ("bfcl", bfcl),
    ("tau2-bench", tau2_bench),
    ("matharena", matharena),
    ("livecodebench", livecodebench),
    ("hle", hle),
    ("general_agentbench", general_agentbench),
    ("osworld", osworld),
    ("swebench", swebench),
    ("harbor_runs", harbor_runs),
    ("toolathlon", toolathlon),
]


class DatasetBuilder:
    """
    Build the release files from the frozen raw-data snapshot.

    The builder downloads the snapshot from Hugging Face when it is not
    available locally, runs each selected benchmark builder, and combines its
    tasks, verifiers, records, and trajectories. It attaches cached SOC and
    NAICS labels, standardizes benchmark groups, model dates, and agent IDs,
    validates references between files, and writes the resulting JSONL files
    to ``data/processed``, which is gitignored.

    Processed tasks are generated rather than edited directly. Add or change
    them by updating a benchmark's raw input, fetch logic, or builder. When
    tasks change, build once to produce ``tasks.jsonl``, run
    ``python -m messier.classify``, and build again to attach the new labels.
    """

    def __init__(self, config: DatasetBuilderConfig | None = None, **kwargs):
        self.config = config if config is not None else DatasetBuilderConfig(**kwargs)

    def build(self, *names: str):
        """
        Build all benchmarks, or only those named in ``names``.

        :param names: Builder names to run. An empty value runs every builder.
        :return: This builder instance.
        """
        # verify the frozen sources and select the requested builders
        fetch_all()
        shutil.rmtree(TASK_FILES_PATH, ignore_errors=True)
        progress_width = min(shutil.get_terminal_size().columns, 120)
        selected_builders = [
            (name, module)
            for name, module in BUILDERS
            if not names or name in names
        ]
        unknown = set(names) - {name for name, _ in BUILDERS}
        if unknown:
            raise ValueError(f"unknown builders: {sorted(unknown)}")

        # run the benchmark builders in parallel
        builder_results = {}
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(module.build): name
                for name, module in selected_builders
            }
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="building benchmarks",
                unit=" benchmarks",
                ncols=progress_width,
                colour="blue",
            )
            for future in progress:
                name = futures[future]
                builder_results[name] = future.result()
                progress.set_postfix_str(name)

        # merge tasks, verifier definitions, and records
        tasks: dict[tuple[str, str], Task] = {}
        verifiers: dict[tuple[str, str], VerifierDefinition] = {}
        records = []
        task_counts = Counter()
        next_trial = Counter()
        trial_numbers = {}

        for name, _ in selected_builders:
            built_tasks, built_verifiers, built_records = builder_results[name]
            for record in built_records:
                if record.record_type == RecordType.SOURCE_SUMMARY:
                    records.append(record)
                    continue

                trial_key = (
                    name,
                    record.benchmark,
                    record.task_id,
                    record.agent_id,
                    record.trial,
                )
                agent_task = (
                    record.benchmark,
                    record.task_id,
                    record.agent_id,
                )
                if trial_key not in trial_numbers:
                    trial_numbers[trial_key] = next_trial[agent_task]
                    next_trial[agent_task] += 1

                record.trial = trial_numbers[trial_key]
                records.append(record)

            for task in built_tasks:
                key = (task.benchmark, task.task_id)
                if key in tasks:
                    raise ValueError(f"duplicate task: {task.benchmark}/{task.task_id}")

                task.benchmark_group = BenchmarkGroup(BENCHMARK_GROUPS[task.benchmark])
                if task.scoring_rule is None:
                    rule = DEFAULT_SCORING_RULES.get(task.benchmark)
                    task.scoring_rule = ScoringRule(rule) if rule else None

                tasks[key] = task
                task_counts[task.benchmark] += 1

            for verifier in built_verifiers:
                key = (verifier.benchmark, verifier.verifier_id)
                if key in verifiers:
                    raise ValueError(f"duplicate verifier: {verifier.benchmark}/{verifier.verifier_id}")

                verifiers[key] = verifier

        # attach task classifications
        cached_labels = {}
        if CLASSIFICATIONS_PATH.exists():
            for line in CLASSIFICATIONS_PATH.open():
                if line.strip():
                    row = json.loads(line)
                    cached_labels[(row["benchmark"], row["task_id"])] = (
                        row.get("soc_code"),
                        row.get("naics_code"),
                    )

        label_counts = Counter()
        for key, task in tasks.items():
            soc_code = task.soc_code
            naics_code = task.naics_code
            if soc_code is not None and naics_code is not None:
                label_counts["source metadata"] += 1
            elif key in cached_labels:
                soc_code, naics_code = cached_labels[key]
                label_counts["task classification"] += 1

            if soc_code is None or naics_code is None:
                label_counts["missing"] += 1

            task.soc_code = soc_code
            task.naics_code = naics_code

        # validate and complete the verifier definitions
        verifiers_by_task = defaultdict(list)
        for verifier in verifiers.values():
            if (verifier.benchmark, verifier.task_id) not in tasks:
                raise ValueError(f"verifier {verifier.benchmark}/{verifier.verifier_id} references an unknown task")

            verifiers_by_task[(verifier.benchmark, verifier.task_id)].append(verifier)

        for key, task in tasks.items():
            task_verifiers = verifiers_by_task[key]
            if task_verifiers:
                task.verifiers = [verifier.verifier for verifier in task_verifiers]
                continue

            if not task.verifiers:
                raise ValueError(f"task {task.benchmark}/{task.task_id} has no verifier")

            for index, verifier in enumerate(task.verifiers):
                verifier_definition = VerifierDefinition(
                    benchmark=task.benchmark,
                    task_id=task.task_id,
                    verifier_id=f"{task.task_id}::verifier:{index}",
                    verifier=verifier,
                    source_platform=task.source_platform,
                    data_provider=task.data_provider,
                )
                verifier_key = (
                    verifier_definition.benchmark,
                    verifier_definition.verifier_id,
                )
                verifiers[verifier_key] = verifier_definition
                task_verifiers.append(verifier_definition)

        # normalize, validate, and write the records
        record_counts = Counter()
        record_keys = set()
        agent_ids = set()
        task_stats = defaultdict(lambda: [0, 0, 0])
        PROCESSED.mkdir(parents=True, exist_ok=True)
        records_tmp = PROCESSED / "records.jsonl.tmp"

        with records_tmp.open("w") as output:
            for record in tqdm(records, desc="writing records", unit=" records", ncols=progress_width, colour="blue"):
                task_key = (record.benchmark, record.task_id)
                if task_key not in tasks:
                    raise ValueError(f"record references unknown task: {record.benchmark}/{record.task_id}")

                verifier_key = (record.benchmark, record.verifier_id)
                if record.verifier_id and verifier_key not in verifiers:
                    raise ValueError(f"record references unknown verifier: {record.benchmark}/{record.verifier_id}")

                record.benchmark_group = BenchmarkGroup(BENCHMARK_GROUPS[record.benchmark])
                resolved_date = resolve_model_date(record.agent_model)
                if record.agent_model in {"multiple", "undisclosed"}:
                    record.model_date = None
                elif resolved_date:
                    if record.model_date and record.model_date != resolved_date:
                        record.metadata = {**record.metadata, "source_model_date": record.model_date}

                    record.model_date = resolved_date

                agent_id = format_agent_id(
                    record.agent_model,
                    record.agent_scaffold,
                    record.model_reasoning_effort,
                )
                if record.agent_id != agent_id:
                    raise ValueError(f"agent_id must be {agent_id!r}, got {record.agent_id!r}")

                agent_ids.add(record.agent_id)
                record_key = (
                    record.benchmark,
                    record.task_id,
                    record.agent_id,
                    record.trial,
                    record.record_type,
                    record.verifier_id,
                )
                if record_key in record_keys:
                    raise ValueError(f"duplicate record: {record_key}")

                record_keys.add(record_key)

                serialized = redact_credentials(record.model_dump_json())
                output.write(serialized + "\n")
                record_counts[record.benchmark] += 1
                stats = task_stats[(record.benchmark, record.task_id)]
                stats[0] += 1
                if record.record_type == RecordType.TRIAL_RESULT:
                    if record.result is not None:
                        stats[1] += 1
                        stats[2] += record.result

        # stream, validate, and write source-provided trajectories
        trajectory_keys = set()
        trajectory_count = 0
        trajectories_tmp = PROCESSED / "trajectories.jsonl.tmp"
        with (
            trajectories_tmp.open("w") as output,
            tqdm(desc="writing trajectories", unit=" trajectories", ncols=progress_width, colour="blue") as progress,
        ):
            for name, module in selected_builders:
                build_trajectories = getattr(module, "trajectories", None)
                if build_trajectories is None:
                    continue

                progress.set_postfix_str(name)
                for trajectory in build_trajectories():
                    task_key = (trajectory.benchmark, trajectory.task_id)
                    if task_key not in tasks:
                        raise ValueError(
                            f"trajectory references unknown task: "
                            f"{trajectory.benchmark}/{trajectory.task_id}"
                        )

                    agent_id = format_agent_id(
                        trajectory.agent_model,
                        trajectory.agent_scaffold,
                        trajectory.model_reasoning_effort,
                    )
                    if trajectory.agent_id != agent_id:
                        raise ValueError(
                            f"trajectory agent_id must be {agent_id!r}, "
                            f"got {trajectory.agent_id!r}"
                        )

                    trial_key = (
                        name,
                        trajectory.benchmark,
                        trajectory.task_id,
                        trajectory.agent_id,
                        trajectory.trial,
                    )
                    if trial_key not in trial_numbers:
                        raise ValueError(
                            f"trajectory has no matching trial: "
                            f"{trajectory.benchmark}/{trajectory.task_id}/"
                            f"{trajectory.agent_id}/{trajectory.trial}"
                        )

                    trajectory.trial = trial_numbers[trial_key]
                    trial_record_key = (
                        trajectory.benchmark,
                        trajectory.task_id,
                        trajectory.agent_id,
                        trajectory.trial,
                        RecordType.TRIAL_RESULT,
                        None,
                    )
                    if trial_record_key not in record_keys:
                        raise ValueError(
                            f"trajectory has no matching trial result: "
                            f"{trajectory.benchmark}/{trajectory.task_id}/"
                            f"{trajectory.agent_id}/{trajectory.trial}"
                        )

                    resolved_date = resolve_model_date(trajectory.agent_model)
                    if trajectory.agent_model in {"multiple", "undisclosed"}:
                        trajectory.model_date = None
                    elif resolved_date:
                        if trajectory.model_date and trajectory.model_date != resolved_date:
                            trajectory.metadata = {**trajectory.metadata, "source_model_date": trajectory.model_date}

                        trajectory.model_date = resolved_date

                    key = (
                        trajectory.benchmark,
                        trajectory.task_id,
                        trajectory.agent_id,
                        trajectory.trial,
                    )
                    if key in trajectory_keys:
                        raise ValueError(f"duplicate trajectory: {key}")

                    trajectory_keys.add(key)
                    trajectory_count += 1
                    serialized = redact_credentials(trajectory.model_dump_json())
                    output.write(serialized + "\n")
                    progress.update()

        # derive task-level counts and mean results
        for key, task in tasks.items():
            n_records, n_trial_results, result_sum = task_stats[key]
            task.n_records = n_records
            task.n_verifiers = len(task.verifiers)
            task.mean_trial_result = (result_sum / n_trial_results if n_trial_results else None)
            task.is_saturated = task.mean_trial_result in {0.0, 1.0}
            task.is_unrecorded = n_records == 0

        # write tasks and verifier definitions atomically
        tasks_tmp = PROCESSED / "tasks.jsonl.tmp"
        verifiers_tmp = PROCESSED / "verifiers.jsonl.tmp"
        with tasks_tmp.open("w") as task_file:
            for task in tqdm(tasks.values(), desc="writing tasks", unit=" tasks", ncols=progress_width, colour="blue"):
                serialized = redact_credentials(task.model_dump_json())
                task_file.write(serialized + "\n")

        with verifiers_tmp.open("w") as verifier_file:
            for verifier in tqdm(verifiers.values(), desc="writing verifiers", unit=" verifiers", ncols=progress_width, colour="blue"):
                serialized = redact_credentials(verifier.model_dump_json())
                verifier_file.write(serialized + "\n")

        records_tmp.replace(RECORDS_PATH)
        tasks_tmp.replace(TASKS_PATH)
        verifiers_tmp.replace(VERIFIERS_PATH)
        trajectories_tmp.replace(TRAJECTORIES_PATH)

        # print: report the resulting corpus counts
        console = Console()
        corpus_table = Table(title="Corpus", box=box.SIMPLE_HEAVY, show_header=False)
        corpus_table.add_column("Measure", style="blue")
        corpus_table.add_column("Count", justify="right")
        corpus_table.add_row("Benchmarks", f"{len(task_counts):,}")
        corpus_table.add_row("Agents", f"{len(agent_ids):,}")
        corpus_table.add_row("Tasks", f"{sum(task_counts.values()):,}")
        corpus_table.add_row("Verifiers", f"{len(verifiers):,}")
        corpus_table.add_row("Records", f"{sum(record_counts.values()):,}")
        corpus_table.add_row("Trajectories", f"{trajectory_count:,}")
        console.print(corpus_table)

        benchmark_table = Table(title="Benchmarks", box=box.SIMPLE_HEAVY, header_style="bold blue")
        benchmark_table.add_column("Benchmark")
        benchmark_table.add_column("Tasks", justify="right")
        benchmark_table.add_column("Records", justify="right")
        for benchmark in sorted(task_counts):
            benchmark_table.add_row(benchmark, f"{task_counts[benchmark]:,}", f"{record_counts[benchmark]:,}")

        benchmark_table.add_row(
            "Total",
            f"{sum(task_counts.values()):,}",
            f"{sum(record_counts.values()):,}",
            style="bold",
        )
        console.print(benchmark_table)

        labels = ", ".join(f"{name}: {count:,}" for name, count in label_counts.items())
        console.print(f"[bold blue]Labels[/bold blue]  {labels}\n")
        return self
