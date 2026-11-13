import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMAS = {
    "classifications": pa.schema([
        ("benchmark", pa.string()),
        ("task_id", pa.string()),
        ("soc_code", pa.string()),
        ("naics_code", pa.string()),
        ("soc_source", pa.string()),
        ("naics_source", pa.string()),
        ("votes", pa.string()),
        ("adjudicator_model", pa.string()),
        ("adjudicator_rationale", pa.string()),
        ("adjudicator_confidence", pa.bool_()),
    ]),
    "records": pa.schema([
        ("benchmark", pa.string()),
        ("benchmark_group", pa.string()),
        ("task_id", pa.string()),
        ("verifier_id", pa.string()),
        ("record_type", pa.string()),
        ("agent_id", pa.string()),
        ("agent_model", pa.string()),
        ("model_reasoning_effort", pa.string()),
        ("agent_scaffold", pa.string()),
        ("model_date", pa.string()),
        ("trial", pa.int64()),
        ("result", pa.float64()),
        ("metadata", pa.string()),
        ("source_platform", pa.string()),
        ("data_provider", pa.string()),
    ]),
    "tasks": pa.schema([
        ("benchmark", pa.string()),
        ("benchmark_group", pa.string()),
        ("task_id", pa.string()),
        ("environment_id", pa.string()),
        ("environment_description", pa.string()),
        ("environment_metadata", pa.string()),
        ("task_description", pa.string()),
        ("task_metadata", pa.string()),
        ("gold_answer", pa.string()),
        ("task_date", pa.string()),
        ("verifiers", pa.string()),
        ("human", pa.string()),
        ("difficulty_label", pa.string()),
        ("source_platform", pa.string()),
        ("data_provider", pa.string()),
        ("soc_code", pa.string()),
        ("naics_code", pa.string()),
        ("n_records", pa.int64()),
        ("n_verifiers", pa.int64()),
        ("scoring_rule", pa.string()),
        ("mean_trial_result", pa.float64()),
        ("is_saturated", pa.bool_()),
        ("is_unrecorded", pa.bool_()),
    ]),
    "trajectories": pa.schema([
        ("benchmark", pa.string()),
        ("task_id", pa.string()),
        ("agent_id", pa.string()),
        ("agent_model", pa.string()),
        ("model_reasoning_effort", pa.string()),
        ("agent_scaffold", pa.string()),
        ("model_date", pa.string()),
        ("trial", pa.int64()),
        ("content", pa.string()),
        ("metadata", pa.string()),
        ("source_platform", pa.string()),
        ("data_provider", pa.string()),
    ]),
    "verifiers": pa.schema([
        ("benchmark", pa.string()),
        ("task_id", pa.string()),
        ("verifier_id", pa.string()),
        ("description", pa.string()),
        ("verifier", pa.string()),
        ("metadata", pa.string()),
        ("gold_answer", pa.string()),
        ("source_platform", pa.string()),
        ("data_provider", pa.string()),
    ]),
}
JSON_FIELDS = {
    "classifications": {"votes"},
    "records": {"metadata"},
    "tasks": {"environment_metadata", "task_metadata", "gold_answer", "verifiers", "human"},
    "trajectories": {"metadata"},
    "verifiers": {"verifier", "metadata", "gold_answer"},
}


def export_viewer_file(source: Path, destination: Path, name: str) -> None:
    schema = SCHEMAS[name]
    schema_fields = set(schema.names)
    json_fields = JSON_FIELDS[name]
    temporary = destination.with_suffix(".parquet.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    with source.open() as file, pq.ParquetWriter(
        temporary,
        schema,
        compression="zstd",
    ) as writer:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            unexpected = set(row) - schema_fields
            if unexpected:
                raise ValueError(f"unexpected {name} fields: {sorted(unexpected)}")
            for field in json_fields:
                if row.get(field) is not None:
                    row[field] = json.dumps(
                        row[field],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )

            rows.append(row)
            if len(rows) == 1_000:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                rows.clear()

        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))

    temporary.replace(destination)


def export_viewer_files(sources: dict[str, Path], destination: Path) -> None:
    for name, source in sources.items():
        export_viewer_file(source, destination / f"{name}.parquet", name)
