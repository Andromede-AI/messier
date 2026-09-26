"""
Import MathArena questions and model results.

Proof scores are retained as numerical verifier results and full credit defines success.
"""

import json
import pandas as pd
from ..standardize import format_agent_id, standardize_model
from ..dates import resolve_model_date
from ...io import RAW
from ...models import (
    BuildResult,
    ActionSpaceType,
    SourcePlatform,
    EnvironmentStateAccess,
    EnvironmentStateType,
    Record,
    RecordType,
    ScoringRule,
    Task,
    Trajectory,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)
from .constants import (
    ACTION_SPACE_DESCRIPTION,
    BENCHMARK,
    COMPETITIONS,
    ENV_DESCRIPTIONS,
    HUMAN_GRADED_PROOFS,
    MATHARENA_COMMITS,
    MAX_ATTEMPTS_PER_CELL,
    SCAFFOLD_SUFFIXES,
)


def build() -> BuildResult:
    root = RAW / "matharena"
    tasks = []
    verifiers = []
    records = []
    for slug, competition_date, result_kind in COMPETITIONS:
        result_paths = sorted(root.glob(f"{slug}_outputs*.parquet"))
        if not result_paths:
            continue

        results = pd.concat(
            [pd.read_parquet(path) for path in result_paths],
            ignore_index=True,
        )
        results = results[results["model_name"] != "Grok 4 (Specific Prompt)"]
        results = standardize_task_ids(results, slug)
        tasks_by_problem = results.drop_duplicates(subset=["_task_id"])

        for _, task_row in tasks_by_problem.iterrows():
            task_id = task_row["_task_id"]
            problem_id = task_id.rsplit(".", 1)[-1]
            if result_kind == "binary":
                gold_answer = str(task_row["gold_answer"])
                verifier_spec = [
                    {"type": VerifierType.EXACT_MATCH, "mode": VerifierMode.FINAL}
                ]
            else:
                grading_details = json.loads(task_row["grading_details_judge_1"])
                gold_answer = [
                    {
                        "title": detail.get("title"),
                        "description": detail["grading_scheme_desc"],
                        "max_points": detail["max_points"],
                    }
                    for detail in grading_details
                ]
                verifier_spec = [
                    {
                        "type": VerifierType.HUMAN_LABEL
                        if slug in HUMAN_GRADED_PROOFS
                        else VerifierType.LLM_JUDGE,
                        "mode": VerifierMode.FINAL,
                        "name": "proof_judge",
                        "scoring": "avg_judge_fraction_of_max",
                        "threshold": 1.0,
                    }
                ]
            scoring_rule = (
                ScoringRule.DIRECT
                if result_kind == "binary"
                else ScoringRule.THRESHOLD
            )
            task_metadata = {"problem_idx": problem_id, "kind": result_kind}
            if slug == "apex_shortlist":
                task_metadata["source"] = task_row["source"]

            tasks.append(
                Task(
                    benchmark=BENCHMARK,
                    task_id=task_id,
                    environment_id=slug,
                    environment_description=ENV_DESCRIPTIONS[slug],
                    environment_metadata={
                        "action_space": {
                            "types": [ActionSpaceType.TEXT],
                            "format": "exact_match"
                            if result_kind == "binary"
                            else "proof",
                            "description": ACTION_SPACE_DESCRIPTION,
                        },
                        "environment_state": {
                            "type": EnvironmentStateType.NONE,
                            "access": EnvironmentStateAccess.NONE,
                            "ref": f"hf://datasets/MathArena/{slug}_outputs@{MATHARENA_COMMITS[slug]}",
                        },
                        "competition": slug,
                        "competition_date": competition_date,
                    },
                    task_description=task_row.get("problem"),
                    gold_answer=gold_answer,
                    task_metadata=task_metadata,
                    task_date=competition_date,
                    verifiers=verifier_spec,
                    scoring_rule=scoring_rule,
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )
            verifiers.append(
                VerifierDefinition(
                    benchmark=BENCHMARK,
                    task_id=task_id,
                    verifier_id=f"{task_id}::verifier:0",
                    verifier=verifier_spec[0],
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )

        results = results[results["idx_answer"] < MAX_ATTEMPTS_PER_CELL]

        for _, result_row in results.iterrows():
            verifier_result = float(result_row["correct"])
            raw_model, scaffold = split_model_scaffold(str(result_row["model_name"]))
            model, reasoning_effort = standardize_model(raw_model)
            model_date = resolve_model_date(model, raw_model)
            agent_id = format_agent_id(model, scaffold, reasoning_effort)
            task_id = result_row["_task_id"]
            trial = int(result_row["idx_answer"])
            metadata = {
                "raw_model": str(result_row["model_name"]),
                "model_config": result_row.get("model_config"),
                "input_tokens": int(result_row["input_tokens"])
                if pd.notna(result_row.get("input_tokens"))
                else None,
                "output_tokens": int(result_row["output_tokens"])
                if pd.notna(result_row.get("output_tokens"))
                else None,
                "cost": float(result_row["cost"])
                if pd.notna(result_row.get("cost"))
                else None,
            }
            if slug == "apex_shortlist":
                metadata["source"] = result_row["source"]
                metadata["source_problem_idx"] = str(result_row["problem_idx"])

            records.append(
                Record(
                    benchmark=BENCHMARK,
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=trial,
                    result=int(verifier_result == 1.0),
                    metadata=metadata,
                    source_platform=SourcePlatform.HUGGINGFACE,
                )
            )
            if result_kind != "binary":
                records.append(
                    Record(
                        benchmark=BENCHMARK,
                        task_id=task_id,
                        verifier_id=f"{task_id}::verifier:0",
                        record_type=RecordType.VERIFIER_RESULT,
                        agent_id=agent_id,
                        agent_model=model,
                        model_reasoning_effort=reasoning_effort,
                        agent_scaffold=scaffold,
                        model_date=model_date,
                        trial=trial,
                        result=verifier_result,
                        metadata=metadata,
                        source_platform=SourcePlatform.HUGGINGFACE,
                    )
                )

    return tasks, verifiers, records


def trajectories():
    root = RAW / "matharena"

    for slug, _, _ in COMPETITIONS:
        paths = sorted(root.glob(f"{slug}_outputs*.parquet"))
        if not paths:
            continue

        results = pd.concat(
            [pd.read_parquet(path) for path in paths],
            ignore_index=True,
        )
        results = results[results["model_name"] != "Grok 4 (Specific Prompt)"]
        results = standardize_task_ids(results, slug)
        results = results[results["idx_answer"] < MAX_ATTEMPTS_PER_CELL]
        for _, row in results.iterrows():
            raw_model, scaffold = split_model_scaffold(str(row["model_name"]))
            model, reasoning_effort = standardize_model(raw_model)
            agent_id = format_agent_id(model, scaffold, reasoning_effort)
            task_id = row["_task_id"]
            trial = int(row["idx_answer"])
            content = row.get("all_messages")
            if not isinstance(content, str) or not content:
                continue

            metadata = {
                "competition": slug,
                "attempt": int(row["idx_answer"]),
            }

            if slug == "apex_shortlist":
                metadata["source"] = row["source"]
                metadata["source_problem_idx"] = str(row["problem_idx"])

            yield Trajectory(
                benchmark=BENCHMARK,
                task_id=task_id,
                agent_id=agent_id,
                agent_model=model,
                model_reasoning_effort=reasoning_effort,
                agent_scaffold=scaffold,
                model_date=resolve_model_date(model, raw_model),
                trial=trial,
                content=content,
                metadata=metadata,
                source_platform=SourcePlatform.HUGGINGFACE,
            )


def split_model_scaffold(name: str) -> tuple[str, str | None]:
    for suffix, scaffold in SCAFFOLD_SUFFIXES.items():
        if name.endswith(suffix):
            return name.removesuffix(suffix).strip(), scaffold

    return name, None


def standardize_task_ids(results: pd.DataFrame, slug: str) -> pd.DataFrame:
    results = results.copy()
    results["_task_id"] = slug + "." + results["problem_idx"].astype(str)
    if slug != "apex_shortlist":
        return results

    if "source" not in results or results["source"].isna().any():
        raise ValueError("APEX requires a source for every result")

    for field in ("problem", "gold_answer"):
        if (results.groupby("source")[field].nunique(dropna=False) > 1).any():
            raise ValueError(f"APEX source maps to conflicting {field}")

    counts = results.groupby(["source", "problem_idx"]).size()
    most_common = counts[counts == counts.groupby(level="source").transform("max")]
    if most_common.index.get_level_values("source").duplicated().any():
        raise ValueError("APEX source has no unique problem index")

    source_to_index = most_common.reset_index().set_index("source")["problem_idx"]
    if source_to_index.duplicated().any():
        raise ValueError("APEX sources share a problem index")

    results["_task_id"] = results["source"].map(lambda source: f"{slug}.{source_to_index[source]}")
    return results
