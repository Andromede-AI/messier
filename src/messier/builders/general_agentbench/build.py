"""
Import BrowseComp, MathHay, MCPBench, and WebVoyager evaluation data.

MCPBench dimension scores are retained beside its thresholded trial results.
"""

import json
import pandas as pd
from ..standardize import format_agent_id, standardize_model
from ..dates import resolve_model_date, resolve_task_date
from ...io import RAW
from ...models import (
    BuildResult,
    DataProvider,
    SourcePlatform,
    Record,
    RecordType,
    Task,
    Trajectory,
    VerifierDefinition,
    VerifierMode,
    VerifierType,
)
from .constants import BENCHMARK_SPECS, MCP_BINARY_THRESHOLD, MCP_DIMENSIONS, MCP_SCORE_TOTAL, SPECS



def build() -> BuildResult:
    tasks: dict[tuple[str, str], Task] = {}
    verifiers: dict[tuple[str, str], VerifierDefinition] = {}
    records: list[Record] = []

    for path, benchmark, row in source_rows():
        details = row["eval_details"]
        if isinstance(details, str):
            details = json.loads(details)

        details = details or {}
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        tool_registry = row["tool_registry"]
        if isinstance(tool_registry, str):
            tool_registry = json.loads(tool_registry)

        system_prompt = next(
            (
                message["content"]
                for message in messages
                if message.get("role") == "system" and message.get("content")
            ),
            None,
        )
        task_id = str(row["task_id"])
        key = (benchmark, task_id)
        if key not in tasks:
            spec = SPECS[benchmark]
            source_description = details.get(spec["desc_key"])
            user_prompt = next(
                (
                    message["content"]
                    for message in messages
                    if message.get("role") == "user" and message.get("content")
                ),
                None,
            )
            if benchmark == "mcpbench" and user_prompt and source_description:
                description = (
                    f"{user_prompt}\n\n"
                    f"Task specification:\n{source_description}"
                )
            else:
                description = user_prompt or source_description

            gold_answer = (details.get(spec["gold_key"]) if spec["gold_key"] else None)
            if isinstance(gold_answer, str) and not gold_answer.strip():
                gold_answer = None

            task_metadata = {
                "source_model_label": str(row["source_model"]),
                "extra_context": {"system_prompt": system_prompt} if system_prompt else None,
            }
            if str(row["domain"]) != benchmark:
                task_metadata["domain"] = str(row["domain"])
            if benchmark == "mathhay":
                task_metadata.update(
                    {
                        "task_type": details.get("task_type"),
                        "context_length": details.get("context_length"),
                        "num_relevant_docs": details.get("num_relevant_docs"),
                        "num_irrelevant_docs": details.get("num_irrelevant_docs"),
                    }
                )
            elif benchmark == "mcpbench":
                task_metadata["server_name"] = details.get("server_name")

            if benchmark == "mcpbench":
                task_verifiers = [
                    {
                        "type": spec["verifier"],
                        "mode": VerifierMode.SEQUENTIAL,
                        "role": dimension,
                    }
                    for dimension in MCP_DIMENSIONS
                ]
            else:
                task_verifiers = [
                    {
                        "type": spec["verifier"],
                        "mode": spec["verifier_mode"],
                    }
                ]

            tasks[key] = Task(
                benchmark=benchmark,
                task_id=task_id,
                environment_id=benchmark,
                environment_description=spec["env_desc"],
                environment_metadata={
                    "action_space": {
                        "types": spec["action_types"],
                        "description": spec["action_desc"],
                        "tool_registry": tool_registry,
                    },
                    "environment_state": {
                        "type": spec["state_type"],
                        "access": spec["state_access"],
                        "ref": f"cxcmu-agent-trajectories://{benchmark}/{row['domain']}",
                        "snapshot": None,
                    },
                },
                task_description=str(description).strip() if description else None,
                task_metadata=task_metadata,
                gold_answer=gold_answer,
                task_date=resolve_task_date(benchmark),
                verifiers=task_verifiers,
                source_platform=SourcePlatform.HUGGINGFACE,
                data_provider=DataProvider.GENERAL_AGENTBENCH,
            )
            if benchmark == "mcpbench":
                for dimension in MCP_DIMENSIONS:
                    verifier_id = f"{task_id}::{dimension}"
                    verifiers[(benchmark, verifier_id)] = VerifierDefinition(
                        benchmark=benchmark,
                        task_id=task_id,
                        verifier_id=verifier_id,
                        verifier={
                            "type": VerifierType.LLM_JUDGE,
                            "mode": VerifierMode.SEQUENTIAL,
                            "role": dimension,
                        },
                        metadata={
                            "dimension": dimension,
                            "threshold_pct": 50,
                            "threshold_basis": "mcpbench:0-10 score",
                        },
                        source_platform=SourcePlatform.HUGGINGFACE,
                        data_provider=DataProvider.GENERAL_AGENTBENCH,
                    )

        source_model = row["source_model"]
        reward = float(row["reward"])
        model, reasoning_effort = standardize_model(source_model)
        scaffold = "general-agentbench"
        agent_id = format_agent_id(model, scaffold, reasoning_effort)
        model_date = resolve_model_date(model, source_model)
        trial = int(row["pass"]) - 1
        metadata = {
            "num_turns": int(row["num_turns"]),
            "harness": "cxcmu+distraction",
            "source_reward": reward,
        }
        if benchmark == "mathhay":
            metadata.update(
                {
                    "is_correct": details.get("is_correct"),
                    "numerical_match": details.get("numerical_match"),
                    "llm_judge": details.get("llm_judge"),
                }
            )
        elif benchmark == "mcpbench":
            evaluation = details.get("evaluation") or {}
            metadata.update(
                {
                    "task_completion_score": evaluation.get("task_completion_score"),
                    "tool_selection_score": evaluation.get("tool_selection_score"),
                    "planning_score": evaluation.get("planning_effectiveness_and_efficiency_score"),
                }
            )

        if benchmark == "mcpbench":
            evaluation = details.get("evaluation") or {}
            dimension_results = [evaluation.get(dimension) for dimension in MCP_DIMENSIONS]
            has_all_dimensions = all(value is not None for value in dimension_results)
            if not has_all_dimensions:
                continue

            records.append(
                Record(
                    benchmark=benchmark,
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=trial,
                    result=int(reward >= MCP_BINARY_THRESHOLD),
                    metadata=metadata,
                    source_platform=SourcePlatform.HUGGINGFACE,
                    data_provider=DataProvider.GENERAL_AGENTBENCH,
                )
            )
            for dimension, value in zip(MCP_DIMENSIONS, dimension_results):
                source_result = float(value)
                source_fraction = source_result / MCP_SCORE_TOTAL # normalize the source's 0-10 score
                records.append(
                    Record(
                        benchmark=benchmark,
                        task_id=task_id,
                        verifier_id=f"{task_id}::{dimension}",
                        record_type=RecordType.VERIFIER_RESULT,
                        agent_id=agent_id,
                        agent_model=model,
                        model_reasoning_effort=reasoning_effort,
                        agent_scaffold=scaffold,
                        model_date=model_date,
                        trial=trial,
                        result=source_fraction,
                        metadata={
                            **metadata,
                            "dimension": dimension,
                            "source_result": source_result,
                            "source_total": MCP_SCORE_TOTAL,
                            "source_fraction": source_fraction,
                        },
                        source_platform=SourcePlatform.HUGGINGFACE,
                        data_provider=DataProvider.GENERAL_AGENTBENCH,
                    )
                )
        else:
            records.append(
                Record(
                    benchmark=benchmark,
                    task_id=task_id,
                    agent_id=agent_id,
                    agent_model=model,
                    model_reasoning_effort=reasoning_effort,
                    agent_scaffold=scaffold,
                    model_date=model_date,
                    trial=trial,
                    result=int(reward == 1.0),
                    metadata=metadata,
                    source_platform=SourcePlatform.HUGGINGFACE,
                    data_provider=DataProvider.GENERAL_AGENTBENCH,
                )
            )

    return list(tasks.values()), list(verifiers.values()), records


def source_rows():
    root = RAW / "general_agentbench"
    for file_stem, benchmark, domain_filter in BENCHMARK_SPECS:
        path = root / f"{file_stem}.parquet"
        if not path.exists():
            continue

        frame = pd.read_parquet(path)
        if domain_filter is not None:
            frame = frame[frame["domain"] == domain_filter]

        for _, row in frame.iterrows():
            yield path, benchmark, row


def trajectories():
    for path, benchmark, row in source_rows():
        details = row["eval_details"]
        if isinstance(details, str):
            details = json.loads(details)

        if benchmark == "mcpbench":
            evaluation = (details or {}).get("evaluation") or {}
            if not all(evaluation.get(dimension) is not None for dimension in MCP_DIMENSIONS):
                continue

        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        source_model = row["source_model"]
        model, reasoning_effort = standardize_model(source_model)
        scaffold = "general-agentbench"
        yield Trajectory(
            benchmark=benchmark,
            task_id=str(row["task_id"]),
            agent_id=format_agent_id(model, scaffold, reasoning_effort),
            agent_model=model,
            model_reasoning_effort=reasoning_effort,
            agent_scaffold=scaffold,
            model_date=resolve_model_date(model, source_model),
            trial=int(row["pass"]) - 1,
            content=json.dumps(messages, ensure_ascii=False),
            metadata={"domain": str(row["domain"]), "source_file": path.name,},
            source_platform=SourcePlatform.HUGGINGFACE,
            data_provider=DataProvider.GENERAL_AGENTBENCH,
        )
