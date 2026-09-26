"""
Compute benchmark frontier results by model release quarter.
"""

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from messier.io import PROCESSED, load_jsonl


OUT = Path(__file__).resolve().parents[1] / "outputs" / "frontier_progress"
FIRST_QUARTER = "2023-01-01"


def quarter_index(date: str) -> int:
    """
    Convert an ISO date into an integer quarter.

    :param date: Date beginning with ``YYYY-MM``.
    :return: Sequential quarter number.
    """
    year = int(date[:4])
    month = int(date[5:7])
    return year * 4 + (month - 1) // 3


def load_agent_task_results() -> pd.DataFrame:
    """
    Average repeated results and identify each agent's model-release quarter.

    :return: One result for each benchmark, task, and agent.
    """
    accumulated_results = defaultdict(lambda: [0.0, 0])
    model_quarter_by_agent = {}

    for record in load_jsonl(PROCESSED / "records.jsonl"):
        if record["record_type"] != "trial_result" or record["result"] is None:
            continue
        if not record.get("model_date"):
            continue

        agent_id = record["agent_id"]
        model_quarter = quarter_index(record["model_date"])
        previous_quarter = model_quarter_by_agent.get(agent_id)
        if previous_quarter is None or model_quarter < previous_quarter:
            model_quarter_by_agent[agent_id] = model_quarter

        key = (record["benchmark"], record["task_id"], agent_id)
        accumulated_results[key][0] += record["result"]
        accumulated_results[key][1] += 1

    rows = []
    for (benchmark, task_id, agent_id), (total, count) in accumulated_results.items():
        rows.append(
            {
                "benchmark": benchmark,
                "task_id": task_id,
                "agent_id": agent_id,
                "model_quarter": model_quarter_by_agent[agent_id],
                "mean_trial_result": total / count,
            }
        )

    return pd.DataFrame(rows)


def calculate_frontier(agent_task_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the best result observed for every task by each quarter.

    For each quarter, only agents whose models were released by that quarter
    are eligible. Their best task results are averaged within each benchmark.

    :param agent_task_results: Averaged result for each task and agent.
    :return: Benchmark frontiers and their standard errors by quarter.
    """
    first_quarter = quarter_index(FIRST_QUARTER)
    last_quarter = int(agent_task_results["model_quarter"].max())
    quarter_indices = list(range(first_quarter, last_quarter + 1))
    quarter_labels = [f"{index // 4}Q{index % 4 + 1}" for index in quarter_indices]

    rows = []
    for cutoff, label in zip(quarter_indices, quarter_labels):
        available = agent_task_results[agent_task_results["model_quarter"] <= cutoff]
        if available.empty:
            continue

        task_frontier = available.groupby(["benchmark", "task_id"])["mean_trial_result"].max().reset_index()
        benchmark_frontier = (
            task_frontier.groupby("benchmark")["mean_trial_result"]
            .agg(mean="mean", std="std", n_tasks="size")
            .reset_index()
        )
        benchmark_frontier["se"] = benchmark_frontier["std"] / np.sqrt(benchmark_frontier["n_tasks"])
        benchmark_frontier["quarter"] = label
        rows.append(benchmark_frontier)

    quarter_results = pd.concat(rows, ignore_index=True)
    frontier = quarter_results.pivot(index="benchmark", columns="quarter", values="mean")
    standard_errors = quarter_results.pivot(index="benchmark", columns="quarter", values="se")
    frontier = frontier.reindex(columns=quarter_labels).sort_index()
    standard_errors = standard_errors.reindex(columns=quarter_labels).sort_index()
    return frontier, standard_errors


def main() -> None:
    """
    Calculate and save the quarterly benchmark frontiers.
    """
    agent_task_results = load_agent_task_results()
    frontier, standard_errors = calculate_frontier(agent_task_results)

    # save files
    OUT.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(OUT / "frontier.csv", float_format="%.4f")
    standard_errors.to_csv(OUT / "frontier_se.csv", float_format="%.4f")

    # print
    print(f"computed the frontier for {len(frontier):,} benchmarks across {len(frontier.columns):,} model-release quarters")
    print(f"wrote frontier.csv and frontier_se.csv to {OUT}")


if __name__ == "__main__":
    main()
