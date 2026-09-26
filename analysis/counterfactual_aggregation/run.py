"""
Compare alternative scoring rules on trials with multiple verifier results.
"""

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from messier.io import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "analysis" / "outputs" / "counterfactual_aggregation"
NEAR_PASS_THRESHOLD = 0.8
BENCHMARKS = {"harveyai-lab", "tau2-bench", "theagentcompany"}


def main() -> None:
    """
    Recompute and summarize the counterfactual trial results.
    """
    # first identify tasks with more than one verifier
    selected_tasks = set()
    for task in load_jsonl(DATA / "tasks.jsonl"):
        if task["benchmark"] in BENCHMARKS and task["n_verifiers"] > 1:
            selected_tasks.add((task["benchmark"], task["task_id"]))

    verifier_ids_by_task = defaultdict(set)
    for verifier in load_jsonl(DATA / "verifiers.jsonl"):
        task = (verifier["benchmark"], verifier["task_id"])
        if task in selected_tasks:
            verifier_ids_by_task[task].add(verifier["verifier_id"])

    # collect the final result and each verifier result for the same trial
    verifier_results = defaultdict(dict)
    trial_results = {}
    for record in load_jsonl(DATA / "records.jsonl"):
        task = (record["benchmark"], record["task_id"])
        if task not in selected_tasks or record["result"] is None:
            continue

        key = (*task, record["agent_id"], record["trial"])

        if record["record_type"] == "trial_result":
            trial_results[key] = int(record["result"])
        elif record["record_type"] == "verifier_result":
            if not 0 <= record["result"] <= 1:
                raise ValueError("counterfactual aggregation requires verifier results between 0 and 1")

            if record["verifier_id"] in verifier_ids_by_task[task]:
                verifier_results[key][record["verifier_id"]] = int(record["result"] == 1)

    # include a trial only when it has a final result and one stored result for every verifier defined for its task
    rows = []
    for (benchmark, task_id, agent_id, trial), results in verifier_results.items():
        key = (benchmark, task_id, agent_id, trial)
        task = (benchmark, task_id)
        if key not in trial_results or set(results) != verifier_ids_by_task[task]:
            continue

        values = list(results.values())
        soft = float(np.mean(values))
        all_pass = int(all(values))
        rows.append(
            {
                "benchmark": benchmark,
                "task_id": task_id,
                "agent_id": agent_id,
                "trial": trial,
                "trial_result": trial_results[key],
                "all_pass": all_pass,
                "soft": soft,
                "threshold_50": int(soft >= 0.5),
                "near_pass": int(not all_pass and soft >= NEAR_PASS_THRESHOLD),
            }
        )

    rescored = pd.DataFrame(rows)
    if rescored.empty:
        raise RuntimeError("no complete multi-verifier trials found")

    # average each scoring rule across trials, then compare agent ordering
    summary = (
        rescored.groupby("benchmark")
        .agg(
            n_trials=("all_pass", "size"),
            n_agents=("agent_id", "nunique"),
            trial_result=("trial_result", "mean"),
            all_pass=("all_pass", "mean"),
            soft=("soft", "mean"),
            threshold_50=("threshold_50", "mean"),
            near_pass=("near_pass", "mean"),
        )
        .reset_index()
    )
    correlations = []
    for benchmark, group in rescored.groupby("benchmark"):
        by_agent = group.groupby("agent_id")[["all_pass", "soft"]].mean()
        rho = spearmanr(by_agent["all_pass"], by_agent["soft"]).statistic
        correlations.append({"benchmark": benchmark, "rho_soft_vs_allpass": float(rho)})

    # save files
    summary = summary.merge(pd.DataFrame(correlations), on="benchmark")
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)

    # print
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
