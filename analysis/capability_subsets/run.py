"""
Analysis for project page figs and tables.
"""

import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analysis.epoch_eci.run import join_against_epoch, prepare_trials
from messier import IRT, tasks
from messier.io import PROCESSED, load_jsonl

ROOT = Path(__file__).resolve().parents[2]
IRT_DIR = ROOT / "analysis" / "outputs" / "irt"
EPOCH_DIR = ROOT / "data" / "raw" / "epoch"
OUT_DIR = ROOT / "analysis" / "outputs" / "capability_subsets"
PAGE_DATA = ROOT / "page" / "static" / "data" / "results.json"
FRONTIER_DATA = ROOT / "analysis" / "outputs" / "frontier_progress" / "frontier.csv"
MIN_TASKS_PER_MODEL = 10
MIN_BENCHMARK_GROUPS_PER_MODEL = 2

BENCHMARK_GROUPS = ["programming", "enterprise", "research", "gui", "function_calling"]

ECI_COMPARISONS = {
    "general": ("General", EPOCH_DIR / "eci_scores.csv", None),
    "coding":  ("Coding", EPOCH_DIR / "eci_scores_coding.csv", "15-1200"),
    "math":    ("Mathematics", EPOCH_DIR / "eci_scores_math.csv", "15-2000"),
}

INDUSTRIES = {
    "professional": ("Professional, Scientific, and Technical Services", "54"),
    "finance":      ("Finance and Insurance", "52"),
    "healthcare":   ("Health Care and Social Assistance", "62"),
}


def main() -> None:
    """
    Recompute every distribution and comparison shown on the project page.
    """
    trials = prepare_trials()
    trials["task_key"] = trials["benchmark"] + "::" + trials["task_id"].astype(str)
    irt = IRT.load(IRT_DIR)
    overall = pd.DataFrame({"agent_id": irt.agents, "overall_theta": irt.theta})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # summarize the data represented in the page figures
    coverage = trials.groupby(["agent_id", "benchmark"])["task_key"].nunique().unstack(fill_value=0)
    coverage = coverage.reindex(index=irt.agents, fill_value=0)
    tasks_per_benchmark = trials.groupby("benchmark")["task_key"].nunique()
    benchmark_order = (coverage > 0).sum().sort_values(ascending=False).index.tolist()
    model_order = pd.DataFrame(
        {"benchmarks": (coverage > 0).sum(axis=1), "tasks": coverage.sum(axis=1)}
    ).sort_values(["benchmarks", "tasks"], ascending=False).index.tolist()
    coverage = coverage.loc[model_order, benchmark_order]

    task_rows = list(load_jsonl(PROCESSED / "tasks.jsonl"))
    benchmark_group = {row["benchmark"]: row["benchmark_group"] for row in task_rows}
    tasks_by_group = Counter(row["benchmark_group"] for row in task_rows)
    benchmarks_by_group = {
        group: len({row["benchmark"] for row in task_rows if row["benchmark_group"] == group})
        for group in BENCHMARK_GROUPS
    }
    scoring_rules = Counter(row["scoring_rule"] for row in task_rows)
    verifier_types = Counter(row["verifier"]["type"] for row in load_jsonl(PROCESSED / "verifiers.jsonl"))

    agent_ids = set()
    model_names = set()
    scaffold_names = set()
    for record in load_jsonl(PROCESSED / "records.jsonl"):
        if record.get("agent_id"):
            agent_ids.add(record["agent_id"])
        if record.get("agent_model"):
            model_names.add(record["agent_model"])
        if record.get("agent_scaffold"):
            scaffold_names.add(record["agent_scaffold"])

    # combine benchmark frontiers into benchmark-group summaries
    frontier = pd.read_csv(FRONTIER_DATA, index_col=0)
    frontier_results = []
    for group in BENCHMARK_GROUPS:
        group_benchmarks = [benchmark for benchmark in frontier.index if benchmark_group.get(benchmark) == group]
        values = frontier.loc[group_benchmarks]
        means = values.mean(axis=0, skipna=True)
        counts = values.count(axis=0)
        standard_errors = values.std(axis=0, skipna=True) / np.sqrt(counts)
        points = []
        for quarter in frontier.columns:
            if pd.isna(means[quarter]):
                continue
            points.append(
                {
                    "quarter": quarter,
                    "mean": float(means[quarter]),
                    "se": float(standard_errors[quarter]) if counts[quarter] > 1 else 0.0,
                }
            )
        frontier_results.append({"group": group, "points": points})

    # compare overall, coding, and mathematics ability with Epoch ECI
    eci_results = []
    eci_summaries = []
    for slug, (title, eci_path, soc_code) in ECI_COMPARISONS.items():
        if soc_code is None:
            estimates = overall.rename(columns={"overall_theta": "theta"})
        else:
            estimates = irt.refit_theta(trials[trials["soc"] == soc_code], epochs=irt.config.epochs)

        matched = join_against_epoch(estimates, eci_path)
        rho = float(spearmanr(matched["theta"], matched["eci"]).statistic)
        matched[["model", "agent_id", "theta", "eci"]].to_csv(OUT_DIR / f"eci_{slug}.csv", index=False)
        eci_summaries.append(
            {"slug": slug, "title": title, "rho": rho, "n_models": len(matched), "soc_code": soc_code}
        )
        eci_results.append(
            {
                "slug": slug,
                "title": title,
                "rho": rho,
                "n": len(matched),
                "points": matched[["model", "theta", "eci"]].to_dict(orient="records"),
            }
        )

    # repeat the fixed-difficulty fit for each selected industry
    labels = tasks(source="local")[["benchmark", "task_id", "benchmark_group", "naics_code"]]
    trials = trials.drop(columns="soc").merge(labels, on=["benchmark", "task_id"], how="left")
    industry_results = []
    industry_summaries = []
    for slug, (title, naics_code) in INDUSTRIES.items():
        subset = trials[trials["naics_code"] == naics_code].reset_index(drop=True)
        estimates = irt.refit_theta(subset, epochs=irt.config.epochs)
        task_counts = subset.groupby("agent_id")["task_key"].nunique().rename("n_tasks")
        group_counts = subset.groupby("agent_id")["benchmark_group"].nunique().rename("n_benchmark_groups")
        estimates = estimates.merge(overall, on="agent_id").merge(task_counts, on="agent_id").merge(group_counts, on="agent_id")
        shown = estimates[
            (estimates["n_tasks"] >= MIN_TASKS_PER_MODEL)
            & (estimates["n_benchmark_groups"] >= MIN_BENCHMARK_GROUPS_PER_MODEL)
        ].copy()
        rho = float(spearmanr(shown["overall_theta"], shown["theta"]).statistic)

        estimates.to_csv(OUT_DIR / f"industry_{slug}.csv", index=False)
        industry_summaries.append(
            {
                "slug": slug,
                "title": title,
                "naics_code": naics_code,
                "rho": rho,
                "n_trials": len(subset),
                "n_tasks": subset["task_key"].nunique(),
                "n_models": subset["agent_id"].nunique(),
                "n_models_shown": len(shown),
                "minimum_tasks_per_model": MIN_TASKS_PER_MODEL,
                "minimum_benchmark_groups_per_model": MIN_BENCHMARK_GROUPS_PER_MODEL,
            }
        )
        industry_results.append(
            {
                "slug": slug,
                "title": title,
                "rho": rho,
                "n": len(shown),
                "minimum_tasks_per_model": MIN_TASKS_PER_MODEL,
                "minimum_benchmark_groups_per_model": MIN_BENCHMARK_GROUPS_PER_MODEL,
                "points": shown.rename(columns={"theta": "industry_theta"})[
                    ["agent_id", "overall_theta", "industry_theta", "n_tasks", "n_benchmark_groups"]
                ].to_dict(orient="records"),
            }
        )

    # save files
    summaries = {"eci": eci_summaries, "industries": industry_summaries}
    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    page_results = {
        "composition": {
            "benchmark_groups": [
                {
                    "group": group,
                    "benchmarks": benchmarks_by_group[group],
                    "tasks": tasks_by_group[group],
                }
                for group in BENCHMARK_GROUPS
            ],
            "identities": {
                "models": len(model_names),
                "scaffolds": len(scaffold_names),
                "agents": len(agent_ids),
            },
            "scoring_rules": dict(scoring_rules),
            "verifier_types": dict(verifier_types),
            "frontier": frontier_results,
        },
        "irt": {
            "theta": irt.theta.tolist(),
            "beta": irt.beta.tolist(),
            "coverage": {
                "models": model_order,
                "benchmarks": benchmark_order,
                "benchmark_groups": [benchmark_group[benchmark] for benchmark in benchmark_order],
                "observed": coverage.to_numpy().tolist(),
                "tasks_per_benchmark": tasks_per_benchmark.loc[benchmark_order].tolist(),
            },
        },
        "eci": eci_results,
        "industries": industry_results,
    }
    PAGE_DATA.parent.mkdir(parents=True, exist_ok=True)
    PAGE_DATA.write_text(json.dumps(page_results, separators=(",", ":")) + "\n")
    # print
    print(f"wrote {OUT_DIR}")
    print(f"wrote {PAGE_DATA}")


if __name__ == "__main__":
    main()
