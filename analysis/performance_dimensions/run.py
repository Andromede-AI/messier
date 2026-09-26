"""
Measure shared dimensions in model performance across benchmarks and benchmark groups.
"""

import json
import os
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from messier import IRT
from messier.io import load_jsonl


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "analysis" / "outputs" / "performance_dimensions"

MIN_TASKS_PER_MODEL_BENCHMARK = 3
MIN_BENCHMARKS_PER_MODEL = 2
MIN_MODELS_PER_BENCHMARK = (5, 10, 15, 20)

# the auxiliary group fits use 500 epochs, whereas the main IRT fit uses 2,000
IRT_EPOCHS = int(os.environ.get("MESSIER_FACTOR_IRT_EPOCHS", "500"))
MIN_RESULTS_PER_MODEL_GROUP = 30
MIN_MODELS_PER_GROUP = 20
MIN_GROUPS_PER_MODEL = 2


def load_trials() -> pd.DataFrame:
    """
    Load the binary results and agent fields used by both analyses.

    :return: Scored results with a disclosed model identity.
    """
    columns = ("benchmark", "benchmark_group", "task_id", "agent_model", "agent_scaffold", "result")
    rows = []
    for record in load_jsonl(DATA / "records.jsonl"):
        if record["record_type"] != "trial_result" or record["result"] is None:
            continue
        if record.get("agent_model") in {None, "multiple", "undisclosed"}:
            continue

        rows.append({column: record.get(column) for column in columns})

    trials = pd.DataFrame(rows)
    if trials.empty:
        raise ValueError("no scored trial results found")

    trials["agent_scaffold"] = trials["agent_scaffold"].fillna("")
    return trials


def best_scaffold_results(trials: pd.DataFrame, min_tasks: int = 1) -> pd.DataFrame:
    """
    Select one scaffold for each model and benchmark.

    Results from repeated trials of the same task are averaged first. The task
    scores are then averaged over all task IDs for each model, scaffold, and
    benchmark. The scaffold with the highest benchmark score is selected.

    :param trials: Results with model and scaffold identifiers.
    :param min_tasks: Minimum number of tasks required for selection.
    :return: One selected scaffold row for each model and benchmark.
    """
    keys = ["benchmark", "agent_model", "agent_scaffold"]
    per_task = trials.groupby(keys + ["task_id"], as_index=False)["result"].mean()
    scaffold_results = per_task.groupby(keys, as_index=False).agg(
        mean_trial_result=("result", "mean"),
        n_tasks=("task_id", "nunique"),
    )
    scaffold_results = scaffold_results[scaffold_results["n_tasks"] >= min_tasks]
    return scaffold_results.sort_values(
        ["benchmark", "agent_model", "mean_trial_result", "n_tasks", "agent_scaffold"],
        ascending=[True, True, False, False, True],
    ).drop_duplicates(["benchmark", "agent_model"])


def filter_coverage(scores: pd.DataFrame, min_models: int) -> pd.DataFrame:
    """
    Retain benchmarks and models that satisfy both coverage requirements.

    Removing one benchmark can make a model ineligible and vice versa, so the
    two filters repeat until the matrix shape stops changing.

    :param scores: Model rows and benchmark columns.
    :param min_models: Minimum represented models per benchmark.
    :return: Matrix after the coverage filters stabilize.
    """
    previous_shape = None
    while scores.shape != previous_shape:
        previous_shape = scores.shape
        scores = scores.loc[:, scores.notna().sum() >= min_models]
        scores = scores.loc[scores.notna().sum(axis=1) >= MIN_BENCHMARKS_PER_MODEL]

    return scores.sort_index(axis=1)


def pca_variance(values: pd.DataFrame) -> pd.DataFrame:
    """
    Measure the variance explained by each principal component.

    Missing values are replaced with zero after each input column has been
    standardized by its caller, so zero represents that column's mean.

    :param values: Standardized model rows and evaluation columns.
    :return: Explained and cumulative variance by component.
    """
    matrix = SimpleImputer(strategy="constant", fill_value=0).fit_transform(values)
    explained = PCA().fit(matrix).explained_variance_ratio_
    return pd.DataFrame(
        {
            "pc": range(1, len(explained) + 1),
            "explained_variance": explained,
            "cumulative_variance": explained.cumsum(),
        }
    )


def benchmark_structure(trials: pd.DataFrame) -> list[dict]:
    """
    Apply PCA to model rankings across individual benchmarks.

    :param trials: Scored results with model and scaffold identifiers.
    :return: PCA summaries at each benchmark-coverage threshold.
    """
    selected = best_scaffold_results(trials, min_tasks=MIN_TASKS_PER_MODEL_BENCHMARK)
    scores = selected.pivot(index="agent_model", columns="benchmark", values="mean_trial_result")

    rows = []
    for min_models in MIN_MODELS_PER_BENCHMARK:
        covered = filter_coverage(scores, min_models)
        if covered.shape[0] < 2 or covered.shape[1] < 2:
            continue

        ranks = covered.rank(axis=0)
        standardized_ranks = (ranks - ranks.mean()) / ranks.std(ddof=0)
        variance = pca_variance(standardized_ranks)
        rows.append(
            {
                "min_models_per_benchmark": min_models,
                "n_models": int(covered.shape[0]),
                "n_benchmarks": int(covered.shape[1]),
                "pc1_variance": float(variance.loc[0, "explained_variance"]),
                "pcs_for_80pct_variance": int(variance.loc[variance["cumulative_variance"] >= 0.8, "pc"].iloc[0]),
            }
        )

    return rows


def fit_group(group: str, trials: pd.DataFrame) -> pd.DataFrame | None:
    """
    Fit and standardize model ability within one benchmark group.

    :param group: Benchmark-group identifier.
    :param trials: Results retained after scaffold selection.
    :return: Standardized model abilities, or ``None`` without enough models.
    """
    selected = trials[trials["benchmark_group"] == group].copy()
    counts = selected.groupby("agent_id").size()
    selected = selected[selected["agent_id"].isin(counts[counts >= MIN_RESULTS_PER_MODEL_GROUP].index)]

    selected["task_key"] = selected["benchmark"] + "::" + selected["task_id"].astype(str)
    varying = selected.groupby("task_key")["result"].nunique()
    selected = selected[selected["task_key"].isin(varying[varying > 1].index)]
    if selected["agent_id"].nunique() < MIN_MODELS_PER_GROUP:
        return None

    irt = IRT(model_type="1pl", epochs=IRT_EPOCHS).fit(selected)
    abilities = pd.DataFrame({"benchmark_group": group, "model": irt.agents, "theta": irt.theta})
    abilities["theta_z"] = (abilities["theta"] - abilities["theta"].mean()) / abilities["theta"].std(ddof=0)
    return abilities


def benchmark_group_structure(trials: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Apply PCA to model abilities fitted separately in each benchmark group.

    :param trials: Scored results with model and scaffold identifiers.
    :return: Summary and variance table for the benchmark-group PCA.
    """
    keys = ["benchmark", "agent_model", "agent_scaffold"]
    best = best_scaffold_results(trials)
    selected = trials.merge(best[keys], on=keys)
    selected["agent_id"] = selected["agent_model"]

    fits = []
    for group in sorted(selected["benchmark_group"].unique()):
        fitted = fit_group(group, selected)
        if fitted is not None:
            fits.append(fitted)

    abilities = pd.concat(fits, ignore_index=True)
    wide = abilities.pivot(index="model", columns="benchmark_group", values="theta_z")
    wide = wide.loc[wide.notna().sum(axis=1) >= MIN_GROUPS_PER_MODEL]
    if wide.shape[0] < 2 or wide.shape[1] < 2:
        raise ValueError("not enough model and benchmark-group coverage for PCA")

    variance = pca_variance(wide)
    summary = {
        "unit": "model",
        "n_models": int(wide.shape[0]),
        "n_benchmark_groups": int(wide.shape[1]),
        "benchmark_groups": sorted(wide.columns),
        "pc1_variance": float(variance.loc[0, "explained_variance"]),
        "pcs_for_80pct_variance": int(variance.loc[variance["cumulative_variance"] >= 0.8, "pc"].iloc[0]),
    }
    return summary, variance


def main() -> None:
    """
    Run the benchmark-level and benchmark-group PCA analyses.
    """
    trials = load_trials()
    benchmark_summary = benchmark_structure(trials)
    group_summary, group_variance = benchmark_group_structure(trials)
    summary = {
        "benchmark_scores": {"unit": "model", "sensitivity": benchmark_summary},
        "benchmark_group_abilities": group_summary,
    }

    # save files
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(benchmark_summary).to_csv(OUT / "benchmark_sensitivity.csv", index=False)
    group_variance.to_csv(OUT / "benchmark_group_pca.csv", index=False)

    # print
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
