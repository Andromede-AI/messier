import json
from pathlib import Path
import numpy as np
import pandas as pd
from messier import IRT, tasks
from messier.io import PROCESSED, load_jsonl

OUT = Path(__file__).resolve().parents[1] / "outputs" / "human_time_calibration"
BENCHMARKS = ("hcast", "rebench", "swaa")
BOOTSTRAPS = 1_000
SEED = 42


def main() -> None:

    # retain METR tasks with a positive median human completion time
    task_table = tasks(source="local")
    timed = task_table[
        task_table["benchmark"].isin(BENCHMARKS)
        & task_table["human"].apply(lambda value: bool(value) and (value.get("minutes_median") or 0) > 0)
    ]
    timed_keys = set(zip(timed["benchmark"], timed["task_id"]))

    rows = []
    for record in load_jsonl(PROCESSED / "records.jsonl"):
        if record["record_type"] != "trial_result" or record["result"] is None:
            continue
        if record.get("agent_model") in {None, "multiple", "undisclosed"}:
            continue
        if (record["benchmark"], record["task_id"]) not in timed_keys:
            continue

        rows.append({key: record.get(key) for key in ("agent_model", "agent_scaffold", "benchmark", "task_id", "result")})

    # average trials within each task, then average tasks within each
    # model, scaffold, and benchmark before selecting the strongest scaffold
    trials = pd.DataFrame(rows)
    trials["agent_scaffold"] = trials["agent_scaffold"].fillna("")
    keys = ["agent_model", "benchmark", "agent_scaffold"]
    per_task = trials.groupby(keys + ["task_id"], as_index=False)["result"].mean()
    scaffold_results = (
        per_task.groupby(keys, as_index=False)
        .agg(mean_trial_result=("result", "mean"), n_tasks=("task_id", "nunique"))
        .sort_values(
            [
                "agent_model",
                "benchmark",
                "mean_trial_result",
                "n_tasks",
                "agent_scaffold",
            ],
            ascending=[True, True, False, False, True],
        )
        .drop_duplicates(["agent_model", "benchmark"])
    )
    trials = trials.merge(scaffold_results[keys], on=keys)

    # the BRIDGE comparison is model-level after scaffold selection
    trials["agent_id"] = trials["agent_model"]
    trials["task_key"] = trials["benchmark"] + "::" + trials["task_id"].astype(str)
    varying = trials.groupby("task_key")["result"].nunique()
    trials = trials[trials["task_key"].isin(varying[varying > 1].index)].reset_index(drop=True)

    # use 2PL task difficulty and its uncertainty (as in the original paper)
    irt = IRT(model_type="2pl", epochs=2_000).fit(trials)
    beta = dict(zip(irt.task_ids, irt.beta))
    beta_posterior_std = dict(zip(irt.task_ids, irt.beta_posterior_std))
    calibration = []
    for row in timed.to_dict("records"):
        task_key = f"{row['benchmark']}::{row['task_id']}"
        if task_key in beta:
            calibration.append(
                {
                    "beta": beta[task_key],
                    "beta_posterior_std": beta_posterior_std[task_key],
                    "log_minutes": np.log(row["human"]["minutes_median"]),
                }
            )

    calibration = pd.DataFrame(calibration)
    x = calibration["beta"].to_numpy()
    y = calibration["log_minutes"].to_numpy()
    variance = calibration["beta_posterior_std"].to_numpy() ** 2
    weights = 1 / np.clip(variance, 1e-6, None)

    def slope(indices: np.ndarray) -> float:
        """
        Fit the weighted log-minutes slope for selected task rows.
        """
        xi, yi, wi = x[indices], y[indices], weights[indices]
        x_mean = np.average(xi, weights=wi)
        y_mean = np.average(yi, weights=wi)
        numerator = np.sum(wi * (xi - x_mean) * (yi - y_mean))
        denominator = np.sum(wi * (xi - x_mean) ** 2)
        return float(numerator / denominator)

    # resample tasks and refit the slope for each bootstrap sample
    indices = np.arange(len(calibration))
    fitted_slope = slope(indices)
    rng = np.random.default_rng(SEED)
    bootstrap_slopes = [slope(rng.choice(indices, len(indices), replace=True)) for _ in range(BOOTSTRAPS)]
    ci_low, ci_high = np.percentile(bootstrap_slopes, [2.5, 97.5])
    summary = {
        "n_tasks": len(calibration),
        "slope": fitted_slope,
        "ci_95": [float(ci_low), float(ci_high)],
        "factor_per_beta": float(np.exp(fitted_slope)),
    }

    # save files
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # print
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
