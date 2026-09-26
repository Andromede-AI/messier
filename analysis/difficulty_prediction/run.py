"""
Evaluate task-difficulty prediction from text embeddings.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
EMBED = "BAAI/bge-base-en-v1.5"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
FOLDS = 5
SEED = 42


def load() -> pd.DataFrame:
    """
    Join task text with the difficulty values from the main IRT fit.

    :return: One row per fitted task with both prediction targets and inputs.
    """
    tasks = pd.read_json(ROOT / "data/processed/tasks.jsonl", lines=True)
    state = json.loads((ROOT / "analysis/outputs/irt/state.json").read_text())
    beta = dict(zip(state.get("tasks", state.get("items", [])), state["beta"]))
    keys = tasks["benchmark"].astype(str) + "::" + tasks["task_id"].astype(str)
    tasks["beta"] = keys.map(beta)
    df = tasks.dropna(subset=["beta"]).query("not is_unrecorded").reset_index(drop=True)
    df["beta_z"] = df.groupby("benchmark")["beta"].transform(
        lambda x: (x - x.mean()) / (x.std() or 1.0)
    )

    # derive the action description from the stored action space
    def action_text(metadata):
        if not isinstance(metadata, dict):
            return ""

        action_space = metadata.get("action_space") or {}
        description = action_space.get("description") or ""
        extras = {
            key: value
            for key, value in action_space.items()
            if key not in ("type", "description")
        }
        if not extras:
            return description

        return f"{description}\n\n{json.dumps(extras, default=str)}"

    df["action_description"] = df["environment_metadata"].map(action_text)

    def as_text(value):
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        return json.dumps(value, default=str)

    # truncate before tokenization while preserving BGE's 512-token input
    df["text_instruction"] = df["task_description"].map(as_text).str.slice(stop=8192)
    df["text_full"] = (
        df["task_description"].map(as_text)
        + "\n\n"
        + df["environment_description"].map(as_text)
        + "\n\n"
        + df["action_description"].map(as_text)
        + "\n\n"
        + df["gold_answer"].map(as_text)
    ).str.slice(stop=8192)

    return df


def predict(features: np.ndarray, target: np.ndarray, benchmarks: np.ndarray, estimator: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce out-of-fold predictions from Ridge or the benchmark-mean baseline.

    :param features: Task embedding matrix.
    :param target: Difficulty value for each task.
    :param benchmarks: Benchmark name for each task.
    :param estimator: ``ridge`` or ``bench_mean``.
    :return: Prediction and fold assignment for every task.
    """
    predictions = np.full_like(target, np.nan, dtype=float)
    folds = np.full(len(target), -1, dtype=int)
    benchmark_series = pd.Series(benchmarks)
    splitter = KFold(FOLDS, shuffle=True, random_state=SEED)

    for fold, (train, test) in enumerate(splitter.split(np.arange(len(target)))):
        folds[test] = fold
        if estimator == "ridge":
            if np.unique(target[train]).size < 2:
                continue

            model = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), RidgeCV(alphas=ALPHAS))
            predictions[test] = model.fit(features[train], target[train]).predict(features[test])

        else:
            means = (
                pd.Series(target[train])
                .groupby(benchmark_series.iloc[train].values)
                .mean()
                .to_dict()
            )
            overall_mean = float(target[train].mean())
            predictions[test] = [means.get(benchmark, overall_mean) for benchmark in benchmark_series.iloc[test]]

    return predictions, folds


def spearman_per_fold(target: np.ndarray, predictions: np.ndarray, folds: np.ndarray) -> tuple[float, float]:
    """
    Summarize rank correlation across the cross-validation folds.

    :param target: Observed task difficulty values.
    :param predictions: Out-of-fold predictions.
    :param folds: Fold assignment for each task.
    :return: Mean and sample standard deviation of fold correlations.
    """
    rhos = []
    for fold in range(FOLDS):
        mask = (folds == fold) & ~np.isnan(predictions)
        if mask.sum() < 2 or np.unique(predictions[mask]).size < 2:
            continue

        rho = spearmanr(target[mask], predictions[mask]).statistic
        rhos.append(float(rho))

    if not rhos:
        return float("nan"), float("nan")

    return float(np.mean(rhos)), float(np.std(rhos, ddof=1))


def main() -> None:
    """
    Embed both text representations and evaluate both difficulty targets.
    """
    task_table = load()
    print(f"{len(task_table)} tasks across {task_table.benchmark.nunique()} benchmarks")

    encoder = SentenceTransformer(EMBED)
    benchmarks = task_table["benchmark"].to_numpy()
    targets = {
        "beta_raw": task_table["beta"].to_numpy(float),
        "beta_z": task_table["beta_z"].to_numpy(float),
    }

    # print
    print(f"\n{'target':12s}  {'features':10s}  {'kf-model':>16s}  {'kf-base':>16s}")
    for feature_name, column in (("instruction", "text_instruction"), ("all", "text_full")):
        features = np.asarray(
            encoder.encode(
                task_table[column].tolist(),
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        for name, target in targets.items():
            cells = []
            for estimator in ("ridge", "bench_mean"):
                predictions, folds = predict(features, target, benchmarks, estimator)
                mean_rho, std_rho = spearman_per_fold(target, predictions, folds)
                cells.append(f"{mean_rho:+.3f} ± {std_rho:.3f}")

            print(f"{name:12s}  {feature_name:10s}  {cells[0]:>16s}  {cells[1]:>16s}")


if __name__ == "__main__":
    main()
