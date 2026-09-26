"""
Compare MESSIER model abilities with Epoch ECI rankings.
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from messier import IRT, tasks
from messier.io import PROCESSED, load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "outputs" / "epoch_eci"
IRT_DIR = REPO_ROOT / "analysis" / "outputs" / "irt"
IRT_2PL_DIR = REPO_ROOT / "analysis" / "outputs" / "irt_2pl"
ECI_DIR = REPO_ROOT / "data" / "raw" / "epoch"

# each comparison defines its label, Epoch file, and optional SOC filter
DOMAINS = [
    ("general", "General", ECI_DIR / "eci_scores.csv", None),
    ("coding", "Coding (vs. SWE-ECI)", ECI_DIR / "eci_scores_coding.csv", "15-1200"),
    ("math", "Math (vs. Math-ECI)", ECI_DIR / "eci_scores_math.csv", "15-2000"),
]


def normalize(name: str) -> str:
    """
    Normalize a model name for cross-source matching.

    :param name: Source model name.
    :return: Normalized model name.
    """
    if not name:
        return ""

    normalized = str(name).strip().lower()
    normalized = re.sub(r"\s*\([^)]*\)", "", normalized)
    for ch in " /_.:":
        normalized = normalized.replace(ch, "-")

    normalized = re.sub(r"-?(\d{8}|\d{4}-\d{2}-\d{2})$", "", normalized)
    normalized = re.sub(r"-(instruct|chat|base|preview|exp)$", "", normalized)
    normalized = re.sub(r"\b(claude|gpt|gemini|llama|qwen|kimi|mistral|grok|deepseek|phi|gemma)(\d)", r"\1-\2", normalized)
    normalized = re.sub(r"(\d)(opus|sonnet|haiku|mini|nano|flash|pro|turbo)\b", r"\1-\2", normalized)
    return re.sub(r"-+", "-", normalized).strip("-")


def keep_best_scaffold(trials: pd.DataFrame) -> pd.DataFrame:
    """
    For each model, scaffold, benchmark, and task, results are first averaged
    over the trial index. For each model, scaffold, and benchmark, those task
    scores are then averaged over task IDs. The scaffold with the highest
    benchmark score is retained for that model and benchmark.

    :param trials: Trial results with model and scaffold identifiers.
    :return: Trial results for the selected scaffold of each model and benchmark.
    """
    trials = trials.copy()
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
    selected = trials.merge(scaffold_results[keys], on=keys)
    # the paper fits model ability after selecting one scaffold per benchmark.
    selected["agent_id"] = selected["agent_model"]
    return selected.reset_index(drop=True)


def drop_constant_tasks(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Remove tasks solved by every model or by no model.

    :param trials: Trial results to filter.
    :return: Trial results from tasks with both successes and failures.
    """
    task_key = trials["benchmark"] + "::" + trials["task_id"].astype(str)
    n_results = trials.groupby(task_key)["result"].nunique()
    return trials[task_key.isin(n_results[n_results > 1].index)].reset_index(drop=True)


def prepare_trials() -> pd.DataFrame:
    """
    Load and prepare the trial results used by the paper's IRT fit.

    :return: Selected model results with SOC labels attached.
    """
    task_table = tasks(source="local")
    soc = task_table[["benchmark", "task_id", "soc_code"]].rename(columns={"soc_code": "soc"})
    parent_keys = set(map(tuple, soc[["benchmark", "task_id"]].itertuples(index=False, name=None)))

    # stream only the fields needed by IRT to keep memory bounded
    columns = ("agent_id", "agent_model", "agent_scaffold", "benchmark", "task_id", "result")
    rows = []
    for record in load_jsonl(PROCESSED / "records.jsonl"):
        if (record["benchmark"], record["task_id"]) not in parent_keys:
            continue
        if record.get("record_type") != "trial_result":
            continue
        if record.get("result") is None:
            continue
        if record.get("agent_model") in {None, "multiple", "undisclosed"}:
            continue

        rows.append({column: record.get(column) for column in columns})

    trials = pd.DataFrame(rows)
    trials = drop_constant_tasks(keep_best_scaffold(trials))
    trials = trials.merge(soc, on=["benchmark", "task_id"], how="left")
    return trials


def fit_irt(trials: pd.DataFrame, epochs: int = 2000) -> IRT:
    """
    Fit and save the paper's model-level 1PL and 2PL models.

    :param trials: Prepared model-level trial results.
    :param epochs: Number of optimization epochs.
    :return: Fitted 1PL model.
    """
    n_tasks = len(trials[["benchmark", "task_id"]].drop_duplicates())
    # print
    print(
        f"[step 1] fitting IRT on {len(trials):,} trial results, "
        f"{trials['agent_id'].nunique():,} models, and "
        f"{n_tasks:,} tasks"
    )
    irt = IRT(model_type="1pl", epochs=epochs).fit(trials)

    # save files
    IRT_DIR.mkdir(parents=True, exist_ok=True)
    irt.save(IRT_DIR)

    irt_2pl = IRT(model_type="2pl", epochs=epochs).fit(trials)
    IRT_2PL_DIR.mkdir(parents=True, exist_ok=True)
    irt_2pl.save(IRT_2PL_DIR)
    rho = spearmanr(irt.theta, irt_2pl.theta).statistic
    print(f"  1PL vs 2PL model ordering: rho = {rho:.3f}")
    return irt


def domain_theta(irt: IRT, trials: pd.DataFrame, soc_filter: str | None) -> pd.DataFrame:
    """
    Estimate model ability for one SOC domain.

    :param irt: Full-corpus fitted IRT model.
    :param trials: Prepared trial results with SOC labels.
    :param soc_filter: SOC code to retain, or ``None`` for the full fit.
    :return: Model ability estimates.
    """
    if soc_filter is None:
        return pd.DataFrame({"agent_id": irt.agents, "theta": irt.theta})
    sub = trials[trials["soc"] == soc_filter]
    print(
        f"  refit on {len(sub):,} trial results across {sub['task_id'].nunique():,} tasks "
        f"({sub['benchmark'].nunique()} benches) where soc_code == {soc_filter!r}"
    )
    return irt.refit_theta(sub, epochs=irt.config.epochs)


def join_against_epoch(theta: pd.DataFrame, eci_path: Path) -> pd.DataFrame:
    """
    Match MESSIER model abilities with Epoch ECI scores.

    :param theta: MESSIER model ability estimates.
    :param eci_path: Epoch ECI score file.
    :return: Matched MESSIER and Epoch rows.
    """
    messier = theta.assign(key=theta["agent_id"].map(normalize))
    messier = messier.sort_values("theta", ascending=False).drop_duplicates("key").reset_index(drop=True)
    epoch = pd.read_csv(eci_path)
    epoch["key"] = epoch["model"].map(normalize)
    epoch = epoch.sort_values("eci", ascending=False).drop_duplicates("key").reset_index(drop=True)
    return messier.merge(epoch, on="key", how="inner")


def bootstrap_rho(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """
    Estimate Spearman correlation and its bootstrap interval.

    :param x: First set of values.
    :param y: Second set of values.
    :param n_boot: Number of bootstrap resamples.
    :param seed: Random seed.
    :return: Correlation and lower and upper confidence limits.
    """
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, n, n)
        rhos[k] = spearmanr(x[i], y[i]).statistic

    lo, hi = np.nanpercentile(rhos, [2.5, 97.5])
    return rho, float(lo), float(hi)


def pairwise_concordance(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """
    Measure agreement between pairwise model orderings.

    :param x: First set of model scores.
    :param y: Second set of model scores.
    :return: Concordance rate and number of non-tied pairs.
    """
    n = len(x)
    if n < 2:
        return float("nan"), 0

    iu = np.triu_indices(n, k=1)
    dx = (x[:, None] - x[None, :])[iu]
    dy = (y[:, None] - y[None, :])[iu]
    mask = (dx != 0) & (dy != 0)
    pairs = int(mask.sum())
    if pairs == 0:
        return float("nan"), 0

    agree = int((np.sign(dx[mask]) == np.sign(dy[mask])).sum())
    return agree / pairs, pairs


def make_figure(rows: list[dict], path: Path) -> None:
    """
    Plot the three MESSIER and Epoch ECI comparisons used in the paper.

    :param rows: Matched values and summary statistics for each comparison.
    :param path: Destination for the PDF figure and its PNG copy.
    """
    # plot
    blue = "#3a55d6"
    beige = "#c8c2b2"
    ink = "#1c1d20"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    for axis, row in zip(axes, rows):
        x = row["matched"]["theta"].to_numpy()
        y = row["matched"]["eci"].to_numpy()
        axis.set_axisbelow(True)
        axis.grid(True, color=beige, linewidth=0.5, zorder=0)
        if len(x) >= 3:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 30)
            axis.plot(xs, slope * xs + intercept, color=ink, lw=0.9, alpha=0.45, zorder=2)

        axis.scatter(
            x,
            y,
            s=28,
            color=blue,
            edgecolors=ink,
            linewidths=0.6,
            alpha=0.95,
            zorder=3
        )
        for side in ("left", "bottom"):
            axis.spines[side].set_linewidth(0.6)
            axis.spines[side].set_color(ink)

        axis.tick_params(axis="both", length=2.5, width=0.5)
        axis.set_title(row["title"], fontsize=11, fontweight="bold", color=ink)
        axis.set_xlabel(r"MESSIER $\theta$ (logit)")
        axis.set_ylabel("Epoch ECI")
        axis.text(
            0.04,
            0.96,
            f"ρ = {row['rho']:.2f}  [{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]\n"
            f"concordance = {row['concordance']:.0%}\n"
            f"n = {row['n']}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontfamily="monospace",
            fontsize=8.5,
            color=ink,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=ink,
                linewidth=0.5,
            ),
        )
    # save files
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Fit IRT, compare its abilities with Epoch ECI, and write the results.
    """

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = prepare_trials()
    irt = fit_irt(trials)

    # compare the full fit and two fixed-difficulty SOC subset fits
    rows = []
    for slug, title, eci_path, soc_filter in DOMAINS:
        # print
        print(f"\n[{slug}] {title}")
        theta = domain_theta(irt, trials, soc_filter)
        matched = join_against_epoch(theta, eci_path)
        if len(matched) < 3:
            print(f"  skipped: only {len(matched)} models matched")
            continue

        messier_theta = matched["theta"].to_numpy()
        epoch_eci = matched["eci"].to_numpy()
        rho, ci_low, ci_high = bootstrap_rho(messier_theta, epoch_eci)
        concordance, n_pairs = pairwise_concordance(messier_theta, epoch_eci)
        print(
            f"  ρ = {rho:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]   "
            f"concordance = {concordance:.1%} ({n_pairs} pairs)   "
            f"n = {len(matched)}"
        )

        matched[["model", "agent_id", "theta", "eci", "key"]].to_csv(OUT_DIR / f"matched_{slug}.csv", index=False)
        rows.append(
            {
                "slug": slug,
                "title": title,
                "rho": rho,
                "ci_lo": ci_low,
                "ci_hi": ci_high,
                "concordance": concordance,
                "n_pairs": n_pairs,
                "n": len(matched),
                "soc_filter": soc_filter,
                "matched": matched,
            }
        )

    # save files
    summary_cols = [
        "slug",
        "rho",
        "ci_lo",
        "ci_hi",
        "concordance",
        "n_pairs",
        "n",
        "soc_filter",
    ]
    pd.DataFrame([{key: row[key] for key in summary_cols} for row in rows]).to_csv(OUT_DIR / "results.csv", index=False)
    summary = [{key: row[key] for key in ["title"] + summary_cols} for row in rows]
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(rows, OUT_DIR / "fig_eci.pdf")
    print(f"\nwrote: {OUT_DIR}/{{results.csv, summary.json, fig_eci.pdf, matched_*.csv}}")


if __name__ == "__main__":
    main()
