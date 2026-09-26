from pathlib import Path
import pandas as pd
from eci import compute_eci_scores, fit_eci_model, load_benchmark_data

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "epoch"
RAW_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_CSV_URL = "https://epoch.ai/data/eci_benchmarks.csv"


def fit_subset(df, anchor_benchmark, bootstrap_samples=100):
    """
    Fit Epoch's ECI model to one selected set of benchmarks.

    :param df: Epoch model-by-benchmark result table.
    :param anchor_benchmark: Benchmark that fixes the ECI scale.
    :param bootstrap_samples: Bootstrap samples passed to the ECI package.
    :return: ECI scores and available uncertainty columns for each model.
    """
    subset = df.copy()
    n_models = subset["model"].nunique()
    print(
        f"  refitting on {len(subset)} (model, benchmark) rows across "
        f"{n_models} models  anchor={anchor_benchmark!r}"
    )
    if anchor_benchmark not in subset["benchmark"].unique():
        raise ValueError(f"anchor {anchor_benchmark!r} missing from subset")

    model_fit, benchmark_fit = fit_eci_model(subset, anchor_benchmark=anchor_benchmark, bootstrap_samples=bootstrap_samples)

    eci_df, _ = compute_eci_scores(model_fit, benchmark_fit)
    model_col = "Model" if "Model" in eci_df.columns else "model"
    keep = [
        c
        for c in [
            model_col,
            "eci",
            "eci_ci_low",
            "eci_ci_high",
            "capability",
            "capability_se",
            "capability_ci_low",
            "capability_ci_high",
        ]
        if c in eci_df.columns
    ]
    return eci_df[keep].rename(columns={model_col: "model"})


def main():
    # print progress
    print(f"Loading {BENCHMARK_CSV_URL} ...")
    df = load_benchmark_data(BENCHMARK_CSV_URL)
    print(f"Loaded {len(df):,} (model, benchmark) rows.")

    print("\n=== General ECI (full Epoch substrate) ===")
    out_general = fit_subset(df, "Winogrande")

    # release dates apply to the general output
    raw = pd.read_csv(BENCHMARK_CSV_URL)
    dates = (
        raw.dropna(subset=["date"])
        .groupby("model", as_index=False)["date"]
        .min()
        .rename(columns={"date": "release_date"})
    )

    out_general = out_general.merge(dates, on="model", how="left")

    print("\n=== Coding ECI (is_coding=True subset) ===")
    coding_benchmarks = df[df["is_coding"]]["benchmark"].unique()
    print(f"  coding benchmarks ({len(coding_benchmarks)}): {sorted(coding_benchmarks)}")

    out_coding = fit_subset(df[df["is_coding"]], "SWE-Bench verified")

    print("\n=== Math ECI (is_math=True subset) ===")
    math_benchmarks = df[df["is_math"]]["benchmark"].unique()
    print(f"  math benchmarks ({len(math_benchmarks)}): {sorted(math_benchmarks)}")
    out_math = fit_subset(df[df["is_math"]], "MATH level 5")

    # save files
    out_general.to_csv(RAW_DIR / "eci_scores.csv", index=False)
    out_coding.to_csv(RAW_DIR / "eci_scores_coding.csv", index=False)
    out_math.to_csv(RAW_DIR / "eci_scores_math.csv", index=False)
    # print
    print(f"\nWrote {RAW_DIR}/eci_scores.csv ({len(out_general)} rows)")
    print(f"Wrote {RAW_DIR}/eci_scores_coding.csv ({len(out_coding)} rows)")
    print(f"Wrote {RAW_DIR}/eci_scores_math.csv ({len(out_math)} rows)")


if __name__ == "__main__":
    main()
