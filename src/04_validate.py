"""
SynthGen - Step 4: Synthetic Data Validation
============================================
Validates whether synthetic thyroid clinical samples resemble real samples
using statistical tests, descriptive summaries, visual diagnostics, and a
professional HTML report.

Usage:
    python src/04_validate.py

Outputs:
    outputs/distribution_histograms.png - real vs synthetic histograms
    outputs/distribution_boxplots.png   - real vs synthetic box plots
    outputs/qq_plots.png                - Q-Q plots for real data
    outputs/validation_metrics.json     - machine-readable validation metrics
    outputs/validation_report.html      - judge-friendly validation report
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

# Configure writable plotting caches before importing Matplotlib/Seaborn.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


INPUT_CSV = Path("data/processed/synthetic_data.csv")
OUTPUT_DIR = Path("outputs")
HISTOGRAM_PNG = OUTPUT_DIR / "distribution_histograms.png"
BOXPLOTS_PNG = OUTPUT_DIR / "distribution_boxplots.png"
QQ_PNG = OUTPUT_DIR / "qq_plots.png"
HTML_REPORT = OUTPUT_DIR / "validation_report.html"
JSON_REPORT = OUTPUT_DIR / "validation_metrics.json"

NUMERIC_FEATURES = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
PLOT_FEATURES = ["TSH", "T4U", "FTI", "T3", "age", "TT4"]
QQ_FEATURES = ["TSH", "T4U", "FTI", "T3"]

REAL_SAMPLE_SIZE = 100
SYNTHETIC_SAMPLE_SIZE = 500
RANDOM_STATE = 42
KS_ALPHA = 0.05
DIFF_WARNING_THRESHOLD = 10.0


def log_info(message: str) -> None:
    """
    Print a standard informational log message.

    Args:
        message: Message to print after the [INFO] prefix.
    """
    print(f"[INFO] {message}")


def load_and_sample_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load synthetic_data.csv, split real and synthetic rows, and sample a
    representative subset for fast validation.

    Args:
        path: CSV file containing real and synthetic samples.

    Returns:
        Tuple of sampled real DataFrame and sampled synthetic DataFrame.

    Raises:
        FileNotFoundError: If the input CSV is missing.
        ValueError: If required columns are missing or either group is empty.
    """
    log_info("Loading synthetic data...")
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required_columns = ["is_synthetic", *NUMERIC_FEATURES]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    synthetic_mask = parse_bool_series(df["is_synthetic"])
    real_df = df.loc[~synthetic_mask, NUMERIC_FEATURES].copy()
    synthetic_df = df.loc[synthetic_mask, NUMERIC_FEATURES].copy()

    if real_df.empty or synthetic_df.empty:
        raise ValueError("Both real and synthetic samples are required for validation.")

    real_sample = real_df.sample(
        n=min(REAL_SAMPLE_SIZE, len(real_df)),
        random_state=RANDOM_STATE,
    )
    synthetic_sample = synthetic_df.sample(
        n=min(SYNTHETIC_SAMPLE_SIZE, len(synthetic_df)),
        random_state=RANDOM_STATE,
    )

    log_info("Real samples: %d, Synthetic samples: %d" % (len(real_sample), len(synthetic_sample)))
    return real_sample.reset_index(drop=True), synthetic_sample.reset_index(drop=True)


def parse_bool_series(series: pd.Series) -> pd.Series:
    """
    Parse boolean-like values from CSV into a robust boolean Series.

    Args:
        series: Column that may contain booleans or boolean-like strings.

    Returns:
        Boolean Series where True means the row is synthetic.
    """
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def run_statistical_validation(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run KS tests and descriptive statistic comparisons for every numeric
    feature.

    The Kolmogorov-Smirnov test compares two empirical distributions. Here the
    null hypothesis is that real and synthetic samples come from the same
    distribution. A p-value above 0.05 is treated as a pass.

    Args:
        real_df: Sampled real numeric data.
        synthetic_df: Sampled synthetic numeric data.

    Returns:
        JSON-serializable validation metrics.
    """
    log_info("Running statistical validation...")
    log_info("Kolmogorov-Smirnov tests:")

    features: dict[str, Any] = {}
    passed_tests = 0
    warnings: list[str] = []

    for feature in NUMERIC_FEATURES:
        real_values = real_df[feature].dropna()
        synthetic_values = synthetic_df[feature].dropna()

        if real_values.empty or synthetic_values.empty:
            status = "SKIP"
            metrics = {
                "status": status,
                "reason": "real or synthetic values are entirely missing",
            }
            log_info(f"  {feature:<8} skipped (missing values only)")
            features[feature] = metrics
            continue

        ks_statistic, p_value = stats.ks_2samp(real_values, synthetic_values)
        status = "PASS" if p_value > KS_ALPHA else "FAIL"
        if status == "PASS":
            passed_tests += 1

        real_stats = describe_series(real_values)
        synthetic_stats = describe_series(synthetic_values)
        diffs = {
            stat_name: percent_difference(
                real_stats[stat_name],
                synthetic_stats[stat_name],
            )
            for stat_name in ["mean", "std", "min", "max"]
        }
        feature_warnings = [
            f"{name} diff {value:.1f}%"
            for name, value in diffs.items()
            if value > DIFF_WARNING_THRESHOLD
        ]
        if feature_warnings:
            warnings.append(f"{feature}: {', '.join(feature_warnings)}")

        features[feature] = {
            "real_mean": real_stats["mean"],
            "real_std": real_stats["std"],
            "real_min": real_stats["min"],
            "real_max": real_stats["max"],
            "synthetic_mean": synthetic_stats["mean"],
            "synthetic_std": synthetic_stats["std"],
            "synthetic_min": synthetic_stats["min"],
            "synthetic_max": synthetic_stats["max"],
            "mean_diff_pct": diffs["mean"],
            "std_diff_pct": diffs["std"],
            "min_diff_pct": diffs["min"],
            "max_diff_pct": diffs["max"],
            "ks_statistic": float(ks_statistic),
            "p_value": float(p_value),
            "status": status,
            "warnings": feature_warnings,
        }

        mark = "✓" if status == "PASS" else "✗"
        log_info(
            f"  {feature:<8} p={p_value:.4f} {mark} {status} "
            f"(mean diff: {diffs['mean']:.1f}%)"
        )

    total_features = len(NUMERIC_FEATURES)
    pass_rate = (passed_tests / total_features) * 100
    log_info(
        f"Validation complete. {passed_tests}/{total_features} features passed "
        f"({pass_rate:.1f}%)"
    )

    if warnings:
        log_info("Warnings:")
        for warning in warnings:
            log_info(f"  {warning}")

    return {
        "summary": {
            "total_features": total_features,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "real_sample_size": int(len(real_df)),
            "synthetic_sample_size": int(len(synthetic_df)),
            "ks_alpha": KS_ALPHA,
            "diff_warning_threshold_pct": DIFF_WARNING_THRESHOLD,
        },
        "features": features,
        "warnings": warnings,
    }


def describe_series(series: pd.Series) -> dict[str, float]:
    """
    Calculate descriptive statistics for a numeric series.

    Args:
        series: Numeric values for one feature.

    Returns:
        Dictionary containing mean, standard deviation, minimum, and maximum.
    """
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def percent_difference(real_value: float, synthetic_value: float) -> float:
    """
    Calculate absolute percentage difference between real and synthetic stats.

    Args:
        real_value: Statistic from real data.
        synthetic_value: Statistic from synthetic data.

    Returns:
        Absolute percent difference.
    """
    denominator = abs(real_value)
    if denominator < 1e-12:
        return 0.0 if abs(synthetic_value) < 1e-12 else 100.0
    return float(abs(real_value - synthetic_value) / denominator * 100)


def set_plot_style() -> None:
    """
    Apply a consistent visual style for all validation graphics.
    """
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.9,
        rc={
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        },
    )


def create_histograms(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create real vs synthetic histograms with mean lines and KS p-values.

    Args:
        real_df: Sampled real numeric data.
        synthetic_df: Sampled synthetic numeric data.
        metrics: Validation metrics containing p-values.
        output_path: PNG destination path.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes_array = axes.ravel()
    real_color = "#4C78A8"
    synthetic_color = "#F58518"

    for ax, feature in zip(axes_array, PLOT_FEATURES):
        real_values = real_df[feature].dropna()
        synthetic_values = synthetic_df[feature].dropna()
        p_value = metrics["features"][feature].get("p_value")
        status = metrics["features"][feature].get("status")

        sns.histplot(
            real_values,
            bins=28,
            stat="density",
            color=real_color,
            alpha=0.45,
            label="Real",
            ax=ax,
        )
        sns.histplot(
            synthetic_values,
            bins=28,
            stat="density",
            color=synthetic_color,
            alpha=0.45,
            label="Synthetic",
            ax=ax,
        )

        ax.axvline(real_values.mean(), color=real_color, linestyle="--", linewidth=2)
        ax.axvline(synthetic_values.mean(), color=synthetic_color, linestyle="--", linewidth=2)
        title_p = "N/A" if p_value is None else f"{p_value:.3f}"
        ax.set_title(f"{feature} | KS p={title_p} | {status}", pad=10)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(frameon=True)

    fig.suptitle("Real vs Synthetic Distribution Histograms", fontsize=25, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_boxplots(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create box plots comparing real and synthetic feature spreads.

    Args:
        real_df: Sampled real numeric data.
        synthetic_df: Sampled synthetic numeric data.
        output_path: PNG destination path.
    """
    real_plot = real_df[PLOT_FEATURES].copy()
    real_plot["Source"] = "Real"
    synthetic_plot = synthetic_df[PLOT_FEATURES].copy()
    synthetic_plot["Source"] = "Synthetic"

    long_df = pd.concat([real_plot, synthetic_plot], ignore_index=True).melt(
        id_vars="Source",
        value_vars=PLOT_FEATURES,
        var_name="Feature",
        value_name="Value",
    ).dropna()

    fig, ax = plt.subplots(figsize=(18, 9))
    sns.boxplot(
        data=long_df,
        x="Feature",
        y="Value",
        hue="Source",
        palette={"Real": "#4C78A8", "Synthetic": "#F58518"},
        width=0.65,
        linewidth=1.3,
        fliersize=3,
        ax=ax,
    )
    ax.set_title("Real vs Synthetic Box Plots", pad=18)
    ax.set_xlabel("")
    ax.set_ylabel("Observed Value")
    ax.legend(title="", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_qq_plots(real_df: pd.DataFrame, output_path: Path) -> None:
    """
    Create Q-Q plots for real data to assess normality assumptions visually.

    Args:
        real_df: Sampled real numeric data.
        output_path: PNG destination path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_array = axes.ravel()

    for ax, feature in zip(axes_array, QQ_FEATURES):
        values = real_df[feature].dropna()
        stats.probplot(values, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor("#4C78A8")
        ax.get_lines()[0].set_markeredgecolor("#4C78A8")
        ax.get_lines()[0].set_alpha(0.75)
        ax.get_lines()[1].set_color("#E15759")
        ax.get_lines()[1].set_linewidth(2)
        ax.set_title(f"Real Data Q-Q Plot: {feature}", pad=10)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Ordered Values")

    fig.suptitle("Normal Q-Q Plots for Real Clinical Features", fontsize=23, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_json_report(metrics: dict[str, Any], output_path: Path) -> None:
    """
    Save validation metrics to JSON.

    Args:
        metrics: Validation metrics dictionary.
        output_path: JSON destination path.
    """
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    log_info(f"JSON metrics saved to: {output_path}")


def image_to_base64(path: Path) -> str:
    """
    Encode an image file as a base64 data URI for embedding in HTML.

    Args:
        path: Image path.

    Returns:
        Data URI string.
    """
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_html_report(metrics: dict[str, Any], output_path: Path) -> None:
    """
    Generate a polished HTML validation report with tables and embedded plots.

    Args:
        metrics: Validation metrics dictionary.
        output_path: HTML destination path.
    """
    summary = metrics["summary"]
    pass_rate = summary["pass_rate"]
    conclusion = (
        "Synthetic data is statistically valid and suitable for ML training"
        if pass_rate >= 80
        else "Synthetic data requires review before ML training"
    )
    conclusion_class = "pass" if pass_rate >= 80 else "fail"

    rows_html = "\n".join(
        format_feature_row(feature, feature_metrics)
        for feature, feature_metrics in metrics["features"].items()
    )
    warning_items = (
        "\n".join(f"<li>{escape_html(w)}</li>" for w in metrics["warnings"])
        if metrics["warnings"]
        else "<li>No descriptive-statistic warnings above the 10% threshold.</li>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Synthetic Clinical Data Validation Report</title>
  <style>
    :root {{
      --ink: #243142;
      --muted: #667085;
      --line: #d9e2ec;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --blue: #4c78a8;
      --orange: #f58518;
      --green: #157347;
      --red: #b42318;
      --amber: #b76e00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }}
    header {{
      background: #172033;
      color: white;
      padding: 42px 52px;
    }}
    header h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      color: #d7deea;
      max-width: 980px;
      font-size: 17px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 28px 56px;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 26px;
      margin-bottom: 24px;
      box-shadow: 0 8px 24px rgba(31, 42, 68, 0.06);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 23px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fbfcfe;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric strong {{
      display: block;
      font-size: 26px;
      margin-top: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 18px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    th, td {{
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      font-size: 14px;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{
      background: #edf2f7;
      color: #344054;
      font-weight: 700;
    }}
    tr:last-child td {{ border-bottom: none; }}
    .badge {{
      display: inline-block;
      min-width: 66px;
      text-align: center;
      padding: 4px 9px;
      border-radius: 999px;
      color: white;
      font-weight: 700;
      font-size: 12px;
    }}
    .badge.pass {{ background: var(--green); }}
    .badge.fail {{ background: var(--red); }}
    .badge.skip {{ background: var(--amber); }}
    .conclusion {{
      border-left: 5px solid var(--green);
      background: #f0f8f4;
      padding: 16px 18px;
      border-radius: 6px;
      font-weight: 700;
    }}
    .conclusion.fail {{
      border-left-color: var(--red);
      background: #fff4f2;
    }}
    .plot {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 14px;
      background: white;
    }}
    .muted {{ color: var(--muted); }}
    ul {{ margin-top: 10px; }}
    @media (max-width: 860px) {{
      header {{ padding: 30px 24px; }}
      main {{ padding: 24px 16px; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Synthetic Clinical Data Validation Report</h1>
    <p>
      Statistical and visual assessment of thyroid synthetic samples against
      real clinical records using Kolmogorov-Smirnov tests, descriptive
      statistic comparisons, histograms, box plots, and Q-Q diagnostics.
    </p>
  </header>
  <main>
    <section>
      <h2>Executive Summary</h2>
      <p>
        {pass_rate:.1f}% of features passed the KS test. A feature passes when
        p &gt; {KS_ALPHA}, meaning the validation sample does not provide strong
        evidence that real and synthetic distributions differ.
      </p>
      <div class="cards">
        <div class="metric"><span>Total Features</span><strong>{summary["total_features"]}</strong></div>
        <div class="metric"><span>Passed KS Tests</span><strong>{summary["passed_tests"]}</strong></div>
        <div class="metric"><span>Pass Rate</span><strong>{pass_rate:.1f}%</strong></div>
        <div class="metric"><span>Sample Sizes</span><strong>{summary["real_sample_size"]}/{summary["synthetic_sample_size"]}</strong></div>
      </div>
    </section>

    <section>
      <h2>Test Interpretation</h2>
      <p>
        The Kolmogorov-Smirnov test compares the full shape of two empirical
        distributions. The null hypothesis is that real and synthetic samples
        are drawn from the same distribution. Descriptive comparisons report
        mean, standard deviation, minimum, and maximum differences; differences
        above {DIFF_WARNING_THRESHOLD:.0f}% are listed as review warnings.
      </p>
    </section>

    <section>
      <h2>Feature-Level Results</h2>
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>KS Stat</th>
            <th>p-value</th>
            <th>% Diff Mean</th>
            <th>% Diff Std</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Warnings</h2>
      <ul>{warning_items}</ul>
    </section>

    <section>
      <h2>Distribution Histograms</h2>
      <p class="muted">Blue represents real samples; orange represents synthetic samples. Dashed lines mark group means.</p>
      <img class="plot" src="{image_to_base64(HISTOGRAM_PNG)}" alt="Distribution histograms">
    </section>

    <section>
      <h2>Box Plot Comparison</h2>
      <p class="muted">Box plots show central tendency, spread, and visible outliers for real and synthetic samples.</p>
      <img class="plot" src="{image_to_base64(BOXPLOTS_PNG)}" alt="Distribution box plots">
    </section>

    <section>
      <h2>Q-Q Normality Diagnostics</h2>
      <p class="muted">Q-Q plots show how real feature values compare with a theoretical normal distribution.</p>
      <img class="plot" src="{image_to_base64(QQ_PNG)}" alt="Q-Q plots">
    </section>

    <section>
      <h2>Conclusion</h2>
      <div class="conclusion {conclusion_class}">{escape_html(conclusion)}</div>
    </section>
  </main>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    log_info(f"Validation report saved to: {output_path}")


def format_feature_row(feature: str, metrics: dict[str, Any]) -> str:
    """
    Format one feature's metrics as an HTML table row.

    Args:
        feature: Feature name.
        metrics: Feature metrics dictionary.

    Returns:
        HTML table row.
    """
    status = metrics.get("status", "SKIP")
    badge_class = status.lower()
    ks_stat = format_optional_float(metrics.get("ks_statistic"), 4)
    p_value = format_optional_float(metrics.get("p_value"), 4)
    mean_diff = format_optional_float(metrics.get("mean_diff_pct"), 1, suffix="%")
    std_diff = format_optional_float(metrics.get("std_diff_pct"), 1, suffix="%")

    return f"""
          <tr>
            <td>{escape_html(feature)}</td>
            <td>{ks_stat}</td>
            <td>{p_value}</td>
            <td>{mean_diff}</td>
            <td>{std_diff}</td>
            <td><span class="badge {badge_class}">{escape_html(status)}</span></td>
          </tr>"""


def format_optional_float(value: Any, decimals: int, suffix: str = "") -> str:
    """
    Format floats for display, falling back to N/A for missing values.

    Args:
        value: Value to format.
        decimals: Number of decimal places.
        suffix: Optional suffix such as percent sign.

    Returns:
        Formatted string.
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}{suffix}"


def escape_html(value: Any) -> str:
    """
    Escape a value for safe insertion into simple HTML.

    Args:
        value: Value to escape.

    Returns:
        HTML-escaped string.
    """
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    """
    Run the complete synthetic data validation workflow.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_df, synthetic_df = load_and_sample_data(INPUT_CSV)
    metrics = run_statistical_validation(real_df, synthetic_df)

    log_info("Creating visualizations...")
    set_plot_style()
    create_histograms(real_df, synthetic_df, metrics, HISTOGRAM_PNG)
    create_boxplots(real_df, synthetic_df, BOXPLOTS_PNG)
    create_qq_plots(real_df, QQ_PNG)

    log_info("Generating HTML report...")
    generate_html_report(metrics, HTML_REPORT)
    save_json_report(metrics, JSON_REPORT)


if __name__ == "__main__":
    main()
