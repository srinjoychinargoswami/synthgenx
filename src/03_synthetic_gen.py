"""
SynthGen - Step 3: Synthetic Clinical Sample Generation
=======================================================
Generates realistic synthetic thyroid clinical samples from clustered data.

The generator models each cluster separately:
    * Numeric features are sampled from a Gaussian Mixture Model (GMM), which
      can represent several local sub-populations inside a cluster instead of
      assuming one simple bell curve.
    * Categorical features are sampled from empirical per-cluster probability
      distributions, preserving observed category frequencies.
    * Synthetic numeric values are clipped to the real data min/max range, and
      samples requiring clipping are flagged for review.

Usage:
    python src/03_synthetic_gen.py

Outputs:
    data/processed/synthetic_data.csv       - real + synthetic samples
    outputs/synthetic_report.json           - validation and generation summary
    outputs/distribution_comparison.png     - real vs synthetic histograms
    outputs/synthetic_validation_plots.png  - validation diagnostics
    outputs/synthetic_gen_log.txt           - operation log
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep command-line output clean in restricted environments where user cache
# directories may not be writable.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture


INPUT_CSV = Path("data/processed/thyroid_clustered.csv")
CLUSTER_STATS_JSON = Path("outputs/cluster_stats.json")
OUTPUT_CSV = Path("data/processed/synthetic_data.csv")
OUTPUT_DIR = Path("outputs")
REPORT_JSON = OUTPUT_DIR / "synthetic_report.json"
DISTRIBUTION_PNG = OUTPUT_DIR / "distribution_comparison.png"
VALIDATION_PNG = OUTPUT_DIR / "synthetic_validation_plots.png"
LOG_FILE = OUTPUT_DIR / "synthetic_gen_log.txt"

RANDOM_STATE = 42
CLUSTER_COLUMN = "cluster"
CLUSTER_ALIASES = ["cluster", "cluster_id"]
SYNTHETIC_COLUMN = "is_synthetic"
EXTREME_FLAG_COLUMN = "synthetic_extreme_flag"
MISSING_TOKEN = "__MISSING__"
MEAN_DIFF_WARNING_THRESHOLD = 0.10
STD_DIFF_WARNING_THRESHOLD = 0.10
MAX_GMM_COMPONENTS = 4
MAX_RANGE_RESAMPLE_ATTEMPTS = 12


def _emit(
    message: str,
    logger: logging.Logger | None = None,
    progress_callback: Any | None = None,
    level: str = "info",
) -> None:
    """
    Emit a progress/debug message to stdout, logger, and optional Streamlit callback.

    Args:
        message: Message to emit.
        logger: Optional configured logger.
        progress_callback: Optional callback accepting one string message.
        level: Logging level name.
    """
    if logger is not None:
        log_fn = getattr(logger, level, logger.info)
        log_fn(message)
    else:
        print(f"[{level.upper()}] {message}")
    if progress_callback is not None:
        try:
            progress_callback(message)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] progress_callback failed: {exc}")


def failure_result(error: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build a standard failure result for Streamlit and CLI callers.

    Args:
        error: Human-readable error message.
        report: Optional partial report.

    Returns:
        Standard result dictionary.
    """
    return {
        "synthetic_data": pd.DataFrame(),
        "report": report or {},
        "success": False,
        "error": error,
    }


def success_result(synthetic_data: pd.DataFrame, report: dict[str, Any]) -> dict[str, Any]:
    """
    Build a standard success result for Streamlit and CLI callers.

    Args:
        synthetic_data: Combined real + synthetic DataFrame.
        report: Generation report.

    Returns:
        Standard result dictionary.
    """
    return {
        "synthetic_data": synthetic_data,
        "report": report,
        "success": True,
        "error": None,
    }


@dataclass
class ClusterModel:
    """
    Container for the fitted generative pieces for one cluster.

    Attributes:
        cluster_id: Integer cluster assignment.
        row_count: Number of real rows in the cluster.
        gmm: Fitted GaussianMixture for non-constant numeric features.
        modeled_numeric_columns: Numeric columns included in the GMM.
        constant_numeric_values: Numeric columns that are constant and should be
            copied as that constant value during generation.
        all_missing_numeric_columns: Numeric columns that are entirely missing
            and should remain missing in generated samples.
        categorical_distributions: Per-column category probabilities.
    """

    cluster_id: int
    row_count: int
    gmm: GaussianMixture | None
    modeled_numeric_columns: list[str]
    constant_numeric_values: dict[str, float]
    all_missing_numeric_columns: list[str]
    categorical_distributions: dict[str, dict[str, float]]


def setup_logging(log_path: Path) -> logging.Logger:
    """
    Configure console and file logging for the synthetic generation pipeline.

    Args:
        log_path: Destination path for the detailed log file.

    Returns:
        Configured logger instance.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("synthgen.synthetic")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def load_clustered_data(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load clustered thyroid data and validate that cluster assignments exist.

    Args:
        path: Path to the clustered CSV.
        logger: Logger used for progress reporting.

    Returns:
        Clustered DataFrame.

    Raises:
        FileNotFoundError: If the clustered CSV does not exist.
        ValueError: If required columns are missing or data is empty.
    """
    logger.info("Loading clustered data...")
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Run src/02_cluster.py first."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    if find_cluster_column(df) is None:
        raise ValueError("Required cluster column is missing ('cluster' or 'cluster_id').")
    df = normalize_cluster_column(df)

    cluster_count = df[CLUSTER_COLUMN].nunique(dropna=True)
    logger.info("Loaded %d samples across %d clusters", len(df), cluster_count)
    return df


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Separate feature columns into numeric and categorical groups.

    The cluster column is not treated as a feature because it controls which
    cluster-specific model generates each synthetic row.

    Args:
        df: Clustered DataFrame.

    Returns:
        Tuple of numeric feature columns and categorical feature columns.
    """
    excluded = {CLUSTER_COLUMN, *CLUSTER_ALIASES, SYNTHETIC_COLUMN, EXTREME_FLAG_COLUMN}
    feature_columns = [col for col in df.columns if col not in excluded]
    numeric_columns = [
        col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_columns = [col for col in feature_columns if col not in numeric_columns]
    numeric_columns = [
        col for col in numeric_columns
        if df[col].notna().sum() > 0 and df[col].nunique(dropna=True) > 0
    ]
    if not numeric_columns:
        raise ValueError("No usable numeric features found for synthetic generation.")
    return numeric_columns, categorical_columns


def load_borderline_cluster(df: pd.DataFrame, stats_path: Path, logger: logging.Logger) -> int:
    """
    Load the borderline cluster identified in step 2, falling back to the
    smallest cluster when the previous report is unavailable.

    Args:
        df: Clustered DataFrame.
        stats_path: Path to outputs/cluster_stats.json.
        logger: Logger used for progress reporting.

    Returns:
        Borderline cluster ID.
    """
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as handle:
                stats_json = json.load(handle)
            if not isinstance(stats_json, dict):
                raise ValueError("cluster_stats.json did not contain a JSON object.")
            borderline = stats_json.get("borderline_analysis", {}).get("borderline_cluster")
            if borderline is not None and int(borderline) in set(df[CLUSTER_COLUMN].astype(int)):
                logger.info("Using borderline cluster from %s: %s", stats_path, borderline)
                return int(borderline)
            logger.warning(
                "cluster_stats.json exists but does not contain a usable borderline_cluster."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load %s safely: %s", stats_path, exc)
    else:
        logger.warning("%s not found. Falling back to smallest cluster.", stats_path)

    fallback = int(df[CLUSTER_COLUMN].value_counts().idxmin())
    logger.warning(
        "Could not read borderline cluster from %s; using smallest cluster %d.",
        stats_path,
        fallback,
    )
    return fallback


def choose_gmm_component_count(cluster_size: int, feature_count: int) -> int:
    """
    Choose a conservative number of GMM components for a cluster.

    GMMs need enough rows to estimate covariance matrices. This rule allows
    richer mixtures for large clusters while avoiding unstable fits for tiny
    clusters.

    Args:
        cluster_size: Number of real samples in the cluster.
        feature_count: Number of numeric features modeled by the GMM.

    Returns:
        Number of Gaussian mixture components.
    """
    if feature_count == 0 or cluster_size < 2:
        return 0
    size_based_limit = max(1, cluster_size // 25)
    sqrt_limit = max(1, int(np.sqrt(cluster_size / 2)))
    return int(min(MAX_GMM_COMPONENTS, size_based_limit, sqrt_limit, cluster_size))


def fit_cluster_models(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    logger: logging.Logger,
) -> dict[int, ClusterModel]:
    """
    Fit GMM and categorical probability models independently for each cluster.

    Args:
        df: Real clustered data.
        numeric_columns: Numeric feature columns to consider.
        categorical_columns: Categorical feature columns to sample.
        logger: Logger used for progress reporting.

    Returns:
        Mapping from cluster ID to fitted ClusterModel.
    """
    logger.info("Fitting GMMs for each cluster...")
    models: dict[int, ClusterModel] = {}

    for cluster_id, cluster_df in df.groupby(CLUSTER_COLUMN, sort=True):
        cluster_int = int(cluster_id)
        if cluster_df.empty:
            logger.warning("Skipping empty cluster %d.", cluster_int)
            continue
        modeled_numeric: list[str] = []
        constant_numeric: dict[str, float] = {}
        all_missing_numeric: list[str] = []

        for col in numeric_columns:
            series = cluster_df[col]
            non_missing = series.dropna()
            if non_missing.empty:
                all_missing_numeric.append(col)
            elif non_missing.nunique() <= 1:
                constant_numeric[col] = float(non_missing.iloc[0])
            else:
                modeled_numeric.append(col)

        gmm: GaussianMixture | None = None
        if modeled_numeric:
            model_matrix = cluster_df[modeled_numeric].copy()
            model_matrix = model_matrix.fillna(model_matrix.median(numeric_only=True))
            n_components = choose_gmm_component_count(len(cluster_df), len(modeled_numeric))
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                reg_covar=1e-5,
                random_state=RANDOM_STATE + cluster_int,
                max_iter=500,
                n_init=3,
            )
            try:
                gmm.fit(model_matrix)
                logger.debug(
                    "Cluster %d GMM: %d components over %d numeric features.",
                    cluster_int,
                    n_components,
                    len(modeled_numeric),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GMM fit failed for cluster %d (%s). Falling back to bootstrap sampling.",
                    cluster_int,
                    exc,
                )
                gmm = None
                for col in modeled_numeric:
                    constant_numeric[col] = float(model_matrix[col].median())
                modeled_numeric = []
        else:
            logger.warning("Cluster %d has no variable numeric features for GMM.", cluster_int)

        categorical_distributions = {
            col: make_categorical_distribution(cluster_df[col])
            for col in categorical_columns
        }

        models[cluster_int] = ClusterModel(
            cluster_id=cluster_int,
            row_count=len(cluster_df),
            gmm=gmm,
            modeled_numeric_columns=modeled_numeric,
            constant_numeric_values=constant_numeric,
            all_missing_numeric_columns=all_missing_numeric,
            categorical_distributions=categorical_distributions,
        )

    if not models:
        raise ValueError("No cluster models could be fitted.")
    return models


def make_categorical_distribution(series: pd.Series) -> dict[str, float]:
    """
    Estimate an empirical categorical probability distribution.

    Missing values are represented internally with MISSING_TOKEN so they can be
    sampled with the same frequency as observed, then converted back to NaN in
    the synthetic DataFrame.

    Args:
        series: Categorical column from one cluster.

    Returns:
        Mapping of category string to probability.
    """
    normalized = series.astype("object").where(series.notna(), MISSING_TOKEN)
    probabilities = normalized.value_counts(normalize=True, dropna=False).sort_index()
    return {str(category): float(prob) for category, prob in probabilities.items()}


def calculate_generation_plan(
    df: pd.DataFrame,
    borderline_cluster: int,
    n_samples: int | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Decide how many synthetic rows to generate for each cluster.

    The identified borderline cluster is amplified 5x to improve representation.
    All other clusters are amplified 2x, including tiny clusters, to preserve
    the original cluster structure without over-dominating outlier groups.

    Args:
        df: Real clustered data.
        borderline_cluster: Cluster that should receive 5x augmentation.

    Returns:
        Mapping from cluster ID to generation metadata.
    """
    plan: dict[int, dict[str, Any]] = {}
    counts = df[CLUSTER_COLUMN].value_counts().sort_index()
    if n_samples is not None and n_samples > 0:
        total_real = int(counts.sum())
        allocated = 0
        for i, (cluster_id, count) in enumerate(counts.items()):
            cluster_int = int(cluster_id)
            if i == len(counts) - 1:
                synthetic_count = max(1, int(n_samples - allocated))
            else:
                synthetic_count = max(1, int(round(n_samples * int(count) / total_real)))
                allocated += synthetic_count
            label = "borderline" if cluster_int == borderline_cluster else "normal"
            if count < 50 and cluster_int != borderline_cluster:
                label = "tiny"
            plan[cluster_int] = {
                "real_count": int(count),
                "multiplier": round(synthetic_count / max(int(count), 1), 3),
                "synthetic_count": int(synthetic_count),
                "label": label,
            }
        return plan

    for cluster_id, count in counts.items():
        cluster_int = int(cluster_id)
        multiplier = 5 if cluster_int == borderline_cluster else 2
        label = "borderline" if cluster_int == borderline_cluster else "normal"
        if count < 50 and cluster_int != borderline_cluster:
            label = "tiny"
        plan[cluster_int] = {
            "real_count": int(count),
            "multiplier": multiplier,
            "synthetic_count": int(count * multiplier),
            "label": label,
        }
    return plan


def generate_synthetic_samples(
    models: dict[int, ClusterModel],
    generation_plan: dict[int, dict[str, Any]],
    real_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate synthetic rows by sampling each cluster's numeric GMM and
    categorical probability distributions.

    Args:
        models: Fitted model per cluster.
        generation_plan: Synthetic sample counts per cluster.
        real_df: Real dataset used for ranges and column ordering.
        numeric_columns: Numeric feature columns.
        categorical_columns: Categorical feature columns.
        logger: Logger used for progress reporting.

    Returns:
        Synthetic DataFrame and generation diagnostics.
    """
    logger.info("Generating synthetic samples...")
    rng = np.random.default_rng(RANDOM_STATE)
    numeric_ranges = {
        col: {
            "min": _json_number(real_df[col].min(skipna=True)),
            "max": _json_number(real_df[col].max(skipna=True)),
        }
        for col in numeric_columns
    }

    synthetic_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {
        "clusters": {},
        "total_extreme_rows": 0,
        "extreme_counts_by_feature": {col: 0 for col in numeric_columns},
    }

    for cluster_id, plan in generation_plan.items():
        if cluster_id not in models:
            logger.warning("Generation plan references missing cluster %s; skipping.", cluster_id)
            continue
        model = models[cluster_id]
        n_samples = int(plan["synthetic_count"])
        if n_samples <= 0:
            logger.warning("Cluster %s requested non-positive sample count; skipping.", cluster_id)
            continue
        synthetic_cluster = pd.DataFrame(index=np.arange(n_samples))
        extreme_flags = np.zeros(n_samples, dtype=bool)
        feature_extreme_counts: dict[str, int] = {}

        if model.gmm is not None and model.modeled_numeric_columns:
            numeric_sample_df = sample_numeric_with_range_retries(
                model.gmm,
                model.modeled_numeric_columns,
                numeric_ranges,
                n_samples,
            )

            for col in model.modeled_numeric_columns:
                min_value = numeric_ranges[col]["min"]
                max_value = numeric_ranges[col]["max"]
                if min_value is None or max_value is None:
                    synthetic_cluster[col] = np.nan
                    continue

                out_of_range = (
                    (numeric_sample_df[col] < min_value)
                    | (numeric_sample_df[col] > max_value)
                )
                feature_extreme_counts[col] = int(out_of_range.sum())
                diagnostics["extreme_counts_by_feature"][col] += int(out_of_range.sum())
                extreme_flags |= out_of_range.to_numpy()
                synthetic_cluster[col] = numeric_sample_df[col].clip(min_value, max_value)

        for col, value in model.constant_numeric_values.items():
            synthetic_cluster[col] = value
            feature_extreme_counts[col] = 0

        for col in model.all_missing_numeric_columns:
            synthetic_cluster[col] = np.nan
            feature_extreme_counts[col] = 0

        for col in numeric_columns:
            if col not in synthetic_cluster.columns:
                synthetic_cluster[col] = np.nan

        for col in categorical_columns:
            distribution = model.categorical_distributions[col]
            categories = np.array(list(distribution.keys()), dtype=object)
            probabilities = np.array(list(distribution.values()), dtype=float)
            probabilities = probabilities / probabilities.sum()
            sampled = rng.choice(categories, size=n_samples, p=probabilities)
            synthetic_cluster[col] = pd.Series(sampled).replace(MISSING_TOKEN, np.nan)

        synthetic_cluster[CLUSTER_COLUMN] = cluster_id
        synthetic_cluster[SYNTHETIC_COLUMN] = True
        synthetic_cluster[EXTREME_FLAG_COLUMN] = extreme_flags

        ordered_columns = [
            col for col in real_df.columns if col != SYNTHETIC_COLUMN
        ] + [SYNTHETIC_COLUMN, EXTREME_FLAG_COLUMN]
        synthetic_cluster = synthetic_cluster.reindex(columns=ordered_columns)
        if synthetic_cluster.empty:
            logger.warning("Cluster %s generated an empty synthetic frame; skipping.", cluster_id)
            continue
        synthetic_frames.append(synthetic_cluster)

        extreme_count = int(extreme_flags.sum())
        diagnostics["total_extreme_rows"] += extreme_count
        diagnostics["clusters"][str(cluster_id)] = {
            **plan,
            "extreme_rows_before_clipping": extreme_count,
            "extreme_counts_by_feature": feature_extreme_counts,
            "modeled_numeric_columns": model.modeled_numeric_columns,
            "constant_numeric_columns": list(model.constant_numeric_values.keys()),
            "all_missing_numeric_columns": model.all_missing_numeric_columns,
        }
        logger.info(
            "  Cluster %d (%s): Generated %d samples (%sx from %d real)",
            cluster_id,
            plan["label"],
            n_samples,
            plan["multiplier"],
            plan["real_count"],
        )

    if not synthetic_frames:
        raise ValueError("No synthetic samples were generated.")
    synthetic_df = pd.concat(synthetic_frames, ignore_index=True)
    if synthetic_df.empty:
        raise ValueError("Synthetic DataFrame is empty after generation.")
    logger.info("Total synthetic samples: %d", len(synthetic_df))
    return synthetic_df, diagnostics


def sample_numeric_with_range_retries(
    gmm: GaussianMixture,
    modeled_numeric_columns: list[str],
    numeric_ranges: dict[str, dict[str, float | None]],
    n_samples: int,
) -> pd.DataFrame:
    """
    Sample numeric values from a fitted GMM, retrying rows outside real ranges.

    GMMs can occasionally draw plausible-but-extreme tails beyond the empirical
    clinical range. A retry loop preserves the GMM's covariance structure better
    than clipping every invalid draw immediately. Remaining invalid values after
    a small number of attempts are clipped and flagged by the caller.

    Args:
        gmm: Fitted GaussianMixture model.
        modeled_numeric_columns: Columns represented by the GMM.
        numeric_ranges: Real-data min/max bounds per numeric feature.
        n_samples: Number of synthetic rows to draw.

    Returns:
        DataFrame of sampled numeric values.
    """
    sampled_numeric, _ = gmm.sample(n_samples)
    sample_df = pd.DataFrame(sampled_numeric, columns=modeled_numeric_columns)

    for _ in range(MAX_RANGE_RESAMPLE_ATTEMPTS):
        invalid_mask = numeric_out_of_range_mask(sample_df, modeled_numeric_columns, numeric_ranges)
        invalid_count = int(invalid_mask.sum())
        if invalid_count == 0:
            break

        replacement_numeric, _ = gmm.sample(invalid_count)
        replacement_df = pd.DataFrame(
            replacement_numeric,
            columns=modeled_numeric_columns,
            index=sample_df.index[invalid_mask],
        )
        sample_df.loc[invalid_mask, modeled_numeric_columns] = replacement_df

    return sample_df


def numeric_out_of_range_mask(
    sample_df: pd.DataFrame,
    modeled_numeric_columns: list[str],
    numeric_ranges: dict[str, dict[str, float | None]],
) -> pd.Series:
    """
    Identify rows where any modeled numeric feature is outside real min/max.

    Args:
        sample_df: Sampled numeric values.
        modeled_numeric_columns: Numeric columns represented in sample_df.
        numeric_ranges: Real-data min/max bounds per numeric feature.

    Returns:
        Boolean Series marking rows with at least one out-of-range value.
    """
    invalid_mask = pd.Series(False, index=sample_df.index)
    for col in modeled_numeric_columns:
        min_value = numeric_ranges[col]["min"]
        max_value = numeric_ranges[col]["max"]
        if min_value is None or max_value is None:
            continue
        invalid_mask |= (sample_df[col] < min_value) | (sample_df[col] > max_value)
    return invalid_mask


def combine_real_and_synthetic(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine real and synthetic samples into one DataFrame with origin flags.

    Args:
        real_df: Original clustered data.
        synthetic_df: Generated synthetic samples.

    Returns:
        Combined DataFrame.
    """
    real_flagged = real_df.copy()
    real_flagged[SYNTHETIC_COLUMN] = False
    real_flagged[EXTREME_FLAG_COLUMN] = False

    ordered_columns = list(real_df.columns) + [SYNTHETIC_COLUMN, EXTREME_FLAG_COLUMN]
    real_flagged = real_flagged.reindex(columns=ordered_columns)
    synthetic_df = synthetic_df.reindex(columns=ordered_columns)
    return pd.concat([real_flagged, synthetic_df], ignore_index=True)


def calibrate_synthetic_numeric_moments(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_columns: list[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Lightly calibrate synthetic numeric columns to match real global moments.

    Cluster-level GMM sampling is the primary generator, but intentional cluster
    rebalancing can shift overall means and standard deviations. This affine
    correction aligns each synthetic numeric feature with the real dataset's
    global mean/std while preserving sample ranking and most distributional
    shape. Values are clipped to empirical clinical ranges after calibration.

    Args:
        real_df: Original clustered data.
        synthetic_df: Synthetic-only rows before calibration.
        numeric_columns: Numeric feature columns to calibrate.
        logger: Logger used for progress reporting.

    Returns:
        Calibrated synthetic DataFrame and calibration diagnostics.
    """
    calibrated = synthetic_df.copy()
    diagnostics: dict[str, Any] = {}
    logger.info("Calibrating synthetic numeric moments against real data...")

    for col in numeric_columns:
        real_values = real_df[col].dropna()
        synthetic_values = calibrated[col].dropna()

        if real_values.empty or synthetic_values.empty:
            diagnostics[col] = {"status": "SKIP", "reason": "missing values only"}
            continue

        real_mean = float(real_values.mean())
        real_std = float(real_values.std(ddof=1))
        synthetic_mean = float(synthetic_values.mean())
        synthetic_std = float(synthetic_values.std(ddof=1))

        if real_std == 0 or synthetic_std == 0 or np.isnan(real_std) or np.isnan(synthetic_std):
            diagnostics[col] = {"status": "SKIP", "reason": "constant feature"}
            continue

        calibrated_values = ((calibrated[col] - synthetic_mean) / synthetic_std) * real_std
        calibrated_values = calibrated_values + real_mean

        min_value = float(real_values.min())
        max_value = float(real_values.max())
        out_of_range = (
            calibrated_values.notna()
            & ((calibrated_values < min_value) | (calibrated_values > max_value))
        )
        calibrated.loc[out_of_range, EXTREME_FLAG_COLUMN] = True
        calibrated[col] = calibrated_values.clip(min_value, max_value)

        diagnostics[col] = {
            "status": "CALIBRATED",
            "real_mean": real_mean,
            "real_std": real_std,
            "pre_calibration_synthetic_mean": synthetic_mean,
            "pre_calibration_synthetic_std": synthetic_std,
            "post_calibration_clipped_values": int(out_of_range.sum()),
        }

    total_clipped = int(calibrated[EXTREME_FLAG_COLUMN].sum())
    logger.info("Moment calibration complete; %d synthetic rows carry extreme flags.", total_clipped)
    return calibrated, diagnostics


def validate_synthetic_data(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_columns: list[str],
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Compare real and synthetic numeric distributions and flag large differences.

    Validation includes mean/std percent differences and a two-sample
    Kolmogorov-Smirnov test for each numeric feature with non-missing values.

    Args:
        real_df: Original clustered data.
        synthetic_df: Synthetic-only rows.
        numeric_columns: Numeric feature columns to validate.
        logger: Logger used for progress reporting.

    Returns:
        JSON-serializable validation summary.
    """
    logger.info("Validating synthetic data...")
    feature_reports: dict[str, Any] = {}
    warnings: list[str] = []

    for col in numeric_columns:
        real_values = real_df[col].dropna()
        synthetic_values = synthetic_df[col].dropna()

        if real_values.empty or synthetic_values.empty:
            feature_reports[col] = {
                "status": "SKIP",
                "reason": "real or synthetic values are entirely missing",
            }
            logger.info("  %s: skipped (missing in real or synthetic data)", col)
            continue

        real_mean = float(real_values.mean())
        synth_mean = float(synthetic_values.mean())
        real_std = float(real_values.std(ddof=1))
        synth_std = float(synthetic_values.std(ddof=1))
        mean_diff = relative_difference(synth_mean, real_mean)
        std_diff = relative_difference(synth_std, real_std)
        ks_stat, ks_pvalue = stats.ks_2samp(real_values, synthetic_values)

        status = "PASS"
        feature_warnings: list[str] = []
        if mean_diff > MEAN_DIFF_WARNING_THRESHOLD:
            status = "WARN"
            feature_warnings.append(f"mean differs by {mean_diff:.1%}")
        if std_diff > STD_DIFF_WARNING_THRESHOLD:
            status = "WARN"
            feature_warnings.append(f"std differs by {std_diff:.1%}")

        if feature_warnings:
            warnings.append(f"{col}: {', '.join(feature_warnings)}")

        feature_reports[col] = {
            "status": status,
            "real_mean": real_mean,
            "synthetic_mean": synth_mean,
            "mean_diff_pct": float(mean_diff),
            "real_std": real_std,
            "synthetic_std": synth_std,
            "std_diff_pct": float(std_diff),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "warnings": feature_warnings,
        }
        logger.info("  %s: mean diff = %.1f%% %s", col, mean_diff * 100, status)

    if warnings:
        logger.warning("Validation warnings: %d feature(s) exceeded thresholds.", len(warnings))
        for warning in warnings:
            logger.warning("  %s", warning)
    else:
        logger.info("All numeric validation checks passed.")

    return {
        "thresholds": {
            "mean_diff_warning_pct": MEAN_DIFF_WARNING_THRESHOLD,
            "std_diff_warning_pct": STD_DIFF_WARNING_THRESHOLD,
        },
        "features": feature_reports,
        "warnings": warnings,
        "passed": len(warnings) == 0,
    }


def relative_difference(candidate: float, reference: float) -> float:
    """
    Calculate a stable relative difference for validation metrics.

    Args:
        candidate: Synthetic statistic.
        reference: Real statistic.

    Returns:
        Absolute relative difference.
    """
    denominator = max(abs(reference), 1e-8)
    return float(abs(candidate - reference) / denominator)


def choose_top_features(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numeric_columns: list[str],
    n: int = 6,
) -> list[str]:
    """
    Select the most informative numeric features for validation plots.

    Features are ranked by real-data variance, with all-missing and constant
    features excluded.

    Args:
        real_df: Original clustered data.
        synthetic_df: Synthetic-only rows.
        numeric_columns: Candidate numeric columns.
        n: Maximum number of features to select.

    Returns:
        Ranked list of feature names.
    """
    scores: dict[str, float] = {}
    for col in numeric_columns:
        real_values = real_df[col].dropna()
        synthetic_values = synthetic_df[col].dropna()
        if real_values.empty or synthetic_values.empty or real_values.nunique() <= 1:
            continue
        scores[col] = float(real_values.var())
    return sorted(scores, key=scores.get, reverse=True)[:n]


def set_plot_style() -> None:
    """
    Apply a consistent visual style for publication-quality validation plots.
    """
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.95,
        rc={
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        },
    )


def plot_distribution_comparison(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    top_features: list[str],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """
    Save side-by-side real vs synthetic histograms for top numeric features.

    Args:
        real_df: Original clustered data.
        synthetic_df: Synthetic-only rows.
        top_features: Numeric features to plot.
        output_path: Destination PNG path.
        logger: Logger used for progress reporting.
    """
    if not top_features:
        logger.warning("No eligible features found for distribution comparison plots.")
        return

    plot_frames = []
    for source_name, source_df in [("Real", real_df), ("Synthetic", synthetic_df)]:
        frame = source_df[top_features].copy()
        frame["source"] = source_name
        plot_frames.append(frame)
    long_df = pd.concat(plot_frames, ignore_index=True).melt(
        id_vars="source",
        value_vars=top_features,
        var_name="feature",
        value_name="value",
    ).dropna()

    n_cols = 3
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.5 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()
    palette = {"Real": "#4C78A8", "Synthetic": "#F58518"}

    for ax, feature in zip(axes_array, top_features):
        feature_df = long_df[long_df["feature"] == feature]
        sns.histplot(
            data=feature_df,
            x="value",
            hue="source",
            bins=35,
            stat="density",
            common_norm=False,
            element="step",
            fill=True,
            alpha=0.35,
            palette=palette,
            ax=ax,
        )
        ax.set_title(feature, pad=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    for ax in axes_array[len(top_features):]:
        ax.axis("off")

    fig.suptitle("Real vs Synthetic Numeric Distributions", y=1.02, fontsize=24)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_validation_diagnostics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    validation: dict[str, Any],
    top_features: list[str],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """
    Save a multi-panel validation figure with boxplots, percent-difference bars,
    and cluster count comparisons.

    Args:
        real_df: Original clustered data.
        synthetic_df: Synthetic-only rows.
        validation: Validation report dictionary.
        top_features: Numeric features to visualize.
        output_path: Destination PNG path.
        logger: Logger used for progress reporting.
    """
    if not top_features:
        logger.warning("No eligible features found for validation diagnostic plots.")
        return

    feature_reports = validation["features"]
    diff_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "mean_diff_pct": feature_reports[feature]["mean_diff_pct"] * 100,
                "std_diff_pct": feature_reports[feature]["std_diff_pct"] * 100,
            }
            for feature in top_features
            if feature_reports.get(feature, {}).get("status") != "SKIP"
        ]
    )

    cluster_counts = pd.concat(
        [
            real_df[CLUSTER_COLUMN].value_counts().rename("Real"),
            synthetic_df[CLUSTER_COLUMN].value_counts().rename("Synthetic"),
        ],
        axis=1,
    ).fillna(0).sort_index()
    cluster_long = cluster_counts.reset_index(names="cluster").melt(
        id_vars="cluster",
        var_name="source",
        value_name="count",
    )

    box_frames = []
    for source_name, source_df in [("Real", real_df), ("Synthetic", synthetic_df)]:
        frame = source_df[top_features].copy()
        frame["source"] = source_name
        box_frames.append(frame)
    box_long = pd.concat(box_frames, ignore_index=True).melt(
        id_vars="source",
        value_vars=top_features,
        var_name="feature",
        value_name="value",
    ).dropna()

    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax_counts = fig.add_subplot(grid[0, 0])
    ax_diff = fig.add_subplot(grid[0, 1])
    ax_box = fig.add_subplot(grid[1, :])

    sns.barplot(
        data=cluster_long,
        x="cluster",
        y="count",
        hue="source",
        palette={"Real": "#4C78A8", "Synthetic": "#F58518"},
        ax=ax_counts,
    )
    ax_counts.set_title("Sample Counts by Cluster")
    ax_counts.set_xlabel("Cluster")
    ax_counts.set_ylabel("Count")
    ax_counts.legend(title="")

    if not diff_df.empty:
        diff_long = diff_df.melt(
            id_vars="feature",
            value_vars=["mean_diff_pct", "std_diff_pct"],
            var_name="metric",
            value_name="difference_pct",
        )
        diff_long["metric"] = diff_long["metric"].map(
            {"mean_diff_pct": "Mean", "std_diff_pct": "Std Dev"}
        )
        sns.barplot(
            data=diff_long,
            x="feature",
            y="difference_pct",
            hue="metric",
            palette={"Mean": "#59A14F", "Std Dev": "#E15759"},
            ax=ax_diff,
        )
        ax_diff.axhline(10, color="#2F2F2F", linestyle="--", linewidth=1.3)
        ax_diff.set_title("Validation Percent Differences")
        ax_diff.set_xlabel("")
        ax_diff.set_ylabel("Difference (%)")
        ax_diff.tick_params(axis="x", rotation=35)
        ax_diff.legend(title="")
    else:
        ax_diff.axis("off")

    sns.boxplot(
        data=box_long,
        x="feature",
        y="value",
        hue="source",
        palette={"Real": "#4C78A8", "Synthetic": "#F58518"},
        width=0.65,
        linewidth=1.2,
        fliersize=2.0,
        ax=ax_box,
    )
    ax_box.set_title("Real vs Synthetic Feature Spread")
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Value")
    ax_box.legend(title="")

    fig.suptitle("Synthetic Data Validation Summary", y=0.99, fontsize=25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_report(report: dict[str, Any], path: Path, logger: logging.Logger) -> None:
    """
    Save the synthetic generation and validation report as JSON.

    Args:
        report: JSON-serializable report dictionary.
        path: Output path.
        logger: Logger used for progress reporting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Validation report: %s", path)


def _json_number(value: Any) -> float | None:
    """
    Convert pandas/numpy numeric values into JSON-safe Python floats.

    Args:
        value: Numeric value that may be missing.

    Returns:
        Float value, or None when the input is NaN.
    """
    if pd.isna(value):
        return None
    return float(value)


def validate_clustered_dataframe(df: pd.DataFrame | None) -> tuple[bool, str]:
    """
    Validate clustered input before synthetic generation.

    Args:
        df: Candidate clustered DataFrame.

    Returns:
        Tuple of is_valid flag and message.
    """
    if df is None:
        return False, "clustered_data is None."
    if not isinstance(df, pd.DataFrame):
        return False, f"clustered_data must be a pandas DataFrame, got {type(df).__name__}."
    if df.empty:
        return False, "clustered_data is empty."
    cluster_col = find_cluster_column(df)
    if cluster_col is None:
        return False, "clustered_data is missing required cluster column ('cluster' or 'cluster_id')."
    if df[cluster_col].isna().all():
        return False, "cluster column contains only missing values."
    if df[cluster_col].nunique(dropna=True) < 1:
        return False, "clustered_data does not contain any valid clusters."
    numeric_cols = [
        col
        for col in df.columns
        if col not in {CLUSTER_COLUMN, SYNTHETIC_COLUMN, EXTREME_FLAG_COLUMN}
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].notna().sum() > 0
    ]
    if not numeric_cols:
        return False, "clustered_data has no usable numeric columns."
    return True, "clustered_data validation passed."


def find_cluster_column(df: pd.DataFrame) -> str | None:
    """
    Find the cluster assignment column using supported aliases.

    Args:
        df: Candidate clustered DataFrame.

    Returns:
        Column name if found, otherwise None.
    """
    for col in CLUSTER_ALIASES:
        if col in df.columns:
            return col
    return None


def normalize_cluster_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has the canonical 'cluster' column.

    Args:
        df: Clustered DataFrame that may use 'cluster_id'.

    Returns:
        Copy of DataFrame with canonical cluster column.
    """
    normalized = df.copy()
    cluster_col = find_cluster_column(normalized)
    if cluster_col is None:
        return normalized
    if cluster_col != CLUSTER_COLUMN:
        normalized[CLUSTER_COLUMN] = normalized[cluster_col]
    return normalized


def build_report(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    borderline_cluster: int,
    generation_plan: dict[int, dict[str, Any]],
    generation_diagnostics: dict[str, Any],
    calibration_diagnostics: dict[str, Any],
    validation: dict[str, Any],
    top_features: list[str],
    input_path: str,
    output_csv: Path,
    report_json: Path,
    distribution_png: Path,
    validation_png: Path,
) -> dict[str, Any]:
    """
    Build the JSON-serializable synthetic generation report.

    Args:
        real_df: Real clustered data.
        synthetic_df: Synthetic-only data.
        combined_df: Combined real + synthetic data.
        numeric_columns: Numeric feature names.
        categorical_columns: Categorical feature names.
        borderline_cluster: Borderline cluster id.
        generation_plan: Per-cluster generation plan.
        generation_diagnostics: Generation diagnostics.
        calibration_diagnostics: Calibration diagnostics.
        validation: Validation metrics.
        top_features: Features selected for plots.
        input_path: Input data source description.
        output_csv: Synthetic CSV output path.
        report_json: Report JSON output path.
        distribution_png: Distribution plot path.
        validation_png: Validation plot path.

    Returns:
        Report dictionary.
    """
    return {
        "input": {
            "path": input_path,
            "real_rows": int(len(real_df)),
            "real_columns": int(real_df.shape[1]),
            "clusters": {
                str(int(k)): int(v)
                for k, v in real_df[CLUSTER_COLUMN].value_counts().sort_index().items()
            },
        },
        "generation": {
            "borderline_cluster": int(borderline_cluster),
            "plan": {str(k): v for k, v in generation_plan.items()},
            "total_synthetic_rows": int(len(synthetic_df)),
            "combined_rows": int(len(combined_df)),
            **generation_diagnostics,
        },
        "calibration": calibration_diagnostics,
        "features": {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "top_plot_features": top_features,
        },
        "validation": validation,
        "outputs": {
            "synthetic_csv": str(output_csv),
            "report_json": str(report_json),
            "distribution_comparison_png": str(distribution_png),
            "synthetic_validation_plots_png": str(validation_png),
        },
    }


def run_synthetic_gen(
    clustered_data: pd.DataFrame | None,
    n_samples: int = 500,
    borderline_cluster: int = 1,
    output_dir: str = "outputs",
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """
    Generate synthetic clinical samples from clustered data.

    This is the Streamlit-safe public entry point. It never returns None: every
    outcome is a dict with synthetic_data, report, success, and error keys.

    Args:
        clustered_data: DataFrame with a required cluster column.
        n_samples: Number of synthetic rows to generate. If <= 0, the historical
            cluster multiplier plan is used.
        borderline_cluster: Fallback borderline cluster id.
        output_dir: Directory for JSON reports and plots.
        progress_callback: Optional callback receiving progress messages.

    Returns:
        Standard result dictionary:
        {
            "synthetic_data": DataFrame,
            "report": dict,
            "success": bool,
            "error": str | None,
        }
    """
    output_path = Path(output_dir)
    log_path = output_path / "synthetic_gen_log.txt"
    logger = setup_logging(log_path)

    try:
        _emit("Starting run_synthetic_gen().", logger, progress_callback)
        output_path.mkdir(parents=True, exist_ok=True)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        report_json = output_path / "synthetic_report.json"
        distribution_png = output_path / "distribution_comparison.png"
        validation_png = output_path / "synthetic_validation_plots.png"
        stats_path = output_path / "cluster_stats.json"

        is_valid, message = validate_clustered_dataframe(clustered_data)
        _emit(message, logger, progress_callback, "info" if is_valid else "error")
        if not is_valid:
            return failure_result(message)

        real_df = normalize_cluster_column(clustered_data)
        _emit(
            f"Input shape: {real_df.shape[0]} rows, {real_df.shape[1]} columns.",
            logger,
            progress_callback,
        )

        numeric_columns, categorical_columns = infer_feature_columns(real_df)
        _emit(
            f"Detected {len(numeric_columns)} numeric and {len(categorical_columns)} categorical features.",
            logger,
            progress_callback,
        )

        if stats_path.exists():
            _emit(f"Found cluster stats at {stats_path}.", logger, progress_callback)
            selected_borderline = load_borderline_cluster(real_df, stats_path, logger)
        else:
            _emit(
                f"{stats_path} not found. Using provided borderline_cluster={borderline_cluster}.",
                logger,
                progress_callback,
                "warning",
            )
            cluster_values = set(real_df[CLUSTER_COLUMN].dropna().astype(int))
            selected_borderline = (
                int(borderline_cluster)
                if int(borderline_cluster) in cluster_values
                else int(real_df[CLUSTER_COLUMN].value_counts().idxmin())
            )

        _emit("Fitting per-cluster GMM and categorical models.", logger, progress_callback)
        models = fit_cluster_models(real_df, numeric_columns, categorical_columns, logger)

        requested_samples = int(n_samples) if n_samples is not None else 0
        plan_samples = requested_samples if requested_samples > 0 else None
        generation_plan = calculate_generation_plan(real_df, selected_borderline, plan_samples)
        _emit(f"Generation plan prepared for {len(generation_plan)} clusters.", logger, progress_callback)

        _emit("Sampling synthetic records.", logger, progress_callback)
        synthetic_df, generation_diagnostics = generate_synthetic_samples(
            models,
            generation_plan,
            real_df,
            numeric_columns,
            categorical_columns,
            logger,
        )
        if synthetic_df is None or synthetic_df.empty:
            return failure_result("Synthetic generation produced no rows.")

        _emit("Calibrating numeric moments.", logger, progress_callback)
        synthetic_df, calibration_diagnostics = calibrate_synthetic_numeric_moments(
            real_df,
            synthetic_df,
            numeric_columns,
            logger,
        )

        _emit("Combining real and synthetic data.", logger, progress_callback)
        combined_df = combine_real_and_synthetic(real_df, synthetic_df)
        if combined_df.empty or SYNTHETIC_COLUMN not in combined_df.columns:
            return failure_result("Combined DataFrame failed validation after merge.")

        _emit("Validating synthetic distribution quality.", logger, progress_callback)
        validation = validate_synthetic_data(real_df, synthetic_df, numeric_columns, logger)
        top_features = choose_top_features(real_df, synthetic_df, numeric_columns, n=6)
        _emit(f"Top plot features: {top_features}", logger, progress_callback)

        _emit("Creating validation plots.", logger, progress_callback)
        set_plot_style()
        try:
            plot_distribution_comparison(
                real_df,
                synthetic_df,
                top_features,
                distribution_png,
                logger,
            )
            plot_validation_diagnostics(
                real_df,
                synthetic_df,
                validation,
                top_features,
                validation_png,
                logger,
            )
        except Exception as exc:  # noqa: BLE001
            _emit(f"Plot creation failed but data generation succeeded: {exc}", logger, progress_callback, "warning")

        report = build_report(
            real_df=real_df,
            synthetic_df=synthetic_df,
            combined_df=combined_df,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            borderline_cluster=selected_borderline,
            generation_plan=generation_plan,
            generation_diagnostics=generation_diagnostics,
            calibration_diagnostics=calibration_diagnostics,
            validation=validation,
            top_features=top_features,
            input_path="<provided DataFrame>",
            output_csv=OUTPUT_CSV,
            report_json=report_json,
            distribution_png=distribution_png,
            validation_png=validation_png,
        )

        _emit("Saving synthetic CSV and report.", logger, progress_callback)
        try:
            combined_df.to_csv(OUTPUT_CSV, index=False)
            save_report(report, report_json, logger)
        except Exception as exc:  # noqa: BLE001
            error = f"Failed to save outputs: {exc}"
            _emit(error, logger, progress_callback, "error")
            return failure_result(error, report)

        _emit(
            f"Synthetic generation complete: {len(synthetic_df)} synthetic rows, {len(combined_df)} combined rows.",
            logger,
            progress_callback,
        )
        return success_result(combined_df, report)

    except Exception as exc:  # noqa: BLE001
        error = f"Synthetic generation failed: {exc}"
        if logger:
            logger.exception(error)
        _emit(error, logger, progress_callback, "error")
        return failure_result(error)


def main() -> None:
    """
    Execute the complete synthetic data generation workflow.

    Steps:
        1. Load clustered real data.
        2. Fit per-cluster GMMs and categorical distributions.
        3. Generate synthetic rows with cluster-specific multipliers.
        4. Clip numeric values to realistic real-data ranges and flag clipped rows.
        5. Validate synthetic distributions against real distributions.
        6. Save combined data, JSON report, plots, and logs.
    """
    logger = setup_logging(LOG_FILE)
    logger.info("SynthGen - Step 3: Synthetic generation started.")
    try:
        clustered_df = load_clustered_data(INPUT_CSV, logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not load clustered data: %s", exc)
        raise

    # CLI mode preserves the historical full amplification behavior by passing
    # n_samples=0. Streamlit can pass a positive n_samples for faster runs.
    result = run_synthetic_gen(
        clustered_df,
        n_samples=0,
        borderline_cluster=1,
        output_dir=str(OUTPUT_DIR),
    )
    if not result["success"]:
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
