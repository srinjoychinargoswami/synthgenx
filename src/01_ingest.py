"""
SynthGen - Step 1: Data Ingestion (Multi-Dataset)
==================================================
Downloads or loads clinical datasets (UCI Thyroid or UCI Heart Disease),
cleans, standardises, and persists them for downstream pipeline steps.

Supported datasets
------------------
  thyroid  – UCI Thyroid Disease (allhypo / allhyper / thyroid0387)
  heart    – UCI Heart Disease (Cleveland, 13 features + target)

Usage (CLI):
    python src/01_ingest.py [--dataset thyroid|heart] [--input /path/to/file.data]

Callable function (for app.py / notebook):
    from src.01_ingest import run_ingest
    clean_df, stats = run_ingest(raw_file="data/raw/thyroid_disease.data",
                                  output_dir="data/processed")
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Dataset schemas
# ---------------------------------------------------------------------------

THYROID_COLUMNS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine",
    "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery",
    "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium",
    "goitre", "tumor", "hypopituitary", "psych", "TSH_measured", "TSH",
    "T3_measured", "T3", "TT4_measured", "TT4", "T4U_measured", "T4U",
    "FTI_measured", "FTI", "TBG_measured", "TBG", "referral_source", "target",
]
THYROID_NUMERIC = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
THYROID_CATEGORICAL = [c for c in THYROID_COLUMNS if c not in THYROID_NUMERIC]

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
HEART_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
HEART_CATEGORICAL = [c for c in HEART_COLUMNS if c not in HEART_NUMERIC]

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"
UCI_THYROID_URLS = [
    f"{UCI_BASE}/thyroid-disease/allhypo.data",
    f"{UCI_BASE}/thyroid-disease/allhyper.data",
    f"{UCI_BASE}/thyroid-disease/thyroid0387.data",
]
UCI_HEART_URLS = [
    f"{UCI_BASE}/heart-disease/processed.cleveland.data",
]

REQUEST_TIMEOUT = 20


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _make_logger(log_path: Path, name: str = "synthgen.ingest") -> logging.Logger:
    """Create a dual (file + console) logger, clearing stale handlers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    fmt_con = logging.Formatter("[%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_con)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Dataset-type detection
# ---------------------------------------------------------------------------

def detect_dataset_type(raw_file: Optional[str]) -> str:
    """
    Infer dataset type from filename.

    Args:
        raw_file: Path string or None.

    Returns:
        'thyroid' or 'heart'
    """
    if raw_file and "heart" in Path(raw_file).name.lower():
        return "heart"
    return "thyroid"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _try_download_urls(urls: list[str], logger: logging.Logger,
                        pipe_strip: bool = False) -> Optional[str]:
    """Try a list of UCI URLs and return raw text of first success."""
    for url in urls:
        logger.info("Attempting download: %s", url)
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            logger.info("Download succeeded from %s", url)
            text = r.text
            if pipe_strip:
                lines = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if "|" in line:
                        line = line.split("|")[0].strip().rstrip(".")
                    lines.append(line)
                text = "\n".join(lines)
            return text
        except requests.RequestException as exc:
            logger.warning("Download failed (%s): %s", url, exc)
    return None


def _parse_csv_text(text: str, col_names: list[str]) -> pd.DataFrame:
    """Parse raw CSV text into a DataFrame with named columns."""
    df = pd.read_csv(
        io.StringIO(text),
        header=None,
        na_values=["?", " ?", "? ", ""],
    )
    n = min(len(col_names), df.shape[1])
    df.columns = col_names[:n]
    return df


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def download_thyroid(logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Download and parse UCI Thyroid dataset."""
    text = _try_download_urls(UCI_THYROID_URLS, logger, pipe_strip=True)
    if text is None:
        return None
    df = _parse_csv_text(text, THYROID_COLUMNS)
    logger.info("Thyroid download parsed: %s", df.shape)
    return df


def download_heart(logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Download and parse UCI Heart Disease (Cleveland) dataset."""
    text = _try_download_urls(UCI_HEART_URLS, logger, pipe_strip=False)
    if text is None:
        return None
    df = _parse_csv_text(text, HEART_COLUMNS)
    logger.info("Heart download parsed: %s", df.shape)
    return df


def load_local_file(path: str, dataset_type: str,
                     logger: logging.Logger) -> pd.DataFrame:
    """
    Load a dataset from a local .data or .csv file.

    Args:
        path: Filesystem path.
        dataset_type: 'thyroid' or 'heart'.
        logger: Logger instance.

    Returns:
        Loaded DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local file not found: {path}")
    logger.info("Loading local file: %s", path)
    df = pd.read_csv(p, header=None if p.suffix == ".data" else "infer",
                     na_values=["?", " ?", "? ", ""])
    col_names = THYROID_COLUMNS if dataset_type == "thyroid" else HEART_COLUMNS
    if list(df.columns) == list(range(df.shape[1])):
        n = min(len(col_names), df.shape[1])
        df.columns = col_names[:n]
    logger.info("Local load complete: %s", df.shape)
    return df


def acquire_data(raw_file: Optional[str], dataset_type: str,
                  logger: logging.Logger) -> pd.DataFrame:
    """
    Primary acquisition dispatcher: UCI download → local fallback.

    Args:
        raw_file: Optional local file path fallback.
        dataset_type: 'thyroid' or 'heart'.
        logger: Logger instance.

    Returns:
        Raw DataFrame.
    """
    df = None
    if dataset_type == "heart":
        df = download_heart(logger)
    else:
        df = download_thyroid(logger)

    if df is None:
        if raw_file:
            df = load_local_file(raw_file, dataset_type, logger)
        else:
            raise RuntimeError(
                f"Acquisition failed for dataset '{dataset_type}'. "
                "Re-run with --input <path>."
            )
    if df is None or df.empty:
        raise ValueError("Acquired dataset is empty.")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def remove_high_missingness_rows(df: pd.DataFrame, threshold: float,
                                   logger: logging.Logger) -> pd.DataFrame:
    """Drop rows with > threshold fraction of missing values."""
    frac = df.isnull().mean(axis=1)
    before = len(df)
    df = df[frac <= threshold].copy()
    dropped = before - len(df)
    logger.info("Removed %d row(s) with >%.0f%% missing.", dropped, threshold * 100)
    return df


def remove_duplicate_rows(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate row(s).", before - len(df))
    return df


def impute_missing_values(df: pd.DataFrame, numeric_cols: list[str],
                           categorical_cols: list[str],
                           logger: logging.Logger) -> pd.DataFrame:
    """
    Fill missing values: median for numeric, mode for categorical.
    Logs fill counts per column.
    """
    df = df.copy()
    for col in [c for c in numeric_cols if c in df.columns]:
        n = df[col].isnull().sum()
        if n:
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            logger.info("  Numeric '%s': filled %d → median=%.4f", col, n, fill)

    for col in [c for c in categorical_cols if c in df.columns]:
        n = df[col].isnull().sum()
        if n:
            mode = df[col].mode()
            fill = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill)
            logger.info("  Categorical '%s': filled %d → mode='%s'", col, n, fill)
    return df


def clean_dataset(df: pd.DataFrame, numeric_cols: list[str],
                   categorical_cols: list[str],
                   logger: logging.Logger) -> pd.DataFrame:
    """Full cleaning pipeline: missingness filter → dedup → imputation."""
    logger.info("--- Cleaning  input: %s ---", df.shape)
    df = remove_high_missingness_rows(df, 0.5, logger)
    df = remove_duplicate_rows(df, logger)
    df = impute_missing_values(df, numeric_cols, categorical_cols, logger)
    logger.info("Post-cleaning shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def standardize_numeric(df: pd.DataFrame, numeric_cols: list[str],
                          logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    """Apply MinMaxScaler [0,1] to present numeric columns."""
    df = df.copy()
    present = [c for c in numeric_cols if c in df.columns]
    if present:
        df[present] = MinMaxScaler().fit_transform(df[present])
        logger.info("MinMaxScaler applied to: %s", present)
    return df, present


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_metadata(scaled: list[str], all_cols: list[str],
                    path: Path, logger: logging.Logger) -> None:
    """Save feature type metadata to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "numeric_columns": scaled,
        "categorical_columns": [c for c in all_cols if c not in scaled],
        "scaler": "MinMaxScaler",
        "scale_range": [0, 1],
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Feature metadata → %s", path)


# ---------------------------------------------------------------------------
# Public API  (callable from app.py / other scripts)
# ---------------------------------------------------------------------------

def run_ingest(
    raw_file: Optional[str] = None,
    dataset_type: Optional[str] = None,
    output_dir: str = "data/processed",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    End-to-end ingestion pipeline callable from app.py or other scripts.

    Args:
        raw_file:          Path to local .data/.csv file (used as fallback or
                           primary source if dataset_type cannot be inferred).
        dataset_type:      'thyroid' or 'heart'. Auto-detected from raw_file
                           name if not provided.
        output_dir:        Directory for output CSV, log, and metadata files.
        progress_callback: Optional fn(message: str, fraction: float) for UI.

    Returns:
        Tuple of (clean_df, summary_stats_dict).
    """
    t0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds_type = dataset_type or detect_dataset_type(raw_file) or "thyroid"
    prefix = ds_type  # e.g. 'thyroid' or 'heart'

    log_path = out / f"{prefix}_ingest_log.txt"
    logger = _make_logger(log_path)
    logger.info("SynthGen ingest started  dataset=%s", ds_type)

    def _cb(msg: str, frac: float) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, frac)

    # Schema
    if ds_type == "heart":
        num_cols, cat_cols = HEART_NUMERIC, HEART_CATEGORICAL
    else:
        num_cols, cat_cols = THYROID_NUMERIC, THYROID_CATEGORICAL

    _cb("Acquiring data…", 0.05)
    df_raw = acquire_data(raw_file, ds_type, logger)

    # Align unknown columns to schema
    present_num = [c for c in num_cols if c in df_raw.columns]
    present_cat = [c for c in cat_cols if c in df_raw.columns]
    known = set(present_num) | set(present_cat)
    for col in df_raw.columns:
        if col not in known:
            if pd.api.types.is_numeric_dtype(df_raw[col]):
                present_num.append(col)
            else:
                present_cat.append(col)

    _cb("Cleaning data…", 0.30)
    df_clean = clean_dataset(df_raw, present_num, present_cat, logger)

    _cb("Standardising features…", 0.65)
    df_final, scaled = standardize_numeric(df_clean, present_num, logger)

    # Save outputs
    clean_csv = out / f"{prefix}_clean.csv"
    df_final.to_csv(clean_csv, index=False)
    logger.info("Clean CSV → %s", clean_csv)
    _save_metadata(scaled, list(df_final.columns), out / "feature_metadata.json",
                   logger)

    elapsed = time.perf_counter() - t0
    stats: dict[str, Any] = {
        "dataset_type": ds_type,
        "rows": int(df_final.shape[0]),
        "columns": int(df_final.shape[1]),
        "numeric_columns": scaled,
        "output_csv": str(clean_csv),
        "elapsed_sec": round(elapsed, 2),
    }
    _cb(f"Ingest complete: {df_final.shape[0]} rows, {df_final.shape[1]} cols "
        f"({elapsed:.1f}s)", 1.0)
    return df_final, stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SynthGen Step 1 — Data Ingestion")
    p.add_argument("--dataset", choices=["thyroid", "heart"], default=None)
    p.add_argument("--input", "-i", metavar="PATH", default=None,
                   help="Local .data/.csv fallback path")
    return p.parse_args()


def main() -> None:
    """CLI wrapper around run_ingest()."""
    args = _parse_args()
    try:
        df, stats = run_ingest(raw_file=args.input, dataset_type=args.dataset)
        print(f"\n✓ Data ingestion complete. {stats['rows']} rows, "
              f"{stats['columns']} columns  → {stats['output_csv']}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
