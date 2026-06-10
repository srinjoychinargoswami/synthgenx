"""
SynthGen - Step 2: KMeans Clustering (Multi-Dataset)
=====================================================
Identifies natural clusters in any cleaned clinical dataset using numeric
features detected dynamically. Saves cluster assignments, statistics, and
publication-quality visualisations.

Usage (CLI):
    python src/02_cluster.py [--input data/processed/thyroid_clean.csv]

Callable function (for app.py):
    from src.02_cluster import run_clustering
    clustered_df, stats, pca_fig = run_clustering(clean_df, n_clusters=4,
                                                   output_dir="outputs")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Column names to always exclude from clustering features
_LABEL_HINTS = {
    "id", "patient_id", "record_id", "sample_id",
    "target", "label", "class", "diagnosis", "disease", "condition",
    "pipeline_run_id", "_dbt_model",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _make_logger(log_path: Path, name: str = "synthgen.cluster") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_numeric_features(df: pd.DataFrame,
                              logger: logging.Logger) -> list[str]:
    """
    Dynamically select numeric columns suitable for clustering.

    Excludes ID-like, label-like, and low-variance columns.

    Args:
        df: Input DataFrame.
        logger: Logger instance.

    Returns:
        List of usable numeric column names.
    """
    candidates = list(df.select_dtypes(include=[np.number]).columns)
    usable = []
    excluded = []
    for col in candidates:
        if col.lower() in _LABEL_HINTS or col.lower().endswith("_id"):
            excluded.append(col)
            continue
        if df[col].nunique() < 3:
            excluded.append(col)
            continue
        if df[col].std() < 1e-9:
            excluded.append(col)
            continue
        usable.append(col)

    logger.info("Clustering features (%d): %s", len(usable), usable)
    if excluded:
        logger.debug("Excluded from clustering: %s", excluded)
    return usable


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def fit_kmeans(X: np.ndarray, n_clusters: int,
               random_state: int = 42) -> KMeans:
    """Fit and return a KMeans model."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init="auto", max_iter=300)
    km.fit(X)
    return km


def assign_labels(df: pd.DataFrame, labels: np.ndarray,
                   n_clusters: int) -> pd.DataFrame:
    """
    Add cluster_id and diagnosis_class columns to DataFrame.

    Cluster→class mapping varies by dataset size and centroid position but
    uses a generic ordinal scheme when dataset-specific mapping isn't available.
    """
    df = df.copy()
    df["cluster_id"] = labels

    # Generic label scheme (overridable downstream)
    generic = {0: "group_A", 1: "group_B", 2: "group_C", 3: "group_D",
               4: "group_E", 5: "group_F", 6: "group_G", 7: "group_H"}

    # Dataset-aware mapping
    thyroid_map = {0: "normal", 1: "borderline",
                   2: "hyperthyroid", 3: "hypothyroid"}
    heart_map   = {0: "no_disease", 1: "mild", 2: "moderate", 3: "severe"}

    # Auto-detect by checking column presence
    cols = set(df.columns)
    if {"TSH", "T3", "TT4"}.issubset(cols) or {"tsh", "t3", "tt4"}.issubset(cols):
        label_map = thyroid_map
    elif {"trestbps", "chol", "thalach"}.issubset(cols):
        label_map = heart_map
    else:
        label_map = generic

    df["diagnosis_class"] = df["cluster_id"].map(
        lambda x: label_map.get(x, f"group_{x}")
    )
    return df


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_pca(X_scaled: np.ndarray, labels: np.ndarray,
             feature_names: list[str], output_dir: Path) -> plt.Figure:
    """PCA 2D scatter coloured by cluster."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(labels)))
    for i, lab in enumerate(np.unique(labels)):
        mask = labels == lab
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=palette[i], label=f"Cluster {lab}", alpha=0.6, s=20)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA Projection by Cluster")
    ax.legend(title="Cluster", loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_plot.png", dpi=150)
    return fig


def plot_feature_distributions(df: pd.DataFrame, feature_cols: list[str],
                                 output_dir: Path) -> plt.Figure:
    """Box plots of top features per cluster."""
    top_feats = feature_cols[:min(6, len(feature_cols))]
    n = len(top_feats)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, feat in enumerate(top_feats):
        ax = axes_flat[i]
        df.boxplot(column=feat, by="cluster_id", ax=ax,
                   patch_artist=True, notch=False)
        ax.set_title(feat)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Value")
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Feature Distributions by Cluster", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "boxplots.png", dpi=150, bbox_inches="tight")
    return fig


def plot_cluster_counts(df: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Bar chart of sample counts per cluster."""
    counts = df["cluster_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=sns.color_palette("tab10", len(counts)))
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Count")
    ax.set_title("Samples per Cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_counts.png", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_cluster_stats(df: pd.DataFrame, feature_cols: list[str],
                            km: KMeans) -> dict[str, Any]:
    """Compile cluster statistics into a JSON-serialisable dict."""
    counts = df["cluster_id"].value_counts().sort_index().to_dict()
    per_cluster: dict[str, Any] = {}
    for cid in sorted(counts.keys()):
        subset = df[df["cluster_id"] == cid][feature_cols]
        per_cluster[str(cid)] = {
            "count": int(counts[cid]),
            "diagnosis_class": str(df[df["cluster_id"] == cid]["diagnosis_class"].iloc[0]),
            "means": {c: round(float(subset[c].mean()), 4) for c in feature_cols},
            "stds": {c: round(float(subset[c].std()), 4) for c in feature_cols},
        }
    return {
        "n_clusters": int(km.n_clusters),
        "inertia": round(float(km.inertia_), 4),
        "feature_columns": feature_cols,
        "per_cluster": per_cluster,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_clustering(
    clean_data: pd.DataFrame,
    n_clusters: int = 4,
    output_dir: str = "outputs",
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[pd.DataFrame, dict[str, Any], plt.Figure]:
    """
    Full clustering pipeline callable from app.py or other scripts.

    Args:
        clean_data:        Cleaned, standardised DataFrame from run_ingest().
        n_clusters:        Number of K-means clusters.
        output_dir:        Directory for plots, stats JSON, and log.
        progress_callback: Optional fn(message, fraction) for UI progress.

    Returns:
        Tuple of (clustered_df, cluster_stats_dict, pca_figure).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = _make_logger(out / "cluster_log.txt")

    def _cb(msg: str, frac: float) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, frac)

    _cb("Selecting features…", 0.05)
    feat_cols = select_numeric_features(clean_data, logger)
    if len(feat_cols) < 2:
        raise ValueError("Need at least 2 numeric features for clustering.")

    _cb("Scaling features…", 0.15)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_data[feat_cols].fillna(0))

    _cb(f"Fitting KMeans (k={n_clusters})…", 0.30)
    km = fit_kmeans(X_scaled, n_clusters)
    labels = km.labels_

    _cb("Assigning cluster labels…", 0.55)
    clustered_df = assign_labels(clean_data, labels, n_clusters)

    _cb("Computing statistics…", 0.65)
    stats = compute_cluster_stats(clustered_df, feat_cols, km)
    (out / "cluster_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    _cb("Generating plots…", 0.75)
    pca_fig = plot_pca(X_scaled, labels, feat_cols, out)
    plot_feature_distributions(clustered_df, feat_cols, out)
    plot_cluster_counts(clustered_df, out)

    # Save clustered CSV
    # Infer prefix from an existing column or fall back to 'dataset'
    prefix = "thyroid" if "TSH" in clustered_df.columns or "tsh" in clustered_df.columns else (
        "heart" if "trestbps" in clustered_df.columns else "dataset"
    )
    out_csv = Path("data/processed") / f"{prefix}_clustered.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clustered_df.to_csv(out_csv, index=False)
    logger.info("Clustered CSV → %s", out_csv)

    _cb(f"Clustering complete: {n_clusters} clusters over {len(feat_cols)} features.",
        1.0)
    return clustered_df, stats, pca_fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="SynthGen Step 2 — Clustering")
    p.add_argument("--input", default="data/processed/thyroid_clean.csv")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    clustered, stats, _ = run_clustering(df, n_clusters=args.k,
                                          output_dir=args.output_dir)
    print(f"\n✓ Clustering complete. {args.k} clusters, {len(df)} rows.")
    for cid, info in stats["per_cluster"].items():
        print(f"  Cluster {cid} ({info['diagnosis_class']}): {info['count']} samples")


if __name__ == "__main__":
    main()
