"""
SynthGenX — Streamlit Application
===================================
Interactive UI for the Synthetic Clinical Data Generation Pipeline.

Features:
  • Multi-dataset support (Thyroid, Heart Disease, custom CSV upload)
  • Full pipeline: Ingest → Cluster → Synthetic Gen → Validate → LLM Notes
  • Real-time progress indicators
  • System status sidebar with integration check
  • Downloadable outputs
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional
import uuid 

# Add src/ to path so pipeline modules are importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SynthGenX - Synthetic Clinical Data Generator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached pipeline imports
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _import_pipeline() -> dict[str, Any]:
    """
    Import all pipeline modules once and cache the callables.
    Returns a dict of callable functions keyed by module name.
    """
    fns: dict[str, Any] = {}
    try:
        from importlib.util import spec_from_file_location, module_from_spec
        src = Path(__file__).parent / "src"

        def _load(name: str, file: str):
            spec = spec_from_file_location(name, src / file)
            if spec and spec.loader:
                mod = module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                return mod
            return None

        m1 = _load("ingest",    "01_ingest.py");        fns["ingest"]    = m1.run_ingest    if m1 else None
        m2 = _load("cluster",   "02_cluster.py");       fns["cluster"]   = m2.run_clustering if m2 else None
        m3 = _load("synthgen",  "03_synthetic_gen.py"); fns["synthgen"]  = m3.run_synthetic_gen if m3 else None
        m4 = _load("validate",  "04_validate.py")
        if m4 and hasattr(m4, "run_validation"):
            fns["validate"] = m4.run_validation
        elif m4:
            fns["validate"] = _make_validation_adapter(m4)
        else:
            fns["validate"] = None
        m5 = _load("llm_notes", "03b_llm_notes.py");    fns["llm_notes"] = m5.run_llm_notes if m5 else None
    except Exception as exc:
        st.warning(f"Some pipeline modules could not be loaded: {exc}")
    return fns


def _make_validation_adapter(validate_module: types.ModuleType):
    """
    Adapt src/04_validate.py's actual function set to the app's expected
    callable shape: (metrics, plots, html).
    """
    def _adapter(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        output_dir: str = "outputs",
        progress_callback=None,
    ) -> tuple[dict, list, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def cb(msg: str, frac: float) -> None:
            if progress_callback:
                progress_callback(msg, frac)

        cb("Preparing validation samples…", 0.10)
        real_df = real_data.copy()
        synth_df = synthetic_data.copy()
        if "is_synthetic" in synth_df.columns:
            mask = _bool_mask(synth_df["is_synthetic"])
            synth_df = synth_df.loc[mask].copy()

        numeric = [
            col for col in getattr(validate_module, "NUMERIC_FEATURES", [])
            if col in real_df.columns and col in synth_df.columns
        ]
        if not numeric:
            numeric = sorted(
                set(real_df.select_dtypes(include=[np.number]).columns)
                & set(synth_df.select_dtypes(include=[np.number]).columns)
            )
        if not numeric:
            raise ValueError("No shared numeric columns available for validation.")

        real_sample = real_df[numeric].dropna(how="all").sample(
            n=min(100, len(real_df)), random_state=42
        )
        synth_sample = synth_df[numeric].dropna(how="all").sample(
            n=min(500, len(synth_df)), random_state=42
        )

        # Rebind module-level feature/output settings for generic app calls.
        validate_module.NUMERIC_FEATURES = numeric
        validate_module.PLOT_FEATURES = numeric[:6]
        validate_module.QQ_FEATURES = numeric[:4]
        validate_module.OUTPUT_DIR = out
        validate_module.HISTOGRAM_PNG = out / "distribution_histograms.png"
        validate_module.BOXPLOTS_PNG = out / "distribution_boxplots.png"
        validate_module.QQ_PNG = out / "qq_plots.png"
        validate_module.HTML_REPORT = out / "validation_report.html"
        validate_module.JSON_REPORT = out / "validation_metrics.json"

        cb("Running KS tests…", 0.35)
        metrics = validate_module.run_statistical_validation(real_sample, synth_sample)
        cb("Generating validation plots…", 0.70)
        validate_module.set_plot_style()
        validate_module.create_histograms(real_sample, synth_sample, metrics, validate_module.HISTOGRAM_PNG)
        validate_module.create_boxplots(real_sample, synth_sample, validate_module.BOXPLOTS_PNG)
        validate_module.create_qq_plots(real_sample, validate_module.QQ_PNG)
        validate_module.generate_html_report(metrics, validate_module.HTML_REPORT)
        validate_module.save_json_report(metrics, validate_module.JSON_REPORT)
        cb("Validation complete.", 1.0)

        html = validate_module.HTML_REPORT.read_text(encoding="utf-8")
        return metrics, [], html

    return _adapter


@st.cache_data(show_spinner=False)
def _run_integration_check() -> dict[str, Any]:
    """Run and cache the integration check."""
    try:
        from importlib.util import spec_from_file_location, module_from_spec
        ic_path = Path(__file__).parent / "integration_check.py"
        spec = spec_from_file_location("integration_check", ic_path)
        if spec and spec.loader:
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            return mod.run_integration_check()
    except Exception as exc:
        return {"all_critical_pass": False, "checks": [],
                "summary": {}, "critical_failures": [str(exc)]}
    return {"all_critical_pass": True, "checks": [], "summary": {}}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return user configuration dict."""
    st.sidebar.markdown("## 🧬 SynthGenX")
    st.sidebar.markdown("SynthGenX Configuration")
    st.sidebar.caption("Generate realistic synthetic clinical datasets for ML training.")
    cfg: dict[str, Any] = {}
    cfg["run_full_pipeline"] = st.sidebar.button(
        "🚀 Run Full Pipeline",
        type="primary",
        use_container_width=True,
        help="Run ingest, clustering, synthetic generation, validation, and optional LLM notes in sequence.",
    )
    if st.sidebar.button("Reset Results", use_container_width=True):
        st.session_state.pop("pipeline_state", None)
        st.session_state.pop("full_pipeline_status", None)
        st.rerun()
    st.sidebar.divider()

    # ── System status ──────────────────────────────────────────────────────
    with st.sidebar.expander("System Status", expanded=False):
        check = _run_integration_check()
        ICONS = {"ok": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}
        if check.get("checks"):
            for c in check["checks"]:
                icon = ICONS.get(c["status"], "❓")
                st.markdown(f"{icon} {c['message']}")
        s = check.get("summary", {})
        st.caption(f"{s.get('ok',0)} ok · {s.get('warn',0)} warn · {s.get('fail',0)} fail")

    st.sidebar.divider()

    # ── Dataset selection ──────────────────────────────────────────────────
    st.sidebar.markdown("### Dataset")
    uploaded = st.sidebar.file_uploader(
        "Upload your own CSV", type=["csv"],
        help="At least 100 rows recommended for clustering."
    )

    if uploaded is not None:
        cfg["source"] = "upload"
        cfg["uploaded_file"] = uploaded
        cfg["dataset_type"] = "generic"
        st.sidebar.success(f"Uploaded: {uploaded.name}")
    else:
        cfg["source"] = "builtin"
        cfg["dataset_type"] = st.sidebar.selectbox(
            "Dataset", ["thyroid", "heart"],
            format_func=lambda x: "🦋 Thyroid Disease" if x == "thyroid" else "❤️ Heart Disease",
        )
        local_file = st.sidebar.text_input(
            "Local data file (optional fallback)",
            placeholder=f"data/raw/{cfg['dataset_type']}_disease.data",
        )
        cfg["local_file"] = local_file or None

    st.sidebar.divider()

    # ── Pipeline parameters ────────────────────────────────────────────────
    st.sidebar.markdown("### Pipeline Settings")
    cfg["n_clusters"]  = st.sidebar.slider("Clusters (K)", 2, 8, 4)
    cfg["n_samples"]   = st.sidebar.slider("Synthetic samples", 100, 2000, 500, step=100)

    st.sidebar.divider()

    # ── LLM settings ──────────────────────────────────────────────────────
    st.sidebar.markdown("### LLM Clinical Notes")
    cfg["api_key"] = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required to generate clinical notes with Claude.",
    )
    cfg["llm_model"] = st.sidebar.selectbox(
        "Model",
        ["claude-haiku-4-5-20251001"],
    )
    cfg["llm_limit"] = st.sidebar.slider("Max notes to generate", 10, 500, 200, step=10)

    st.sidebar.divider()
    st.sidebar.caption("SynthGenX · Hackathon 2026")
    return cfg


# ---------------------------------------------------------------------------
# Custom upload handling
# ---------------------------------------------------------------------------

def load_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load, validate, and preview a user-uploaded CSV.

    Returns (df, metadata_dict).
    """
    df = pd.read_csv(uploaded_file)

    if len(df) < 10:
        st.error("File has fewer than 10 rows — too small to process.")
        st.stop()
    if len(df) < 100:
        st.warning(f"Only {len(df)} rows — clustering quality may be limited.")

    # Auto-detect column types
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    # Impute missing values inline
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    meta = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "dataset_type": "generic",
        "output_csv": "uploaded_data",
    }
    return df, meta


def _bool_mask(series: pd.Series) -> pd.Series:
    """Parse bool-like values after CSV round-trips."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


# ---------------------------------------------------------------------------
# Pipeline step runners (with Streamlit progress)
# ---------------------------------------------------------------------------

def run_ingest_step(cfg: dict[str, Any]) -> tuple[Optional[pd.DataFrame], dict]:
    """Run ingest step, return (df, stats)."""
    fns = _import_pipeline()
    if fns.get("ingest") is None:
        st.error("src/01_ingest.py could not be loaded.")
        return None, {}

    progress_bar = st.progress(0.0, text="Starting ingestion…")
    status_text  = st.empty()

    def cb(msg: str, frac: float):
        progress_bar.progress(min(frac, 1.0), text=msg)
        status_text.caption(msg)

    try:
        df, stats = fns["ingest"](
            raw_file=cfg.get("local_file"),
            dataset_type=cfg.get("dataset_type", "thyroid"),
            output_dir="data/processed",
            progress_callback=cb,
        )
        progress_bar.progress(1.0, text="Ingestion complete ✓")
        return df, stats
    except Exception as exc:
        st.error(f"Ingestion failed: {exc}")
        return None, {}


def run_cluster_step(df: pd.DataFrame,
                     cfg: dict[str, Any]) -> tuple[Optional[pd.DataFrame], dict, Any]:
    """Run clustering step, return (clustered_df, stats, pca_fig)."""
    fns = _import_pipeline()
    if fns.get("cluster") is None:
        st.error("src/02_cluster.py could not be loaded.")
        return None, {}, None

    progress_bar = st.progress(0.0, text="Starting clustering…")
    status_text  = st.empty()

    def cb(msg: str, frac: float):
        progress_bar.progress(min(frac, 1.0), text=msg)
        status_text.caption(msg)

    try:
        clustered, stats, pca_fig = fns["cluster"](
            clean_data=df,
            n_clusters=cfg.get("n_clusters", 4),
            output_dir="outputs",
            progress_callback=cb,
        )
        progress_bar.progress(1.0, text="Clustering complete ✓")
        return clustered, stats, pca_fig
    except Exception as exc:
        st.error(f"Clustering failed: {exc}")
        return None, {}, None


def run_synthgen_step(clustered: pd.DataFrame,
                      cfg: dict[str, Any]) -> tuple[Optional[pd.DataFrame], dict, Any]:
    """Run synthetic generation step."""
    fns = _import_pipeline()
    if fns.get("synthgen") is None:
        st.error("src/03_synthetic_gen.py could not be loaded.")
        return None, {}, None

    progress_bar = st.progress(0.0, text="Starting synthetic generation…")
    status_text  = st.empty()

    def cb(msg: str, frac: float | None = None):
        # src/03_synthetic_gen.py now calls progress_callback(message), while
        # older steps call progress_callback(message, fraction). Support both.
        current = 0.15 if frac is None else min(frac, 1.0)
        progress_bar.progress(current, text=msg)
        status_text.caption(msg)

    try:
        result = fns["synthgen"](
            clustered_data=clustered,
            n_samples=cfg.get("n_samples", 500),
            borderline_cluster=cfg.get("borderline_cluster", 1),
            output_dir="outputs",
            progress_callback=cb,
        )
        if not isinstance(result, dict):
            st.error(
                "Synthetic generation failed: run_synthetic_gen returned an "
                f"unexpected {type(result).__name__}; expected dict."
            )
            return None, {}, None

        if not result.get("success", False):
            st.error(f"Synthetic generation failed: {result.get('error', 'Unknown error')}")
            return None, result.get("report", {}) or {}, None

        synth_df = result.get("synthetic_data")
        report = result.get("report", {}) or {}
        if not isinstance(synth_df, pd.DataFrame):
            st.error(
                "Synthetic generation failed: result['synthetic_data'] is not "
                f"a DataFrame (got {type(synth_df).__name__})."
            )
            return None, report, None
        if synth_df.empty:
            st.error("Synthetic generation failed: result['synthetic_data'] is empty.")
            return None, report, None

        progress_bar.progress(1.0, text="Synthetic generation complete ✓")
        return synth_df, report, None
    except Exception as exc:
        st.error(f"Synthetic generation failed: {exc}")
        return None, {}, None


def run_validate_step(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                      cfg: dict[str, Any]) -> tuple[dict, list, str]:
    """Run validation step."""
    fns = _import_pipeline()
    if fns.get("validate") is None:
        st.error("src/04_validate.py could not be loaded.")
        return {}, [], ""

    progress_bar = st.progress(0.0, text="Validating distributions…")
    status_text  = st.empty()

    def cb(msg: str, frac: float):
        progress_bar.progress(min(frac, 1.0), text=msg)
        status_text.caption(msg)

    try:
        synth_for_validation = synth_df
        if "is_synthetic" in synth_df.columns:
            synth_for_validation = synth_df.loc[_bool_mask(synth_df["is_synthetic"])].copy()
            if synth_for_validation.empty:
                st.error("Validation failed: no rows with is_synthetic == True found.")
                return {}, [], ""

        metrics, plots, html = fns["validate"](
            real_data=real_df,
            synthetic_data=synth_for_validation,
            output_dir="outputs",
            progress_callback=cb,
        )
        progress_bar.progress(1.0, text="Validation complete ✓")
        return metrics, plots, html
    except Exception as exc:
        st.error(f"Validation failed: {exc}")
        return {}, [], ""


def run_llm_step(synth_df: pd.DataFrame,
                 cfg: dict[str, Any]) -> tuple[Optional[pd.DataFrame], list[str]]:
    """Run LLM clinical notes step."""
    fns = _import_pipeline()
    api_key = cfg.get("api_key", "").strip()

    if not api_key:
        st.info("No Anthropic API key provided — skipping clinical note generation.")
        return synth_df, []

    if fns.get("llm_notes") is None:
        st.error("src/03b_llm_notes.py could not be loaded.")
        return synth_df, []

    progress_bar = st.progress(0.0, text="Generating clinical notes…")
    status_text  = st.empty()
    limit = cfg.get("llm_limit", 200)

    def cb(msg: str, frac: float):
        progress_bar.progress(min(frac, 1.0), text=msg)
        status_text.caption(msg)

    try:
        input_for_notes = synth_df
        original_index = None
        if "is_synthetic" in synth_df.columns:
            mask = _bool_mask(synth_df["is_synthetic"])
            input_for_notes = synth_df.loc[mask].copy()
            original_index = input_for_notes.index.copy()
            if input_for_notes.empty:
                st.error("LLM note generation failed: no synthetic rows found.")
                return synth_df, []

        df_out, sample_notes = fns["llm_notes"](
            synthetic_data=input_for_notes,
            api_key=api_key,
            model=cfg.get("llm_model", "claude-haiku-4-5-20251001"),
            limit=limit,
            dataset_type=cfg.get("dataset_type", "thyroid"),
            output_dir="data/processed",
            progress_callback=cb,
        )
        if original_index is not None and "clinical_note" in df_out.columns:
            merged = synth_df.copy()
            if "clinical_note" not in merged.columns:
                merged["clinical_note"] = ""
            note_values = df_out["clinical_note"].to_numpy()
            merged.loc[original_index[:len(note_values)], "clinical_note"] = note_values
            df_out = merged
        progress_bar.progress(1.0, text=f"Generated {limit} clinical notes ✓")
        return df_out, sample_notes
    except ImportError:
        st.warning("Anthropic SDK not installed. Run: `pip install anthropic`")
        return synth_df, []
    except Exception as exc:
        st.error(f"LLM note generation failed: {exc}")
        return synth_df, []


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def render_ingest_results(df: pd.DataFrame, stats: dict) -> None:
    """Display ingest step results."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{stats.get('rows', len(df)):,}")
    c2.metric("Columns", stats.get('columns', len(df.columns)))
    c3.metric("Numeric features", len(stats.get('numeric_columns', [])))

    with st.expander("Data preview"):
        st.dataframe(df.head(10), use_container_width=True)
    with st.expander("Summary statistics"):
        st.dataframe(df.describe(), use_container_width=True)


def render_cluster_results(clustered: pd.DataFrame, stats: dict,
                            pca_fig: Any) -> None:
    """Display clustering results."""
    per_cluster = stats.get("per_cluster", {})

    cols = st.columns(min(4, len(per_cluster)))
    for i, (cid, info) in enumerate(per_cluster.items()):
        if i < len(cols):
            cols[i].metric(
                f"Cluster {cid}",
                info["count"],
                delta=info["diagnosis_class"],
            )

    col_plot, col_data = st.columns([1.2, 1])
    with col_plot:
        if pca_fig is not None:
            st.pyplot(pca_fig)
        elif Path("outputs/pca_plot.png").exists():
            st.image("outputs/pca_plot.png", caption="PCA Projection")

    with col_data:
        st.markdown("**Cluster Summary**")
        rows = []
        for cid, info in per_cluster.items():
            rows.append({
                "Cluster": cid,
                "Class": info["diagnosis_class"],
                "Count": info["count"],
                "Inertia": stats.get("inertia", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if Path("outputs/boxplots.png").exists():
        with st.expander("Feature distributions"):
            st.image("outputs/boxplots.png")


def render_synthgen_results(synth_df: pd.DataFrame, val_results: dict,
                             comp_fig: Any) -> None:
    """Display synthetic generation results."""
    if "is_synthetic" in synth_df.columns:
        synthetic_count = int(_bool_mask(synth_df["is_synthetic"]).sum())
        real_count = int(len(synth_df) - synthetic_count)
    else:
        synthetic_count = len(synth_df)
        real_count = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Synthetic rows", f"{synthetic_count:,}")
    c2.metric("Combined rows", f"{len(synth_df):,}")
    c3.metric("Features", len(synth_df.columns))

    if val_results:
        validation = val_results.get("validation", {})
        status = "PASS" if validation.get("passed") else "REVIEW"
        st.caption(f"Generation report status: {status} · Real rows retained: {real_count:,}")

    with st.expander("Synthetic data preview"):
        if "is_synthetic" in synth_df.columns:
            preview = synth_df.loc[_bool_mask(synth_df["is_synthetic"])].head(10)
        else:
            preview = synth_df.head(10)
        st.dataframe(preview, use_container_width=True)

    if val_results:
        with st.expander("Synthetic generation report"):
            st.json(val_results)

    if comp_fig is not None:
        st.pyplot(comp_fig)


def render_validation_results(metrics: dict, plots: list, html: str) -> None:
    """Display validation metrics and plots."""
    if not metrics:
        st.info("No validation metrics available.")
        return

    # Show top-level metrics as cards
    key_metrics = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float, str)) and not k.startswith("_")}
    cols = st.columns(min(4, max(1, len(key_metrics))))
    for i, (k, v) in enumerate(list(key_metrics.items())[:4]):
        if i < len(cols):
            display_v = f"{v:.4f}" if isinstance(v, float) else str(v)
            cols[i].metric(k.replace("_", " ").title(), display_v)

    # Distribution plots
    for fig in (plots or []):
        try:
            st.pyplot(fig)
        except Exception:
            pass

    # HTML report download
    if html:
        st.download_button(
            "Download Validation Report (HTML)",
            data=html,
            file_name="validation_report.html",
            mime="text/html",
            key=_unique_download_key("validation_report.html", "validation_tab"),
        )


def render_llm_results(df_with_notes: pd.DataFrame,
                        sample_notes: list[str]) -> None:
    """Display generated clinical notes."""
    filled = (df_with_notes.get("clinical_note", pd.Series(dtype=str))
              .str.strip().ne("").sum()
              if "clinical_note" in df_with_notes.columns else 0)

    st.metric("Notes generated", filled)

    if sample_notes:
        st.markdown("#### Sample Clinical Notes")
        for i, note in enumerate(sample_notes[:3], 1):
            with st.expander(f"Note {i}"):
                st.write(note)

    elif "clinical_note" in df_with_notes.columns:
        nonempty = df_with_notes[
            df_with_notes["clinical_note"].str.strip().ne("")
        ]
        if not nonempty.empty:
            st.markdown("#### Sample Clinical Notes")
            for i, (_, row) in enumerate(nonempty.head(3).iterrows(), 1):
                with st.expander(f"Note {i}"):
                    st.write(row["clinical_note"])


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _unique_download_key(file_name: str, suffix: str | int = "") -> str:
    """Return a collision-proof key for Streamlit download buttons."""
    safe_name = str(file_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
    unique = f"{int(time.time() * 1_000_000)}_{uuid.uuid4().hex}"
    return f"download_{safe_name}_{suffix}_{unique}"


def render_downloads(state: dict[str, Any]) -> None:
    """
    Render all available download buttons safely.

    This function can be called multiple times in one Streamlit render cycle
    from both the Downloads tab and the unified Results tab. Every button gets
    a timestamp/UUID key, so duplicate element keys cannot occur.
    """
    st.markdown("### Downloads")
    cols = st.columns(3)

    if not isinstance(state, dict):
        st.error("Download state is invalid; expected a dictionary.")
        return

    i = 0

    def add_download(label: str, data: bytes, file_name: str, mime: str) -> None:
        """Create one uniquely keyed download button."""
        nonlocal i
        if not data:
            st.warning(f"`{file_name}` is empty or unavailable.")
            return
        cols[i % 3].download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=_unique_download_key(file_name, i),
        )
        i += 1

    # DataFrames stored in session state.
    for label, key in [
        ("Clean Data CSV",          "clean_df"),
        ("Clustered Data CSV",       "clustered_df"),
        ("Synthetic Data CSV",       "synth_df"),
        ("Synthetic + Notes CSV",    "synth_notes_df"),
    ]:
        df = state.get(key)
        if df is not None and isinstance(df, pd.DataFrame):
            add_download(
                label=f"⬇️ {label}",
                data=_df_to_csv_bytes(df),
                file_name=f"{key}.csv",
                mime="text/csv",
            )

    # JSON-like objects stored in session state.
    for label, key in [
        ("Cluster Stats JSON", "cluster_stats"),
        ("Synthetic Report JSON", "val_results"),
        ("Validation Metrics JSON", "val_metrics"),
    ]:
        obj = state.get(key)
        if obj:
            try:
                add_download(
                    label=f"⬇️ {label}",
                    data=json.dumps(obj, indent=2, default=str).encode("utf-8"),
                    file_name=f"{key}.json",
                    mime="application/json",
                )
            except (TypeError, ValueError) as exc:
                st.warning(f"Could not serialize `{key}`: {exc}")

    # Files generated by the pipeline.
    file_downloads = [
        ("Cluster Stats File", Path("outputs/cluster_stats.json"), "application/json"),
        ("Synthetic Report File", Path("outputs/synthetic_report.json"), "application/json"),
        ("Validation Metrics File", Path("outputs/validation_metrics.json"), "application/json"),
        ("Validation HTML Report", Path("outputs/validation_report.html"), "text/html"),
        ("PCA Plot", Path("outputs/pca_plot.png"), "image/png"),
        ("Cluster Boxplots", Path("outputs/boxplots.png"), "image/png"),
        ("Cluster Counts", Path("outputs/cluster_counts.png"), "image/png"),
        ("Distribution Comparison", Path("outputs/distribution_comparison.png"), "image/png"),
        ("Synthetic Validation Plot", Path("outputs/synthetic_validation_plots.png"), "image/png"),
        ("Validation Histograms", Path("outputs/distribution_histograms.png"), "image/png"),
        ("Validation Boxplots", Path("outputs/distribution_boxplots.png"), "image/png"),
        ("Q-Q Plots", Path("outputs/qq_plots.png"), "image/png"),
    ]
    for label, path, mime in file_downloads:
        if path.exists() and path.is_file():
            try:
                add_download(
                    label=f"⬇️ {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                )
            except OSError as exc:
                st.warning(f"Could not read `{path}`: {exc}")

    if "_zip_all_results" in globals():
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            add_download(
                label="⬇️ Download All Results ZIP",
                data=_zip_all_results(state),
                file_name=f"synthgenx_results_{timestamp}.zip",
                mime="application/zip",
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not create zip bundle: {exc}")

    if i == 0:
        st.info("No downloadable outputs are available yet. Run the pipeline first.")


def run_full_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the full SynthGenX pipeline and persist outputs in session state.

    This additive helper reuses the existing step functions, so the original
    tab-by-tab workflow remains available. It stops on the first failed step and
    returns a simple success/error status for the sidebar-triggered run.
    """
    state = st.session_state.setdefault("pipeline_state", {})
    status_box = st.container()
    overall = st.progress(0.0, text="Starting SynthGenX full pipeline...")

    def mark(step: str, frac: float) -> None:
        overall.progress(frac, text=step)
        status_box.info(step)

    try:
        mark("Step 1/5: Ingesting data...", 0.05)
        if cfg["source"] == "upload":
            cfg["uploaded_file"].seek(0)
            clean_df, ingest_meta = load_uploaded_csv(cfg["uploaded_file"])
        else:
            clean_df, ingest_meta = run_ingest_step(cfg)
        if clean_df is None:
            raise RuntimeError("Ingestion failed.")
        state["clean_df"] = clean_df
        state["ingest_meta"] = ingest_meta
        state["cfg"] = cfg
        mark("Step 1/5 complete: data ingested.", 0.20)

        mark("Step 2/5: Clustering records...", 0.25)
        clustered_df, cluster_stats, pca_fig = run_cluster_step(clean_df, cfg)
        if clustered_df is None:
            raise RuntimeError("Clustering failed.")
        state["clustered_df"] = clustered_df
        state["cluster_stats"] = cluster_stats
        state["pca_fig"] = pca_fig
        mark("Step 2/5 complete: clusters identified.", 0.40)

        mark("Step 3/5: Generating synthetic data...", 0.45)
        synth_df, synth_report, comp_fig = run_synthgen_step(clustered_df, cfg)
        if synth_df is None:
            raise RuntimeError("Synthetic generation failed.")
        state["synth_df"] = synth_df
        state["val_results"] = synth_report
        state["comp_fig"] = comp_fig
        mark("Step 3/5 complete: synthetic data generated.", 0.65)

        mark("Step 4/5: Validating distributions...", 0.70)
        metrics, plots, html = run_validate_step(clean_df, synth_df, cfg)
        if not metrics:
            raise RuntimeError("Validation failed.")
        state["val_metrics"] = metrics
        state["val_plots"] = plots
        state["val_html"] = html
        mark("Step 4/5 complete: validation finished.", 0.84)

        api_key = cfg.get("api_key", "").strip()
        if api_key:
            mark("Step 5/5: Generating clinical notes...", 0.88)
            notes_df, sample_notes = run_llm_step(synth_df, cfg)
            if notes_df is not None:
                state["synth_notes_df"] = notes_df
                state["sample_notes"] = sample_notes
            mark("Step 5/5 complete: clinical notes generated.", 0.97)
        else:
            state.pop("synth_notes_df", None)
            state.pop("sample_notes", None)
            mark("Step 5/5 skipped: no API key provided.", 0.97)

        overall.progress(1.0, text="SynthGenX full pipeline complete.")
        st.session_state["full_pipeline_status"] = {"success": True, "error": None}
        return {"success": True, "results": state}

    except Exception as exc:
        error = str(exc)
        st.error(f"Full pipeline failed: {error}")
        st.session_state["full_pipeline_status"] = {"success": False, "error": error}
        return {"success": False, "error": error}


def _synthetic_count(df: Optional[pd.DataFrame]) -> int:
    """Return synthetic row count from a combined or synthetic-only DataFrame."""
    if df is None or not isinstance(df, pd.DataFrame):
        return 0
    if "is_synthetic" in df.columns:
        return int(_bool_mask(df["is_synthetic"]).sum())
    return len(df)


def _real_count(clean_df: Optional[pd.DataFrame], synth_df: Optional[pd.DataFrame]) -> int:
    """Return real row count for summary cards."""
    if isinstance(synth_df, pd.DataFrame) and "is_synthetic" in synth_df.columns:
        return int((~_bool_mask(synth_df["is_synthetic"])).sum())
    return len(clean_df) if isinstance(clean_df, pd.DataFrame) else 0


def _zip_all_results(state: dict[str, Any]) -> bytes:
    """Create an in-memory zip containing CSVs, JSON reports, and plot files."""
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, filename in [
            ("clean_df", "clean_data.csv"),
            ("clustered_df", "clustered_data.csv"),
            ("synth_df", "synthetic_data.csv"),
            ("synth_notes_df", "synthetic_data_with_notes.csv"),
        ]:
            df = state.get(key)
            if isinstance(df, pd.DataFrame):
                zf.writestr(filename, df.to_csv(index=False))

        for key, filename in [
            ("cluster_stats", "cluster_stats.json"),
            ("val_results", "synthetic_report.json"),
            ("val_metrics", "validation_metrics.json"),
        ]:
            obj = state.get(key)
            if obj:
                zf.writestr(filename, json.dumps(obj, indent=2, default=str))

        for plot_path in [
            "outputs/pca_plot.png",
            "outputs/boxplots.png",
            "outputs/cluster_counts.png",
            "outputs/distribution_comparison.png",
            "outputs/synthetic_validation_plots.png",
            "outputs/distribution_histograms.png",
            "outputs/distribution_boxplots.png",
            "outputs/qq_plots.png",
            "outputs/validation_report.html",
        ]:
            path = Path(plot_path)
            if path.exists():
                zf.write(path, arcname=path.name)
    buffer.seek(0)
    return buffer.getvalue()


def render_unified_results(state: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Render the additive one-page SynthGenX Results tab defensively."""
    st.markdown("### 📊 SynthGenX Results")

    if not isinstance(state, dict):
        st.error("Results state is invalid; expected a dictionary.")
        return
    if not isinstance(cfg, dict):
        cfg = {}

    clean_df = state.get("clean_df")
    clustered_df = state.get("clustered_df")
    synth_df = state.get("synth_df")
    notes_df = state.get("synth_notes_df")
    metrics = state.get("val_metrics", {})
    cluster_stats = state.get("cluster_stats", {})

    has_any_dataframe = any(
        isinstance(df, pd.DataFrame) and not df.empty
        for df in [clean_df, clustered_df, synth_df, notes_df]
    )
    if not has_any_dataframe:
        st.info("Run the full pipeline or individual steps to populate this Results page.")
        return

    st.markdown("#### Executive Summary")
    pass_rate = 0.0
    if isinstance(metrics, dict):
        summary = metrics.get("summary", {})
        if isinstance(summary, dict):
            pass_rate = float(summary.get("pass_rate", 0) or 0)
        elif "pass_rate" in metrics:
            pass_rate = float(metrics.get("pass_rate", 0) or 0)
    pass_rate_display = pass_rate * 100 if 0 < pass_rate <= 1 else pass_rate

    notes_generated = 0
    if isinstance(notes_df, pd.DataFrame) and "clinical_note" in notes_df.columns:
        notes_generated = int(
            notes_df["clinical_note"].fillna("").astype(str).str.strip().ne("").sum()
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Real samples", f"{_real_count(clean_df, synth_df):,}")
    c2.metric("Synthetic samples", f"{_synthetic_count(synth_df):,}")
    total_samples = len(synth_df) if isinstance(synth_df, pd.DataFrame) else (
        len(clean_df) if isinstance(clean_df, pd.DataFrame) else 0
    )
    c3.metric("Total samples", f"{total_samples:,}")
    c4.metric("Validation pass rate", f"{pass_rate_display:.1f}%")
    c5.metric("LLM notes", f"{notes_generated:,}")

    with st.expander("Section 2 · Data Overview", expanded=True):
        dataset_name = cfg.get("dataset_type") or cfg.get("source") or "unknown"
        st.write(f"**Dataset:** {str(dataset_name).title()}")

        ingest_meta = state.get("ingest_meta", {})
        if not isinstance(ingest_meta, dict):
            ingest_meta = {}
        before_rows = ingest_meta.get(
            "rows",
            len(clean_df) if isinstance(clean_df, pd.DataFrame) else 0,
        )
        after_shape = clean_df.shape if isinstance(clean_df, pd.DataFrame) else ("—", "—")
        st.write(f"**Shape before/after:** {before_rows:,} rows → {after_shape}")

        if isinstance(clustered_df, pd.DataFrame) and not clustered_df.empty:
            cluster_col = None
            for candidate in ["cluster", "cluster_id", "Cluster"]:
                if candidate in clustered_df.columns:
                    cluster_col = candidate
                    break
            if cluster_col:
                counts = clustered_df[cluster_col].value_counts().sort_index()
                st.bar_chart(counts)
            else:
                st.warning("No cluster column found in clustered data.")
        else:
            st.info("Clustered data is not available yet.")

    with st.expander("Section 3 · Clustering Analysis", expanded=True):
        if isinstance(cluster_stats, dict) and cluster_stats:
            per_cluster = cluster_stats.get("per_cluster", {})
            if isinstance(per_cluster, dict) and per_cluster:
                st.dataframe(
                    pd.DataFrame.from_dict(per_cluster, orient="index"),
                    use_container_width=True,
                )
            else:
                st.json(cluster_stats)
        else:
            st.info("Cluster statistics are not available yet.")

        pca_fig = state.get("pca_fig")
        if pca_fig is not None:
            try:
                st.pyplot(pca_fig)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not display PCA figure from memory: {exc}")
        elif Path("outputs/pca_plot.png").exists():
            st.image("outputs/pca_plot.png", caption="PCA Plot")
        else:
            st.info("PCA plot is not available yet.")

    with st.expander("Section 4 · Synthetic Generation", expanded=True):
        for plot_path, caption in [
            ("outputs/distribution_comparison.png", "Distribution Comparison"),
            ("outputs/synthetic_validation_plots.png", "Synthetic Validation Plots"),
        ]:
            if Path(plot_path).exists():
                st.image(plot_path, caption=caption)

        if isinstance(synth_df, pd.DataFrame) and not synth_df.empty:
            if "is_synthetic" in synth_df.columns:
                preview = synth_df.loc[_bool_mask(synth_df["is_synthetic"])].head(10)
                if preview.empty:
                    preview = synth_df.head(10)
            else:
                preview = synth_df.head(10)
            st.dataframe(preview, use_container_width=True)
        else:
            st.info("Synthetic data is not available yet.")

    with st.expander("Section 5 · Validation Results", expanded=True):
        if isinstance(metrics, dict) and isinstance(metrics.get("features"), dict):
            rows = []
            for feature, vals in metrics["features"].items():
                vals = vals if isinstance(vals, dict) else {}
                rows.append(
                    {
                        "Feature": feature,
                        "KS statistic": vals.get("ks_statistic", vals.get("statistic")),
                        "p-value": vals.get("p_value", vals.get("ks_pvalue")),
                        "Mean diff %": vals.get("mean_diff_pct", vals.get("mean_difference_pct")),
                        "Status": vals.get("status", "UNKNOWN"),
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Validation metrics are not available yet.")

        for plot_path, caption in [
            ("outputs/distribution_histograms.png", "Distribution Histograms"),
            ("outputs/distribution_boxplots.png", "Distribution Boxplots"),
            ("outputs/qq_plots.png", "Q-Q Plots"),
        ]:
            if Path(plot_path).exists():
                st.image(plot_path, caption=caption)

    with st.expander("Section 6 · Clinical Notes", expanded=notes_generated > 0):
        if notes_generated == 0:
            st.info("No clinical notes generated yet. Add an API key and run the LLM step or full pipeline.")
        elif isinstance(notes_df, pd.DataFrame) and "clinical_note" in notes_df.columns:
            nonempty = notes_df[
                notes_df["clinical_note"].fillna("").astype(str).str.strip().ne("")
            ]
            for idx, (_, row) in enumerate(nonempty.head(5).iterrows(), 1):
                st.write(f"**Note {idx}:** {row['clinical_note']}")
            st.markdown("**All generated notes**")
            st.dataframe(nonempty, use_container_width=True)

    st.markdown("#### Section 7 · Download All")
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zip_name = f"synthgenx_results_{timestamp}.zip"
        st.download_button(
            "Download All Results",
            data=_zip_all_results(state),
            file_name=zip_name,
            mime="application/zip",
            type="primary",
            key=_unique_download_key(zip_name, "unified_zip"),
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not create the combined results zip: {exc}")

    render_downloads(state)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entry-point."""

    cfg = render_sidebar()
    fns = _import_pipeline()  # warm up imports

    # Session state for pipeline outputs
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = {}
    state = st.session_state.pipeline_state

    if cfg.get("run_full_pipeline"):
        st.markdown("### 🚀 SynthGenX Full Pipeline")
        run_full_pipeline(cfg)
        st.divider()

    # ── Header ─────────────────────────────────────────────────────────────
    ds_label = {"thyroid": "🦋 Thyroid Disease",
                "heart":   "❤️ Heart Disease",
                "generic": "📂 Custom Dataset"}.get(cfg["dataset_type"], cfg["dataset_type"])
    st.markdown(f"### 🧬 SynthGenX  ·  {ds_label}")
    st.caption("SynthGenX: Generate realistic synthetic clinical datasets for ML training.")
    st.divider()

    # ── Tabs ────────────────────────────────────────────────────────────────
    tabs = st.tabs(["1 · Ingest", "2 · Cluster", "3 · Synthesise",
                    "4 · Validate", "5 · Clinical Notes", "📥 Downloads", "📊 Results"])

    # ════════════════════════════════════════════════════════════════════════
    # Tab 1 — Ingest
    # ════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Data Ingestion")

        if cfg["source"] == "upload":
            # Custom CSV upload path
            st.info(f"Using uploaded file: **{cfg['uploaded_file'].name}**")
            with st.expander("Data Preview", expanded=True):
                df_preview = pd.read_csv(cfg["uploaded_file"])
                st.write(f"Shape: {df_preview.shape}")
                st.dataframe(df_preview.head(10), use_container_width=True)

            if st.button("Process Uploaded Data", type="primary"):
                with st.spinner("Processing…"):
                    cfg["uploaded_file"].seek(0)
                    df, meta = load_uploaded_csv(cfg["uploaded_file"])
                    state["clean_df"]    = df
                    state["ingest_meta"] = meta
                    state["cfg"]         = cfg
                st.success(f"✓ Processed {meta['rows']:,} rows, {meta['columns']} columns")
                render_ingest_results(df, meta)
        else:
            # Built-in dataset path
            st.markdown(f"**Dataset:** {ds_label}")
            if cfg.get("local_file"):
                st.info(f"Using local file: `{cfg['local_file']}`")
            else:
                st.info("Will attempt to download from UCI repository.")

            if st.button("Run Ingestion", type="primary"):
                with st.spinner("Ingesting data…"):
                    df, stats = run_ingest_step(cfg)
                if df is not None:
                    state["clean_df"]    = df
                    state["ingest_meta"] = stats
                    state["cfg"]         = cfg
                    st.success(f"✓ {stats.get('rows', 0):,} rows loaded")

        if state.get("clean_df") is not None and not tabs[0].__class__.__name__ == "hidden":
            render_ingest_results(state["clean_df"], state.get("ingest_meta", {}))

    # ════════════════════════════════════════════════════════════════════════
    # Tab 2 — Cluster
    # ════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### K-Means Clustering")

        if state.get("clean_df") is None:
            st.warning("Complete the Ingest step first.")
        else:
            df_clean = state["clean_df"]
            st.caption(f"Input: {len(df_clean):,} rows · K={cfg['n_clusters']} clusters")

            if st.button("Run Clustering", type="primary"):
                with st.spinner("Clustering…"):
                    clustered, stats, pca_fig = run_cluster_step(df_clean, cfg)
                if clustered is not None:
                    state["clustered_df"]  = clustered
                    state["cluster_stats"] = stats
                    state["pca_fig"]       = pca_fig
                    st.success("✓ Clustering complete")

            if state.get("clustered_df") is not None:
                render_cluster_results(
                    state["clustered_df"],
                    state.get("cluster_stats", {}),
                    state.get("pca_fig"),
                )

    # ════════════════════════════════════════════════════════════════════════
    # Tab 3 — Synthesise
    # ════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Synthetic Data Generation")

        if state.get("clustered_df") is None:
            st.warning("Complete the Cluster step first.")
        else:
            clustered = state["clustered_df"]
            st.caption(f"Input: {len(clustered):,} real records · "
                       f"Target: {cfg['n_samples']:,} synthetic samples")

            if st.button("Generate Synthetic Data", type="primary"):
                with st.spinner("Generating…"):
                    synth_df, val_results, comp_fig = run_synthgen_step(clustered, cfg)
                if synth_df is not None:
                    state["synth_df"]     = synth_df
                    state["val_results"]  = val_results
                    state["comp_fig"]     = comp_fig
                    if "is_synthetic" in synth_df.columns:
                        generated = int(_bool_mask(synth_df["is_synthetic"]).sum())
                    else:
                        generated = len(synth_df)
                    st.success(f"✓ {generated:,} synthetic records generated")

            if state.get("synth_df") is not None:
                render_synthgen_results(
                    state["synth_df"],
                    state.get("val_results", {}),
                    state.get("comp_fig"),
                )

    # ════════════════════════════════════════════════════════════════════════
    # Tab 4 — Validate
    # ════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Distribution Validation")

        if state.get("clean_df") is None or state.get("synth_df") is None:
            st.warning("Complete Ingest and Synthesise steps first.")
        else:
            real_n  = len(state["clean_df"])
            if "is_synthetic" in state["synth_df"].columns:
                synth_n = int(_bool_mask(state["synth_df"]["is_synthetic"]).sum())
            else:
                synth_n = len(state["synth_df"])
            st.caption(f"Real: {real_n:,} rows · Synthetic: {synth_n:,} rows")

            if st.button("Run Validation", type="primary"):
                with st.spinner("Validating…"):
                    metrics, plots, html = run_validate_step(
                        state["clean_df"], state["synth_df"], cfg
                    )
                state["val_metrics"] = metrics
                state["val_plots"]   = plots
                state["val_html"]    = html
                st.success("✓ Validation complete")

            if state.get("val_metrics"):
                render_validation_results(
                    state["val_metrics"],
                    state.get("val_plots", []),
                    state.get("val_html", ""),
                )

    # ════════════════════════════════════════════════════════════════════════
    # Tab 5 — Clinical Notes (LLM)
    # ════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### LLM Clinical Note Generation")

        if state.get("synth_df") is None:
            st.warning("Complete the Synthesise step first.")
        else:
            synth_df = state["synth_df"]
            api_key = cfg.get("api_key", "").strip()

            if not api_key:
                st.info("Enter your Anthropic API key in the sidebar to enable this step.")
            else:
                st.caption(
                    f"Will generate notes for up to **{cfg['llm_limit']}** records "
                    f"using **{cfg['llm_model']}**."
                )

            if api_key and st.button("Generate Clinical Notes", type="primary"):
                with st.spinner("Calling Claude API…"):
                    df_with_notes, sample_notes = run_llm_step(synth_df, cfg)
                if df_with_notes is not None:
                    state["synth_notes_df"] = df_with_notes
                    state["sample_notes"]   = sample_notes
                    st.success("✓ Clinical notes generated")

            if state.get("synth_notes_df") is not None:
                render_llm_results(
                    state["synth_notes_df"],
                    state.get("sample_notes", []),
                )

                # Live note browser
                df_notes = state["synth_notes_df"]
                if "clinical_note" in df_notes.columns:
                    with st.expander("Browse all generated notes"):
                        filled = df_notes[
                            df_notes["clinical_note"].str.strip().ne("")
                        ].reset_index(drop=True)
                        if not filled.empty:
                            idx = st.slider("Record", 0, len(filled) - 1, 0)
                            row = filled.iloc[idx]
                            col_a, col_b = st.columns([1, 1])
                            with col_a:
                                st.markdown("**Source data**")
                                num_cols = filled.select_dtypes(
                                    include=[np.number]).columns.tolist()
                                st.dataframe(
                                    row[num_cols].to_frame().T,
                                    use_container_width=True,
                                )
                            with col_b:
                                st.markdown("**Generated note**")
                                st.info(row["clinical_note"])

    # ════════════════════════════════════════════════════════════════════════
    # Tab 6 — Downloads
    # ════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### Download Outputs")

        if not any(state.get(k) is not None for k in
                   ["clean_df", "clustered_df", "synth_df", "synth_notes_df"]):
            st.info("Run pipeline steps to generate downloadable outputs.")
        else:
            render_downloads(state)

            # Pipeline summary card
            st.divider()
            st.markdown("#### Pipeline Summary")
            summary_rows = []
            for label, key in [
                ("Clean data",          "clean_df"),
                ("Clustered data",      "clustered_df"),
                ("Synthetic data",      "synth_df"),
                ("Synthetic + notes",   "synth_notes_df"),
            ]:
                df = state.get(key)
                if isinstance(df, pd.DataFrame):
                    notes_col = "clinical_note" in df.columns
                    filled = int(df["clinical_note"].str.strip().ne("").sum()) if notes_col else None
                    summary_rows.append({
                        "Output": label,
                        "Rows": len(df),
                        "Columns": len(df.columns),
                        "Notes generated": filled if filled is not None else "—",
                    })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows),
                             use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # Tab 7 — Unified Results
    # ════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        render_unified_results(state, cfg)


if __name__ == "__main__":
    main()
