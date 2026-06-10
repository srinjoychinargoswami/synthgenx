"""
SynthGen - Step 3b: LLM Clinical Note Generation
=================================================
Generates realistic clinical notes for synthetic patient records using the
Anthropic Claude API. Supports thyroid and heart disease datasets with
dataset-appropriate clinical context prompts.

Usage (CLI):
    python src/03b_llm_notes.py --input data/processed/synthetic_data.csv \
        --api-key sk-ant-... --model claude-haiku-4-5-20251001 --limit 200

Callable function (for app.py):
    from src.03b_llm_notes import run_llm_notes
    df_with_notes, sample_notes = run_llm_notes(
        synthetic_data=df,
        api_key="sk-ant-...",
        model="claude-haiku-4-5-20251001",
        limit=200,
        dataset_type="thyroid",
        progress_callback=lambda msg, frac: None,
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset-specific prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_THYROID = """You are a clinical endocrinologist writing a concise
electronic health record note. Given structured lab values, generate a
2-3 sentence clinical impression note that reads like a real physician
note. Be specific about hormone levels and clinical significance.
Use appropriate medical terminology. Do not use patient names."""

_SYSTEM_HEART = """You are a cardiologist writing a concise electronic health
record note. Given structured cardiac risk factors and stress test results,
generate a 2-3 sentence clinical impression note that reads like a real
physician note. Be specific about cardiovascular findings and risk.
Use appropriate medical terminology. Do not use patient names."""

_SYSTEM_GENERIC = """You are a physician writing a concise electronic health
record note. Given structured clinical data, generate a 2-3 sentence
clinical impression note that reads like a real physician note.
Use appropriate medical terminology. Do not use patient names."""


def _get_system_prompt(dataset_type: str) -> str:
    if dataset_type == "thyroid":
        return _SYSTEM_THYROID
    elif dataset_type == "heart":
        return _SYSTEM_HEART
    return _SYSTEM_GENERIC


def _build_user_prompt(row: pd.Series, dataset_type: str) -> str:
    """
    Convert a DataFrame row into a structured clinical prompt.

    Args:
        row: A single patient record as a pandas Series.
        dataset_type: 'thyroid', 'heart', or 'generic'.

    Returns:
        User prompt string for the LLM.
    """
    if dataset_type == "thyroid":
        # Extract available thyroid-relevant fields
        fields = []
        for col in ["age", "sex", "TSH", "tsh", "T3", "t3", "TT4", "tt4",
                    "T4U", "t4u", "FTI", "fti", "diagnosis_class",
                    "on_thyroxine", "on_antithyroid_medication"]:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    fields.append(f"{col}: {val}")
        prompt = ("Generate a clinical note for a thyroid patient with the "
                  "following lab values:\n" + "\n".join(fields))

    elif dataset_type == "heart":
        fields = []
        for col in ["age", "sex", "cp", "trestbps", "chol", "fbs",
                    "thalach", "exang", "oldpeak", "diagnosis_class",
                    "ca", "thal"]:
            if col in row.index:
                val = row[col]
                if pd.notna(val):
                    fields.append(f"{col}: {val}")
        prompt = ("Generate a clinical note for a cardiac patient with the "
                  "following values:\n" + "\n".join(fields))

    else:
        # Generic: use all numeric and string columns
        fields = [f"{k}: {v}" for k, v in row.items()
                  if pd.notna(v) and not str(k).startswith("_")]
        prompt = ("Generate a clinical note for a patient with the "
                  "following data:\n" + "\n".join(fields[:20]))

    return prompt


# ---------------------------------------------------------------------------
# Anthropic client helper
# ---------------------------------------------------------------------------

def _call_anthropic(client: Any, system: str, user: str, model: str) -> str:
    """
    Call Anthropic Messages API and return the text response.

    Args:
        client: Instantiated anthropic.Anthropic client.
        system: System prompt string.
        user: User prompt string.
        model: Model identifier (e.g. 'claude-haiku-4-5-20251001').

    Returns:
        Generated text or empty string on failure.
    """
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_llm_notes(
    synthetic_data: pd.DataFrame,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    limit: int = 200,
    dataset_type: str = "thyroid",
    output_dir: str = "data/processed",
    rate_limit_delay: float = 0.1,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate clinical notes for synthetic patient records using Claude.

    Args:
        synthetic_data:    DataFrame of synthetic records.
        api_key:           Anthropic API key (sk-ant-...).
        model:             Claude model identifier.
        limit:             Maximum number of rows to annotate.
        dataset_type:      'thyroid', 'heart', or 'generic'.
        output_dir:        Directory to save annotated CSV.
        rate_limit_delay:  Seconds to sleep between API calls.
        progress_callback: Optional fn(message, fraction) for UI progress.

    Returns:
        Tuple of (annotated_df, list_of_sample_notes).

    Raises:
        ImportError: If the anthropic package is not installed.
        ValueError: If synthetic_data is empty or api_key is blank.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for LLM note generation. "
            "Install it with: pip install anthropic"
        ) from exc

    if not api_key or not api_key.strip():
        raise ValueError("A valid Anthropic API key is required.")
    if synthetic_data.empty:
        raise ValueError("synthetic_data DataFrame is empty.")

    # Setup
    client = anthropic.Anthropic(api_key=api_key.strip())
    system_prompt = _get_system_prompt(dataset_type)
    df = synthetic_data.copy()

    # Ensure clinical_note column exists
    if "clinical_note" not in df.columns:
        df["clinical_note"] = ""

    n_to_process = min(limit, len(df))
    generated_notes: list[str] = []
    errors = 0

    def _cb(msg: str, frac: float) -> None:
        if progress_callback:
            progress_callback(msg, frac)

    for i in range(n_to_process):
        row = df.iloc[i]
        user_prompt = _build_user_prompt(row, dataset_type)

        try:
            note = _call_anthropic(client, system_prompt, user_prompt, model)
            df.at[df.index[i], "clinical_note"] = note
            if i < 3:
                generated_notes.append(note)

        except Exception as exc:  # noqa: BLE001
            err_msg = f"[API error row {i}]: {exc}"
            df.at[df.index[i], "clinical_note"] = err_msg
            errors += 1
            if errors > 10:
                _cb("Too many API errors — stopping note generation.", i / n_to_process)
                break

        # Progress update every 10 records
        if (i + 1) % 10 == 0 or (i + 1) == n_to_process:
            _cb(f"{i + 1}/{n_to_process} samples processed…",
                (i + 1) / n_to_process)

        time.sleep(rate_limit_delay)

    # Save annotated CSV
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / "synthetic_data_with_notes.csv"
    df.to_csv(out_csv, index=False)
    _cb(f"Saved annotated CSV → {out_csv}", 1.0)

    return df, generated_notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="SynthGen Step 3b — LLM Notes")
    p.add_argument("--input", default="data/processed/synthetic_data.csv")
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--dataset", default="thyroid",
                   choices=["thyroid", "heart", "generic"])
    p.add_argument("--output-dir", default="data/processed")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    print(f"Generating notes for {min(args.limit, len(df))} records…")

    def cb(msg: str, _: float) -> None:
        print(msg)

    df_out, samples = run_llm_notes(
        synthetic_data=df,
        api_key=args.api_key,
        model=args.model,
        limit=args.limit,
        dataset_type=args.dataset,
        output_dir=args.output_dir,
        progress_callback=cb,
    )
    print(f"\n✓ Done. {len(df_out)} rows annotated.")
    if samples:
        print("\nSample notes:")
        for i, note in enumerate(samples, 1):
            print(f"\n--- Note {i} ---\n{note}")


if __name__ == "__main__":
    main()
