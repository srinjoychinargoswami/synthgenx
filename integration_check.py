"""
SynthGen — Integration Check
=============================
Verifies that all pipeline components, directories, and dependencies are
present and importable. Run standalone or imported by app.py at startup.

Usage:
    python integration_check.py
    python integration_check.py --json    # machine-readable output

Returns exit code 0 if all critical checks pass, 1 if any critical check fails.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Check definitions
# ---------------------------------------------------------------------------

SRC_FILES = [
    "src/01_ingest.py",
    "src/02_cluster.py",
    "src/03_synthetic_gen.py",
    "src/03b_llm_notes.py",
    "src/04_validate.py",
]

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "outputs",
    "src",
]

REQUIRED_PACKAGES: list[tuple[str, bool]] = [
    # (import_name, is_critical)
    ("pandas",          True),
    ("numpy",           True),
    ("sklearn",         True),
    ("matplotlib",      True),
    ("seaborn",         True),
    ("requests",        True),
    ("streamlit",       True),
    ("anthropic",       False),   # optional — LLM disabled if absent
]

MIN_PYTHON = (3, 8)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_python() -> dict[str, Any]:
    version = sys.version_info
    ok = (version.major, version.minor) >= MIN_PYTHON
    return {
        "name": "Python version",
        "status": "ok" if ok else "fail",
        "critical": True,
        "detail": f"{version.major}.{version.minor}.{version.micro}",
        "message": (f"Python {version.major}.{version.minor} ✓"
                    if ok else
                    f"Python {version.major}.{version.minor} — need ≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"),
    }


def _check_packages() -> list[dict[str, Any]]:
    results = []
    for pkg, critical in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(pkg)
        found = spec is not None
        if found:
            try:
                mod = importlib.import_module(pkg)
                version = getattr(mod, "__version__", "?")
            except Exception:
                version = "?"
        else:
            version = None
        results.append({
            "name": f"Package: {pkg}",
            "status": "ok" if found else ("fail" if critical else "warn"),
            "critical": critical,
            "detail": version,
            "message": (f"{pkg} {version} ✓" if found
                        else f"{pkg} not installed" +
                             (" (optional — LLM disabled)" if not critical else "")),
        })
    return results


def _check_directories() -> list[dict[str, Any]]:
    results = []
    for d in REQUIRED_DIRS:
        p = Path(d)
        exists = p.exists() and p.is_dir()
        results.append({
            "name": f"Directory: {d}",
            "status": "ok" if exists else "warn",
            "critical": False,
            "detail": str(p.resolve()) if exists else None,
            "message": (f"{d}/ ✓" if exists else f"{d}/ not found (will be created)"),
        })
        if not exists:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
    return results


def _check_src_files() -> list[dict[str, Any]]:
    results = []
    for rel in SRC_FILES:
        p = Path(rel)
        exists = p.exists() and p.is_file()
        importable = False
        error_msg = ""
        if exists:
            spec = importlib.util.spec_from_file_location("_tmp_check", p)
            if spec and spec.loader:
                try:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[arg-type]
                    importable = True
                except Exception as exc:
                    error_msg = str(exc)

        status = "ok" if (exists and importable) else (
            "warn" if exists else "fail"
        )
        results.append({
            "name": f"Module: {rel}",
            "status": status,
            "critical": True,
            "detail": error_msg or None,
            "message": (f"{rel} loaded ✓" if (exists and importable)
                        else f"{rel} import error: {error_msg}" if exists
                        else f"{rel} not found"),
        })
    return results


def _check_data_files() -> list[dict[str, Any]]:
    """Check for presence of raw data files (non-critical — can be downloaded)."""
    raw_files = [
        "data/raw/thyroid_disease.data",
        "data/raw/heart_disease.data",
    ]
    results = []
    for f in raw_files:
        p = Path(f)
        exists = p.exists()
        results.append({
            "name": f"Data file: {f}",
            "status": "ok" if exists else "info",
            "critical": False,
            "detail": f"{p.stat().st_size:,} bytes" if exists else None,
            "message": (f"{f} present ✓" if exists
                        else f"{f} absent (will download from UCI)"),
        })
    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_integration_check() -> dict[str, Any]:
    """
    Run all integration checks and return a structured results dict.

    Returns:
        Dict with keys:
            all_critical_pass (bool)
            checks (list of check result dicts)
            summary (dict with counts by status)
    """
    all_checks: list[dict[str, Any]] = []
    all_checks.append(_check_python())
    all_checks.extend(_check_packages())
    all_checks.extend(_check_directories())
    all_checks.extend(_check_src_files())
    all_checks.extend(_check_data_files())

    counts: dict[str, int] = {"ok": 0, "warn": 0, "fail": 0, "info": 0}
    for c in all_checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    critical_failures = [
        c for c in all_checks if c["status"] == "fail" and c["critical"]
    ]

    return {
        "all_critical_pass": len(critical_failures) == 0,
        "checks": all_checks,
        "summary": counts,
        "critical_failures": [c["name"] for c in critical_failures],
    }


def print_report(results: dict[str, Any]) -> None:
    """Pretty-print the integration check results to stdout."""
    ICONS = {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "ℹ"}
    COLORS = {
        "ok":   "\033[92m",   # green
        "warn": "\033[93m",   # yellow
        "fail": "\033[91m",   # red
        "info": "\033[94m",   # blue
    }
    RESET = "\033[0m"

    print("\n" + "=" * 56)
    print("  SynthGen Integration Check")
    print("=" * 56)
    for c in results["checks"]:
        icon = ICONS.get(c["status"], "?")
        color = COLORS.get(c["status"], "")
        detail = f"  [{c['detail']}]" if c.get("detail") else ""
        print(f"  {color}{icon}{RESET} {c['message']}{detail}")

    s = results["summary"]
    print("-" * 56)
    print(f"  {s.get('ok',0)} ok  |  {s.get('warn',0)} warnings  |  "
          f"{s.get('fail',0)} failures  |  {s.get('info',0)} info")
    if results["all_critical_pass"]:
        print("\033[92m  ✓ All critical checks passed — ready to run!\033[0m")
    else:
        print("\033[91m  ✗ Critical failures:\033[0m")
        for name in results["critical_failures"]:
            print(f"     • {name}")
    print("=" * 56 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="SynthGen Integration Check")
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON instead of pretty-print")
    args = p.parse_args()

    results = run_integration_check()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)

    sys.exit(0 if results["all_critical_pass"] else 1)


if __name__ == "__main__":
    main()