#!/usr/bin/env python3
"""
Error Cross-Tab Analysis — Blueprint Section C.6
═════════════════════════════════════════════════
Reads per-pipeline telemetry and judge results, then produces the
paradigm × query-type cross-tab from blueprint C.6.

Usage:
    python scripts/analyze_errors.py --results-dir results/
    python scripts/analyze_errors.py --results-dir results/ --domain finance

Outputs:
  - Console: formatted cross-tab table
  - results/error_crosstab.json: machine-readable cross-tab
  - results/string_metrics_comparison.json: F1/EM comparison across pipelines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def load_crosstab_files(results_dir: Path, domain: str | None = None) -> list[dict]:
    """Load all error_crosstab.json files from results directory."""
    pattern = f"*_{domain}_error_crosstab.json" if domain else "*_error_crosstab.json"
    files = sorted(results_dir.glob(pattern))
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def load_string_metric_files(results_dir: Path, domain: str | None = None) -> list[dict]:
    """Load all string_metrics.json files."""
    pattern = f"*_{domain}_string_metrics.json" if domain else "*_string_metrics.json"
    files = sorted(results_dir.glob(pattern))
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def build_crosstab(
    crosstab_data: list[dict],
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Build paradigm × query-type × error-category cross-tab.
    Returns: {pipeline: {query_type: {error_cat: count}}}
    """
    result = {}
    for entry in crosstab_data:
        pipeline = entry["pipeline"]
        result[pipeline] = entry.get("error_crosstab", {})
    return result


def print_crosstab(crosstab: dict[str, dict[str, dict[str, int]]]) -> None:
    """Print formatted cross-tab table to console."""
    print("\n" + "=" * 80)
    print("  ERROR CROSS-TAB: Paradigm × Query-Type (Blueprint C.6)")
    print("=" * 80)

    for pipeline, by_type in sorted(crosstab.items()):
        print(f"\n  Pipeline: {pipeline}")
        print(f"  {'Query Type':<20} {'correct':>8} {'format_fail':>12} {'other':>8} {'total':>8}")
        print(f"  {'-'*60}")
        total_correct = 0
        total_all = 0
        for q_type, errors in sorted(by_type.items()):
            n_correct = errors.get("correct", 0)
            n_format = errors.get("format_failure", 0)
            n_other = sum(v for k, v in errors.items() if k not in ("correct", "format_failure"))
            n_total = sum(errors.values())
            total_correct += n_correct
            total_all += n_total
            print(f"  {q_type:<20} {n_correct:>8} {n_format:>12} {n_other:>8} {n_total:>8}")
        if total_all > 0:
            print(f"  {'TOTAL':<20} {total_correct:>8} {'':>12} {'':>8} {total_all:>8} "
                  f"  (acc={total_correct/total_all:.1%})")


def print_string_metrics(metrics_data: list[dict]) -> None:
    """Print F1/EM comparison table."""
    print("\n" + "=" * 80)
    print("  STRING METRICS COMPARISON: F1/EM (Blueprint C.3)")
    print("=" * 80)
    print(f"  {'Pipeline':<30} {'Domain':<12} {'F1':>8} {'EM':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*78}")
    for entry in sorted(metrics_data, key=lambda x: x.get("metrics", {}).get("f1", 0), reverse=True):
        m = entry.get("metrics", {})
        print(
            f"  {entry['pipeline']:<30} {entry['domain']:<12} "
            f"{m.get('f1', 0):>8.4f} {m.get('exact_match', 0):>8.4f} "
            f"{m.get('precision', 0):>10.4f} {m.get('recall', 0):>8.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Error cross-tab analysis (Blueprint C.6)",
    )
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--domain", type=str, default=None,
                        choices=["finance", "legal", "technical"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    # Load cross-tab data
    crosstab_files = load_crosstab_files(results_dir, args.domain)
    if crosstab_files:
        crosstab = build_crosstab(crosstab_files)
        print_crosstab(crosstab)

        # Save combined cross-tab
        out_path = results_dir / "error_crosstab.json"
        with open(out_path, "w") as f:
            json.dump(crosstab, f, indent=2)
        logger.info(f"Cross-tab saved: {out_path}")
    else:
        logger.warning("No error cross-tab files found. Run benchmark first.")

    # Load + print string metrics
    metrics_files = load_string_metric_files(results_dir, args.domain)
    if metrics_files:
        print_string_metrics(metrics_files)

        # Save comparison
        out_path = results_dir / "string_metrics_comparison.json"
        with open(out_path, "w") as f:
            json.dump(metrics_files, f, indent=2)
        logger.info(f"Metrics comparison saved: {out_path}")

    print()


if __name__ == "__main__":
    main()
