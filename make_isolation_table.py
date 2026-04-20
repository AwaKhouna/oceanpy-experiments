#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from parameters import DATASETS
except Exception:  # pragma: no cover - keeps the script usable outside the repo root.
    DATASETS = []


VARIANTS = {
    "CP-noIF": "cp_plain",
    "CP-IF": "cp_iso",
    "MIP-noIF": "mip_plain",
    "MIP-IF": "mip_iso",
}
METRICS = ("Plausibility (%)", "Mean time")


def load_entries(results_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable results file {path}: {exc}")
            continue

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            print(f"Skipping unexpected payload in {path}: expected list or dict.")
            continue

        for item in payload:
            if isinstance(item, dict):
                item = dict(item)
                item["_source_file"] = str(path)
                entries.append(item)
    return entries


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def summarize_variant_from_explanations(
    entries: list[dict[str, Any]],
    variant: str,
) -> tuple[float | None, float | None]:
    times: list[float] = []
    found = 0
    plausible = 0

    for entry in entries:
        explanations = entry.get("explanations")
        if not isinstance(explanations, list):
            continue

        for explanation in explanations:
            if not isinstance(explanation, dict):
                continue

            time_value = finite_float(explanation.get(f"{variant}_time"))
            if time_value is not None and time_value >= 0:
                times.append(time_value)

            if explanation.get(f"{variant}_cf") is not None:
                found += 1
                if explanation.get(f"{variant}_plausible") is True:
                    plausible += 1

    plausibility = None if found == 0 else 100.0 * plausible / found
    mean_time = None if not times else float(np.mean(times))
    return plausibility, mean_time


def summarize_variant_from_entry_summaries(
    entries: list[dict[str, Any]],
    variant: str,
) -> tuple[float | None, float | None]:
    prevalence_values: list[tuple[float, float]] = []
    time_values: list[tuple[float, float]] = []

    for entry in entries:
        prevalence = finite_float(entry.get(f"{variant}_plausibility_prevalence"))
        cf_count = finite_float(entry.get(f"{variant}_cf_count"))
        if prevalence is not None and cf_count is not None and cf_count > 0:
            prevalence_values.append((100.0 * prevalence, cf_count))

        mean_time = finite_float(entry.get(f"{variant}_mean_time"))
        query_count = finite_float(entry.get("query_count"))
        if mean_time is not None and query_count is not None and query_count > 0:
            time_values.append((mean_time, query_count))

    plausibility = (
        None
        if not prevalence_values
        else float(
            np.average(
                [value for value, _ in prevalence_values],
                weights=[weight for _, weight in prevalence_values],
            )
        )
    )
    mean_time = (
        None
        if not time_values
        else float(
            np.average(
                [value for value, _ in time_values],
                weights=[weight for _, weight in time_values],
            )
        )
    )
    return plausibility, mean_time


def summarize_variant(
    entries: list[dict[str, Any]],
    variant: str,
) -> tuple[float | None, float | None]:
    plausibility, mean_time = summarize_variant_from_explanations(entries, variant)
    if plausibility is not None or mean_time is not None:
        return plausibility, mean_time
    return summarize_variant_from_entry_summaries(entries, variant)


def ordered_datasets(found: set[str], include_missing: bool) -> list[str]:
    order: list[str] = []
    known = list(DATASETS) if include_missing else [dataset for dataset in DATASETS if dataset in found]
    for dataset in known:
        if dataset not in order:
            order.append(dataset)
    for dataset in sorted(found):
        if dataset not in order:
            order.append(dataset)
    return order


def build_table(entries: list[dict[str, Any]], include_missing: bool) -> pd.DataFrame:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        dataset = entry.get("dataset")
        if isinstance(dataset, str):
            by_dataset.setdefault(dataset, []).append(entry)

    rows: list[dict[tuple[str, str], Any]] = []
    index: list[str] = []
    for dataset in ordered_datasets(set(by_dataset), include_missing):
        dataset_entries = by_dataset.get(dataset, [])
        row: dict[tuple[str, str], Any] = {}
        for label, variant in VARIANTS.items():
            plausibility, mean_time = summarize_variant(dataset_entries, variant)
            row[(label, "Plausibility (%)")] = plausibility
            row[(label, "Mean time")] = mean_time
        rows.append(row)
        index.append(dataset)

    columns = pd.MultiIndex.from_product([VARIANTS.keys(), METRICS])
    return pd.DataFrame(rows, index=pd.Index(index, name="Dataset"), columns=columns)


def flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    flat = table.copy()
    flat.columns = [f"{method} {metric}" for method, metric in table.columns]
    return flat.reset_index()


def format_table(table: pd.DataFrame, digits: int) -> pd.DataFrame:
    formatted = table.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the CP/MIP noIF/IF plausibility and time table."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/rf/isolation"),
        help="Directory containing isolation_*.json result files.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("plots/isolation_table.csv"),
        help="Path for the flattened CSV table.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=Path("plots/isolation_table.tex"),
        help="Path for the LaTeX table with grouped columns.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include every dataset from parameters.py, leaving missing result rows blank.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal places used in the LaTeX and console tables.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = load_entries(args.results_dir)
    table = build_table(entries, include_missing=args.include_missing)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)

    flatten_columns(table).to_csv(args.csv_output, index=False)
    formatted = format_table(table, args.digits)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*DataFrame.to_latex.*",
            category=FutureWarning,
        )
        latex = formatted.to_latex(
            multicolumn=True,
            multirow=True,
            escape=False,
            na_rep="",
        )
    args.latex_output.write_text(
        latex
    )

    print(format_table(table, args.digits).to_string())
    print(f"\nSaved CSV table to {args.csv_output}")
    print(f"Saved LaTeX table to {args.latex_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
