#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from actionability_metrics import ACTIONABILITY, SCHEMA_VERSION as AC_SCHEMA_VERSION

try:
    from parameters import DATASETS, DATASET_ACRONYMS
except Exception:  # pragma: no cover - keeps the script usable outside the repo root.
    DATASETS = []
    DATASET_ACRONYMS = {}


VARIANTS = {
    "CP-noIF": "cp_plain",
    "MIP-noIF": "mip_plain",
    "CP-IF": "cp_iso",
    "MIP-IF": "mip_iso",
}
METRICS = (r"P (\%)", "D ratio", "T (s)")
ADD_BUILD_TIME = True


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

        build_time = (
            finite_float(entry.get(f"{variant}_build_time")) if ADD_BUILD_TIME else 0.0
        )
        if ADD_BUILD_TIME and variant.endswith("_plain"):
            target_build_times = entry.get(f"{variant}_target_build_times", {})
            if target_build_times:
                # Count only the first recorded baseline build.
                build_time = float(next(iter(target_build_times.values())))

        for explanation in explanations:
            if not isinstance(explanation, dict):
                continue

            elapsed = finite_float(explanation.get(f"{variant}_time"))
            if elapsed is not None and build_time is not None and min(elapsed, build_time) >= 0:
                times.append(elapsed + build_time)

            if explanation.get(f"{variant}_cf") is not None:
                found += 1
                if explanation.get(f"{variant}_plausible") is True:
                    plausible += 1

    plausibility = None if found == 0 else 100.0 * plausible / found
    median_time = None if not times else float(np.median(times))
    # mean_time = None if not times else float(np.mean(times))
    return plausibility, median_time


def summarize_variant(
    entries: list[dict[str, Any]],
    variant: str,
) -> tuple[float | None, float | None]:
    plausibility, median_time = summarize_variant_from_explanations(entries, variant)
    assert plausibility is None or 0.0 <= plausibility <= 100.0, (
        f"Invalid plausibility {plausibility} for variant {variant}"
    )
    assert median_time is None or median_time >= 0.0, (
        f"Invalid median time {median_time} for variant {variant}"
    )

    return plausibility, median_time


def ordered_datasets(found: set[str], include_missing: bool) -> list[str]:
    order: list[str] = []
    known = (
        list(DATASETS)
        if include_missing
        else [dataset for dataset in DATASETS if dataset in found]
    )
    for dataset in known:
        if dataset not in order:
            order.append(dataset)
    for dataset in sorted(found):
        if dataset not in order:
            order.append(dataset)
    return order


def mean_distance_ratio(entries: list[dict[str, Any]], variant: str) -> float | None:
    method, suffix = variant.split("_", 1)
    ratios = []
    for entry in entries:
        for row in entry.get("explanations", []):
            if any(row.get(f"{method}_{s}_cf") is None or
                   row.get(f"{method}_{s}_valid") is not True for s in ("plain", "iso")):
                continue
            if row.get(f"{method}_iso_plausible") is not True:
                continue
            plain = finite_float(row.get(f"{method}_plain_objective"))
            constrained = finite_float(row.get(f"{method}_iso_objective"))
            if plain is None or constrained is None or plain <= 0 or constrained < 0:
                continue
            ratio = constrained / plain
            if np.isfinite(ratio):
                ratios.append(1.0 if suffix == "plain" else ratio)
    return float(np.mean(ratios)) if ratios else None


def matching_ac_baselines(entries, actionability_entries):
    match_keys = ("dataset", "seed", "voting", "n_estimators", "max_depth",
                  "norm", "threads", "timeout", "query_count")
    baselines = []
    for entry in entries:
        matches = [e for e in actionability_entries
                   if e.get("complete") is True and e.get("schema_version") == AC_SCHEMA_VERSION
                   and all(e.get(k) == entry.get(k) for k in match_keys)]
        if len(matches) != 1:
            raise ValueError(f"Expected one matching actionability result for {entry['dataset']}, "
                             f"seed {entry.get('seed')}; found {len(matches)}")
        baselines.extend(matches)
    return baselines


def build_table(entries: list[dict[str, Any]], include_missing: bool, *,
                actionability_entries=None) -> pd.DataFrame:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        dataset = entry.get("dataset")
        if isinstance(dataset, str):
            by_dataset.setdefault(dataset, []).append(entry)

    rows: list[dict[tuple[str, str], Any]] = []
    index: list[str] = []
    for dataset in ordered_datasets(set(by_dataset), include_missing):
        dataset_entries = by_dataset.get(dataset, [])
        ac_baselines = None
        if actionability_entries is not None and dataset in ACTIONABILITY:
            ac_baselines = matching_ac_baselines(dataset_entries, actionability_entries)
        row: dict[tuple[str, str], Any] = {}
        for label, variant in VARIANTS.items():
            plausibility, median_time = summarize_variant(dataset_entries, variant)
            if variant.endswith("_plain") and ac_baselines is not None:
                _, median_time = summarize_variant_from_explanations(ac_baselines, variant)
            row[(label, r"P (\%)")] = plausibility
            row[(label, "T (s)")] = median_time
            row[(label, "D ratio")] = mean_distance_ratio(dataset_entries, variant)
        rows.append(row)
        index.append(DATASET_ACRONYMS[dataset])

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
        description="Create the CP/MIP noIF/IF plausibility, distance-ratio and time table."
    )
    parser.add_argument("--actionability-results-dir", type=Path,
                        default=Path("results/rf/actionability"),
                        help="Matching noAC results used for noIF times on the five AC datasets")
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
    table = build_table(entries, include_missing=args.include_missing,
                        actionability_entries=load_entries(args.actionability_results_dir))

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
    args.latex_output.write_text(latex)

    print(format_table(table, args.digits).to_string())
    print("D ratio: mean paired withIF/noIF distance; noIF=1 on the same eligible pairs.")
    print("noIF times use matching noAC results on AC datasets; other IF metrics use isolation results.")
    print(f"\nSaved CSV table to {args.csv_output}")
    print(f"Saved LaTeX table to {args.latex_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
