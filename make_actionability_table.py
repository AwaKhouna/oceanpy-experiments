#!/usr/bin/env python3
"""Build the actionability distance-ratio/time table from paired JSON results."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from actionability_metrics import ACTIONABILITY, SCHEMA_VERSION, paired_ratio, pair_is_optimal
from make_isolation_table import load_entries, finite_float, flatten_columns, format_table
from parameters import DATASETS, DATASET_ACRONYMS

VARIANTS = {"CP-noAC": "cp_plain", "MIP-noAC": "mip_plain",
            "CP-AC": "cp_actionable", "MIP-AC": "mip_actionable"}
METRICS = ("D ratio", "T (s)")


def build_table(entries, include_missing=False, *, voting="SOFT", optimal_only=False):
    selected = [e for e in entries if e.get("schema_version") == SCHEMA_VERSION
                and e.get("complete") is True and e.get("voting") == voting
                and e.get("dataset") in ACTIONABILITY]
    # Different seeds may be pooled; different solver budgets or datasets must not
    # silently receive extra weight simply because reruns coexist in a JSON file.
    configs = {}
    seen = set()
    for entry in selected:
        dataset = entry["dataset"]
        config = tuple(entry.get(k) for k in ("n_estimators", "max_depth", "norm",
                       "threads", "timeout", "query_count", "dataset_sha256", "ocean_revision", "scale"))
        if dataset in configs and configs[dataset] != config:
            raise ValueError(f"Mixed configurations for {dataset}; select a homogeneous results directory")
        configs[dataset] = config
        identity = (dataset, entry.get("seed"))
        if identity in seen:
            raise ValueError(f"Duplicate dataset/seed entry: {identity}")
        seen.add(identity)
    datasets = [d for d in DATASETS if d in ACTIONABILITY and (include_missing or d in configs)]
    rows = []
    for dataset in datasets:
        group = [e for e in selected if e["dataset"] == dataset]
        row = {}
        for label, variant in VARIANTS.items():
            method, suffix = variant.split("_", 1)
            ratios, times = [], []
            for entry in group:
                build = finite_float(entry.get(f"{variant}_build_time"))
                for explanation in entry.get("explanations", []):
                    elapsed = finite_float(explanation.get(f"{variant}_time"))
                    if elapsed is not None and build is not None and min(elapsed, build) >= 0:
                        times.append(elapsed + build)
                    ratio, _ = paired_ratio(explanation, method)
                    if ratio is not None and (not optimal_only or pair_is_optimal(explanation, method)):
                        ratios.append(1.0 if suffix == "plain" else ratio)
            row[(label, "D ratio")] = float(np.mean(ratios)) if ratios else None
            row[(label, "T (s)")] = float(np.median(times)) if times else None
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index([DATASET_ACRONYMS[d] for d in datasets], name="Dataset"),
                        columns=pd.MultiIndex.from_product([VARIANTS, METRICS]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/rf/actionability"))
    parser.add_argument("--csv-output", type=Path, default=Path("plots/actionability_table.csv"))
    parser.add_argument("--latex-output", type=Path, default=Path("plots/actionability_table.tex"))
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--voting", choices=("SOFT", "HARD"), default="SOFT")
    parser.add_argument("--optimal-only", action="store_true", help="Use only proven-optimal pairs for distance ratios")
    parser.add_argument("--digits", type=int, default=2)
    args = parser.parse_args()
    table = build_table(load_entries(args.results_dir), args.include_missing,
                        voting=args.voting, optimal_only=args.optimal_only)
    if table.empty:
        parser.error("No complete actionability results for the selected voting mode")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    flatten_columns(table).to_csv(args.csv_output, index=False)
    formatted = format_table(table, args.digits)
    args.latex_output.write_text(formatted.to_latex(multicolumn=True, multirow=True, escape=False))
    print(formatted.to_string())
    print("D ratio: mean of paired distances; noAC=1 on the same eligible pairs.")
    print("T (s): median solve + explainer build time across all queries, as in isolation.")
    print(f"Saved {args.csv_output} and {args.latex_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
