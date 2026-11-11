#!/usr/bin/env python3
"""Paired CP/MIP actionability experiment at RF=100 trees, depth=5, L1."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from tqdm import tqdm
from ocean import ConstraintProgrammingExplainer, MixedIntegerProgramExplainer

from actionability_metrics import (
    ACTIONABILITY, SCHEMA_VERSION, VALIDATION_TOLERANCE, paired_ratio, pair_is_optimal,
)
from isolation_experiment import (
    ROOT, RF_N_ESTIMATORS, RF_MAX_DEPTH, NORM, close_explainer,
    load_saved_entries, make_explainer, sample_queries,
)
from parameters import DATASETS, N_SAMPLES, N_THREADS, SEEDS, TIMEOUT, VOTING
from run_experiment import load_dataset, get_node_count, get_split_levels
from utils import train_model

EXPERIMENT_IDS = tuple(i for i, name in enumerate(DATASETS, 1) if name in ACTIONABILITY)
VARIANTS = tuple((f"{method}_{suffix}", method, constrained)
                 for method in ("cp", "mip")
                 for suffix, constrained in (("plain", False), ("actionable", True)))
RESULT_KEYS = ("cf", "objective", "status", "time", "valid", "callback", "error",
               "prediction", "actionability_valid", "actionability_violations",
               "processed_l1_distance", "changed_feature_count", "solver_objective",
               "best_bound", "relative_gap")
IDENTITY_KEYS = ("schema_version", "experiment_id", "dataset", "model_type",
                 "n_estimators", "max_depth", "norm", "seed", "voting",
                 "query_count", "threads", "timeout", "actionability",
                 "dataset_sha256", "ocean_revision", "scale")


def require_actionability() -> None:
    for cls in (ConstraintProgrammingExplainer, MixedIntegerProgramExplainer):
        if "actionability" not in inspect.signature(cls.explain).parameters:
            raise RuntimeError(
                "This OCEAN installation lacks CP/MIP actionability. Update OCEAN "
                "to commit 48b103f or newer and reinstall it editable (see README)."
            )


def complete(entry: dict[str, Any]) -> bool:
    rows = entry.get("explanations", [])
    return (entry.get("complete") is True and len(rows) == entry.get("query_count")
            and all(f"{variant}_build_time" in entry for variant, _, _ in VARIANTS)
            and all(all(f"{variant}_{key}" in row for variant, _, _ in VARIANTS
                        for key in RESULT_KEYS) for row in rows))


def same_run(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a.get(key) == b.get(key) for key in IDENTITY_KEYS)


def save_entry(path: Path, result: dict[str, Any]) -> None:
    entries = [entry for entry in load_saved_entries(path) if not same_run(entry, result)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(entries + [result], indent=2, allow_nan=False))
    temporary.replace(path)


def validate_actionability(query, cf, feature_indices, rules) -> list[str]:
    violations = []
    for name, rule in rules.items():
        indices = feature_indices[name]
        delta = cf[indices] - query[indices]
        ok = (np.all(np.abs(delta) <= VALIDATION_TOLERANCE) if rule == "fixed"
              else np.all(delta >= -VALIDATION_TOLERANCE))
        if not ok:
            violations.append(name)
    return violations


def explain_one(query, target, *, explainer, model, rules, feature_indices,
                constrained, seed, threads, timeout) -> dict[str, Any]:
    metric = dict.fromkeys(RESULT_KEYS)
    metric.update(callback=[], actionability_violations=[])
    started = time.perf_counter()
    try:
        cf = explainer.explain(
            query, y=target, norm=NORM, actionability=rules if constrained else None,
            return_callback=True, max_time=timeout, random_seed=seed,
            num_workers=threads, verbose=False, clean_up=False,
        )
        metric["time"] = time.perf_counter() - started
        metric["status"] = explainer.get_solving_status()
        callback = getattr(explainer, "callback", None)
        metric["callback"] = [] if callback is None else callback.sollist
        if isinstance(explainer, ConstraintProgrammingExplainer):
            bound = explainer.solver.BestObjectiveBound() / explainer._obj_scale
        else:
            bound = explainer.ObjBound
        metric["best_bound"] = float(bound) if np.isfinite(bound) else None
        if cf is not None:
            point = np.asarray(cf.to_numpy(), dtype=float).reshape(-1)
            if point.shape != query.shape or not np.all(np.isfinite(point)):
                raise ValueError("Solver returned a malformed/non-finite counterfactual")
            violations = validate_actionability(query, point, feature_indices, rules)
            prediction = int(model.predict([point])[0])
            objective = float(explainer.get_distance())
            if not np.isfinite(objective):
                raise ValueError("Solver returned a non-finite distance")
            solver_objective = float(explainer.get_objective_value())
            metric["solver_objective"] = solver_objective
            if metric["best_bound"] is not None and solver_objective != 0:
                metric["relative_gap"] = abs(solver_objective - bound) / abs(solver_objective)
            metric.update(
                cf=point.tolist(), objective=objective, prediction=prediction,
                valid=prediction == target, actionability_valid=not violations,
                actionability_violations=violations,
                processed_l1_distance=float(np.abs(point - query).sum()),
                changed_feature_count=sum(
                    bool(np.any(np.abs(point[indices] - query[indices]) > VALIDATION_TOLERANCE))
                    for indices in feature_indices.values()),
            )
    except Exception as exc:
        metric.update(time=time.perf_counter() - started,
                      status=f"ERROR:{type(exc).__name__}", error=str(exc))
    finally:
        # A subsequent query must never inherit constraints from a failed solve.
        explainer.cleanup()
    return metric


def add_summaries(result: dict[str, Any]) -> None:
    rows = result["explanations"]
    for variant, _, _ in VARIANTS:
        found = [r for r in rows if r.get(f"{variant}_cf") is not None]
        for key, value in {
            "cf_count": len(found),
            "valid_count": sum(r.get(f"{variant}_valid") is True for r in found),
            "actionability_valid_count": sum(r.get(f"{variant}_actionability_valid") is True for r in found),
            "status_counts": dict(Counter(r[f"{variant}_status"] for r in rows)),
            "mean_distance": float(np.mean([r[f"{variant}_objective"] for r in found])) if found else None,
            "mean_time": float(np.mean([r[f"{variant}_time"] for r in rows])) if rows else None,
        }.items():
            result[f"{variant}_{key}"] = value
    for method in ("cp", "mip"):
        ratios, optimal_ratios, reasons = [], [], Counter()
        for row in rows:
            ratio, reason = paired_ratio(row, method)
            optimal = ratio is not None and pair_is_optimal(row, method)
            row[f"{method}_distance_ratio"] = ratio
            row[f"{method}_distance_ratio_reason"] = reason
            row[f"{method}_pair_optimal"] = optimal
            reasons[reason] += 1
            if ratio is not None:
                ratios.append(ratio)
                if optimal:
                    optimal_ratios.append(ratio)
        result[f"{method}_paired_summary"] = {
            "included_count": len(ratios), "excluded_count": len(rows) - len(ratios),
            "reason_counts": dict(reasons),
            "mean_distance_ratio": float(np.mean(ratios)) if ratios else None,
            "median_distance_ratio": float(np.median(ratios)) if ratios else None,
            "optimal_pair_count": len(optimal_ratios),
            "optimal_mean_distance_ratio": float(np.mean(optimal_ratios)) if optimal_ratios else None,
            "ratio_below_one_count": sum(r < 1 - VALIDATION_TOLERANCE for r in ratios),
        }


def run_dataset_experiment(experiment_id: int, *, seed: int, threads: int,
                           timeout: int, voting: str, query_count: int,
                           overwrite: bool, results_dir: Path) -> Path:
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(f"Choose a dataset with actionability features: {EXPERIMENT_IDS}")
    require_actionability()
    dataset = DATASETS[experiment_id - 1]
    rules = ACTIONABILITY[dataset]
    (X, y), mapper = load_dataset(dataset, scale=True)
    names = X.columns.get_level_values(0)
    feature_indices = {name: np.flatnonzero(names == name).tolist() for name in mapper}
    for name in rules:
        if not feature_indices.get(name):
            raise ValueError(f"Actionability feature {name!r} missing from {dataset}")
    path = results_dir / f"actionability_{experiment_id}.json"
    import ocean
    ocean_dir = Path(ocean.__file__).resolve().parent.parent
    revision = subprocess.run(["git", "-C", str(ocean_dir), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    result = dict(
        schema_version=SCHEMA_VERSION, experiment_id=experiment_id, dataset=dataset,
        model_type="rf", n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        norm=NORM, seed=seed, voting=voting, threads=threads, timeout=timeout,
        scale=True,
        scaling_definition="Continuous features: (x - min) / (max - min) - 0.5, fitted on the full dataset; discrete and encoded features unchanged",
        query_count=min(query_count, len(X)), actionability=rules,
        dataset_sha256=hashlib.sha256((ROOT / "datasets" / f"{dataset}.csv").read_bytes()).hexdigest(),
        ocean_revision=revision.stdout.strip() if revision.returncode == 0 else None,
        ocean_path=str(ocean_dir), variants=[v for v, _, _ in VARIANTS],
        feature_columns=[list(c) if isinstance(c, tuple) else c for c in X.columns],
        feature_indices=feature_indices, validation_tolerance=VALIDATION_TOLERANCE,
        distance_definition="OCEAN get_distance(), norm=1, scale=True",
        ratio_definition="Arithmetic mean of per-query actionable/plain distances for valid paired CFs; baseline > 0",
        reference="https://arxiv.org/pdf/2106.06631 (Section 4.1)",
        complete=False,
    )
    if not overwrite and any(same_run(e, result) and complete(e) for e in load_saved_entries(path)):
        print(f"Skipping experiment {experiment_id} ({dataset}): matching result exists.")
        return path
    started = time.perf_counter()
    model, accuracy = train_model(X, y, model_type="rf", n_estimators=RF_N_ESTIMATORS,
                                 max_depth=RF_MAX_DEPTH, seed=seed, voting=voting,
                                 return_accuracy=True)
    result.update(model_build_time=time.perf_counter() - started, accuracy=float(accuracy),
                  n_features=X.shape[1], split_levels=get_split_levels(model), nodes=get_node_count(model))
    pairs = sample_queries(X, model, seed=seed, query_count=query_count)
    sampled = X.sample(n=query_count, random_state=seed) if query_count < len(X) else X
    rows = [dict(query=q.tolist(), target=t, query_index=int(index),
                 query_prediction=int(model.predict([q])[0]))
            for (q, t), index in zip(pairs, sampled.index)]
    result["explanations"] = rows
    for variant, method, constrained in VARIANTS:
        explainer, build_time = make_explainer(method, model=model, mapper=mapper,
                                             isolation=None, isolation_threshold=None)
        result[f"{variant}_build_time"] = build_time
        try:
            for row, (query, target) in tqdm(zip(rows, pairs), total=len(rows),
                                            desc=f"{dataset} | {variant}"):
                metric = explain_one(query, target, explainer=explainer, model=model,
                                     rules=rules, feature_indices=feature_indices,
                                     constrained=constrained, seed=seed,
                                     threads=threads, timeout=timeout)
                row.update({f"{variant}_{key}": value for key, value in metric.items()})
        finally:
            close_explainer(explainer)
            del explainer
            gc.collect()
        save_entry(path, result)  # Preserve finished variants if a later build fails.
    add_summaries(result)
    result["complete"] = True
    save_entry(path, result)
    print(f"Saved actionability results to {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", nargs="+", type=int, choices=EXPERIMENT_IDS,
                        default=list(EXPERIMENT_IDS), help="Dataset IDs from parameters.py: 1, 3, 4, 5, 10")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--threads", type=int, default=N_THREADS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--queries", type=int, default=N_SAMPLES)
    parser.add_argument("--voting", choices=VOTING, default="SOFT")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/rf/actionability")
    args = parser.parse_args()
    if min(args.queries, args.threads, args.timeout) <= 0:
        parser.error("queries, threads, and timeout must be positive")
    if args.threads > (os.cpu_count() or 1):
        parser.error("requested threads exceed available CPUs")
    for experiment_id in args.experiment:
        run_dataset_experiment(experiment_id, seed=args.seed, threads=args.threads,
                               timeout=args.timeout, voting=args.voting,
                               query_count=args.queries, overwrite=args.overwrite,
                               results_dir=args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
