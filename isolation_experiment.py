#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tqdm import tqdm


from ocean import (  # noqa: E402
    ConstraintProgrammingExplainer,
    MixedIntegerProgramExplainer,
)
from parameters import DATASETS, N_SAMPLES, SEEDS, TIMEOUT, VOTING, N_THREADS
from run_experiment import (  # noqa: E402
    METHOD_RESULT_KEYS,
    choose_random_label,
    get_node_count,
    get_split_levels,
    load_dataset,
    normalize_voting,
)
from utils import train_model  # noqa: E402

ROOT = Path(__file__).resolve().parent

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 5
N_ISOLATORS = 100
NORM = 1
TARGET_OUTLIER_RATE = 0.10
DEFAULT_THREADS = N_THREADS
VARIANTS = (
    ("cp_plain", "cp", False),
    ("cp_iso", "cp", True),
    ("mip_plain", "mip", False),
    ("mip_iso", "mip", True),
)
VARIANT_NAMES = tuple(name for name, _, _ in VARIANTS)
QUERY_RESULT_KEYS = METHOD_RESULT_KEYS + (
    "error",
    "isolation_score",
    "isolation_score_percentile",
    "plausible",
)
EXPERIMENT_IDS = tuple(range(1, len(DATASETS) + 1))
SCHEMA_VERSION = 3


def get_dataset_from_experiment_id(experiment_id: int) -> str:
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(
            f"Experiment ID must be between 1 and {len(DATASETS)}, got {experiment_id}."
        )
    return DATASETS[experiment_id - 1]


def results_path(experiment_id: int) -> Path:
    return ROOT / "results" / "rf" / "isolation" / f"isolation_{experiment_id}.json"


def load_saved_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    raise ValueError(f"Unexpected results payload in {path}: {type(payload)}")


def entry_identity(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("schema_version"),
        entry.get("experiment_id"),
        entry.get("dataset"),
        entry.get("model_type"),
        entry.get("n_estimators"),
        entry.get("max_depth"),
        entry.get("n_isolators"),
        entry.get("norm"),
        entry.get("seed"),
        entry.get("voting", "SOFT"),
        entry.get("query_count"),
        entry.get("target_outlier_rate"),
    )


def is_complete_entry(entry: dict[str, Any]) -> bool:
    explanations = entry.get("explanations")
    if not isinstance(explanations, list):
        return False
    if len(explanations) != entry.get("query_count"):
        return False
    for key in ("isolation_build_time", "target_isolators"):
        if key not in entry:
            return False
    for variant in VARIANT_NAMES:
        if f"{variant}_build_time" not in entry:
            return False
        if f"{variant}_plausibility_prevalence" not in entry:
            return False
    for explanation in explanations:
        if not isinstance(explanation, dict):
            return False
        for shared_key in (
            "query",
            "target",
            "target_isolator_training_count",
            "query_isolation_score",
            "query_isolation_score_percentile",
            "query_plausible",
        ):
            if shared_key not in explanation:
                return False
        for variant in VARIANT_NAMES:
            for key in QUERY_RESULT_KEYS:
                if f"{variant}_{key}" not in explanation:
                    return False
    return True


def existing_complete_entry(
    path: Path,
    *,
    experiment_id: int,
    dataset: str,
    seed: int,
    voting: str,
    query_count: int,
) -> dict[str, Any] | None:
    expected = (
        SCHEMA_VERSION,
        experiment_id,
        dataset,
        "rf",
        RF_N_ESTIMATORS,
        RF_MAX_DEPTH,
        N_ISOLATORS,
        NORM,
        seed,
        voting,
        query_count,
        TARGET_OUTLIER_RATE,
    )
    for entry in load_saved_entries(path):
        if entry_identity(entry) == expected and is_complete_entry(entry):
            return entry
    return None


def save_entry(path: Path, result: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_saved_entries(path)
    identity = entry_identity(result)
    payload: list[dict[str, Any]] = []
    replaced = False

    for entry in entries:
        if entry_identity(entry) != identity:
            payload.append(entry)
            continue
        payload.append(result if overwrite or not is_complete_entry(entry) else entry)
        replaced = True

    if not replaced:
        payload.append(result)

    path.write_text(json.dumps(payload, indent=2))


def sample_queries(
    X: pd.DataFrame,
    model: Any,
    *,
    seed: int,
    query_count: int,
) -> list[tuple[np.ndarray, int]]:
    test_data = (
        X.sample(n=query_count, random_state=seed) if query_count < len(X) else X
    )
    pairs: list[tuple[np.ndarray, int]] = []
    for row_index in test_data.index:
        query = test_data.loc[row_index].to_numpy(dtype=float).flatten()
        target = choose_random_label(model.predict([query])[0], model.n_classes_, seed)
        pairs.append((query, int(target)))
    return pairs


def fit_isolation_forest(
    X: pd.DataFrame,
    *,
    seed: int,
) -> tuple[IsolationForest, float]:
    t0 = time.time()
    isolation = IsolationForest(
        n_estimators=N_ISOLATORS,
        random_state=seed,
        contamination="auto",
    )
    isolation.fit(X)
    return isolation, time.time() - t0


def build_target_isolators(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], float]:
    target_isolators: dict[int, dict[str, Any]] = {}
    total_build_time = 0.0
    for target_class in sorted(int(label) for label in np.unique(y)):
        class_data = X.loc[y == target_class]
        if class_data.empty:
            raise ValueError(f"No samples available for target class {target_class}.")
        isolation, build_time = fit_isolation_forest(class_data, seed=seed)
        scores = np.asarray(isolation.decision_function(class_data), dtype=float)
        threshold, decision_level, achieved_outlier_rate = search_isolation_threshold(
            scores,
            isolation=isolation,
            target_outlier_rate=TARGET_OUTLIER_RATE,
        )
        total_build_time += build_time
        target_isolators[target_class] = {
            "isolation": isolation,
            "build_time": build_time,
            "training_count": int(len(class_data)),
            "max_samples": int(isolation.max_samples_),
            "offset": float(isolation.offset_),
            "threshold": threshold,
            "decision_level": decision_level,
            "target_outlier_rate": TARGET_OUTLIER_RATE,
            "achieved_outlier_rate": achieved_outlier_rate,
            "scores": scores,
            "sorted_scores": np.sort(scores),
        }
    return target_isolators, total_build_time


def decision_level_from_threshold(
    isolation: IsolationForest,
    threshold: float,
) -> float:
    return float(-threshold - isolation.offset_)


def outlier_rate_at_threshold(
    scores: np.ndarray,
    *,
    isolation: IsolationForest,
    threshold: float,
) -> float:
    level = decision_level_from_threshold(isolation, threshold)
    return float(np.mean(scores < level))


def search_isolation_threshold(
    scores: np.ndarray,
    *,
    isolation: IsolationForest,
    target_outlier_rate: float = TARGET_OUTLIER_RATE,
    iterations: int = 80,
) -> tuple[float, float, float]:
    low = np.nextafter(0.0, 1.0)
    high = 1.0
    best_threshold = high
    best_rate = outlier_rate_at_threshold(scores, isolation=isolation, threshold=high)
    best_gap = abs(best_rate - target_outlier_rate)

    for _ in range(iterations):
        mid = (low + high) / 2.0
        rate = outlier_rate_at_threshold(scores, isolation=isolation, threshold=mid)
        gap = abs(rate - target_outlier_rate)
        if gap < best_gap:
            best_threshold = mid
            best_rate = rate
            best_gap = gap
        if rate > target_outlier_rate:
            low = mid
        else:
            high = mid

    decision_level = decision_level_from_threshold(isolation, best_threshold)
    return float(best_threshold), float(decision_level), float(best_rate)


def isolation_score_percentile(
    score: float,
    sorted_training_scores: np.ndarray,
) -> float:
    rank = np.searchsorted(sorted_training_scores, score, side="right")
    return float(rank / len(sorted_training_scores))


def evaluate_plausibility(
    point: np.ndarray,
    *,
    columns: pd.Index,
    isolation: IsolationForest,
    sorted_training_scores: np.ndarray,
    decision_level: float,
) -> tuple[float, float, bool]:
    frame = pd.DataFrame([point], columns=columns)
    score = float(isolation.decision_function(frame)[0])
    prevalence = isolation_score_percentile(score, sorted_training_scores)
    plausible = bool(score >= decision_level)
    return score, prevalence, plausible


def target_isolator_metadata(
    target_isolators: dict[int, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    metadata: dict[str, dict[str, float | int]] = {}
    for target_class, context in target_isolators.items():
        metadata[str(target_class)] = {
            "training_count": context["training_count"],
            "max_samples": context["max_samples"],
            "offset": context["offset"],
            "threshold": context["threshold"],
            "decision_level": context["decision_level"],
            "target_outlier_rate": context["target_outlier_rate"],
            "achieved_outlier_rate": context["achieved_outlier_rate"],
            "build_time": context["build_time"],
        }
    return metadata


def group_queries_by_target(
    query_target_pairs: list[tuple[np.ndarray, int]],
) -> dict[int, list[tuple[int, np.ndarray, int]]]:
    grouped: dict[int, list[tuple[int, np.ndarray, int]]] = {}
    for idx, (query, target) in enumerate(query_target_pairs):
        grouped.setdefault(int(target), []).append((idx, query, int(target)))
    return grouped


def make_explainer(
    method: str,
    *,
    model: Any,
    mapper: Any,
    isolation: IsolationForest | None,
    isolation_threshold: float | None,
) -> tuple[Any, float]:
    t0 = time.time()
    if method == "cp":
        kwargs: dict[str, Any] = {}
        if isolation is not None:
            kwargs["isolation"] = isolation
            kwargs["isolation_threshold"] = isolation_threshold
        explainer = ConstraintProgrammingExplainer(model, mapper=mapper, **kwargs)
    elif method == "mip":
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        kwargs = {"env": env}
        if isolation is not None:
            kwargs["isolation"] = isolation
            kwargs["isolation_threshold"] = isolation_threshold
        explainer = MixedIntegerProgramExplainer(model, mapper=mapper, **kwargs)
    else:
        raise ValueError(f"Unknown explainer type: {method}")
    return explainer, time.time() - t0


def close_explainer(explainer: Any) -> None:
    try:
        explainer.cleanup()
    except Exception:
        pass
    dispose = getattr(explainer, "dispose", None)
    if callable(dispose):
        try:
            dispose()
        except Exception:
            pass


def assign_metric(
    explanation: dict[str, Any],
    *,
    variant: str,
    metric: dict[str, Any],
) -> None:
    if explanation["query"] != metric["query"]:
        raise ValueError(f"Query mismatch while assigning {variant} results.")
    if explanation["target"] != metric["target"]:
        raise ValueError(f"Target mismatch while assigning {variant} results.")
    for key in QUERY_RESULT_KEYS:
        explanation[f"{variant}_{key}"] = metric[key]


def initialize_explanations(
    query_target_pairs: list[tuple[np.ndarray, int]],
    *,
    columns: pd.Index,
    target_isolators: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    for query, target in query_target_pairs:
        context = target_isolators[int(target)]
        query_score, query_prevalence, query_plausible = evaluate_plausibility(
            query,
            columns=columns,
            isolation=context["isolation"],
            sorted_training_scores=context["sorted_scores"],
            decision_level=context["decision_level"],
        )
        explanation: dict[str, Any] = {
            "query": query.tolist(),
            "target": int(target),
            "target_isolator_training_count": context["training_count"],
            "query_isolation_score": query_score,
            "query_isolation_score_percentile": query_prevalence,
            "query_plausible": query_plausible,
        }
        for variant in VARIANT_NAMES:
            for key in QUERY_RESULT_KEYS:
                explanation[f"{variant}_{key}"] = [] if key == "callback" else None
        explanations.append(explanation)
    return explanations


def explain_one(
    query: np.ndarray,
    target: int,
    *,
    explainer: Any,
    model: Any,
    use_isolation: bool,
    seed: int,
    threads: int,
    timeout: int,
    columns: pd.Index,
    isolation: IsolationForest,
    sorted_training_scores: np.ndarray,
    decision_level: float,
) -> dict[str, Any]:
    t0 = time.time()
    try:
        cf = explainer.explain(
            query,
            y=target,
            norm=NORM,
            return_callback=True,
            max_time=timeout,
            random_seed=seed,
            num_workers=threads,
            verbose=False,
            clean_up=True,
        )
        elapsed = time.time() - t0
        callback = getattr(explainer, "callback", None)
        sollist = [] if callback is None else callback.sollist
    except Exception as exc:
        try:
            explainer.cleanup()
        except Exception:
            pass
        return {
            "query": query.tolist(),
            "cf": None,
            "objective": None,
            "status": f"ERROR:{type(exc).__name__}",
            "valid": None,
            "target": int(target),
            "time": time.time() - t0,
            "callback": [],
            "error": str(exc),
            "isolation_score": None,
            "isolation_score_percentile": None,
            "plausible": None,
        }

    if cf is None:
        return {
            "query": query.tolist(),
            "cf": None,
            "objective": None,
            "status": explainer.get_solving_status(),
            "valid": None,
            "target": int(target),
            "time": elapsed,
            "callback": sollist,
            "error": None,
            "isolation_score": None,
            "isolation_score_percentile": None,
            "plausible": None,
        }

    counterfactual = cf.to_numpy()
    isolation_score, prevalence, plausible = evaluate_plausibility(
        counterfactual,
        columns=columns,
        isolation=isolation,
        sorted_training_scores=sorted_training_scores,
        decision_level=decision_level,
    )
    if use_isolation:
        plausible = True
    return {
        "query": query.tolist(),
        "cf": counterfactual.tolist(),
        "objective": explainer.get_distance(),
        "status": explainer.get_solving_status(),
        "valid": int(target) == int(model.predict([counterfactual])[0]),
        "target": int(target),
        "time": elapsed,
        "callback": sollist,
        "error": None,
        "isolation_score": isolation_score,
        "isolation_score_percentile": prevalence,
        "plausible": plausible,
    }


def variant_summary(
    explanations: list[dict[str, Any]],
    *,
    variant: str,
    use_isolation: bool,
) -> dict[str, Any]:
    found = [exp for exp in explanations if exp.get(f"{variant}_cf") is not None]
    valid = [exp for exp in found if exp.get(f"{variant}_valid") is True]
    plausible_count = sum(1 for exp in found if exp.get(f"{variant}_plausible") is True)
    distances = [
        float(exp[f"{variant}_objective"])
        for exp in found
        if exp.get(f"{variant}_objective") is not None
    ]
    times = [
        float(exp[f"{variant}_time"])
        for exp in explanations
        if exp.get(f"{variant}_time") is not None
    ]

    if not found:
        plausibility_prevalence = None
    elif use_isolation:
        plausibility_prevalence = 1.0
    else:
        plausibility_prevalence = plausible_count / len(found)

    return {
        "cf_count": len(found),
        "valid_count": len(valid),
        "plausible_count": len(found) if use_isolation else plausible_count,
        "plausibility_prevalence": plausibility_prevalence,
        "mean_distance": float(np.mean(distances)) if distances else None,
        "mean_time": float(np.mean(times)) if times else None,
    }


def run_dataset_experiment(
    experiment_id: int,
    *,
    seed: int,
    threads: int,
    timeout: int,
    voting: str,
    query_count: int,
    overwrite: bool,
) -> Path:
    dataset = get_dataset_from_experiment_id(experiment_id)
    path = results_path(experiment_id)

    (X, y), mapper = load_dataset(dataset)

    if (
        not overwrite
        and existing_complete_entry(
            path,
            experiment_id=experiment_id,
            dataset=dataset,
            seed=seed,
            voting=voting,
            query_count=query_count,
        )
        is not None
    ):
        print(
            f"Skipping experiment {experiment_id} ({dataset}): matching result exists."
        )
        return path

    target_isolators, isolation_build_time = build_target_isolators(
        X,
        y,
        seed=seed,
    )

    model, accuracy = train_model(
        X,
        y,
        model_type="rf",
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        seed=seed,
        voting=voting,
        return_accuracy=True,
    )

    query_target_pairs = sample_queries(X, model, seed=seed, query_count=query_count)
    grouped_queries = group_queries_by_target(query_target_pairs)
    explanations = initialize_explanations(
        query_target_pairs,
        columns=X.columns,
        target_isolators=target_isolators,
    )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model_type": "rf",
        "variants": list(VARIANT_NAMES),
        "norm": NORM,
        "n_estimators": RF_N_ESTIMATORS,
        "max_depth": RF_MAX_DEPTH,
        "n_isolators": N_ISOLATORS,
        "target_outlier_rate": TARGET_OUTLIER_RATE,
        "target_isolators": target_isolator_metadata(target_isolators),
        "seed": seed,
        "voting": voting,
        "threads": threads,
        "timeout": timeout,
        "accuracy": accuracy,
        "n_features": X.shape[1],
        "split_levels": get_split_levels(model),
        "nodes": get_node_count(model),
        "query_count": len(query_target_pairs),
        "isolation_build_time": isolation_build_time,
        "explanations": explanations,
    }

    for variant, method, use_isolation in VARIANTS:
        isolation_label = "with isolation" if use_isolation else "plain"
        print(
            f"Building {variant} explainer for experiment {experiment_id} "
            f"({dataset}, {isolation_label})"
        )
        variant_build_time = 0.0
        target_build_times: dict[str, float] = {}
        for target_class, indexed_queries in grouped_queries.items():
            context = target_isolators[target_class]
            explainer, build_time = make_explainer(
                method,
                model=model,
                mapper=mapper,
                isolation=context["isolation"] if use_isolation else None,
                isolation_threshold=context["threshold"] if use_isolation else None,
            )
            variant_build_time += build_time
            target_build_times[str(target_class)] = build_time

            description = (
                f"exp={experiment_id} | {dataset} | {variant} | target={target_class}"
            )
            try:
                for idx, query, target in tqdm(indexed_queries, desc=description):
                    metric = explain_one(
                        query,
                        target,
                        explainer=explainer,
                        model=model,
                        use_isolation=use_isolation,
                        seed=seed,
                        threads=threads,
                        timeout=timeout,
                        columns=X.columns,
                        isolation=context["isolation"],
                        sorted_training_scores=context["sorted_scores"],
                        decision_level=context["decision_level"],
                    )
                    assign_metric(explanations[idx], variant=variant, metric=metric)
            finally:
                close_explainer(explainer)
                del explainer
                gc.collect()

        result[f"{variant}_build_time"] = variant_build_time
        result[f"{variant}_target_build_times"] = target_build_times
        summary = variant_summary(
            explanations,
            variant=variant,
            use_isolation=use_isolation,
        )
        for key, value in summary.items():
            result[f"{variant}_{key}"] = value

    save_entry(path, result, overwrite=overwrite)
    print(f"Saved isolation results to {path.relative_to(ROOT)}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        nargs="+",
        type=int,
        choices=EXPERIMENT_IDS,
        default=list(EXPERIMENT_IDS),
        help=(
            "Experiment ID(s) to run, using the dataset order in parameters.py "
            f"(1..{len(DATASETS)}). Default: all experiments."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEEDS[0],
        help=f"Random seed for RF, IF, and query sampling. Default: {SEEDS[0]}",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Solver threads per explanation. Default: {DEFAULT_THREADS}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT,
        help=f"Time limit per explanation in seconds. Default: {TIMEOUT}",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=N_SAMPLES,
        help=f"Number of queries per dataset. Default: {N_SAMPLES}",
    )
    parser.add_argument(
        "--voting",
        type=str,
        choices=VOTING,
        default="SOFT",
        help="RF voting mode. Default: SOFT",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing matching isolation entry.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads > os.cpu_count():
        raise ValueError(
            f"Requested {args.threads} threads, but only {os.cpu_count()} are available."
        )
    if args.queries <= 0:
        raise ValueError(f"Expected a positive query count, got {args.queries}.")

    voting = normalize_voting(args.voting, "rf")
    for experiment_id in args.experiment:
        run_dataset_experiment(
            experiment_id,
            seed=args.seed,
            threads=args.threads,
            timeout=args.timeout,
            voting=voting,
            query_count=args.queries,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
