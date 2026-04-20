#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

from parameters import (  # noqa: E402
    DATASETS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    N_SAMPLES,
    SEEDS,
    TIMEOUT,
    VOTING,
)
from run_experiment import (  # noqa: E402
    METHOD_RESULT_KEYS,
    choose_random_label,
    get_node_count,
    get_split_levels,
    load_dataset,
    make_explainer,
    make_mace_placeholder,
    normalize_voting,
)
from utils import train_model  # noqa: E402

METHODS = ("cp", "mip", "mace")
NORMS = (0, 2)
QUERY_RESULT_KEYS = METHOD_RESULT_KEYS + ("error",)
EXPERIMENT_IDS = tuple(range(1, len(DATASETS) + 1))


def get_dataset_from_experiment_id(experiment_id: int) -> str:
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(
            f"Experiment ID must be between 1 and {len(DATASETS)}, got {experiment_id}."
        )
    return DATASETS[experiment_id - 1]


def l1_reference_path(
    dataset: str,
    seed: int,
    voting: str,
) -> Path:
    suffix = "_HARD" if voting == "HARD" else ""
    return (
        ROOT
        / "results"
        / "rf"
        / (
            f"exp_{dataset}_{DEFAULT_N_ESTIMATORS}_{DEFAULT_MAX_DEPTH}_{seed}{suffix}.json"
        )
    )


def multinorm_results_path(experiment_id: int) -> Path:
    return ROOT / "results" / "rf" / "multinorms" / f"multinorms_{experiment_id}.json"


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
        entry.get("experiment_id"),
        entry.get("dataset"),
        entry.get("model_type"),
        entry.get("n_estimators"),
        entry.get("max_depth"),
        entry.get("seed"),
        entry.get("voting", "SOFT"),
        tuple(entry.get("methods", METHODS)),
        tuple(entry.get("norms", NORMS)),
        entry.get("query_count"),
    )


def is_complete_entry(entry: dict[str, Any]) -> bool:
    explanations = entry.get("explanations")
    if not isinstance(explanations, list):
        return False
    if len(explanations) != entry.get("query_count"):
        return False
    for method in METHODS:
        if f"{method}_build_time" not in entry:
            return False
        if method == "mace" and f"{method}_supported" not in entry:
            return False
    for explanation in explanations:
        if not isinstance(explanation, dict):
            return False
        for method in METHODS:
            for norm in NORMS:
                for key in QUERY_RESULT_KEYS:
                    if f"{method}_{norm}_{key}" not in explanation:
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
        experiment_id,
        dataset,
        "rf",
        DEFAULT_N_ESTIMATORS,
        DEFAULT_MAX_DEPTH,
        seed,
        voting,
        METHODS,
        NORMS,
        query_count,
    )
    for entry in load_saved_entries(path):
        if entry_identity(entry) == expected and is_complete_entry(entry):
            return entry
    return None


def save_entry(results_path: Path, result: dict[str, Any], overwrite: bool) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_saved_entries(results_path)
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

    results_path.write_text(json.dumps(payload, indent=2))


def load_l1_queries(
    dataset: str,
    *,
    seed: int,
    voting: str,
    query_count: int,
) -> tuple[list[tuple[np.ndarray, int]], str | None]:
    reference_path = l1_reference_path(dataset, seed, voting)
    if not reference_path.exists():
        return [], None

    for entry in load_saved_entries(reference_path):
        explanations = entry.get("explanations")
        if (
            entry.get("dataset") != dataset
            or entry.get("n_estimators") != DEFAULT_N_ESTIMATORS
            or entry.get("max_depth") != DEFAULT_MAX_DEPTH
            or entry.get("seed") != seed
            or entry.get("voting", "SOFT") != voting
            or not isinstance(explanations, list)
            or len(explanations) < query_count
        ):
            continue

        pairs: list[tuple[np.ndarray, int]] = []
        for explanation in explanations[:query_count]:
            query = explanation.get("query")
            target = explanation.get("target")
            if query is None or target is None:
                pairs = []
                break
            pairs.append((np.asarray(query, dtype=float), int(target)))
        if len(pairs) == query_count:
            return pairs, str(reference_path.relative_to(ROOT))

    return [], None


def sample_queries(
    X: Any,
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
        query = test_data.loc[row_index].to_numpy().flatten()
        target = choose_random_label(model.predict([query])[0], model.n_classes_, seed)
        pairs.append((query, int(target)))
    return pairs


def initialize_explanations(
    query_target_pairs: list[tuple[np.ndarray, int]],
) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    for query, target in query_target_pairs:
        explanation: dict[str, Any] = {
            "query": query.tolist(),
            "target": int(target),
        }
        for method in METHODS:
            for norm in NORMS:
                for key in QUERY_RESULT_KEYS:
                    explanation[f"{method}_{norm}_{key}"] = (
                        [] if key == "callback" else None
                    )
        explanations.append(explanation)
    return explanations


def explain_one(
    query: np.ndarray,
    target: int,
    *,
    explainer: Any,
    model: Any,
    seed: int,
    threads: int,
    norm: int,
    timeout: int,
) -> dict[str, Any]:
    t0 = time.time()
    try:
        cf = explainer.explain(
            query,
            y=target,
            norm=norm,
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
        }

    counterfactual = cf.to_numpy()
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
    }


def merge_metrics(
    explanations: list[dict[str, Any]],
    *,
    method: str,
    norm: int,
    metrics: list[dict[str, Any]],
) -> None:
    if len(explanations) != len(metrics):
        raise ValueError(
            f"Instance count mismatch for {method} norm {norm}: "
            f"{len(explanations)} expected, got {len(metrics)}"
        )
    for explanation, metric in zip(explanations, metrics):
        if explanation["query"] != metric["query"]:
            raise ValueError(f"Query mismatch while merging {method} norm {norm}.")
        if explanation["target"] != metric["target"]:
            raise ValueError(f"Target mismatch while merging {method} norm {norm}.")
        for key in QUERY_RESULT_KEYS:
            explanation[f"{method}_{norm}_{key}"] = metric[key]


def build_query_target_pairs(
    dataset: str,
    *,
    X: Any,
    model: Any,
    seed: int,
    voting: str,
    query_count: int,
    reuse_l1_queries: bool,
) -> tuple[list[tuple[np.ndarray, int]], str, str | None]:
    if reuse_l1_queries:
        query_target_pairs, reference_path = load_l1_queries(
            dataset,
            seed=seed,
            voting=voting,
            query_count=query_count,
        )
        if query_target_pairs:
            return query_target_pairs, "existing_l1_results", reference_path

    return sample_queries(X, model, seed=seed, query_count=query_count), "sampled", None


def run_dataset_experiment(
    experiment_id: int,
    *,
    seed: int,
    threads: int,
    timeout: int,
    voting: str,
    query_count: int,
    overwrite: bool,
    reuse_l1_queries: bool,
) -> Path:
    dataset = get_dataset_from_experiment_id(experiment_id)
    results_path = multinorm_results_path(experiment_id)
    if (
        not overwrite
        and existing_complete_entry(
            results_path,
            experiment_id=experiment_id,
            dataset=dataset,
            seed=seed,
            voting=voting,
            query_count=query_count,
        )
        is not None
    ):
        print(f"Skipping {dataset}: matching multinorm result already exists.")
        return results_path

    (X, y), mapper = load_dataset(dataset)
    model, accuracy = train_model(
        X,
        y,
        model_type="rf",
        n_estimators=DEFAULT_N_ESTIMATORS,
        max_depth=DEFAULT_MAX_DEPTH,
        seed=seed,
        voting=voting,
        return_accuracy=True,
    )

    query_target_pairs, query_source, reference_path = build_query_target_pairs(
        dataset,
        X=X,
        model=model,
        seed=seed,
        voting=voting,
        query_count=query_count,
        reuse_l1_queries=reuse_l1_queries,
    )
    explanations = initialize_explanations(query_target_pairs)

    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model_type": "rf",
        "methods": list(METHODS),
        "norms": list(NORMS),
        "reference_norm": 1,
        "n_estimators": DEFAULT_N_ESTIMATORS,
        "max_depth": DEFAULT_MAX_DEPTH,
        "seed": seed,
        "voting": voting,
        "threads": threads,
        "timeout": timeout,
        "accuracy": accuracy,
        "n_features": X.shape[1],
        "split_levels": get_split_levels(model),
        "nodes": get_node_count(model),
        "query_count": len(query_target_pairs),
        "query_source": query_source,
        "l1_reference_path": reference_path,
        "mace_supported": False,
        "mace_reason": None,
        "explanations": explanations,
    }

    for method in METHODS:
        print(f"Building {method} explainer for {dataset}")
        if method == "mace":
            try:
                explainer, build_time = make_explainer(
                    model,
                    mapper,
                    method,
                    data=X,
                    target=y,
                    dataset_name=dataset,
                    voting=voting,
                )
                result["mace_build_time"] = build_time
                result["mace_supported"] = True
                result.pop("mace_reason", None)
            except Exception as exc:
                placeholder = make_mace_placeholder(str(exc))
                result["mace_build_time"] = placeholder["build_time"]
                result["mace_supported"] = placeholder["supported"]
                result["mace_reason"] = placeholder["reason"]
                continue
        else:
            explainer, build_time = make_explainer(
                model,
                mapper,
                method,
                data=X,
                target=y,
                dataset_name=dataset,
                voting=voting,
            )
            result[f"{method}_build_time"] = build_time

        for norm in NORMS:
            metrics: list[dict[str, Any]] = []
            description = f"{dataset} | {method} | norm={norm}"
            for query, target in tqdm(query_target_pairs, desc=description):
                metrics.append(
                    explain_one(
                        query,
                        target,
                        explainer=explainer,
                        model=model,
                        seed=seed,
                        threads=threads,
                        norm=norm,
                        timeout=timeout,
                    )
                )
            merge_metrics(explanations, method=method, norm=norm, metrics=metrics)

    save_entry(results_path, result, overwrite=overwrite)
    print(f"Saved multinorm results to {results_path.relative_to(ROOT)}")
    return results_path


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
        help=f"Random seed for the RF and the query sampling. Default: {SEEDS[0]}",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Solver threads per explanation. Default: 1",
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
        help=f"Number of queries to run per dataset. Default: {N_SAMPLES}",
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
        help="Overwrite an existing matching multinorm entry.",
    )
    parser.add_argument(
        "--no-reuse-l1-queries",
        action="store_true",
        help="Do not reuse query/target pairs from an existing L1 result file.",
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
            reuse_l1_queries=not args.no_reuse_l1_queries,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
