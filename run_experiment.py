#!/usr/bin/env python3
import sys
import os
import json
import time
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Literal
import gurobipy as gp
import numpy as np
import pandas as pd
import ortools.sat.python.cp_model as _cp
import ortools
from tqdm import tqdm
from parameters import (
    DATASETS,
    SEEDS,
    MODELS,
    N_ESTIMATORS,
    MAX_DEPTHS,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_MAX_DEPTH,
    VOTING,
    TIMEOUT,
    N_SAMPLES,
)
from utils import train_model, get_split_levels, get_node_count, parse_dataset
from mace_explainer import MACEExplainer
from ocean import (
    MixedIntegerProgramExplainer,
    ConstraintProgrammingExplainer,
    MaxSATExplainer,
)
import warnings

# from ocean.datasets import load_adult, load_compas, load_credit
from ocean.typing import BaseExplainableEnsemble

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logger = logging.getLogger(__name__)
METHOD_RESULT_KEYS = ("objective", "cf", "status", "time", "valid", "callback")
SETUPS = ("setup1", "setup2")


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    seed: int
    method: str
    voting: Literal["SOFT", "HARD"]
    setup: str

## For compatibility with ortools 9.10 and earlier versions
if ortools.__version__ < "9.11":
    _orig_sum = _cp.LinearExpr.Sum

    def _sum_wrapper(*args):
        # If they passed one iterable, call it directly; else pack varargs into a list
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            return _orig_sum(args[0])
        else:
            return _orig_sum(list(args))

    _cp.LinearExpr.Sum = _sum_wrapper


def load_dataset(
    dataset: str,
    scale: bool = False,
    return_mapper: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    return parse_dataset(
        dataset,
        scale=scale,
        return_mapper=return_mapper,
    )
    # if dataset == "COMPAS":
    #     return load_compas(scale=scale, return_mapper=return_mapper)
    # elif dataset == "Adult":
    #     return load_adult(scale=scale, return_mapper=return_mapper)
    # elif dataset == "Credit":
    #     return load_credit(scale=scale, return_mapper=return_mapper)
    # else:
    #     raise ValueError(f"Unknown dataset: {dataset}")


def set_thread_envvars(threads: int) -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(threads)


def available_methods(model_type: Literal["rf", "xgb"]) -> List[str]:
    if model_type == "xgb":
        return [method for method in MODELS if method != "mace"]
    return list(MODELS)


def normalize_voting(
    voting: str | None,
    model_type: Literal["rf", "xgb"],
) -> Literal["SOFT", "HARD"]:
    if model_type == "xgb":
        if voting is not None and voting.upper() != "SOFT":
            raise ValueError("XGBoost experiments only support SOFT voting.")
        return "SOFT"
    if voting is None:
        return "SOFT"
    normalized = voting.upper()
    if normalized not in VOTING:
        raise ValueError(f"Unknown voting type: {voting}")
    return normalized  # type: ignore[return-value]


def get_result_filename(
    dataset: str,
    n_estimators: int,
    max_depth: int,
    seed: int,
    *,
    model_type: Literal["rf", "xgb"],
    voting: str | None = None,
) -> str:
    normalized_voting = normalize_voting(voting, model_type)
    suffix = "_HARD" if model_type == "rf" and normalized_voting == "HARD" else ""
    return f"{model_type}/exp_{dataset}_{n_estimators}_{max_depth}_{seed}{suffix}.json"


def result_identity(result: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        result.get("dataset"),
        result.get("model_type"),
        result.get("n_estimators"),
        result.get("max_depth"),
        result.get("seed"),
        result.get("voting", "SOFT"),
    )


def result_methods(result: Dict[str, Any]) -> List[str]:
    return [method for method in MODELS if f"{method}_build_time" in result]


def merge_result_entry(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    methods_to_merge: List[str],
) -> Dict[str, Any]:
    merged = dict(existing)
    for field in (
        "dataset",
        "model_type",
        "n_estimators",
        "max_depth",
        "split_levels",
        "nodes",
        "seed",
        "threads",
        "accuracy",
        "n_features",
        "voting",
    ):
        if field in incoming:
            merged[field] = incoming[field]

    existing_explanations = merged.get("explanations")
    incoming_explanations = incoming.get("explanations")
    if not isinstance(existing_explanations, list) or not existing_explanations:
        merged["explanations"] = incoming_explanations
        existing_explanations = merged.get("explanations")

    if not isinstance(existing_explanations, list) or not isinstance(
        incoming_explanations, list
    ):
        raise ValueError("Expected list-based explanations when merging results.")

    if len(existing_explanations) != len(incoming_explanations):
        raise ValueError(
            "Explanation count mismatch while merging saved results: "
            f"{len(existing_explanations)} vs {len(incoming_explanations)}"
        )

    for existing_explanation, incoming_explanation in zip(
        existing_explanations, incoming_explanations
    ):
        if existing_explanation.get("query") != incoming_explanation.get("query"):
            raise ValueError("Query mismatch while merging saved results.")
        if existing_explanation.get("target") != incoming_explanation.get("target"):
            raise ValueError("Target mismatch while merging saved results.")
        for method in methods_to_merge:
            for key in METHOD_RESULT_KEYS:
                metric_key = f"{method}_{key}"
                existing_explanation[metric_key] = incoming_explanation.get(metric_key)

    for method in methods_to_merge:
        build_time_key = f"{method}_build_time"
        if build_time_key in incoming:
            merged[build_time_key] = incoming[build_time_key]
        supported_key = f"{method}_supported"
        reason_key = f"{method}_reason"
        if supported_key in incoming:
            merged[supported_key] = incoming[supported_key]
        else:
            merged.pop(supported_key, None)
        if reason_key in incoming:
            merged[reason_key] = incoming[reason_key]
        else:
            merged.pop(reason_key, None)
    return merged


def coalesce_results(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    order: List[Tuple[Any, ...]] = []
    for entry in entries:
        key = result_identity(entry)
        if key not in merged:
            merged[key] = entry
            order.append(key)
            continue
        merged[key] = merge_result_entry(merged[key], entry, result_methods(entry))
    return [merged[key] for key in order]


def load_results_file(results_path: Path) -> List[Dict[str, Any]]:
    if not results_path.exists():
        return []
    with open(results_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected results payload in {results_path}: {type(payload)}")
    return coalesce_results([item for item in payload if isinstance(item, dict)])


def setup_configurations(setup: str) -> List[Tuple[int, int]]:
    if setup == "setup1":
        return [(DEFAULT_N_ESTIMATORS, max_depth) for max_depth in MAX_DEPTHS]
    if setup == "setup2":
        return [(n_estimators, DEFAULT_MAX_DEPTH) for n_estimators in N_ESTIMATORS]
    raise ValueError(f"Unknown setup: {setup}")


def check_experiment(
    dataset: str,
    n_estimators: int,
    max_depth: int,
    seed: int,
    model_type: Literal["rf", "xgb"] = "rf",
    voting: str | None = None,
    methods: List[str] | None = None,
) -> bool:
    selected_methods = methods or available_methods(model_type)
    normalized_voting = normalize_voting(voting, model_type)
    filename = get_result_filename(
        dataset,
        n_estimators,
        max_depth,
        seed,
        model_type=model_type,
        voting=normalized_voting,
    )
    results_path = Path(f"results/{filename}")
    if not results_path.exists():
        return False

    for res in load_results_file(results_path):
        explanations = res.get("explanations")
        has_all_methods = False
        if isinstance(explanations, dict):
            has_all_methods = all(method in explanations for method in selected_methods)
        elif isinstance(explanations, list):
            has_all_methods = all(
                f"{method}_build_time" in res for method in selected_methods
            )
        if (
            res["n_estimators"] == n_estimators
            and res["max_depth"] == max_depth
            and res["seed"] == seed
            and res.get("voting", "SOFT") == normalized_voting
            and has_all_methods
        ):
            return True
    return False


def make_explainer(
    model: BaseExplainableEnsemble,
    mapper: Any,
    explainer_type: str,
    data: pd.DataFrame | None = None,
    target: pd.Series | None = None,
    dataset_name: str | None = None,
    voting: Literal["SOFT", "HARD"] = "SOFT",
) -> Any:
    t0 = time.time()
    if explainer_type == "cp":
        exp = ConstraintProgrammingExplainer(model, mapper=mapper)

    elif explainer_type == "mip":
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        exp = MixedIntegerProgramExplainer(model, mapper=mapper, env=env)

    elif explainer_type == "maxsat":
        exp = MaxSATExplainer(
            model,
            mapper=mapper,
            hard_voting=voting == "HARD",
        )

    elif explainer_type == "mace":
        if data is None or target is None:
            raise ValueError("MACEExplainer requires data and target.")
        exp = MACEExplainer(
            model,
            mapper=mapper,
            data=data,
            target=target,
            dataset_name=dataset_name or "dataset",
        )

    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
    build_time = time.time() - t0
    return exp, build_time


def explain_one(
    query: np.ndarray,
    y: int,
    explainer: (
        MixedIntegerProgramExplainer
        | ConstraintProgrammingExplainer
        | MaxSATExplainer
        | MACEExplainer
    ),
    seed: int,
    threads: int,
    model: Any,
) -> Dict[str, Any]:
    is_maxsat_explainer = getattr(explainer, "Type", None) == MaxSATExplainer.Type
    t0 = time.time()
    cf = explainer.explain(
        query,
        y=y,
        norm=1,
        return_callback=True,
        max_time=TIMEOUT,
        random_seed=seed,
        num_workers=threads if not is_maxsat_explainer else None,
        verbose=False,
        clean_up=True,
    )
    t_final = time.time() - t0
    callback = getattr(explainer, "callback", None)
    if callback is not None:
        sollist = callback.sollist
    elif is_maxsat_explainer:
        sollist = (
            []
            if cf is None
            else [{"objective_value": explainer.get_distance(), "time": t_final}]
        )
    else:
        sollist = []
    if cf is None:
        return {
            "query": query.tolist(),
            "cf": None,
            "objective": None,
            "status": explainer.get_solving_status(),
            "valid": None,
            "target": int(y),
            "time": t_final,
            "callback": sollist,
        }
    return {
        "query": query.tolist(),
        "cf": cf.to_numpy().tolist(),
        "objective": explainer.get_distance(),
        "status": explainer.get_solving_status(),
        "valid": int(y) == int(model.predict([cf.to_numpy()])[0]),
        "target": int(y),
        "time": t_final,
        "callback": sollist,
    }


def choose_random_label(y: int, n_classes: int, seed: int) -> int:
    if n_classes == 2:
        return 1 - y
    else:
        labels = list(range(n_classes))
        labels.pop(y)
        np.random.seed(seed)
        return np.random.choice(labels)


def initialize_explanations(
    test_data: pd.DataFrame,
    targets: List[int],
) -> List[Dict[str, Any]]:
    explanations: List[Dict[str, Any]] = []
    for i, target in zip(test_data.index, targets):
        explanation: Dict[str, Any] = {
            "query": test_data.loc[i].to_numpy().flatten().tolist(),
            "target": int(target),
        }
        for method in MODELS:
            explanation[f"{method}_objective"] = None
            explanation[f"{method}_cf"] = None
            explanation[f"{method}_status"] = None
            explanation[f"{method}_time"] = None
            explanation[f"{method}_valid"] = None
            explanation[f"{method}_callback"] = []
        explanations.append(explanation)
    return explanations


def merge_method_metrics(
    explanations: List[Dict[str, Any]],
    explainer_type: str,
    metrics: List[Dict[str, Any]],
) -> None:
    if len(explanations) != len(metrics):
        raise ValueError(
            f"Instance count mismatch for {explainer_type}: "
            f"{len(explanations)} expected, got {len(metrics)}"
        )
    for explanation, metric in zip(explanations, metrics):
        if explanation["query"] != metric["query"]:
            raise ValueError(f"Query mismatch while merging {explainer_type} results.")
        if explanation["target"] != metric["target"]:
            raise ValueError(f"Target mismatch while merging {explainer_type} results.")
        for key in METHOD_RESULT_KEYS:
            explanation[f"{explainer_type}_{key}"] = metric[key]


def get_performance_metrics(
    model: Any,
    data: pd.DataFrame,
    target: pd.Series,
    mapper: Any,
    explainer_type: str,
    seed: int,
    threads: int,
    dataset_name: str,
    voting: Literal["SOFT", "HARD"] = "SOFT",
    test_data: pd.DataFrame | None = None,
    targets: List[int] | None = None,
) -> Tuple[List[Dict[str, Any]], float]:
    explainer, build_time = make_explainer(
        model,
        mapper,
        explainer_type,
        data=data,
        target=target,
        dataset_name=dataset_name,
        voting=voting,
    )
    metrics: List[Dict[str, Any]] = []
    if test_data is None:
        test_data = (
            data.sample(n=N_SAMPLES, random_state=seed)
            if N_SAMPLES < len(data)
            else data
        )
    if targets is not None and len(test_data) != len(targets):
        raise ValueError(
            f"Target count mismatch: {len(test_data)} queries, {len(targets)} targets"
        )
    for j, i in enumerate(tqdm(test_data.index)):
        q = test_data.loc[i].to_numpy().flatten()
        y = (
            targets[j]
            if targets is not None
            else choose_random_label(model.predict([q])[0], model.n_classes_, seed)
        )
        res = explain_one(q, y, explainer, seed=seed, threads=threads, model=model)
        metrics.append(res)
    return metrics, build_time


def make_mace_placeholder(reason: str) -> Dict[str, Any]:
    return {
        "build_time": 0.0,
        "supported": False,
        "reason": reason,
    }


def run_experiment(
    dataset_path: str,
    n_estimators: int,
    max_depth: int,
    seed: int,
    threads: int,
    filename: str,
    model_type: Literal["rf", "xgb"] = "rf",
    voting: str | None = None,
    methods: List[str] | None = None,
) -> Dict[str, Any]:
    selected_methods = methods or available_methods(model_type)
    normalized_voting = normalize_voting(voting, model_type)
    logger.info(
        "Dataset=%s | n_estimators=%d | max_depth=%s | seed=%d | voting=%s | threads=%d",
        dataset_path,
        n_estimators,
        max_depth,
        seed,
        normalized_voting,
        threads,
    )
    print(
        f"Running experiment on {dataset_path} with n_estimators={n_estimators}, "
        f"max_depth={max_depth}, seed={seed}, voting={normalized_voting}, threads={threads}"
    )

    (X, y), mapper = load_dataset(dataset_path)
    # parse_dataset(dataset_path, return_mapper=True)
    model = train_model(
        X,
        y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        seed=seed,
        model_type=model_type,
        voting=normalized_voting,
    )
    acc = model.score(X, y)
    test_data = X.sample(n=N_SAMPLES, random_state=seed) if N_SAMPLES < len(X) else X
    targets = [
        choose_random_label(
            model.predict([test_data.loc[i].to_numpy().flatten()])[0],
            model.n_classes_,
            seed,
        )
        for i in test_data.index
    ]
    out: Dict[str, Any] = {
        "dataset": dataset_path,
        "model_type": model_type,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "split_levels": get_split_levels(model),
        "nodes": get_node_count(model),
        "seed": seed,
        "voting": normalized_voting,
        "threads": threads,
        "accuracy": acc,
        "n_features": X.shape[1],
        "explanations": initialize_explanations(test_data, targets),
    }
    for j, expl_type in enumerate(selected_methods):
        print(f"\t Explaining with :{expl_type}")
        if expl_type == "mace" and model_type != "rf":
            placeholder = make_mace_placeholder(
                "MACE integration is only enabled for rf experiments."
            )
            out[f"{expl_type}_build_time"] = placeholder["build_time"]
            out[f"{expl_type}_supported"] = placeholder["supported"]
            out[f"{expl_type}_reason"] = placeholder["reason"]
        else:
            try:
                metrics, build_time = get_performance_metrics(
                    model,
                    X,
                    y,
                    mapper,
                    expl_type,
                    seed,
                    threads,
                    dataset_path,
                    voting=normalized_voting,
                    test_data=test_data,
                    targets=targets,
                )
                out[f"{expl_type}_build_time"] = build_time
                merge_method_metrics(out["explanations"], expl_type, metrics)
                if expl_type == "mace":
                    out["mace_supported"] = True
                    out.pop("mace_reason", None)
            except ValueError as exc:
                if expl_type != "mace":
                    raise
                placeholder = make_mace_placeholder(str(exc))
                out[f"{expl_type}_build_time"] = placeholder["build_time"]
                out[f"{expl_type}_supported"] = placeholder["supported"]
                out[f"{expl_type}_reason"] = placeholder["reason"]
        if j >= 1:
            check_results(out, model)
        save_results(out, filename=filename, methods_to_merge=[expl_type])
    return out


def check_results(results: Dict[str, Any], model: BaseExplainableEnsemble) -> None:
    for explanation in results["explanations"]:
        mip_optimal = explanation["mip_status"] == "OPTIMAL"
        cp_optimal = explanation["cp_status"] == "OPTIMAL"
        if not mip_optimal or not cp_optimal:
            continue
        if (
            explanation["cp_objective"] is not None
            and explanation["mip_objective"] is not None
        ):
            if abs(explanation["cp_objective"] - explanation["mip_objective"]) >= 1e-4:
                if results["model_type"] == "rf":
                    print_rf_splits_levels(model)
                msg = (
                    "Objective values differ: "
                    f"{explanation['cp_objective']:.4f} vs {explanation['mip_objective']:.4f} \n"
                    f"query={explanation['query']} \n"
                    f"cf_mip={explanation['mip_cf']} \n"
                    f"cf_cp={explanation['cp_cf']}"
                )
                logger.warning(msg)
                warnings.warn(msg, category=UserWarning, stacklevel=2)


def save_results(
    results: Dict[str, Any],
    filename: str,
    methods_to_merge: List[str] | None = None,
) -> None:
    results_path = Path(f"results/{filename}")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new_res = load_results_file(results_path) if results_path.exists() else []
    if methods_to_merge is None:
        new_res.append(results)
        payload = coalesce_results(new_res)
    else:
        payload = []
        merged_entry = results
        merged = False
        for existing in new_res:
            if result_identity(existing) != result_identity(results):
                payload.append(existing)
                continue
            merged_entry = merge_result_entry(existing, results, methods_to_merge)
            payload.append(merged_entry)
            merged = True
        if not merged:
            payload.append(results)
    results_path.write_text(json.dumps(payload, indent=2))
    logger.info("Results saved to %s", results_path)


def get_total_experiments(model_type: Literal["rf", "xgb"]) -> int:
    methods = available_methods(model_type)
    voting_count = len(VOTING) if model_type == "rf" else 1
    return len(DATASETS) * len(SEEDS) * len(methods) * voting_count * len(SETUPS)


def get_experiment_params(
    id: int,
    model_type: Literal["rf", "xgb"],
) -> ExperimentSpec:
    methods = available_methods(model_type)
    setups_per_method = len(SETUPS)
    votings = list(VOTING) if model_type == "rf" else ["SOFT"]
    configs_per_dataset = len(methods) * len(votings) * setups_per_method
    configs_per_seed = len(DATASETS) * configs_per_dataset

    sd = SEEDS[id // configs_per_seed]
    dataset_offset = id % configs_per_seed
    ds = DATASETS[dataset_offset // configs_per_dataset]
    config_offset = dataset_offset % configs_per_dataset
    method = methods[config_offset // (len(votings) * setups_per_method)]
    voting_index = (config_offset // setups_per_method) % len(votings)
    setup = SETUPS[config_offset % setups_per_method]
    return ExperimentSpec(
        dataset=ds,
        seed=sd,
        method=method,
        voting=normalize_voting(votings[voting_index], model_type),
        setup=setup,
    )


def run_experiments(
    experiment_id: int = 1,
    threads: int = 1,
    model_type: Literal["rf", "xgb"] = "rf",
    force: bool = False,
    overwrite: bool = False,
    methods: List[str] | None = None,
) -> None:
    spec = get_experiment_params(experiment_id, model_type)
    selected_methods = [spec.method]
    if methods is not None:
        if len(methods) != 1 or methods[0] != spec.method:
            raise ValueError(
                "Experiment IDs now fix a single explainer method. "
                f"Experiment {experiment_id + 1} maps to method '{spec.method}'."
            )
    if threads > os.cpu_count():
        raise ValueError(
            f"Requested {threads} threads, but only {os.cpu_count()} are available."
        )
    # set_thread_envvars(threads)

    logging.basicConfig(
        filename=f"log_{model_type}.txt",
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    logger.info(
        "Total CPUs=%d",
        threads,
    )
    configurations = setup_configurations(spec.setup)
    print(
        f"Running experiment {experiment_id + 1} with dataset={spec.dataset}, "
        f"seed={spec.seed}, method={spec.method}, voting={spec.voting}, setup={spec.setup}"
    )

    completed = 0
    skipped = 0
    for n_estimators, max_depth in configurations:
        filename = get_result_filename(
            spec.dataset,
            n_estimators,
            max_depth,
            spec.seed,
            model_type=model_type,
            voting=spec.voting,
        )
        already_done = check_experiment(
            spec.dataset,
            n_estimators,
            max_depth,
            spec.seed,
            model_type=model_type,
            voting=spec.voting,
            methods=selected_methods,
        )
        if already_done and not force and not overwrite:
            skipped += 1
            logger.info(
                "Skipping completed configuration: dataset=%s seed=%d method=%s voting=%s setup=%s n_estimators=%d max_depth=%d",
                spec.dataset,
                spec.seed,
                spec.method,
                spec.voting,
                spec.setup,
                n_estimators,
                max_depth,
            )
            continue

        _ = run_experiment(
            dataset_path=spec.dataset,
            n_estimators=n_estimators,
            max_depth=max_depth,
            seed=spec.seed,
            threads=threads,
            model_type=model_type,
            filename=filename,
            voting=spec.voting,
            methods=selected_methods,
        )
        completed += 1

    logger.info(
        "Experiment %d finished: dataset=%s seed=%d method=%s voting=%s setup=%s completed=%d skipped=%d",
        experiment_id + 1,
        spec.dataset,
        spec.seed,
        spec.method,
        spec.voting,
        spec.setup,
        completed,
        skipped,
    )


def print_rf_splits_levels(model: BaseExplainableEnsemble) -> None:
    splits = [set() for _ in range(model.n_features_in_)]
    for i, est in enumerate(model.estimators_):
        tree = est.tree_
        for j in range(tree.node_count):
            if tree.children_left[j] != -1:  # not a leaf
                splits[tree.feature[j]].add(tree.threshold[j])
    for i, s in enumerate(splits):
        print(f"Feature {i} has {len(s)} unique split levels: {sorted(s)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        "-e",
        type=int,
        default=1,
        help="Experiment ID (1-indexed within the selected model type).",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of threads (default: 1)",
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        choices=["rf", "xgb"],
        default="rf",
        help="Model type (default: rf)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun the experiment even if a completed result file already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun this experiment and replace the selected method results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=MODELS,
        default=None,
        help="Optional validation override. Must match the method fixed by the experiment ID.",
    )

    args = parser.parse_args()
    total_experiments = get_total_experiments(args.model_type)
    if not (1 <= args.experiment <= total_experiments):
        raise ValueError(
            f"Experiment ID must be between 1 and {total_experiments} for "
            f"model type '{args.model_type}', got {args.experiment}"
        )

    run_experiments(
        args.experiment - 1,
        args.threads,
        args.model_type,
        args.force,
        args.overwrite,
        args.methods,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
    logger.info("Experiment completed successfully.")
