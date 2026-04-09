#!/usr/bin/env python3
import sys
import os
import json
import time
import logging
import argparse
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


def check_experiment(
    dataset: str,
    n_estimators: int,
    max_depth: int,
    seed: int,
    model_type: Literal["rf", "xgb"] = "rf",
) -> bool:
    filename = f"{model_type}/exp_{dataset}_{n_estimators}_{max_depth}_{seed}.json"
    results_path = Path(f"results/{filename}")
    if not results_path.exists():
        return False

    with open(results_path, "r") as f:
        results = json.load(f)

    for res in results:
        explanations = res.get("explanations")
        has_all_methods = False
        if isinstance(explanations, dict):
            has_all_methods = all(method in explanations for method in MODELS)
        elif isinstance(explanations, list):
            has_all_methods = all(f"{method}_build_time" in res for method in MODELS)
        if (
            res["n_estimators"] == n_estimators
            and res["max_depth"] == max_depth
            and res["seed"] == seed
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
        exp = MaxSATExplainer(model, mapper=mapper)

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
) -> Dict[str, Any]:
    logger.info(
        "Dataset=%s | n_estimators=%d | max_depth=%s | seed=%d | threads=%d",
        dataset_path,
        n_estimators,
        max_depth,
        seed,
        threads,
    )
    print(
        f"Running experiment on {dataset_path} with n_estimators={n_estimators}, "
        f"max_depth={max_depth}, seed={seed}, threads={threads}"
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
        "threads": threads,
        "accuracy": acc,
        "n_features": X.shape[1],
        "explanations": initialize_explanations(test_data, targets),
    }
    for j, expl_type in enumerate(MODELS):
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
        save_results(out, filename=filename, overwrite=True)
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
            if abs(explanation["cp_objective"] - explanation["mip_objective"]) >= 1e-2:
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
    overwrite: bool = False,
) -> None:
    results_path = Path(f"results/{filename}")
    new_res: List[Dict[str, Any]] = []
    if results_path.exists() and not overwrite:
        with open(results_path, "r") as f:
            new_res += json.load(f)
    new_res.append(results)
    results_path.write_text(json.dumps(new_res, indent=2))
    logger.info("Results saved to %s", results_path)


def get_experiment_params(id: int) -> Tuple[str, int, int, int | None]:
    sd = SEEDS[id // (len(DATASETS) * len(N_ESTIMATORS) * len(MAX_DEPTHS))]
    ds = DATASETS[id // (len(N_ESTIMATORS) * len(MAX_DEPTHS)) % len(DATASETS)]
    ne = N_ESTIMATORS[(id // len(MAX_DEPTHS)) % len(N_ESTIMATORS)]
    md = MAX_DEPTHS[id % len(MAX_DEPTHS)]
    return ds, ne, md, sd


def run_experiments(
    experiment_id: int = 1,
    threads: int = 1,
    model_type: Literal["rf", "xgb"] = "rf",
    force: bool = False,
    overwrite: bool = False,
) -> None:
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

    ds, ne, md, sd = get_experiment_params(experiment_id)
    filename = f"{model_type}/exp_{ds}_{ne}_{md}_{sd}.json"
    results_path = Path(f"results/{filename}")

    if (
        not force
        and not overwrite
        and check_experiment(ds, ne, md, sd, model_type=model_type)
    ):
        logger.info("Experiment %d already completed", experiment_id)
        print(
            f"Experiment {experiment_id + 1} already completed for {ds} "
            f"(n_estimators={ne}, max_depth={md}, seed={sd}). "
            "Use --force to rerun or --overwrite to replace the saved file."
        )
        return

    if overwrite and results_path.exists():
        results_path.unlink()
        logger.info("Deleted existing results file %s", results_path)
        print(f"Overwriting existing results file {results_path}")

    print(
        f"Running experiment {experiment_id} with dataset {ds},",
        f" n_estimators={ne}, max_depth={md}, seed={sd}, threads={threads}",
    )
    _ = run_experiment(
        dataset_path=ds,
        n_estimators=ne,
        max_depth=md,
        seed=sd,
        threads=threads,
        model_type=model_type,
        filename=filename,
    )

    # Path("results.json").write_text(json.dumps(all_results, indent=2))
    logger.info(f"Experiment {experiment_id} is done — wrote results to {filename}")


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
        help="Experiment ID (default: 1, between 1 and 900)",
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
        help="Delete the existing results file for this experiment before rerunning.",
    )

    args = parser.parse_args()
    if not (1 <= args.experiment <= 900):
        raise ValueError(
            f"Experiment ID must be between 1 and 900, got {args.experiment}"
        )

    run_experiments(
        args.experiment - 1,
        args.threads,
        args.model_type,
        args.force,
        args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
    logger.info("Experiment completed successfully.")
