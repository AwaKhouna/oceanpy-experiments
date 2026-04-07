#!/usr/bin/env python3
from itertools import zip_longest
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
        if (
            res["n_estimators"] == n_estimators
            and res["max_depth"] == max_depth
            and res["seed"] == seed
            and all(method in res.get("explanations", {}) for method in MODELS)
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
    )
    explainer.cleanup()
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
            "cf": None,
            "objective": None,
            "status": explainer.get_solving_status(),
            "valid": None,
            "target": int(y),
            "time": t_final,
            "callback": sollist,
        }
    return {
        "cf": cf.to_numpy().tolist(),
        "objective": explainer.get_distance(),
        "status": explainer.get_solving_status(),
        "valid": int(y) == int(model.predict([cf.to_numpy()])[0]),
        "target": int(y),
        "time": t_final,
        "callback": sollist,
    }


def choose_random_label(y: int, n_classes: int) -> int:
    if n_classes == 2:
        return 1 - y
    else:
        labels = list(range(n_classes))
        labels.pop(y)
        return np.random.choice(labels)


def get_performance_metrics(
    model: Any,
    data: pd.DataFrame,
    target: pd.Series,
    mapper: Any,
    explainer_type: str,
    seed: int,
    threads: int,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    explainer, build_time = make_explainer(
        model,
        mapper,
        explainer_type,
        data=data,
        target=target,
        dataset_name=dataset_name,
    )
    metrics: List[Dict[str, Any]] = []
    test_data = (
        data.sample(n=N_SAMPLES, random_state=seed) if N_SAMPLES < len(data) else data
    )
    for i in tqdm(test_data.index):
        q = test_data.loc[i].to_numpy().flatten()
        y = choose_random_label(model.predict([q])[0], model.n_classes_)
        res = explain_one(q, y, explainer, seed=seed, threads=threads, model=model)
        metrics.append(res)
    return metrics, build_time


def make_mace_placeholder(reason: str) -> Dict[str, Any]:
    return {
        "build_time": 0.0,
        "metrics": [],
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
        "explanations": {},
    }
    for j, expl_type in enumerate(MODELS):
        print(f"\t Explaining with :{expl_type}")
        if expl_type == "mace" and model_type != "rf":
            out["explanations"][expl_type] = make_mace_placeholder(
                "MACE integration is only enabled for rf experiments."
            )
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
                )
                out["explanations"][expl_type] = {
                    "build_time": build_time,
                    "metrics": metrics,
                }
                if expl_type == "mace":
                    out["explanations"][expl_type]["supported"] = True
            except ValueError as exc:
                if expl_type != "mace":
                    raise
                out["explanations"][expl_type] = make_mace_placeholder(str(exc))
        if j >= 1:
            check_results(out)
        save_results(out, filename=filename, overwrite=True)
    return out


def check_results(results: Dict[str, Any]) -> None:
    cp_metrics = results["explanations"]["cp"]["metrics"]
    mip_metrics = results["explanations"]["mip"]["metrics"]
    for cp, mip in zip_longest(cp_metrics, mip_metrics):
        mip_optimal = mip is not None and mip["status"] == "OPTIMAL"
        cp_optimal = cp is not None and cp["status"] == "OPTIMAL"
        if not mip_optimal or not cp_optimal:
            continue
        if cp["objective"] is not None and mip["objective"] is not None:
            assert abs(cp["objective"] - mip["objective"]) < 1e-4, (
                f"Objective values differ: {cp['objective']:.4f} vs {mip['objective']:.4f} cf_mip={mip['cf']} cf_cp={cp['cf']}"
            )


def save_results(
    results: List[Dict[str, Any]],
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

    if not force and not overwrite and check_experiment(ds, ne, md, sd, model_type=model_type):
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
    res = run_experiment(
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
