#!/usr/bin/env python3
import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gurobipy as gp
import numpy as np
import pandas as pd
import ortools.sat.python.cp_model as _cp
import ortools
from parameters import (
    DATASETS,
    SEEDS,
    MODELS,
    N_ESTIMATORS,
    MAX_DEPTHS,
    TIMEOUT,
    N_SAMPLES,
)
from utils import train_model, parse_dataset, get_split_levels, get_node_count
from ocean import MixedIntegerProgramExplainer, ConstraintProgrammingExplainer
import warnings

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
    dataset: str, n_estimators: int, max_depth: int, seed: int
) -> bool:
    filename = f"exp_{dataset}_{n_estimators}_{max_depth}_{seed}.json"
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
        ):
            return True
    return False


def make_explainer(model: Any, mapper: Any, explainer_type: str) -> Any:
    t0 = time.time()
    if explainer_type == "cp":
        exp = ConstraintProgrammingExplainer(model, mapper=mapper)

    elif explainer_type == "mip":
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        exp = MixedIntegerProgramExplainer(model, mapper=mapper, env=env)

    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")
    build_time = time.time() - t0
    return exp, build_time


def explain_one(
    query: np.ndarray,
    y: int,
    explainer: MixedIntegerProgramExplainer | ConstraintProgrammingExplainer,
    seed: int,
    threads: int,
    model: Any,
) -> Dict[str, Any]:
    t0 = time.time()

    if hasattr(explainer, "solver"):  # CPExp
        cf = explainer.explain(
            query,
            y=y,
            norm=1,
            return_callback=True,
            max_time=TIMEOUT,
            random_seed=seed,
            num_workers=threads,
            verbose=False,
        )
        status = explainer.solver.status_name()
    else:  # MIPExp
        cf = explainer.explain(
            query,
            y=y,
            norm=1,
            return_callback=True,
            max_time=TIMEOUT,
            random_seed=seed,
            num_workers=threads,
            verbose=False,
        )
        status = explainer.Status

    return {
        "status": status,
        "time": time.time() - t0,
        "callback": explainer.callback.sollist,
        "valid": int(y) == int(model.predict([cf.to_numpy()])[0])
        if cf is not None
        else None,
        "target": int(y),
    }


def get_performance_metrics(
    model: Any,
    data: pd.DataFrame,
    mapper: Any,
    explainer_type: str,
    seed: int,
    threads: int,
) -> List[Dict[str, Any]]:
    explainer, build_time = make_explainer(model, mapper, explainer_type)
    metrics: List[Dict[str, Any]] = []
    test_data = (
        data.sample(n=N_SAMPLES, random_state=seed) if N_SAMPLES < len(data) else data
    )
    for i in test_data.index:
        q = test_data.loc[i].to_numpy().flatten()
        y = 1 - model.predict([q])[0]
        res = explain_one(q, y, explainer, seed=seed, threads=threads, model=model)
        metrics.append(res)
    return metrics, build_time


def run_experiment(
    dataset_path: str, n_estimators: int, max_depth: int, seed: int, threads: int
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

    (X, y), mapper = parse_dataset(dataset_path, return_mapper=True)
    model = train_model(
        X, y, n_estimators=n_estimators, max_depth=max_depth, seed=seed, n_jobs=threads
    )
    acc = model.score(X, y)
    out: Dict[str, Any] = {
        "dataset": dataset_path,
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
    for expl_type in MODELS:
        metrics, build_time = get_performance_metrics(
            model, X, mapper, expl_type, seed, threads
        )
        out["explanations"][expl_type] = {
            "build_time": build_time,
            "metrics": metrics,
        }
    return out


def save_results(
    results: List[Dict[str, Any]],
    filename: str,
) -> None:
    results_path = Path(f"results/{filename}")
    new_res: List[Dict[str, Any]] = []
    if results_path.exists():
        with open(results_path, "r") as f:
            new_res += json.load(f)
    new_res.append(results)
    results_path.write_text(json.dumps(new_res, indent=2))
    logger.info("Results saved to %s", results_path)


def get_experiment_params(id: int) -> Tuple[str, int, int, int | None]:
    sd = SEEDS[0]
    ds = DATASETS[id // (len(N_ESTIMATORS) * len(MAX_DEPTHS))]
    ne = N_ESTIMATORS[(id // len(MAX_DEPTHS)) % len(N_ESTIMATORS)]
    md = MAX_DEPTHS[id % len(MAX_DEPTHS)]
    return ds, ne, md, sd


def run_experiments(experiment_id: int = 1, threads: int = 1) -> None:
    if threads > os.cpu_count():
        raise ValueError(
            f"Requested {threads} threads, but only {os.cpu_count()} are available."
        )
    set_thread_envvars(threads)

    logging.basicConfig(
        filename="log.txt",
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    logger.info(
        "Total CPUs=%d",
        threads,
    )

    ds, ne, md, sd = get_experiment_params(experiment_id)
    if check_experiment(ds, ne, md, sd):
        logger.info("Experiment %d already completed", experiment_id)
        return
    print(
        f"Running experiment {experiment_id} with dataset {ds},",
        f" n_estimators={ne}, max_depth={md}, seed={sd}, threads={threads}",
    )
    res = run_experiment(
        dataset_path=ds, n_estimators=ne, max_depth=md, seed=sd, threads=threads
    )
    filename = f"exp_{ds}_{ne}_{md}_{sd}.json"
    save_results(res, filename=filename)

    # Path("results.json").write_text(json.dumps(all_results, indent=2))
    logger.info(f"Experiment {experiment_id} is done — wrote results to {filename}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        "-e",
        type=int,
        default=1,
        help="Experiment ID (default: 1, between 1 and 90)",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="Number of threads (default: 1)",
    )

    args = parser.parse_args()
    if not (1 <= args.experiment <= 90):
        raise ValueError(
            f"Experiment ID must be between 1 and 90, got {args.experiment}"
        )

    run_experiments(args.experiment - 1, args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    logger.info("Experiment completed successfully.")
