#!/usr/bin/env python3
import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import gurobipy as gp
import numpy as np
import pandas as pd

from parameters import (
    DATASETS,
    SEEDS,
    MODELS,
    N_ESTIMATORS,
    MAX_DEPTHS,
    TIMEOUT,
    N_SAMPLES,
)
from utils import train_model, parse_dataset
from ocean import MixedIntegerProgramExplainer, ConstraintProgrammingExplainer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def set_thread_envvars(threads: int) -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(threads)


def make_explainer(
    model: Any, mapper: Any, explainer_type: str, seed: int, threads: int
) -> Any:
    if explainer_type == "cp":
        exp = ConstraintProgrammingExplainer(
            model,
            mapper=mapper,
            max_time=TIMEOUT,
            seed=seed,
            n_threads=threads,
        )
        return exp

    elif explainer_type == "mip":
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("TimeLimit", TIMEOUT)
        env.setParam("Seed", seed)
        env.setParam("Threads", threads)
        env.start()
        return MixedIntegerProgramExplainer(model, mapper=mapper, env=env)

    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")


def explain_one(
    query: np.ndarray,
    y: int,
    explainer: MixedIntegerProgramExplainer | ConstraintProgrammingExplainer,
) -> Dict[str, Any]:
    t0 = time.time()

    if hasattr(explainer, "solver"):  # CPExp
        explainer.explain(query, y=y, norm=1, save_callback=True)
        status = explainer.solver.status_name()
    else:  # MIPExp
        explainer.explain(query, y=y, norm=1, return_callback=True)
        status = explainer.Status
    explainer.cleanup()

    return {
        "status": status,
        "time": time.time() - t0,
        "callback": explainer.callback.sollist,
    }


def get_performance_metrics(
    model: Any,
    data: pd.DataFrame,
    mapper: Any,
    explainer_type: str,
    seed: int,
    threads: int,
) -> List[Dict[str, Any]]:
    explainer = make_explainer(model, mapper, explainer_type, seed, threads)
    metrics: List[Dict[str, Any]] = []
    test_data = (
        data.sample(n=N_SAMPLES, random_state=seed) if N_SAMPLES < len(data) else data
    )
    for i in test_data.index:
        q = test_data.loc[i].to_numpy().flatten()
        y = 1 - model.predict([q])[0]
        res = explain_one(q, y, explainer)
        metrics.append(res)
    return metrics


def run_experiment(
    dataset_path: str, n_estimators: int, max_depth: int, seed: int, threads: int
) -> Dict[str, Any]:
    logging.info(
        "Dataset=%s | n_estimators=%d | max_depth=%s | seed=%d | threads=%d",
        dataset_path,
        n_estimators,
        max_depth,
        seed,
        threads,
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
        "seed": seed,
        "threads": threads,
        "accuracy": acc,
        "explanations": {},
    }
    for expl_type in MODELS:
        out["explanations"][expl_type] = get_performance_metrics(
            model, X, mapper, expl_type, seed, threads
        )
    return out


def run_experiments(threads: int) -> None:
    set_thread_envvars(threads)

    total_cpus = os.cpu_count() or 1
    max_workers = max(1, total_cpus // threads)
    logging.info(
        "Total CPUs=%d → threads per worker=%d → workers=%d",
        total_cpus,
        threads,
        max_workers,
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
    )
    combos = [
        (ds, ne, md, sd, threads)
        for ds in DATASETS
        for ne in N_ESTIMATORS
        for md in MAX_DEPTHS
        for sd in SEEDS
    ]

    all_results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as execu:
        futures = {
            execu.submit(run_experiment, ds, ne, md, sd, threads): (ds, ne, md, sd)
            for ds, ne, md, sd, _ in combos
        }
        for fut in as_completed(futures):
            # params = futures[fut]
            # try:
            # except Exception as e:
            #    logging.error("Failed %s: %s", params, e)
            # else:
            all_results.append(fut.result())

    Path("results.json").write_text(json.dumps(all_results, indent=2))
    logging.info("Done — wrote %d results to results.json", len(all_results))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=1,
        help="CPUs per worker process (default: 1)",
    )
    args = parser.parse_args()

    if os.cpu_count() and args.threads > os.cpu_count():
        parser.error(f"--threads ({args.threads}) > available CPUs ({os.cpu_count()})")

    run_experiments(args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
