#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:  # pragma: no cover - model export is optional.
    joblib = None

try:
    from parameters import N_SAMPLES, TIMEOUT
except Exception:  # pragma: no cover - keeps the script usable outside the repo root.
    N_SAMPLES = 50
    TIMEOUT = 900

ROOT = Path(__file__).resolve().parent


METHODS = ("cp", "mip", "mace", "maxsat")
METHOD_LABELS = {
    "cp": "CP",
    "mip": "MIP",
    "mace": "MACE",
    "maxsat": "MAXSAT",
}
PREFERRED_METRICS = (
    "median+build",
    # "median+buildAmortized",
    "median",
)
PER_COUNTERFACTUAL_METRIC = "per-counterfactual"


# Main run configuration. Edit these values instead of passing CLI arguments.
RESULTS_DIR = Path("results")
DATASETS_DIR = Path("datasets")
OUTPUT_CSV = Path("results/meta_classifier_training_data.csv")
PROCESSED_CSV = Path("Decision-Diagrams/processed_data.csv")
DIAGRAM_OUTPUT = Path("plots/meta_classifier_decision_diagram.svg")
MODEL_OUTPUT = Path("results/meta_classifier_decision_diagram.joblib")

TRAIN_ONLY = False
COLLECT_ONLY = False

RANDOM_SEED = 0

DD_SKELETON = [1, 2, 4, 4, 4]
DD_ALPHA = 0.1
DD_RANDOM_SEED = 42
DD_TRAIN_VALIDATION_TEST_SPLIT = [0.5, 0.25, 0.25]
DD_UNIVARIATE_SPLIT = True

NUMERIC_FEATURES = [
    "n_features",
    "n_features_F",
    "n_features_B",
    "n_features_E",
    "n_features_D",
    # "seed",
    # "query_index",
    "n_estimators",
    "max_depth",
    "time_limit",
    "norm",
    "isolation",
    "hard_voting",
    "total_tree_nodes",
    "mean_tree_nodes",
    "max_tree_nodes",
    "total_split_levels",
    "mean_split_levels_per_feature",
    "max_split_levels_per_feature",
]
CATEGORICAL_FEATURES = ["model_type", "voting_type", "preferred_metric"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DD_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column != "voting_type"]
CSV_COLUMNS = ["dataset", "best_explainer", "overall_best_explainer"] + FEATURE_COLUMNS


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def set_reproducibility_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def finite_nonnegative_float(value: Any) -> float | None:
    result = finite_float(value)
    if result is None or result < 0:
        return None
    return result


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.lower() == "none":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def bool_flag(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return int(value != 0)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "with", "iso", "isolation"}:
            return 1
        if normalized in {"false", "no", "n", "0", "without", "plain", "none", "noif"}:
            return 0
    return None


def infer_norm(entry: dict[str, Any]) -> float:
    for key in ("norm", "reference_norm"):
        value = finite_nonnegative_float(entry.get(key))
        if value is not None:
            return value
    return 1.0


def infer_isolation(path: Path, entry: dict[str, Any]) -> int:
    for key in ("isolation", "use_isolation", "with_isolation", "isolation_enabled"):
        value = bool_flag(entry.get(key))
        if value is not None:
            return value
    if any(key in entry for key in ("isolation_build_time", "target_isolators")):
        return 1
    if "isolation" in path.parts:
        return 1
    return 0


def numeric_summary(values: Iterable[Any]) -> dict[str, float | None]:
    finite_values = [
        value
        for raw_value in values
        if (value := finite_nonnegative_float(raw_value)) is not None
    ]
    if not finite_values:
        return {
            "total": None,
            "mean": None,
            "max": None,
        }
    return {
        "total": float(np.sum(finite_values)),
        "mean": float(np.mean(finite_values)),
        "max": float(np.max(finite_values)),
    }


def model_complexity_features(entry: dict[str, Any]) -> dict[str, float | None]:
    node_stats = numeric_summary(entry.get("nodes") or [])
    split_levels = entry.get("split_levels") or {}
    split_values = split_levels.values() if isinstance(split_levels, dict) else []
    split_stats = numeric_summary(split_values)
    return {
        "total_tree_nodes": node_stats["total"],
        "mean_tree_nodes": node_stats["mean"],
        "max_tree_nodes": node_stats["max"],
        "total_split_levels": split_stats["total"],
        "mean_split_levels_per_feature": split_stats["mean"],
        "max_split_levels_per_feature": split_stats["max"],
    }


def is_optimal_status(status: Any) -> bool:
    return status == "OPTIMAL"


def normalize_method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.upper())


def infer_model_type(path: Path, entry: dict[str, Any]) -> str:
    model_type = entry.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type
    parts = set(path.parts)
    if "xgb" in parts:
        return "xgb"
    if "rf" in parts:
        return "rf"
    return "unknown"


def infer_voting_type(path: Path, entry: dict[str, Any], model_type: str) -> str:
    voting = entry.get("voting")
    if isinstance(voting, str) and voting:
        return voting.upper()
    if model_type == "rf" and path.stem.endswith("_HARD"):
        return "HARD"
    return "SOFT"


def infer_hard_voting(voting_type: Any) -> int:
    if not isinstance(voting_type, str):
        return 0
    return int(voting_type.strip().upper() == "HARD")


def is_supported_result_file(path: Path) -> bool:
    return path.name.startswith(("exp_", "multinorms_", "isolation_"))


def iter_result_entries(results_dir: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(results_dir.rglob("*.json")):
        if not is_supported_result_file(path):
            continue
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

        for entry in payload:
            if isinstance(entry, dict):
                yield path, entry


def dataset_feature_type_counts(
    dataset: str,
    *,
    datasets_dir: Path,
) -> dict[str, float | None]:
    path = datasets_dir / f"{dataset}.csv"
    if not path.exists():
        return {
            "n_features_F": None,
            "n_features_B": None,
            "n_features_E": None,
            "n_features_D": None,
        }

    columns = pd.read_csv(path, nrows=0, header=[0, 1]).columns
    types = columns.get_level_values(1)
    return {
        "n_features_F": int(np.sum(types == "F")),
        "n_features_B": int(np.sum(types == "B")),
        "n_features_E": int(np.sum(types == "E")),
        "n_features_D": int(np.sum(types == "D")),
    }


def has_counterfactual(explanation: dict[str, Any], metric_prefix: str) -> bool:
    return explanation.get(f"{metric_prefix}_cf") is not None


def is_valid_counterfactual(explanation: dict[str, Any], metric_prefix: str) -> bool:
    return explanation.get(f"{metric_prefix}_valid") is True


def metric_suffix(value: Any) -> str:
    numeric_value = finite_float(value)
    if numeric_value is not None and numeric_value.is_integer():
        return str(int(numeric_value))
    return str(value)


def configured_methods(entry: dict[str, Any]) -> list[str]:
    raw_methods = entry.get("methods")
    if isinstance(raw_methods, list):
        return [method for method in METHODS if method in raw_methods]
    return [method for method in METHODS if f"{method}_build_time" in entry]


def has_metric_data(
    entry: dict[str, Any],
    *,
    metric_prefix: str,
    build_time_key: str,
    supported_key: str | None = None,
) -> bool:
    explanations = entry.get("explanations")
    build_time = finite_nonnegative_float(entry.get(build_time_key))
    if build_time is not None:
        return True

    if supported_key is not None and entry.get(supported_key) is False:
        return False

    if not isinstance(explanations, list):
        return False

    for explanation in explanations:
        if not isinstance(explanation, dict):
            continue
        if any(
            f"{metric_prefix}_{suffix}" in explanation
            for suffix in ("status", "time", "objective", "callback", "cf", "valid")
        ):
            return True
    return False


def best_anytime_result(
    explanation: dict[str, Any],
    metric_prefix: str,
    *,
    build_time: float,
) -> tuple[float, float] | None:
    candidates: list[tuple[float, float]] = []

    raw_callback = explanation.get(f"{metric_prefix}_callback") or []
    if isinstance(raw_callback, list):
        for item in raw_callback:
            if isinstance(item, dict):
                time_value = finite_nonnegative_float(item.get("time"))
                objective_value = finite_nonnegative_float(item.get("objective_value"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                time_value = finite_nonnegative_float(item[0])
                objective_value = finite_nonnegative_float(item[1])
            else:
                continue
            if time_value is None or objective_value is None:
                continue
            candidates.append((objective_value, build_time + time_value))

    final_time = finite_nonnegative_float(explanation.get(f"{metric_prefix}_time"))
    final_objective = finite_nonnegative_float(
        explanation.get(f"{metric_prefix}_objective")
    )
    if final_time is not None and final_objective is not None:
        candidates.append((final_objective, build_time + final_time))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item)


def median_time(values: list[float]) -> float:
    return float(statistics.median(values))


def build_time_scale(preferred_metric: str, query_count: int) -> float:
    if preferred_metric == "median+build":
        return 1.0
    if preferred_metric == "median+buildAmortized":
        return 1.0 / query_count
    if preferred_metric == "median":
        return 0.0
    raise ValueError(f"Unsupported preferred metric: {preferred_metric}")


def aggregate_method_times(
    entry: dict[str, Any],
    *,
    method_prefixes: dict[str, str],
    build_times: dict[str, float],
    preferred_metric: str,
) -> dict[str, list[float]]:
    explanations = entry.get("explanations")
    if not isinstance(explanations, list):
        return {}

    valid_explanations = [
        explanation for explanation in explanations if isinstance(explanation, dict)
    ]
    query_count = len(valid_explanations)
    if query_count == 0:
        return {}

    build_scale = build_time_scale(preferred_metric, query_count)
    aggregated_times = {method: [] for method in method_prefixes}
    for explanation in valid_explanations:
        for method, metric_prefix in method_prefixes.items():
            build_share = build_times.get(method, 0.0) * build_scale
            total_time = finite_nonnegative_float(
                explanation.get(f"{metric_prefix}_time")
            )
            if total_time is not None:
                aggregated_times[method].append(build_share + total_time)
                continue

            anytime_result = best_anytime_result(
                explanation,
                metric_prefix,
                build_time=build_share,
            )
            if anytime_result is not None:
                aggregated_times[method].append(anytime_result[1])
    return aggregated_times


def optimal_total_time(
    explanation: dict[str, Any],
    metric_prefix: str,
    *,
    build_time: float,
) -> float | None:
    total_time = finite_nonnegative_float(explanation.get(f"{metric_prefix}_time"))
    total_time_with_build = None if total_time is None else build_time + total_time
    if (
        total_time_with_build is None
        or not is_optimal_status(explanation.get(f"{metric_prefix}_status"))
        or not has_counterfactual(explanation, metric_prefix)
        or not is_valid_counterfactual(explanation, metric_prefix)
    ):
        return None
    return total_time_with_build


def choose_query_winner_details(
    explanation: dict[str, Any],
    *,
    method_prefixes: dict[str, str],
    build_times: dict[str, float],
) -> tuple[str, float] | None:
    optimal_candidates: list[tuple[float, int, str]] = []
    anytime_candidates: list[tuple[float, float, int, str]] = []

    for method, metric_prefix in method_prefixes.items():
        build_time = build_times.get(method, 0.0)
        total_time_with_build = optimal_total_time(
            explanation,
            metric_prefix,
            build_time=build_time,
        )
        if total_time_with_build is not None:
            optimal_candidates.append(
                (total_time_with_build, METHODS.index(method), method)
            )

        anytime_result = best_anytime_result(
            explanation,
            metric_prefix,
            build_time=build_time,
        )
        if anytime_result is not None:
            anytime_candidates.append(
                (
                    anytime_result[0],
                    anytime_result[1],
                    METHODS.index(method),
                    method,
                )
            )

    if optimal_candidates:
        winner = min(optimal_candidates)
        return winner[2], winner[0]
    if anytime_candidates:
        winner = min(anytime_candidates)
        return winner[3], winner[1]
    return None


def choose_query_winner(
    explanation: dict[str, Any],
    *,
    method_prefixes: dict[str, str],
    build_times: dict[str, float],
) -> str | None:
    winner_details = choose_query_winner_details(
        explanation,
        method_prefixes=method_prefixes,
        build_times=build_times,
    )
    if winner_details is None:
        return None
    return normalize_method_label(winner_details[0])


def choose_overall_winner(
    entry: dict[str, Any],
    *,
    method_prefixes: dict[str, str],
    build_times: dict[str, float],
    preferred_metric: str,
) -> str | None:
    method_times = aggregate_method_times(
        entry,
        method_prefixes=method_prefixes,
        build_times=build_times,
        preferred_metric=preferred_metric,
    )
    candidates = [method for method, times in method_times.items() if times]
    if not candidates:
        return None

    best_method = min(
        candidates,
        key=lambda method: (
            median_time(method_times[method]),
            METHODS.index(method),
        ),
    )
    return normalize_method_label(best_method)


def collect_query_rows(
    entry: dict[str, Any],
    *,
    base_row: dict[str, Any],
    method_prefixes: dict[str, str],
    build_times: dict[str, float],
    overall_best_explainer: str | None,
    expand_per_cf: bool,
) -> list[dict[str, Any]]:
    if not expand_per_cf:
        rows: list[dict[str, Any]] = []
        for preferred_metric in PREFERRED_METRICS:
            best_method = choose_overall_winner(
                entry,
                method_prefixes=method_prefixes,
                build_times=build_times,
                preferred_metric=preferred_metric,
            )
            if best_method is None:
                continue

            row = dict(base_row)
            row["preferred_metric"] = preferred_metric
            row["overall_best_explainer"] = best_method
            row["best_explainer"] = best_method
            row["query_index"] = -1
            rows.append(row)
        return rows

    explanations = entry.get("explanations")
    if not isinstance(explanations, list):
        return []

    rows: list[dict[str, Any]] = []
    for query_index, explanation in enumerate(explanations):
        if not isinstance(explanation, dict):
            continue
        query_winner = choose_query_winner(
            explanation,
            method_prefixes=method_prefixes,
            build_times=build_times,
        )
        if query_winner is None:
            continue

        row = dict(base_row)
        row["preferred_metric"] = PER_COUNTERFACTUAL_METRIC
        row["overall_best_explainer"] = overall_best_explainer
        row["best_explainer"] = query_winner
        row["query_index"] = query_index
        rows.append(row)
    return rows


def row_from_entry(
    *,
    path: Path,
    entry: dict[str, Any],
    dataset: str,
    type_counts: dict[str, float | None],
    norm: float,
    isolation: int,
) -> dict[str, Any] | None:
    model_type = infer_model_type(path, entry)
    voting_type = infer_voting_type(path, entry, model_type)
    hard_voting = infer_hard_voting(voting_type)

    time_limit = finite_nonnegative_float(entry.get("timeout"))
    if time_limit is None:
        time_limit = float(TIMEOUT)

    n_features = finite_nonnegative_float(entry.get("n_features"))
    if n_features is None:
        typed_counts = [value for value in type_counts.values() if value is not None]
        n_features = float(sum(typed_counts)) if typed_counts else None

    return {
        "dataset": dataset,
        "n_features": n_features,
        "model_type": model_type,
        "voting_type": voting_type,
        "seed": optional_int(entry.get("seed")),
        "n_estimators": optional_int(entry.get("n_estimators")),
        "max_depth": optional_int(entry.get("max_depth")),
        "time_limit": time_limit,
        "norm": norm,
        "isolation": isolation,
        "hard_voting": hard_voting,
        **type_counts,
        **model_complexity_features(entry),
    }


def collect_standard_entry_rows(
    *,
    path: Path,
    entry: dict[str, Any],
    dataset: str,
    type_counts: dict[str, float | None],
    expand_per_cf: bool,
) -> list[dict[str, Any]]:
    methods = configured_methods(entry)
    method_prefixes = {
        method: method
        for method in methods
        if has_metric_data(
            entry,
            metric_prefix=method,
            build_time_key=f"{method}_build_time",
            supported_key=f"{method}_supported",
        )
    }
    build_times = {
        method: finite_nonnegative_float(entry.get(f"{method}_build_time")) or 0.0
        for method in method_prefixes
    }
    base_row = row_from_entry(
        path=path,
        entry=entry,
        dataset=dataset,
        type_counts=type_counts,
        norm=infer_norm(entry),
        isolation=infer_isolation(path, entry),
    )
    if base_row is None:
        return []
    overall_best_explainer = choose_overall_winner(
        entry,
        method_prefixes=method_prefixes,
        build_times=build_times,
        preferred_metric="median+build",
    )
    return collect_query_rows(
        entry,
        base_row=base_row,
        method_prefixes=method_prefixes,
        build_times=build_times,
        overall_best_explainer=overall_best_explainer,
        expand_per_cf=expand_per_cf,
    )


def collect_multinorm_entry_rows(
    *,
    path: Path,
    entry: dict[str, Any],
    dataset: str,
    type_counts: dict[str, float | None],
    expand_per_cf: bool,
) -> list[dict[str, Any]]:
    raw_norms = entry.get("norms")
    if not isinstance(raw_norms, list):
        return []

    rows: list[dict[str, Any]] = []
    methods = configured_methods(entry)
    for raw_norm in raw_norms:
        norm = finite_nonnegative_float(raw_norm)
        if norm is None:
            continue

        method_prefixes = {
            method: f"{method}_{metric_suffix(raw_norm)}"
            for method in methods
            if has_metric_data(
                entry,
                metric_prefix=f"{method}_{metric_suffix(raw_norm)}",
                build_time_key=f"{method}_build_time",
                supported_key=f"{method}_supported",
            )
        }
        build_times = {
            method: finite_nonnegative_float(entry.get(f"{method}_build_time")) or 0.0
            for method in method_prefixes
        }
        base_row = row_from_entry(
            path=path,
            entry=entry,
            dataset=dataset,
            type_counts=type_counts,
            norm=norm,
            isolation=0,
        )
        if base_row is None:
            continue
        overall_best_explainer = choose_overall_winner(
            entry,
            method_prefixes=method_prefixes,
            build_times=build_times,
            preferred_metric="median+build",
        )
        rows.extend(
            collect_query_rows(
                entry,
                base_row=base_row,
                method_prefixes=method_prefixes,
                build_times=build_times,
                overall_best_explainer=overall_best_explainer,
                expand_per_cf=expand_per_cf,
            )
        )

    return rows


def parse_isolation_variant(variant: Any) -> tuple[str, int] | None:
    if not isinstance(variant, str) or "_" not in variant:
        return None
    method, suffix = variant.rsplit("_", 1)
    if method not in METHODS:
        return None
    if suffix == "plain":
        return method, 0
    if suffix in {"iso", "isolation"}:
        return method, 1
    return None


def collect_isolation_entry_rows(
    *,
    path: Path,
    entry: dict[str, Any],
    dataset: str,
    type_counts: dict[str, float | None],
    expand_per_cf: bool,
) -> list[dict[str, Any]]:
    raw_variants = entry.get("variants")
    if not isinstance(raw_variants, list):
        raw_variants = [
            key[: -len("_build_time")]
            for key in entry
            if key.endswith("_build_time") and "_" in key[: -len("_build_time")]
        ]

    grouped_prefixes: dict[int, dict[str, str]] = {}
    for variant in raw_variants:
        parsed = parse_isolation_variant(variant)
        if parsed is None:
            continue
        method, isolation = parsed
        if not has_metric_data(
            entry,
            metric_prefix=str(variant),
            build_time_key=f"{variant}_build_time",
        ):
            continue
        grouped_prefixes.setdefault(isolation, {})[method] = str(variant)

    rows: list[dict[str, Any]] = []
    norm = infer_norm(entry)
    for isolation, method_prefixes in sorted(grouped_prefixes.items()):
        build_times = {
            method: finite_nonnegative_float(entry.get(f"{method_prefix}_build_time"))
            or 0.0
            for method, method_prefix in method_prefixes.items()
        }
        base_row = row_from_entry(
            path=path,
            entry=entry,
            dataset=dataset,
            type_counts=type_counts,
            norm=norm,
            isolation=isolation,
        )
        if base_row is None:
            continue
        overall_best_explainer = choose_overall_winner(
            entry,
            method_prefixes=method_prefixes,
            build_times=build_times,
            preferred_metric="median+build",
        )
        rows.extend(
            collect_query_rows(
                entry,
                base_row=base_row,
                method_prefixes=method_prefixes,
                build_times=build_times,
                overall_best_explainer=overall_best_explainer,
                expand_per_cf=expand_per_cf,
            )
        )

    return rows


def collect_entry_rows(
    *,
    path: Path,
    entry: dict[str, Any],
    dataset: str,
    type_counts: dict[str, float | None],
    expand_per_cf: bool,
) -> list[dict[str, Any]]:
    if "variants" in entry or path.name.startswith("isolation_"):
        return collect_isolation_entry_rows(
            path=path,
            entry=entry,
            dataset=dataset,
            type_counts=type_counts,
            expand_per_cf=expand_per_cf,
        )
    if "norms" in entry or path.name.startswith("multinorms_"):
        return collect_multinorm_entry_rows(
            path=path,
            entry=entry,
            dataset=dataset,
            type_counts=type_counts,
            expand_per_cf=expand_per_cf,
        )
    return collect_standard_entry_rows(
        path=path,
        entry=entry,
        dataset=dataset,
        type_counts=type_counts,
        expand_per_cf=expand_per_cf,
    )


def collect_training_data(
    *,
    results_dir: Path,
    datasets_dir: Path,
    expand_per_cf: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_count_cache: dict[str, dict[str, float | None]] = {}

    for path, entry in iter_result_entries(results_dir):
        dataset = entry.get("dataset")
        if not isinstance(dataset, str):
            continue

        type_counts = feature_count_cache.setdefault(
            dataset,
            dataset_feature_type_counts(dataset, datasets_dir=datasets_dir),
        )

        rows.extend(
            collect_entry_rows(
                path=path,
                entry=entry,
                dataset=dataset,
                type_counts=type_counts,
                expand_per_cf=expand_per_cf,
            )
        )

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


@dataclass
class DecisionDiagramMetaClassifier:
    feature_names: list[str]
    categorical_encoders: dict[str, dict[str, int]]
    class_decoder: dict[int, str]
    solution: Any

    def _prepare_feature_frame(self, configurations: pd.DataFrame) -> pd.DataFrame:
        if configurations.empty:
            raise ValueError("configurations must contain at least one row.")

        missing = [
            column
            for column in self.feature_names
            if column not in configurations.columns
        ]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        data = configurations[self.feature_names].copy()
        for column, mapping in self.categorical_encoders.items():
            values = data[column].fillna("unknown").astype(str)
            unknown = sorted(
                {value for value in values.unique() if value not in mapping}
            )
            if unknown:
                raise ValueError(f"Unknown categorical values for {column}: {unknown}")
            data[column] = values.map(mapping)

        for column in self.feature_names:
            if column in self.categorical_encoders:
                continue
            data[column] = pd.to_numeric(data[column], errors="coerce")

        if data.isna().any().any():
            missing_columns = sorted(data.columns[data.isna().any()].tolist())
            raise ValueError(
                f"Decision-diagram inference received missing values in {missing_columns}"
            )
        return data

    def _normalize_feature_frame(self, data: pd.DataFrame) -> np.ndarray:
        values = data.to_numpy(dtype=float)
        if not getattr(self.solution.data, "normalize_inputs", False):
            return values

        feature_mins = np.asarray(self.solution.data.feature_mins, dtype=float)
        feature_maxs = np.asarray(self.solution.data.feature_maxs, dtype=float)
        denominator = feature_maxs - feature_mins
        zero_range = np.isclose(denominator, 0.0)
        safe_denominator = denominator.copy()
        safe_denominator[zero_range] = 1.0
        normalized = (values - feature_mins) / safe_denominator
        if np.any(zero_range):
            normalized[:, zero_range] = 0.0
        return normalized

    def _predict_one(self, row: np.ndarray) -> str:
        node = self.solution.topology.root_node
        while node not in self.solution.node_class:
            lhs = float(np.dot(self.solution.node_hyperplane[node], row))
            if lhs >= self.solution.node_intercept[node]:
                node = self.solution.node_positive_arc[node]
            else:
                node = self.solution.node_negative_arc[node]
        return self.class_decoder[int(self.solution.node_class[node])]

    def predict(self, configurations: pd.DataFrame) -> np.ndarray:
        prepared = self._prepare_feature_frame(configurations)
        normalized = self._normalize_feature_frame(prepared)
        return np.asarray([self._predict_one(row) for row in normalized], dtype=object)


def load_decision_diagram_components() -> tuple[Any, Any, Any, Any, Any | None]:
    decision_diagram_src = ROOT / "Decision-Diagrams" / "src"
    if str(decision_diagram_src) not in sys.path:
        sys.path.append(str(decision_diagram_src))

    from dataset import Dataset as DDDataset
    from heuristic import Heuristic as DDHeuristic
    from optimizer import Optimizer as DDOptimizer
    from topology import Topology as DDTopology

    try:
        from visualizer import Visualizer as DDVisualizer
    except ModuleNotFoundError:
        DDVisualizer = None

    return DDDataset, DDTopology, DDHeuristic, DDOptimizer, DDVisualizer


def prepare_decision_diagram_training_data(
    data: pd.DataFrame,
    *,
    processed_csv: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, int]], dict[int, str]]:
    if "best_explainer" not in data.columns:
        raise ValueError("Missing target column: best_explainer")

    training_data = data.dropna(subset=["best_explainer"]).copy()
    if training_data.empty:
        raise ValueError("No training rows available after dropping missing targets.")

    missing = [
        column for column in DD_FEATURE_COLUMNS if column not in training_data.columns
    ]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    categorical_encoders: dict[str, dict[str, int]] = {}
    categorical_columns = [
        column for column in DD_FEATURE_COLUMNS if training_data[column].dtype == object
    ]
    for column in categorical_columns:
        values = training_data[column].fillna("unknown").astype(str)
        encoder = {value: index for index, value in enumerate(sorted(values.unique()))}
        training_data[column] = values.map(encoder)
        categorical_encoders[column] = encoder

    for column in DD_FEATURE_COLUMNS:
        if column in categorical_encoders:
            continue
        training_data[column] = pd.to_numeric(training_data[column], errors="coerce")

    target_values = training_data["best_explainer"].fillna("unknown").astype(str)
    target_encoder = {"CP": 0, "MIP": 1, "MAXSAT": 2, "MACE": 3}
    # Moving MACE to be the last class to avoid issues with the decision-diagram implementation's handling of the first class as a default prediction.
    class_decoder = {index: value for value, index in target_encoder.items()}
    training_data["best_explainer"] = target_values.map(target_encoder)

    ordered_columns = DD_FEATURE_COLUMNS + ["best_explainer"]
    processed = training_data[ordered_columns]
    if processed.isna().any().any():
        missing_columns = sorted(processed.columns[processed.isna().any()].tolist())
        raise ValueError(
            f"Decision-diagram training data has missing values in {missing_columns}"
        )

    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    processed = processed.drop(columns=["seed", "query_index"], errors="ignore")
    # remove the 2 row where the best explainer is MACE
    processed = processed[processed["best_explainer"] != target_encoder.get("MACE")]
    processed.to_csv(processed_csv, index=False)
    print(
        f"Saved processed training data with {len(processed)} rows to {processed_csv}",
        f"Encodings :features={categorical_encoders}",
        f"classes={class_decoder}",
        sep="\n",
    )
    return processed, DD_FEATURE_COLUMNS, categorical_encoders, class_decoder


def train_decision_diagram_meta_classifier(
    data: pd.DataFrame,
    *,
    processed_csv: Path,
    diagram_output: Path,
    model_output: Path | None,
    seed: int,
) -> tuple[DecisionDiagramMetaClassifier, bool]:
    (
        _processed_data,
        feature_names,
        categorical_encoders,
        class_decoder,
    ) = prepare_decision_diagram_training_data(data, processed_csv=processed_csv)

    DDDataset, DDTopology, DDHeuristic, DDOptimizer, DDVisualizer = (
        load_decision_diagram_components()
    )

    dataset = DDDataset(
        str(processed_csv),
        train_validation_test_split=DD_TRAIN_VALIDATION_TEST_SPLIT,
        seed=seed,
    )
    topology = DDTopology(DD_SKELETON, dataset)

    heuristic = DDHeuristic(dataset, topology, alpha=DD_ALPHA)
    print(
        "Decision-diagram heuristic accuracy:",
        f"train={heuristic.solution.training_accuracy():.4f}",
        f"test={heuristic.solution.test_accuracy():.4f}",
    )

    optimized = DDOptimizer(
        dataset,
        topology,
        alpha=DD_ALPHA,
        initial_solution=heuristic.solution,
        univariate_split=DD_UNIVARIATE_SPLIT,
    )
    print(
        "Decision-diagram optimized accuracy:",
        f"train={optimized.solution.training_accuracy():.4f}",
        f"test={optimized.solution.test_accuracy():.4f}",
    )

    diagram_saved = False
    if DDVisualizer is None:
        print(
            "graphviz is not available; skipping decision-diagram visualization export."
        )
    else:
        diagram_output.parent.mkdir(parents=True, exist_ok=True)
        visualizer = DDVisualizer(
            dataset,
            optimized.solution,
            feature_names=feature_names,
        )
        visualizer.save(str(diagram_output))
        diagram_saved = True

    model = DecisionDiagramMetaClassifier(
        feature_names=feature_names,
        categorical_encoders=categorical_encoders,
        class_decoder=class_decoder,
        solution=optimized.solution,
    )
    save_trained_model(model, model_output)
    return model, diagram_saved


def save_trained_model(model: Any, model_output: Path | None) -> None:
    if model_output is None:
        return
    if joblib is None:
        print("joblib is not available; skipping model export.")
        return
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)


def recommend_best_explainer(
    configurations: pd.DataFrame,
    model: Any,
) -> str | list[str]:
    """Return the predicted best explainer for one or more configurations.

    The input must contain one row per configuration and all columns in
    FEATURE_COLUMNS. This mirrors the CSV produced by collect_training_data.
    """

    if configurations.empty:
        raise ValueError("configurations must contain at least one row.")

    predictions = model.predict(configurations)
    labels = [str(label) for label in predictions]
    return labels[0] if len(labels) == 1 else labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expand-per-cf",
        action="store_true",
        help=(
            "Expand each experiment entry into one row per counterfactual/query. "
            "Without this flag, each experiment produces one row per preferred "
            f"metric ({len(PREFERRED_METRICS)} rows per experiment)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_reproducibility_seed(RANDOM_SEED)
    if COLLECT_ONLY and TRAIN_ONLY:
        raise ValueError("COLLECT_ONLY and TRAIN_ONLY cannot both be True.")

    if TRAIN_ONLY:
        data = pd.read_csv(OUTPUT_CSV)
    else:
        data = collect_training_data(
            results_dir=RESULTS_DIR,
            datasets_dir=DATASETS_DIR,
            expand_per_cf=args.expand_per_cf,
        )
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(OUTPUT_CSV, index=False)
        row_kind = (
            "counterfactual" if args.expand_per_cf else "metric-conditioned experiment"
        )
        print(f"Saved {len(data)} {row_kind} rows to {OUTPUT_CSV}")

    if COLLECT_ONLY:
        return 0

    if data.empty:
        raise ValueError("No rows available for decision-diagram training.")

    _, diagram_saved = train_decision_diagram_meta_classifier(
        data,
        processed_csv=PROCESSED_CSV,
        diagram_output=DIAGRAM_OUTPUT,
        model_output=MODEL_OUTPUT,
        seed=DD_RANDOM_SEED,
    )
    print(f"Saved decision-diagram processed CSV to {PROCESSED_CSV}")
    if diagram_saved:
        print(f"Saved decision-diagram visualization to {DIAGRAM_OUTPUT}")
    else:
        print("Skipped decision-diagram visualization export.")
    if MODEL_OUTPUT is not None and joblib is not None:
        print(f"Saved trained decision-diagram model to {MODEL_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
