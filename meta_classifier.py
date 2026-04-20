#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree

try:
    import joblib
except Exception:  # pragma: no cover - model export is optional.
    joblib = None

try:
    from parameters import TIMEOUT
except Exception:  # pragma: no cover - keeps the script usable outside the repo root.
    TIMEOUT = 900


_CACHE_DIR = Path(gettempdir()) / "oceanpy-meta-classifier-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("cp", "mip", "mace", "maxsat")
METHOD_LABELS = {
    "cp": "CP",
    "mip": "MIP",
    "mace": "MACE",
    "maxsat": "MAXSAT",
}
DATASET_ALIASES = {
    "Breast-Cancer-Wisconsin": "BreastCancerWisconsin",
    "Credit": "GermanCredit",
}
BASE_NUMERIC_FEATURES = [
    "n_features",
    "n_features_F",
    "n_features_B",
    "n_features_E",
    "n_features_D",
    "n_estimators",
    "max_depth",
    "time_limit",
    "norm",
    "isolation",
    "n_samples",
    "total_tree_nodes",
    "mean_tree_nodes",
    "max_tree_nodes",
    "total_split_levels",
    "mean_split_levels_per_feature",
    "max_split_levels_per_feature",
]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES
CATEGORICAL_FEATURES = ["model_type", "voting_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CSV_COLUMNS = ["dataset", "best_explainer"] + FEATURE_COLUMNS


@dataclass(frozen=True)
class MethodSummary:
    method: str
    build_time: float
    average_time_per_sample: float
    n_samples: int
    success_count: int
    attempted_count: int

    @property
    def success_rate(self) -> float:
        if self.attempted_count <= 0:
            return 0.0
        return self.success_count / self.attempted_count

    def amortized_total_time(self, n_samples: int) -> float:
        return self.build_time + n_samples * self.average_time_per_sample


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


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


def iter_result_entries(results_dir: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(results_dir.rglob("exp_*.json")):
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


def canonical_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset, dataset)


def dataset_feature_type_counts(
    dataset: str,
    *,
    datasets_dir: Path,
) -> dict[str, float | None]:
    canonical = canonical_dataset_name(dataset)
    path = datasets_dir / f"{canonical}.csv"
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


def has_counterfactual(explanation: dict[str, Any], method: str) -> bool:
    return explanation.get(f"{method}_cf") is not None


def is_valid_counterfactual(explanation: dict[str, Any], method: str) -> bool:
    return explanation.get(f"{method}_valid") is True


def summarize_method(
    entry: dict[str, Any],
    method: str,
    *,
    require_optimal: bool,
) -> MethodSummary | None:
    explanations = entry.get("explanations")
    if not isinstance(explanations, list) or not explanations:
        return None

    build_time = finite_nonnegative_float(entry.get(f"{method}_build_time"))
    supported = entry.get(f"{method}_supported", True)
    if supported is False and build_time is None:
        return None

    times: list[float] = []
    success_count = 0
    attempted_count = 0

    for explanation in explanations:
        if not isinstance(explanation, dict):
            continue
        time_value = finite_nonnegative_float(explanation.get(f"{method}_time"))
        method_was_attempted = (
            time_value is not None or f"{method}_status" in explanation
        )
        if method_was_attempted:
            attempted_count += 1
        if time_value is not None:
            times.append(time_value)

        status = explanation.get(f"{method}_status")
        optimal_enough = (not require_optimal) or is_optimal_status(status)
        if (
            has_counterfactual(explanation, method)
            and is_valid_counterfactual(explanation, method)
            and optimal_enough
        ):
            success_count += 1

    if not times:
        return None

    if attempted_count == 0:
        attempted_count = len(times)

    return MethodSummary(
        method=method,
        build_time=0.0 if build_time is None else build_time,
        average_time_per_sample=float(np.mean(times)),
        n_samples=len(explanations),
        success_count=success_count,
        attempted_count=attempted_count,
    )


def choose_best_method(
    summaries: dict[str, MethodSummary],
    *,
    n_samples: int,
    ranking_policy: str,
) -> str | None:
    if not summaries:
        return None

    successful = {
        method: summary
        for method, summary in summaries.items()
        if summary.success_count > 0
    }
    candidates = successful or summaries

    if ranking_policy == "success_then_time":
        best_success_rate = max(summary.success_rate for summary in candidates.values())
        candidates = {
            method: summary
            for method, summary in candidates.items()
            if math.isclose(summary.success_rate, best_success_rate)
        }

    return min(
        candidates,
        key=lambda method: (
            summaries[method].amortized_total_time(n_samples),
            METHODS.index(method),
        ),
    )


def collect_training_data(
    *,
    results_dir: Path,
    datasets_dir: Path,
    require_optimal: bool,
    ranking_policy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_count_cache: dict[str, dict[str, float | None]] = {}

    for path, entry in iter_result_entries(results_dir):
        dataset = entry.get("dataset")
        if not isinstance(dataset, str):
            continue

        model_type = infer_model_type(path, entry)
        voting_type = infer_voting_type(path, entry, model_type)
        type_counts = feature_count_cache.setdefault(
            dataset,
            dataset_feature_type_counts(dataset, datasets_dir=datasets_dir),
        )

        summaries = {
            method: summary
            for method in METHODS
            if (
                summary := summarize_method(
                    entry,
                    method,
                    require_optimal=require_optimal,
                )
            )
            is not None
        }
        if not summaries:
            continue

        n_samples = max(summary.n_samples for summary in summaries.values())
        best_method = choose_best_method(
            summaries,
            n_samples=n_samples,
            ranking_policy=ranking_policy,
        )
        if best_method is None:
            continue

        time_limit = finite_nonnegative_float(entry.get("timeout"))
        if time_limit is None:
            time_limit = float(TIMEOUT)

        n_features = finite_nonnegative_float(entry.get("n_features"))
        if n_features is None:
            typed_counts = [
                value for value in type_counts.values() if value is not None
            ]
            n_features = float(sum(typed_counts)) if typed_counts else None

        max_depth = optional_int(entry.get("max_depth"))
        n_estimators = optional_int(entry.get("n_estimators"))

        row: dict[str, Any] = {
            "dataset": dataset,
            "best_explainer": normalize_method_label(best_method),
            "n_features": n_features,
            "model_type": model_type,
            "voting_type": voting_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "time_limit": time_limit,
            "norm": infer_norm(entry),
            "isolation": infer_isolation(path, entry),
            "n_samples": n_samples,
            **type_counts,
            **model_complexity_features(entry),
        }

        rows.append(row)

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def clean_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def save_trained_model(model: Pipeline, model_output: Path | None) -> None:
    if model_output is None:
        return
    if joblib is None:
        print("joblib is not available; skipping model export.")
        return
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)


def parse_optional_int_grid(spec: str) -> list[int | None]:
    values: list[int | None] = []
    for raw_value in spec.split(","):
        value = raw_value.strip()
        if not value:
            continue
        if value.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(int(value))
    if not values:
        raise ValueError(f"Empty integer grid: {spec!r}")
    return values


def parse_int_grid(spec: str) -> list[int]:
    values = [value for value in parse_optional_int_grid(spec) if value is not None]
    if not values:
        raise ValueError(f"Empty integer grid: {spec!r}")
    return values


def parse_float_grid(spec: str) -> list[float]:
    values: list[float] = []
    for raw_value in spec.split(","):
        value = raw_value.strip()
        if value:
            values.append(float(value))
    if not values:
        raise ValueError(f"Empty float grid: {spec!r}")
    return values


def parse_class_weight_grid(spec: str) -> list[str | None]:
    values: list[str | None] = []
    for raw_value in spec.split(","):
        value = raw_value.strip().lower()
        if not value:
            continue
        if value in {"none", "null"}:
            values.append(None)
        elif value == "balanced":
            values.append("balanced")
        else:
            raise ValueError(f"Unsupported class_weight value: {raw_value!r}")
    if not values:
        raise ValueError(f"Empty class weight grid: {spec!r}")
    return values


def make_greedy_tree(
    *,
    params: dict[str, Any],
    random_state: int,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=params["min_samples_split"],
        ccp_alpha=params["ccp_alpha"],
        class_weight=params["class_weight"],
        random_state=random_state,
    )


def iter_grid_params(
    *,
    max_depths: list[int | None],
    min_samples_leaf_values: list[int],
    min_samples_split_values: list[int],
    ccp_alphas: list[float],
    class_weights: list[str | None],
) -> Iterable[dict[str, Any]]:
    for max_depth in max_depths:
        for min_samples_leaf in min_samples_leaf_values:
            for min_samples_split in min_samples_split_values:
                if min_samples_split < 2 * min_samples_leaf:
                    continue
                for ccp_alpha in ccp_alphas:
                    for class_weight in class_weights:
                        yield {
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "min_samples_split": min_samples_split,
                            "ccp_alpha": ccp_alpha,
                            "class_weight": class_weight,
                        }


def prune_redundant_same_label_splits(estimator: DecisionTreeClassifier) -> int:
    tree = estimator.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    values = tree.value
    collapsed = 0

    def node_label(node: int) -> int:
        return int(np.argmax(values[node][0]))

    def recurse(node: int) -> int | None:
        nonlocal collapsed
        left = int(children_left[node])
        right = int(children_right[node])
        if left == _tree.TREE_LEAF and right == _tree.TREE_LEAF:
            return node_label(node)

        left_label = recurse(left)
        right_label = recurse(right)
        if left_label is not None and left_label == right_label:
            children_left[node] = _tree.TREE_LEAF
            children_right[node] = _tree.TREE_LEAF
            feature[node] = _tree.TREE_UNDEFINED
            threshold[node] = _tree.TREE_UNDEFINED
            collapsed += 1
            return left_label
        return None

    if tree.node_count:
        recurse(0)
    return collapsed


def count_reachable_splits(estimator: DecisionTreeClassifier) -> int:
    tree = estimator.tree_

    def recurse(node: int) -> int:
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == _tree.TREE_LEAF and right == _tree.TREE_LEAF:
            return 0
        return 1 + recurse(left) + recurse(right)

    return recurse(0) if tree.node_count else 0


def count_redundant_same_label_splits(estimator: DecisionTreeClassifier) -> int:
    tree = estimator.tree_
    values = tree.value
    redundant = 0

    def node_label(node: int) -> int:
        return int(np.argmax(values[node][0]))

    def recurse(node: int) -> int | None:
        nonlocal redundant
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == _tree.TREE_LEAF and right == _tree.TREE_LEAF:
            return node_label(node)
        left_label = recurse(left)
        right_label = recurse(right)
        if left_label is not None and left_label == right_label:
            redundant += 1
            return left_label
        return None

    if tree.node_count:
        recurse(0)
    return redundant


def train_decision_tree(
    data: pd.DataFrame,
    *,
    tree_pdf: Path,
    model_output: Path | None,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
    trainer: str,
    test_size: float,
    grid_max_depths: list[int | None],
    grid_min_samples_leaf: list[int],
    grid_min_samples_split: list[int],
    grid_ccp_alphas: list[float],
    grid_class_weights: list[str | None],
) -> Pipeline:
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if "best_explainer" not in data.columns:
        raise ValueError("Missing target column: best_explainer")

    training_data = data.dropna(subset=["best_explainer"]).copy()
    if training_data.empty:
        raise ValueError("No training rows available after dropping missing targets.")

    for column in NUMERIC_FEATURES:
        training_data[column] = pd.to_numeric(training_data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        training_data[column] = training_data[column].fillna("unknown").astype(str)

    x = training_data[FEATURE_COLUMNS]
    y = training_data["best_explainer"].astype(str)

    class_counts = y.value_counts()
    can_split = (
        len(training_data) >= 10 and len(class_counts) >= 2 and class_counts.min() >= 2
    )

    if trainer == "grid":
        if can_split:
            x_train, x_valid, y_train, y_valid = train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            x_train = x_valid = x
            y_train = y_valid = y
            print(
                "Not enough class diversity for a stratified validation split; "
                "grid scores are training scores."
            )

        best_score = -1.0
        best_split_count: int | None = None
        best_params: dict[str, Any] | None = None
        best_validation_report = ""
        tried = 0

        for params in iter_grid_params(
            max_depths=grid_max_depths,
            min_samples_leaf_values=grid_min_samples_leaf,
            min_samples_split_values=grid_min_samples_split,
            ccp_alphas=grid_ccp_alphas,
            class_weights=grid_class_weights,
        ):
            tried += 1
            candidate = Pipeline(
                steps=[
                    ("preprocess", make_preprocessor()),
                    (
                        "tree",
                        make_greedy_tree(params=params, random_state=random_state),
                    ),
                ]
            )
            candidate.fit(x_train, y_train)
            predictions = candidate.predict(x_valid)
            score = accuracy_score(y_valid, predictions)
            split_count = count_reachable_splits(candidate.named_steps["tree"])
            if (
                score > best_score
                or (
                    np.isclose(score, best_score)
                    and (
                        best_split_count is None
                        or split_count < best_split_count
                    )
                )
            ):
                best_score = float(score)
                best_split_count = split_count
                best_params = params
                best_validation_report = classification_report(
                    y_valid,
                    predictions,
                    labels=sorted(y.unique()),
                    zero_division=0,
                )

        if best_params is None:
            raise ValueError("No valid decision-tree hyperparameter combinations were produced.")

        print(f"Grid search tried {tried} decision trees.")
        print(f"Best validation accuracy: {best_score:.4f}")
        print(f"Best hyperparameters: {best_params}")
        print("Best validation report:")
        print(best_validation_report)

        pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "tree",
                    make_greedy_tree(params=best_params, random_state=random_state),
                ),
            ]
        )
        pipeline.fit(x, y)
        tree = pipeline.named_steps["tree"]
        split_count_before = count_reachable_splits(tree)
        collapsed = prune_redundant_same_label_splits(tree)
        split_count_after = count_reachable_splits(tree)
        redundant_after = count_redundant_same_label_splits(tree)
        if redundant_after:
            raise RuntimeError(
                "Redundant same-label splits remain after pruning: "
                f"{redundant_after}"
            )
        train_predictions = pipeline.predict(x)
        print("Full-data training report after pruning:")
        print(
            classification_report(
                y,
                train_predictions,
                labels=sorted(y.unique()),
                zero_division=0,
            )
        )
        print(
            "Pruned redundant same-label splits: "
            f"{collapsed}; reachable splits {split_count_before} -> {split_count_after}; "
            f"remaining redundant splits={redundant_after}"
        )

        tree_pdf.parent.mkdir(parents=True, exist_ok=True)
        transformed_feature_names = [
            clean_feature_name(name)
            for name in pipeline.named_steps["preprocess"].get_feature_names_out()
        ]
        figure_width = max(16.0, 3.5 * ((best_params["max_depth"] or max_depth) + 1))
        figure_height = max(8.0, 2.2 * ((best_params["max_depth"] or max_depth) + 1))
        figure, axis = plt.subplots(figsize=(figure_width, figure_height))
        plot_tree(
            tree,
            feature_names=transformed_feature_names,
            class_names=[str(label) for label in tree.classes_],
            filled=True,
            rounded=True,
            impurity=False,
            proportion=True,
            ax=axis,
        )
        figure.tight_layout()
        figure.savefig(tree_pdf)
        plt.close(figure)

        save_trained_model(pipeline, model_output)
        return pipeline

    pipeline = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "tree",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    class_counts = y.value_counts()
    can_split = (
        len(training_data) >= 10 and len(class_counts) >= 2 and class_counts.min() >= 2
    )
    if can_split:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=random_state,
            stratify=y,
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        print("Validation report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=sorted(y.unique()),
                zero_division=0,
            )
        )
    else:
        pipeline.fit(x, y)
        print(
            "Not enough class diversity for a stratified validation split; fitted on all rows."
        )

    tree_pdf.parent.mkdir(parents=True, exist_ok=True)
    transformed_feature_names = [
        clean_feature_name(name)
        for name in pipeline.named_steps["preprocess"].get_feature_names_out()
    ]
    tree = pipeline.named_steps["tree"]
    figure_width = max(16.0, 3.5 * (max_depth + 1))
    figure_height = max(8.0, 2.2 * (max_depth + 1))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    plot_tree(
        tree,
        feature_names=transformed_feature_names,
        class_names=[str(label) for label in tree.classes_],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        ax=axis,
    )
    figure.tight_layout()
    figure.savefig(tree_pdf)
    plt.close(figure)

    save_trained_model(pipeline, model_output)
    return pipeline


def recommend_best_explainer(
    configurations: pd.DataFrame,
    model: Pipeline,
) -> str | list[str]:
    """Return the predicted best explainer for one or more configurations.

    The input must contain one row per configuration and all columns in
    FEATURE_COLUMNS. This mirrors the CSV produced by collect_training_data.
    """

    if configurations.empty:
        raise ValueError("configurations must contain at least one row.")
    missing = [column for column in FEATURE_COLUMNS if column not in configurations]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    predictions = model.predict(configurations[FEATURE_COLUMNS])
    labels = [str(label) for label in predictions]
    return labels[0] if len(labels) == 1 else labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect experiment-result metadata and train a decision-tree "
            "meta-classifier for the best counterfactual explainer."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing experiment result JSON files.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory containing dataset CSV files with two-line headers.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/meta_classifier_training_data.csv"),
        help="Path where the collected meta-classifier CSV is written.",
    )
    parser.add_argument(
        "--tree-pdf",
        type=Path,
        default=Path("plots/meta_classifier_decision_tree.pdf"),
        help="Path where the decision-tree PDF is written.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("results/meta_classifier_decision_tree.joblib"),
        help="Optional path for the trained sklearn pipeline.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum depth of the decision tree.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=5,
        help="Minimum samples per decision-tree leaf.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for train/test splitting and tree training.",
    )
    parser.add_argument(
        "--trainer",
        choices=("grid", "greedy"),
        default="grid",
        help=(
            "Tree trainer. grid searches sklearn trees and retrains the best "
            "on all data; greedy uses one sklearn tree."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Validation split fraction used by grid and greedy trainers.",
    )
    parser.add_argument(
        "--grid-max-depths",
        type=str,
        default="1,2,3,4",
        help="Comma-separated max_depth values for grid search. Use None for unlimited.",
    )
    parser.add_argument(
        "--grid-min-samples-leaf",
        type=str,
        default="1,2,5,10",
        help="Comma-separated min_samples_leaf values for grid search.",
    )
    parser.add_argument(
        "--grid-min-samples-split",
        type=str,
        default="2,5,10,20",
        help="Comma-separated min_samples_split values for grid search.",
    )
    parser.add_argument(
        "--grid-ccp-alphas",
        type=str,
        default="0,0.0001,0.001,0.005,0.01",
        help="Comma-separated ccp_alpha values for grid search.",
    )
    parser.add_argument(
        "--grid-class-weights",
        type=str,
        default="none,balanced",
        help="Comma-separated class_weight values for grid search: none,balanced.",
    )
    parser.add_argument(
        "--ranking-policy",
        choices=("success_then_time", "time_then_success"),
        default="success_then_time",
        help=(
            "How to label the best explainer. success_then_time first keeps "
            "only the best success-rate methods, then picks the lowest "
            "amortized time."
        ),
    )
    parser.add_argument(
        "--allow-non-optimal",
        action="store_true",
        help="Treat valid non-OPTIMAL counterfactuals as successful.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect and save the CSV; do not train or plot the tree.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train from an existing CSV without recollecting result JSON files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.collect_only and args.train_only:
        raise ValueError("--collect-only and --train-only cannot be used together.")

    if args.train_only:
        data = pd.read_csv(args.output_csv)
    else:
        data = collect_training_data(
            results_dir=args.results_dir,
            datasets_dir=args.datasets_dir,
            require_optimal=not args.allow_non_optimal,
            ranking_policy=args.ranking_policy,
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output_csv, index=False)
        print(f"Saved {len(data)} configuration rows to {args.output_csv}")

    if args.collect_only:
        return 0

    if data.empty:
        raise ValueError("No rows available for decision-tree training.")

    train_decision_tree(
        data,
        tree_pdf=args.tree_pdf,
        model_output=args.model_output,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        trainer=args.trainer,
        test_size=args.test_size,
        grid_max_depths=parse_optional_int_grid(args.grid_max_depths),
        grid_min_samples_leaf=parse_int_grid(args.grid_min_samples_leaf),
        grid_min_samples_split=parse_int_grid(args.grid_min_samples_split),
        grid_ccp_alphas=parse_float_grid(args.grid_ccp_alphas),
        grid_class_weights=parse_class_weight_grid(args.grid_class_weights),
    )
    print(f"Saved decision-tree PDF to {args.tree_pdf}")
    if args.model_output is not None and joblib is not None:
        print(f"Saved trained sklearn pipeline to {args.model_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
