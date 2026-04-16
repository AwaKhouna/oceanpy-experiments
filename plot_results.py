#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from tempfile import gettempdir
from typing import Any

import numpy as np

from parameters import DATASETS, MAX_DEPTHS, MODELS, N_ESTIMATORS, TIMEOUT, VOTING


_CACHE_DIR = Path(gettempdir()) / "oceanpy-plot-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg-cache"))


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPSILON = 1e-12
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
UNSET = object()

METHOD_COLORS = {
    "cp": "#1f77b4",
    "mip": "#ff7f0e",
    "maxsat": "#2ca02c",
    "mace": "#d62728",
}
METHOD_LINESTYLES = {
    "cp": "-",
    "mip": "--",
    "maxsat": "-.",
    "mace": ":",
}
METHOD_LABELS = {
    "cp": "CP",
    "mip": "MIP",
    "maxsat": "MaxSAT",
    "mace": "MACE",
}
PARAMETER_LABELS = {
    "n_estimators": "Number of trees",
    "max_depth": "Max depth",
}


@dataclass(frozen=True)
class ResultBundle:
    times: dict[str, np.ndarray]
    statuses: dict[str, np.ndarray]
    has_cfs: dict[str, np.ndarray]
    callbacks: dict[str, list[list[tuple[float, float]]]]
    total_instances: int

    @property
    def active_time_methods(self) -> list[str]:
        return [method for method in MODELS if has_valid_times(self, method)]

    @property
    def active_callback_methods(self) -> list[str]:
        return [method for method in MODELS if any(self.callbacks.get(method, []))]


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def format_parameter_value(value: Any) -> str:
    return "None" if value is None else str(value)


def time_mode_label(include_build_time: bool) -> str:
    return "with build time" if include_build_time else "solver time only"


def has_finite_values(values: np.ndarray | None) -> bool:
    return values is not None and values.size > 0 and np.isfinite(values).any()


def is_optimal_status(status: Any) -> bool:
    return status == "OPTIMAL"


def has_valid_times(bundle: ResultBundle, method: str) -> bool:
    values = bundle.times.get(method)
    has_cfs = bundle.has_cfs.get(method)
    if values is None or has_cfs is None or values.size == 0:
        return False
    return np.isfinite(values[has_cfs]).any()


def normalize_voting(model_type: str, voting: str | None = None) -> str:
    if model_type == "xgb":
        return "SOFT"
    return (voting or "SOFT").upper()


def result_identity(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("dataset"),
        item.get("model_type"),
        item.get("n_estimators"),
        item.get("max_depth"),
        item.get("seed"),
        item.get("voting", "SOFT"),
    )


def result_methods(item: dict[str, Any]) -> list[str]:
    return [method for method in MODELS if f"{method}_build_time" in item]


def merge_result_entry(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    methods_to_merge: list[str],
) -> dict[str, Any]:
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
    if not isinstance(existing_explanations, list) or not isinstance(
        incoming_explanations, list
    ):
        return merged

    if len(existing_explanations) != len(incoming_explanations):
        return incoming

    for existing_explanation, incoming_explanation in zip(
        existing_explanations, incoming_explanations
    ):
        if existing_explanation.get("query") != incoming_explanation.get("query"):
            return incoming
        if existing_explanation.get("target") != incoming_explanation.get("target"):
            return incoming
        for method in methods_to_merge:
            for key in ("objective", "cf", "status", "time", "valid", "callback"):
                metric_key = f"{method}_{key}"
                existing_explanation[metric_key] = incoming_explanation.get(metric_key)

    for method in methods_to_merge:
        for suffix in ("build_time", "supported", "reason"):
            field = f"{method}_{suffix}"
            if field in incoming:
                merged[field] = incoming[field]
            elif suffix != "build_time":
                merged.pop(field, None)
    return merged


def coalesce_results(items: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for item in items:
        key = result_identity(item)
        if key not in merged:
            merged[key] = item
            order.append(key)
            continue
        merged[key] = merge_result_entry(merged[key], item, result_methods(item))
    return tuple(merged[key] for key in order)


def ensure_output_dir(dataset: str, model_type: str, voting: str | None = None) -> Path:
    output_dir = PLOTS_DIR / model_type
    if model_type == "rf":
        output_dir = output_dir / normalize_voting(model_type, voting).lower()
    output_dir = output_dir / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def to_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def build_time_or_zero(value: Any) -> float:
    build_time = to_finite_float(value)
    if build_time is None or build_time < 0:
        return 0.0
    return build_time


def cap_runtime_for_plot(value: float) -> float:
    return min(value, float(TIMEOUT))


def normalize_elapsed_time(
    value: Any,
    build_time: float,
    *,
    include_build_time: bool,
) -> float:
    runtime = to_finite_float(value)
    if runtime is None or runtime < 0:
        return np.nan
    total = runtime + (build_time if include_build_time else 0.0)
    return max(cap_runtime_for_plot(total), EPSILON)


def filter_items(
    items: tuple[dict[str, Any], ...],
    *,
    n_estimators: Any = UNSET,
    max_depth: Any = UNSET,
    seed: Any = UNSET,
    voting: Any = UNSET,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in items:
        if n_estimators is not UNSET and item.get("n_estimators") != n_estimators:
            continue
        if max_depth is not UNSET and item.get("max_depth") != max_depth:
            continue
        if seed is not UNSET and item.get("seed") != seed:
            continue
        if voting is not UNSET and item.get("voting", "SOFT") != voting:
            continue
        filtered.append(item)
    return filtered


def select_reference_value(
    dataset: str,
    model_type: str,
    *,
    fixed_parameter_name: str,
    preferred_value: Any,
    varying_parameter_name: str,
    varying_values: list[Any],
    seed: Any = UNSET,
    voting: Any = UNSET,
) -> Any:
    candidates = N_ESTIMATORS if fixed_parameter_name == "n_estimators" else MAX_DEPTHS
    items = load_dataset_results(dataset, model_type)
    coverage: dict[Any, int] = {}

    for candidate in candidates:
        filters = {
            "n_estimators": UNSET,
            "max_depth": UNSET,
            "seed": seed,
            "voting": voting,
        }
        filters[fixed_parameter_name] = candidate
        filtered_items = filter_items(items, **filters)
        distinct_values = {
            item.get(varying_parameter_name)
            for item in filtered_items
            if item.get(varying_parameter_name) in varying_values
        }
        coverage[candidate] = len(distinct_values)

    if coverage.get(preferred_value, 0) > 0:
        return preferred_value

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            coverage.get(candidate, 0),
            -candidates.index(candidate),
        ),
        reverse=True,
    )
    best_candidate = ranked_candidates[0]
    return best_candidate if coverage.get(best_candidate, 0) > 0 else preferred_value


@lru_cache(maxsize=None)
def load_dataset_results(dataset: str, model_type: str) -> tuple[dict[str, Any], ...]:
    model_dir = RESULTS_DIR / model_type
    if not model_dir.exists():
        return tuple()

    items: list[dict[str, Any]] = []
    for path in sorted(model_dir.glob(f"exp_{dataset}_*.json")):
        try:
            with path.open("r") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable results file {path}: {exc}")
            continue

        if not isinstance(payload, list):
            print(f"Skipping unexpected payload in {path}: expected a list.")
            continue

        for item in payload:
            if isinstance(item, dict):
                items.append(item)

    return coalesce_results(items)


def get_method_metrics(
    item: dict[str, Any], method: str
) -> tuple[list[dict[str, Any]], float]:
    explanations = item.get("explanations") or []

    if isinstance(explanations, dict):
        method_data = explanations.get(method) or {}
        metrics = method_data.get("metrics") or []
        return list(metrics), build_time_or_zero(method_data.get("build_time"))

    if not isinstance(explanations, list):
        return [], build_time_or_zero(item.get(f"{method}_build_time"))

    metrics: list[dict[str, Any]] = []
    for explanation in explanations:
        data = explanation if isinstance(explanation, dict) else {}
        raw_callback = data.get(f"{method}_callback") or []
        callback = raw_callback if isinstance(raw_callback, list) else []
        metrics.append(
            {
                "cf": data.get(f"{method}_cf"),
                "objective": data.get(f"{method}_objective"),
                "status": data.get(f"{method}_status"),
                "time": data.get(f"{method}_time"),
                "valid": data.get(f"{method}_valid"),
                "target": data.get("target"),
                "callback": [entry for entry in callback if isinstance(entry, dict)],
            }
        )

    return metrics, build_time_or_zero(item.get(f"{method}_build_time"))


def normalize_callback_series(
    metric: dict[str, Any],
    build_time: float,
    *,
    include_build_time: bool,
) -> list[tuple[float, float]]:
    offset = build_time if include_build_time else 0.0
    points: list[tuple[float, float]] = []

    for entry in metric.get("callback") or []:
        time_value = to_finite_float(entry.get("time"))
        objective_value = to_finite_float(entry.get("objective_value"))
        if time_value is None or objective_value is None or time_value < 0:
            continue
        points.append(
            (max(cap_runtime_for_plot(time_value + offset), EPSILON), objective_value)
        )

    final_time = normalize_elapsed_time(
        metric.get("time"),
        build_time,
        include_build_time=include_build_time,
    )
    final_objective = to_finite_float(metric.get("objective"))
    if np.isfinite(final_time) and final_objective is not None:
        points.append((final_time, final_objective))

    if not points:
        return []

    points.sort(key=lambda pair: pair[0])
    deduped: dict[float, float] = {}
    for time_value, objective_value in points:
        previous = deduped.get(time_value)
        deduped[time_value] = (
            objective_value if previous is None else min(previous, objective_value)
        )

    xs = np.fromiter(deduped.keys(), dtype=float)
    ys = np.fromiter(deduped.values(), dtype=float)
    ys = np.minimum.accumulate(ys)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def collect_method_data(
    dataset: str,
    model_type: str,
    *,
    n_estimators: Any = UNSET,
    max_depth: Any = UNSET,
    seed: Any = UNSET,
    voting: Any = UNSET,
    include_build_time: bool = True,
) -> ResultBundle:
    items = filter_items(
        load_dataset_results(dataset, model_type),
        n_estimators=n_estimators,
        max_depth=max_depth,
        seed=seed,
        voting=voting,
    )

    times = {method: [] for method in MODELS}
    statuses = {method: [] for method in MODELS}
    has_cfs = {method: [] for method in MODELS}
    callbacks = {method: [] for method in MODELS}
    total_instances = 0

    for item in items:
        parsed = {method: get_method_metrics(item, method) for method in MODELS}
        lengths = {len(metrics) for metrics, _ in parsed.values()}
        if len(lengths) > 1:
            raise ValueError(
                "Explanation count mismatch in "
                f"{dataset}/{model_type}: {sorted(lengths)}"
            )

        n_instances = lengths.pop() if lengths else 0
        total_instances += n_instances
        for index in range(n_instances):
            for method, (metrics, build_time) in parsed.items():
                metric = metrics[index]
                has_cf = metric.get("cf") is not None
                times[method].append(
                    normalize_elapsed_time(
                        metric.get("time"),
                        build_time,
                        include_build_time=include_build_time,
                    )
                )
                statuses[method].append(metric.get("status"))
                has_cfs[method].append(has_cf)
                callbacks[method].append(
                    []
                    if not has_cf
                    else normalize_callback_series(
                        metric,
                        build_time,
                        include_build_time=include_build_time,
                    )
                )

    return ResultBundle(
        times={
            method: np.array(values, dtype=float) for method, values in times.items()
        },
        statuses={
            method: np.array(values, dtype=object)
            for method, values in statuses.items()
        },
        has_cfs={
            method: np.array(values, dtype=bool) for method, values in has_cfs.items()
        },
        callbacks=callbacks,
        total_instances=total_instances,
    )


def finite_plot_times(
    values: np.ndarray,
    has_cfs: np.ndarray | None = None,
    statuses: np.ndarray | None = None,
    *,
    require_optimal: bool = True,
) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    mask = np.isfinite(values) & (values > 0)
    if has_cfs is not None:
        if has_cfs.shape[0] != values.shape[0]:
            raise ValueError("CF mask must align with time array.")
        mask &= has_cfs
    if require_optimal:
        if statuses is None or statuses.shape[0] != values.shape[0]:
            raise ValueError("Status array must align with time array.")
        mask &= statuses == "OPTIMAL"
    return np.sort(values[mask])


def valid_time_values(
    values: np.ndarray,
    has_cfs: np.ndarray | None = None,
    statuses: np.ndarray | None = None,
    *,
    require_optimal: bool = False,
) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    mask = np.isfinite(values) & (values > 0)
    if has_cfs is not None:
        if has_cfs.shape[0] != values.shape[0]:
            raise ValueError("CF mask must align with time array.")
        mask &= has_cfs
    if require_optimal:
        if statuses is None or statuses.shape[0] != values.shape[0]:
            raise ValueError("Status array must align with time array.")
        mask &= statuses == "OPTIMAL"
    return values[mask]


def compute_performance_profile(
    times_dict: dict[str, np.ndarray],
    has_cfs_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    methods = [
        method
        for method in MODELS
        if has_finite_values(
            valid_time_values(
                times_dict.get(method, np.array([], dtype=float)),
                has_cfs_dict.get(method),
            )
        )
    ]
    if not methods:
        return np.array([], dtype=float), {}

    arrays = []
    for method in methods:
        times = times_dict[method]
        has_cfs = has_cfs_dict[method]
        arrays.append(
            np.where(
                np.isfinite(times) & has_cfs,
                np.maximum(times, EPSILON),
                np.inf,
            )
        )

    times = np.vstack(arrays)
    valid_instances = np.isfinite(times).any(axis=0)
    if not valid_instances.any():
        return np.array([], dtype=float), {
            method: np.array([], dtype=float) for method in methods
        }

    times = times[:, valid_instances]
    best = np.maximum(np.min(times, axis=0), EPSILON)
    ratios = {method: row / best for method, row in zip(methods, times)}

    finite_ratios = [
        ratio[np.isfinite(ratio)]
        for ratio in ratios.values()
        if np.isfinite(ratio).any()
    ]
    if not finite_ratios:
        return np.array([], dtype=float), {
            method: np.array([], dtype=float) for method in methods
        }

    tau_values = np.unique(np.sort(np.concatenate(([1.0], *finite_ratios))))
    profile = {
        method: np.array([np.mean(ratio <= tau) for tau in tau_values])
        for method, ratio in ratios.items()
    }
    return tau_values, profile


def interpolate_callback(
    callback: list[tuple[float, float]],
    times: np.ndarray,
) -> np.ndarray:
    if not callback:
        return np.full(times.shape, np.nan)

    xs = np.array([point[0] for point in callback], dtype=float)
    ys = np.array([point[1] for point in callback], dtype=float)

    if xs.size == 1:
        return np.full(times.shape, ys[0])

    return np.interp(times, xs, ys, left=ys[0], right=ys[-1])


def aggregate_callbacks(
    callbacks_dict: dict[str, list[list[tuple[float, float]]]],
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    methods = [method for method in MODELS if any(callbacks_dict.get(method, []))]
    if not methods:
        return np.array([], dtype=float), {}

    lengths = {len(callbacks_dict[method]) for method in methods}
    if len(lengths) > 1:
        raise ValueError(f"Callback count mismatch across methods: {sorted(lengths)}")

    all_times = sorted(
        {
            time_value
            for method in methods
            for callback in callbacks_dict[method]
            for time_value, _ in callback
        }
    )
    if not all_times:
        return np.array([], dtype=float), {}

    common_times = np.array(all_times, dtype=float)
    interpolated = {
        method: np.vstack(
            [
                interpolate_callback(callback, common_times)
                for callback in callbacks_dict[method]
            ]
        )
        for method in methods
    }

    stacked = np.stack([interpolated[method] for method in methods], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mins = np.nanmin(stacked, axis=(0, 2))
        maxs = np.nanmax(stacked, axis=(0, 2))

    ranges = maxs - mins
    safe_ranges = np.where(np.isfinite(ranges) & (ranges > 0), ranges, 1.0)
    no_data_rows = np.all(~np.isfinite(stacked), axis=(0, 2))

    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in methods:
        normalized = (interpolated[method] - mins[:, None]) / safe_ranges[:, None]
        normalized[no_data_rows] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_values = np.nanmean(normalized, axis=0)
            std_values = np.nanstd(normalized, axis=0)
        stats[method] = (mean_values, std_values)

    return common_times, stats


def plot_profile(
    dataset: str,
    tau_values: np.ndarray,
    profile: dict[str, np.ndarray],
    *,
    model_type: str,
    output_dir: Path,
) -> None:
    if tau_values.size == 0 or not profile:
        return

    figure, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in MODELS:
        rho = profile.get(method)
        if rho is None or rho.size == 0:
            continue
        ax.step(
            tau_values,
            rho,
            where="post",
            label=method_label(method),
            color=METHOD_COLORS.get(method),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
        )

    ax.set_xlabel(r"Performance ratio $\tau$")
    ax.set_ylabel(r"$\rho(\tau)$")
    ax.set_title(f"Performance profile ({dataset}, {model_type})")
    ax.set_xlim(left=1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "profile.pdf")
    plt.close(figure)


def scatter_plot(
    dataset: str,
    left_method: str,
    right_method: str,
    times_dict: dict[str, np.ndarray],
    statuses_dict: dict[str, np.ndarray],
    has_cfs_dict: dict[str, np.ndarray],
    *,
    model_type: str,
    output_dir: Path,
) -> None:
    left_times = times_dict[left_method]
    right_times = times_dict[right_method]
    left_statuses = statuses_dict[left_method]
    right_statuses = statuses_dict[right_method]
    left_has_cf = has_cfs_dict[left_method]
    right_has_cf = has_cfs_dict[right_method]
    mask = (
        np.isfinite(left_times)
        & np.isfinite(right_times)
        & (left_times > 0)
        & (right_times > 0)
        # & left_has_cf
        # & right_has_cf
    )
    if not mask.any():
        return

    xs = left_times[mask]
    ys = right_times[mask]
    left_optimal = left_has_cf[mask] & (left_statuses[mask] == "OPTIMAL")
    right_optimal = right_has_cf[mask] & (right_statuses[mask] == "OPTIMAL")
    lower = float(min(xs.min(), ys.min()))
    upper = float(max(xs.max(), ys.max()))

    figure, ax = plt.subplots(figsize=(5.2, 5.2))
    category_specs = [
        (
            left_optimal & right_optimal,
            "Both OPTIMAL",
            "o",
            "#2ca02c",
        ),
        (
            ~left_optimal & right_optimal,
            f"{method_label(left_method)} non-OPTIMAL only",
            "X",
            METHOD_COLORS.get(left_method, "#444444"),
        ),
        (
            left_optimal & ~right_optimal,
            f"{method_label(right_method)} non-OPTIMAL only",
            "^",
            METHOD_COLORS.get(right_method, "#444444"),
        ),
        (
            ~left_optimal & ~right_optimal,
            "Both non-OPTIMAL",
            "s",
            "#7f7f7f",
        ),
    ]
    for category_mask, label, marker, color in category_specs:
        if not np.any(category_mask):
            continue
        ax.scatter(
            xs[category_mask],
            ys[category_mask],
            alpha=0.75,
            color=color,
            marker=marker,
            label=label,
            edgecolors="black" if marker != "X" else None,
            linewidths=0.4,
        )
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="gray")
    ax.set_xlabel(f"{method_label(left_method)} time (s)")
    ax.set_ylabel(f"{method_label(right_method)} time (s)")
    ax.set_title(
        f"{method_label(left_method)} vs {method_label(right_method)} ({dataset}, {model_type})"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"scatter_{left_method}_vs_{right_method}.pdf")
    plt.close(figure)


def ratio_histogram(
    dataset: str,
    numerator_method: str,
    denominator_method: str,
    times_dict: dict[str, np.ndarray],
    has_cfs_dict: dict[str, np.ndarray],
    *,
    model_type: str,
    output_dir: Path,
) -> None:
    numerator = times_dict[numerator_method]
    denominator = times_dict[denominator_method]
    numerator_has_cf = has_cfs_dict[numerator_method]
    denominator_has_cf = has_cfs_dict[denominator_method]
    mask = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > 0)
        & (denominator > 0)
        & numerator_has_cf
        & denominator_has_cf
    )
    if not mask.any():
        return

    ratios = numerator[mask] / denominator[mask]
    bins = min(30, max(10, int(np.sqrt(ratios.size))))

    figure, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(ratios, bins=bins, alpha=0.75, color=METHOD_COLORS.get(numerator_method))
    ax.axvline(1.0, color="black", linestyle="--")
    ax.set_xlabel(
        f"{method_label(numerator_method)} / {method_label(denominator_method)}"
    )
    ax.set_ylabel("Instances")
    ax.set_title(f"Time ratios ({dataset}, {model_type})")
    ax.grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(
        output_dir / f"ratio_histogram_{numerator_method}_vs_{denominator_method}.pdf"
    )
    plt.close(figure)


def remove_stale_ratio_histograms(
    output_dir: Path,
    active_time_methods: list[str],
) -> None:
    allowed_files = {
        f"ratio_histogram_cp_vs_{method}.pdf"
        for method in active_time_methods
        if method != "cp"
    }
    for path in output_dir.glob("ratio_histogram_*.pdf"):
        if path.name in allowed_files:
            continue
        try:
            path.unlink()
        except OSError as exc:
            print(f"Could not remove stale histogram {path}: {exc}")


def cactus_plot(
    dataset: str,
    times_dict: dict[str, np.ndarray],
    statuses_dict: dict[str, np.ndarray],
    has_cfs_dict: dict[str, np.ndarray],
    total_instances: int,
    *,
    model_type: str,
    output_dir: Path,
    include_build_time: bool,
    filename: str,
) -> None:
    if total_instances == 0:
        return

    figure, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = False
    for method in MODELS:
        sorted_times = finite_plot_times(
            times_dict.get(method, np.array([], dtype=float)),
            has_cfs_dict.get(method),
            statuses_dict.get(method),
            require_optimal=True,
        )
        if sorted_times.size == 0:
            continue
        solved = np.arange(1, sorted_times.size + 1)
        ax.step(
            sorted_times,
            solved,
            where="post",
            label=method_label(method),
            color=METHOD_COLORS.get(method),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solved instances")
    ax.set_title(
        f"Cactus plot ({dataset}, {model_type}, {time_mode_label(include_build_time)})"
    )
    ax.set_xscale("log")
    ax.axvline(
        TIMEOUT, color="black", linestyle="-", alpha=0.65, label=f"{TIMEOUT}s timeout"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / filename)
    plt.close(figure)


def cactus_cdf_plot(
    dataset: str,
    times_dict: dict[str, np.ndarray],
    statuses_dict: dict[str, np.ndarray],
    has_cfs_dict: dict[str, np.ndarray],
    total_instances: int,
    *,
    model_type: str,
    output_dir: Path,
    include_build_time: bool,
    filename: str,
) -> None:
    if total_instances == 0:
        return

    figure, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = False
    for method in MODELS:
        sorted_times = finite_plot_times(
            times_dict.get(method, np.array([], dtype=float)),
            has_cfs_dict.get(method),
            statuses_dict.get(method),
            require_optimal=True,
        )
        if sorted_times.size == 0:
            continue
        cdf = np.arange(1, sorted_times.size + 1) / total_instances
        ax.step(
            sorted_times,
            cdf,
            where="post",
            label=method_label(method),
            color=METHOD_COLORS.get(method),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solved fraction")
    ax.set_title(
        f"Empirical solved CDF ({dataset}, {model_type}, {time_mode_label(include_build_time)})"
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.axvline(
        TIMEOUT, color="black", linestyle="-", alpha=0.65, label=f"{TIMEOUT}s timeout"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / filename)
    plt.close(figure)


def distance_plot(
    dataset: str,
    common_times: np.ndarray,
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    model_type: str,
    output_dir: Path,
) -> None:
    if common_times.size == 0 or not stats:
        return

    figure, ax = plt.subplots(figsize=(6.8, 4.8))
    plotted = False
    for method in MODELS[:2]:  # Focus on CP and MIP for this plot
        method_stats = stats.get(method)
        if method_stats is None:
            continue

        mean_values, std_values = method_stats
        if not np.isfinite(mean_values).any():
            continue

        color = METHOD_COLORS.get(method)
        ax.plot(
            common_times,
            mean_values,
            label=method_label(method),
            color=color,
            linestyle=METHOD_LINESTYLES.get(method, "-"),
        )
        ax.fill_between(
            common_times,
            np.clip(mean_values - std_values, 0.0, 1.0),
            np.clip(mean_values + std_values, 0.0, 1.0),
            alpha=0.18,
            color=color,
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized objective")
    ax.set_xscale("log")
    ax.set_title(f"Objective vs time ({dataset}, {model_type})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "distance_plot.pdf")
    plt.close(figure)


def plot_times_vs_parameter(
    dataset: str,
    parameter_name: str,
    parameter_values: list[Any],
    *,
    model_type: str,
    output_dir: Path,
    n_estimators: Any = UNSET,
    max_depth: Any = UNSET,
    seed: Any = UNSET,
    voting: Any = UNSET,
    include_build_time: bool,
    filename: str,
) -> None:
    medians = {method: [] for method in MODELS}
    lower_quartiles = {method: [] for method in MODELS}
    upper_quartiles = {method: [] for method in MODELS}

    for parameter_value in parameter_values:
        filters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "seed": seed,
            "voting": voting,
            "include_build_time": include_build_time,
        }
        filters[parameter_name] = parameter_value
        bundle = collect_method_data(dataset, model_type, **filters)
        for method in MODELS:
            values = bundle.times[method]
            finite_values = valid_time_values(
                values,
                bundle.has_cfs[method],
                bundle.statuses[method],
                require_optimal=True,
            )
            if finite_values.size == 0:
                medians[method].append(np.nan)
                lower_quartiles[method].append(np.nan)
                upper_quartiles[method].append(np.nan)
                continue
            medians[method].append(float(np.median(finite_values)))
            lower_quartiles[method].append(float(np.quantile(finite_values, 0.25)))
            upper_quartiles[method].append(float(np.quantile(finite_values, 0.75)))

    figure, ax = plt.subplots(figsize=(8.0, 5.0))
    x_values = np.arange(len(parameter_values))
    plotted = False
    for method in MODELS:
        median_values = np.array(medians[method], dtype=float)
        lower_values = np.array(lower_quartiles[method], dtype=float)
        upper_values = np.array(upper_quartiles[method], dtype=float)
        mask = np.isfinite(median_values)
        if not mask.any():
            continue

        color = METHOD_COLORS.get(method)
        ax.plot(
            x_values[mask],
            median_values[mask],
            marker="o",
            label=method_label(method),
            color=color,
            linestyle=METHOD_LINESTYLES.get(method, "-"),
        )
        ax.fill_between(
            x_values[mask],
            np.clip(lower_values[mask], EPSILON, None),
            np.clip(upper_values[mask], EPSILON, None),
            alpha=0.18,
            color=color,
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return

    ax.set_xticks(x_values)
    ax.set_xticklabels([format_parameter_value(value) for value in parameter_values])
    ax.set_xlabel(PARAMETER_LABELS.get(parameter_name, parameter_name))
    ax.set_ylabel("Median time (s)")
    ax.set_yscale("log")
    ax.axhline(
        TIMEOUT, color="black", linestyle="-", alpha=0.65, label=f"{TIMEOUT}s timeout"
    )
    ax.set_title(
        f"Median time vs {PARAMETER_LABELS.get(parameter_name, parameter_name).lower()} "
        f"({dataset}, {model_type}, {time_mode_label(include_build_time)})"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / filename)
    plt.close(figure)


def plot_distance_by_parameter(
    dataset: str,
    parameter_name: str,
    parameter_values: list[Any],
    *,
    model_type: str,
    output_dir: Path,
    filename: str,
    n_estimators: Any = UNSET,
    max_depth: Any = UNSET,
    seed: Any = UNSET,
    voting: Any = UNSET,
) -> None:
    parameter_series: list[
        tuple[Any, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]
    ] = []
    active_methods: set[str] = set()

    for parameter_value in parameter_values:
        filters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "seed": seed,
            "voting": voting,
        }
        filters[parameter_name] = parameter_value
        bundle = collect_method_data(dataset, model_type, **filters)
        common_times, stats = aggregate_callbacks(bundle.callbacks)
        if common_times.size == 0 or not stats:
            continue
        parameter_series.append((parameter_value, common_times, stats))
        active_methods.update(stats)

    ordered_methods = [method for method in MODELS[:2] if method in active_methods]
    if not ordered_methods:
        return

    figure, axis = plt.subplots(1, 1, figsize=(7.0, 4.2), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(parameter_series)))
    axis = axis.flat[0]
    parameter_legend_handles: dict[Any, Any] = {}

    for method in ordered_methods:
        for color, (parameter_value, common_times, stats) in zip(
            colors, parameter_series
        ):
            method_stats = stats.get(method)
            if method_stats is None:
                continue

            mean_values, _ = method_stats
            if not np.isfinite(mean_values).any():
                continue

            line = axis.plot(
                common_times,
                mean_values,
                color=color,
                label=format_parameter_value(parameter_value),
                linestyle=METHOD_LINESTYLES.get(method, "-"),
            )[0]
            parameter_legend_handles.setdefault(parameter_value, line)

    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Normalized objective")
    axis.set_xscale("log")
    axis.grid(True, linestyle="--", alpha=0.5)

    legend_handles = [
        parameter_legend_handles[parameter_value]
        for parameter_value, _, _ in parameter_series
        if parameter_value in parameter_legend_handles
    ]
    legend_labels = [
        format_parameter_value(parameter_value)
        for parameter_value, _, _ in parameter_series
        if parameter_value in parameter_legend_handles
    ]
    if legend_handles:
        for method in ordered_methods:
            method_line = axis.plot(
                [],
                [],
                color="black",
                linestyle=METHOD_LINESTYLES.get(method, "-"),
                label=METHOD_LABELS.get(method, method),
            )[0]
            legend_handles.append(method_line)
            legend_labels.append(METHOD_LABELS.get(method, method))

        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 6),
            title=PARAMETER_LABELS.get(parameter_name, parameter_name),
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_dir / filename)
    plt.close(figure)


def generate_plots_for_dataset(
    dataset: str,
    model_type: str,
    voting: str | None = None,
) -> None:
    normalized_voting = normalize_voting(model_type, voting)
    output_dir = ensure_output_dir(dataset, model_type, normalized_voting)
    estimators_reference_depth = select_reference_value(
        dataset,
        model_type,
        fixed_parameter_name="max_depth",
        preferred_value=7,
        varying_parameter_name="n_estimators",
        varying_values=N_ESTIMATORS,
        voting=normalized_voting,
    )
    depth_reference_estimators = select_reference_value(
        dataset,
        model_type,
        fixed_parameter_name="n_estimators",
        preferred_value=100,
        varying_parameter_name="max_depth",
        varying_values=MAX_DEPTHS,
        voting=normalized_voting,
    )
    bundle_with_build = collect_method_data(
        dataset,
        model_type,
        voting=normalized_voting,
        include_build_time=True,
    )
    bundle_without_build = collect_method_data(
        dataset,
        model_type,
        voting=normalized_voting,
        include_build_time=False,
    )

    if bundle_with_build.total_instances == 0:
        print(
            f"No results found for {dataset}/{model_type}/{normalized_voting.lower()}."
        )
        return

    active_time_methods = bundle_with_build.active_time_methods
    remove_stale_ratio_histograms(output_dir, active_time_methods)
    if active_time_methods:
        cactus_plot(
            dataset,
            bundle_without_build.times,
            bundle_without_build.statuses,
            bundle_without_build.has_cfs,
            bundle_without_build.total_instances,
            model_type=model_type,
            output_dir=output_dir,
            include_build_time=False,
            filename="cactus.pdf",
        )
        cactus_plot(
            dataset,
            bundle_with_build.times,
            bundle_with_build.statuses,
            bundle_with_build.has_cfs,
            bundle_with_build.total_instances,
            model_type=model_type,
            output_dir=output_dir,
            include_build_time=True,
            filename="cactus_with_build.pdf",
        )
        cactus_cdf_plot(
            dataset,
            bundle_without_build.times,
            bundle_without_build.statuses,
            bundle_without_build.has_cfs,
            bundle_without_build.total_instances,
            model_type=model_type,
            output_dir=output_dir,
            include_build_time=False,
            filename="cactus_cdf.pdf",
        )
        cactus_cdf_plot(
            dataset,
            bundle_with_build.times,
            bundle_with_build.statuses,
            bundle_with_build.has_cfs,
            bundle_with_build.total_instances,
            model_type=model_type,
            output_dir=output_dir,
            include_build_time=True,
            filename="cactus_cdf_with_build.pdf",
        )

        tau_values, profile = compute_performance_profile(
            {method: bundle_with_build.times[method] for method in active_time_methods},
            {
                method: bundle_with_build.has_cfs[method]
                for method in active_time_methods
            },
        )
        plot_profile(
            dataset,
            tau_values,
            profile,
            model_type=model_type,
            output_dir=output_dir,
        )

        for left_method, right_method in combinations(active_time_methods, 2):
            scatter_plot(
                dataset,
                left_method,
                right_method,
                bundle_with_build.times,
                bundle_with_build.statuses,
                bundle_with_build.has_cfs,
                model_type=model_type,
                output_dir=output_dir,
            )

        if "cp" in active_time_methods:
            for other_method in active_time_methods:
                if other_method == "cp":
                    continue
                ratio_histogram(
                    dataset,
                    "cp",
                    other_method,
                    bundle_with_build.times,
                    bundle_with_build.has_cfs,
                    model_type=model_type,
                    output_dir=output_dir,
                )

        plot_times_vs_parameter(
            dataset,
            "n_estimators",
            N_ESTIMATORS,
            model_type=model_type,
            output_dir=output_dir,
            max_depth=estimators_reference_depth,
            voting=normalized_voting,
            include_build_time=False,
            filename="times_vs_estimators.pdf",
        )
        plot_times_vs_parameter(
            dataset,
            "n_estimators",
            N_ESTIMATORS,
            model_type=model_type,
            output_dir=output_dir,
            max_depth=estimators_reference_depth,
            voting=normalized_voting,
            include_build_time=True,
            filename="times_vs_estimators_with_build.pdf",
        )
        plot_times_vs_parameter(
            dataset,
            "max_depth",
            MAX_DEPTHS,
            model_type=model_type,
            output_dir=output_dir,
            n_estimators=depth_reference_estimators,
            voting=normalized_voting,
            include_build_time=False,
            filename="times_vs_depth.pdf",
        )
        plot_times_vs_parameter(
            dataset,
            "max_depth",
            MAX_DEPTHS,
            model_type=model_type,
            output_dir=output_dir,
            n_estimators=depth_reference_estimators,
            voting=normalized_voting,
            include_build_time=True,
            filename="times_vs_depth_with_build.pdf",
        )

    common_times, stats = aggregate_callbacks(bundle_with_build.callbacks)
    distance_plot(
        dataset,
        common_times,
        stats,
        model_type=model_type,
        output_dir=output_dir,
    )

    plot_distance_by_parameter(
        dataset,
        "n_estimators",
        N_ESTIMATORS,
        model_type=model_type,
        output_dir=output_dir,
        seed=2,
        filename="estimators_distance_plot.pdf",
        max_depth=estimators_reference_depth,
        voting=normalized_voting,
    )
    plot_distance_by_parameter(
        dataset,
        "max_depth",
        MAX_DEPTHS,
        model_type=model_type,
        output_dir=output_dir,
        n_estimators=depth_reference_estimators,
        filename="depth_distance_plot.pdf",
        voting=normalized_voting,
    )


def main() -> None:
    for dataset in DATASETS:
        for model_type in ("rf", "xgb"):
            votings = VOTING if model_type == "rf" else ["SOFT"]
            for voting in votings:
                print(
                    "Generating plots for dataset:",
                    dataset,
                    "model type:",
                    model_type,
                    "voting:",
                    voting,
                )
                generate_plots_for_dataset(dataset, model_type, voting=voting)


if __name__ == "__main__":
    main()
