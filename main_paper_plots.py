#!/usr/bin/env python3
"""Generate the aggregate figures used in the main paper results section.

The cactus scope is controlled by CACTUS_SCOPE below. The anytime, depth-scaling,
and estimator-scaling plots aggregate all datasets listed in parameters.DATASETS.
The norm boxplot aggregates the available multinorm experiments under
results/rf/multinorms and adds the L1 reference runs for the same datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable

import numpy as np

from parameters import (
    DATASETS,
    DATASET_ACRONYMS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    MAX_DEPTHS,
    MODELS,
    N_ESTIMATORS,
    TIMEOUT,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "rf"
MULTINORM_DIR = RESULTS_DIR / "multinorms"
OUTPUT_DIR = ROOT / "plots" / "rf" / "paper"
STANDARD_RESULTS_DIRS = {
    "rf": ROOT / "results" / "rf",
    "xgb": ROOT / "results" / "xgb",
}

# Cactus figure scope. Use:
# - "default": only the default ensemble configuration (100 trees, depth 5).
# - "all": every standard RF configuration found in results/rf/exp_*.json,
#   i.e., both the depth-scaling and estimator-scaling sweeps.
CACTUS_SCOPE = "default"
CACTUS_OUTPUT_FILENAME = "default_cactus.pdf"
VALID_CACTUS_SCOPES = {"default", "all"}

# Scaling figure median policy. Use:
# - "all_censored": median over all benchmark instances in the selected
#   configuration, with missing, unevaluated, or non-optimal runs counted as
#   TIMEOUT. This avoids hiding failures and is the recommended setting for the
#   main paper.
# - "optimal_only": median only over instances proven optimal.
SCALING_MEDIAN_POLICY = "all_censored"
VALID_SCALING_MEDIAN_POLICIES = {"all_censored", "optimal_only"}
VALID_SCALING_ERROR_BARS = {"quartiles", "mad"}
SCALING_ERROR_BAR_ALIASES = {"quratiles": "quartiles"}

METHOD_LABELS = {
    "cp": "CPCF",
    "mip": "OCEAN",
    "mace": "MACE",
    "maxsat": "MaxSAT",
}
METHOD_COLORS = {
    "cp": "#1f77b4",
    "mip": "#ff7f0e",
    "mace": "#d62728",
    "maxsat": "#2ca02c",
}
METHOD_LINESTYLES = {
    "cp": "-",
    "mip": "--",
    "mace": ":",
    "maxsat": "-.",
}

DEFAULT_NORM_BOXPLOT_METHODS = ("cp",)
VALID_NORM_BOXPLOT_METHODS = ("cp", "mip")
NORM_GRAYS = {
    0: "#ededed",
    1: "#b2b2b2",
    2: "#717171",
}
NORM_ALPHAS = {
    0: 0.1,
    1: 0.5,
    2: 1.0,
}

PAPER_METHODS = [
    method for method in ["cp", "mip", "mace", "maxsat"] if method in MODELS
]
EPSILON = 1e-12

default_fontsize = 16


@dataclass(frozen=True)
class MethodResult:
    status: Any
    total_time: float | None
    objective: float | None
    has_cf: bool
    callbacks: list[tuple[float, float]]
    evaluated: bool


@dataclass(frozen=True)
class InstanceResult:
    dataset: str
    model_type: str
    voting: str
    n_estimators: int
    max_depth: int
    seed: int
    query_index: int
    methods: dict[str, MethodResult]


@dataclass(frozen=True)
class NormResult:
    dataset: str
    norm: int
    method: str
    total_time: float


@dataclass(frozen=True)
class PanelSpec:
    model_type: str
    voting: str
    title: str


PANELS = (
    PanelSpec(model_type="rf", voting="SOFT", title="RF soft voting"),
    PanelSpec(model_type="rf", voting="HARD", title="RF hard voting"),
    PanelSpec(model_type="xgb", voting="SOFT", title="XGB"),
)
METHODS_BY_MODEL = {
    "rf": tuple(
        method for method in ["cp", "mip", "mace", "maxsat"] if method in PAPER_METHODS
    ),
    "xgb": tuple(
        method for method in ["cp", "mip", "maxsat"] if method in PAPER_METHODS
    ),
}


def configure_matplotlib() -> None:
    cache_root = Path(gettempdir()) / "oceanpy-paper-plots"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg-cache"))
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg-cache").mkdir(parents=True, exist_ok=True)


def finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def normalize_time(solve_time: Any, build_time: float) -> float | None:
    value = finite_float(solve_time)
    if value is None or value < 0:
        return None
    return max(EPSILON, min(float(TIMEOUT), value + build_time))


def normalize_callback(
    explanation: dict[str, Any],
    method: str,
    build_time: float,
    final_time: float | None,
    final_objective: float | None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    raw_callback = explanation.get(f"{method}_callback") or []
    if isinstance(raw_callback, list):
        for item in raw_callback:
            if not isinstance(item, dict):
                continue
            time_value = finite_float(item.get("time"))
            objective_value = finite_float(item.get("objective_value"))
            if time_value is None or objective_value is None or time_value < 0:
                continue
            if final_objective is not None:
                objective_value = max(objective_value, final_objective)
            points.append(
                (
                    max(EPSILON, min(float(TIMEOUT), time_value + build_time)),
                    objective_value,
                )
            )

    if final_time is not None and final_objective is not None:
        points.append((final_time, final_objective))

    points.sort(key=lambda pair: pair[0])
    best_points: list[tuple[float, float]] = []
    best_objective = math.inf
    for time_value, objective_value in points:
        best_objective = min(best_objective, objective_value)
        best_points.append((time_value, best_objective))
    return best_points


def terminal_callback_objective(
    explanation: dict[str, Any],
    method: str,
) -> float | None:
    terminal_time: float | None = None
    terminal_objective: float | None = None
    raw_callback = explanation.get(f"{method}_callback") or []
    if not isinstance(raw_callback, list):
        return None
    for item in raw_callback:
        if not isinstance(item, dict):
            continue
        time_value = finite_float(item.get("time"))
        objective_value = finite_float(item.get("objective_value"))
        if time_value is None or objective_value is None or time_value < 0:
            continue
        if terminal_time is None or time_value >= terminal_time:
            terminal_time = time_value
            terminal_objective = objective_value
    return terminal_objective


def shared_terminal_objective(
    explanation: dict[str, Any],
    methods: tuple[str, ...],
) -> float | None:
    if not methods:
        return None

    values: list[float] = []
    for method in methods:
        if (
            explanation.get(f"{method}_status") != "OPTIMAL"
            or explanation.get(f"{method}_cf") is None
        ):
            return None
        objective = finite_float(explanation.get(f"{method}_objective"))
        if objective is not None:
            values.append(objective)
        callback_objective = terminal_callback_objective(explanation, method)
        if callback_objective is not None:
            values.append(callback_objective)

    return max(values) if values else None


def methods_for_model(model_type: str) -> tuple[str, ...]:
    return METHODS_BY_MODEL.get(model_type, tuple(PAPER_METHODS))


def normalize_voting(model_type: str, voting: Any) -> str:
    if model_type == "xgb":
        return "SOFT"
    return str(voting or "SOFT").upper()


def load_standard_results() -> list[InstanceResult]:
    instances: list[InstanceResult] = []
    for default_model_type, results_dir in STANDARD_RESULTS_DIRS.items():
        for path in sorted(results_dir.glob("exp_*.json")):
            with path.open("r") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                continue

            for item in payload:
                if not isinstance(item, dict):
                    continue

                dataset = item.get("dataset")
                if not isinstance(dataset, str):
                    continue

                model_type = str(item.get("model_type") or default_model_type)
                voting = normalize_voting(model_type, item.get("voting"))
                n_estimators = int(item.get("n_estimators"))
                max_depth = int(item.get("max_depth"))
                seed = int(item.get("seed"))
                explanations = item.get("explanations") or []
                if not isinstance(explanations, list):
                    continue

                for query_index, explanation in enumerate(explanations):
                    if not isinstance(explanation, dict):
                        continue
                    paired_anytime_methods = tuple(
                        method
                        for method in ("cp", "mip")
                        if method in methods_for_model(model_type)
                    )
                    paired_terminal_objective = shared_terminal_objective(
                        explanation, paired_anytime_methods
                    )
                    methods: dict[str, MethodResult] = {}
                    for method in methods_for_model(model_type):
                        build_time = (
                            finite_float(item.get(f"{method}_build_time")) or 0.0
                        )
                        total_time = normalize_time(
                            explanation.get(f"{method}_time"), build_time
                        )
                        objective = finite_float(explanation.get(f"{method}_objective"))
                        if (
                            method in paired_anytime_methods
                            and paired_terminal_objective is not None
                        ):
                            objective = paired_terminal_objective
                        status = explanation.get(f"{method}_status")
                        has_cf = explanation.get(f"{method}_cf") is not None
                        evaluated = any(
                            explanation.get(f"{method}_{suffix}") is not None
                            for suffix in ["status", "time", "cf", "valid", "objective"]
                        )
                        callbacks = normalize_callback(
                            explanation,
                            method,
                            build_time,
                            total_time,
                            objective,
                        )
                        methods[method] = MethodResult(
                            status=status,
                            total_time=total_time,
                            objective=objective,
                            has_cf=has_cf,
                            callbacks=callbacks,
                            evaluated=evaluated,
                        )

                    instances.append(
                        InstanceResult(
                            dataset=dataset,
                            model_type=model_type,
                            voting=voting,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            seed=seed,
                            query_index=query_index,
                            methods=methods,
                        )
                    )
    return instances


def default_records(
    instances: list[InstanceResult], panel: PanelSpec
) -> list[InstanceResult]:
    return [
        instance
        for instance in instances
        if instance.model_type == panel.model_type
        and instance.voting == panel.voting
        and instance.n_estimators == DEFAULT_N_ESTIMATORS
        and instance.max_depth == DEFAULT_MAX_DEPTH
    ]


def cactus_records(
    instances: list[InstanceResult],
    panel: PanelSpec,
    *,
    scope: str,
) -> list[InstanceResult]:
    if scope == "default":
        return default_records(instances, panel)
    if scope == "all":
        return [
            instance
            for instance in instances
            if instance.model_type == panel.model_type
            and instance.voting == panel.voting
        ]
    raise ValueError(
        f"Unknown cactus scope {scope!r}. Use one of {sorted(VALID_CACTUS_SCOPES)}."
    )


def solved_times(records: list[InstanceResult], method: str) -> list[float]:
    times = [
        result.total_time
        for instance in records
        for result in [instance.methods[method]]
        if result.status == "OPTIMAL"
        and result.has_cf
        and result.total_time is not None
    ]
    return sorted(float(value) for value in times)


def scaling_times(
    records: list[InstanceResult],
    method: str,
    *,
    policy: str,
) -> list[float]:
    if policy == "optimal_only":
        return [
            result.total_time
            for instance in records
            for result in [instance.methods[method]]
            if result.status == "OPTIMAL"
            and result.has_cf
            and result.total_time is not None
        ]
    if policy == "all_censored":
        values: list[float] = []
        for instance in records:
            result = instance.methods[method]
            if (
                result.status == "OPTIMAL"
                and result.has_cf
                and result.total_time is not None
            ):
                values.append(result.total_time)
            else:
                values.append(float(TIMEOUT))
        return values
    raise ValueError(
        f"Unknown scaling median policy {policy!r}. "
        f"Use one of {sorted(VALID_SCALING_MEDIAN_POLICIES)}."
    )


def median(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return statistics.median(finite_values) if finite_values else float("nan")


def scaling_summary(
    values: list[float],
    error_bars: str | None,
) -> tuple[float, tuple[float, float] | None]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan"), None

    center = statistics.median(finite_values)
    if error_bars == "quartiles":
        q1, q3 = np.percentile(finite_values, [25, 75])
        return center, (max(0.0, center - float(q1)), max(0.0, float(q3) - center))
    if error_bars == "mad":
        mad = statistics.median(abs(value - center) for value in finite_values)
        return center, (mad, mad)
    return center, None


def scaling_filename(filename: str, error_bars: str | None) -> str:
    if error_bars is None:
        return filename
    path = Path(filename)
    return f"{path.stem}_{error_bars}{path.suffix}"


def scaling_error_bars_arg(value: str) -> str:
    normalized = SCALING_ERROR_BAR_ALIASES.get(value, value)
    if normalized not in VALID_SCALING_ERROR_BARS:
        choices = sorted(VALID_SCALING_ERROR_BARS | set(SCALING_ERROR_BAR_ALIASES))
        raise argparse.ArgumentTypeError(f"choose one of {choices}")
    return normalized


def best_at(callbacks: list[tuple[float, float]], time_value: float) -> float | None:
    best_value: float | None = None
    for callback_time, objective_value in callbacks:
        if callback_time > time_value:
            break
        best_value = (
            objective_value if best_value is None else min(best_value, objective_value)
        )
    return best_value


def anytime_optimum(
    instance: InstanceResult,
    methods: tuple[str, ...],
) -> float | None:
    optimal_objectives = [
        result.objective
        for method in methods
        for result in [instance.methods.get(method)]
        if result is not None
        and result.status == "OPTIMAL"
        and result.has_cf
        and result.objective is not None
    ]
    if not optimal_objectives:
        return None
    return min(optimal_objectives)


def jointly_optimal_objective(
    instance: InstanceResult,
    methods: tuple[str, ...],
) -> float | None:
    optimal_objectives: list[float] = []
    for method in methods:
        result = instance.methods.get(method)
        if (
            result is None
            or result.status != "OPTIMAL"
            or not result.has_cf
            or result.objective is None
        ):
            return None
        optimal_objectives.append(result.objective)

    if not optimal_objectives:
        return None
    if max(optimal_objectives) - min(optimal_objectives) >= 1e-4:
        return None
    return min(optimal_objectives)


def plot_cactus(
    instances: list[InstanceResult],
    output_dir: Path,
    *,
    scope: str,
    filename: str,
) -> None:
    import matplotlib.pyplot as plt

    legend_style = {
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "#d9d9d9",
        "framealpha": 0.94,
        "fontsize": default_fontsize,
    }
    figure, axes = plt.subplots(1, len(PANELS), figsize=(15.4, 4.0), sharey=True)
    legend_handles: dict[str, Any] = {}
    for n, (axis, panel) in enumerate(zip(axes, PANELS)):
        records = cactus_records(instances, panel, scope=scope)
        denominator = len(records)
        for method in methods_for_model(panel.model_type):
            times = solved_times(records, method)
            if not times:
                continue
            y_values = np.arange(1, len(times) + 1) / denominator
            (line,) = axis.step(
                times,
                y_values,
                where="post",
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=3.0,
            )
            legend_handles.setdefault(method, line)
        axis.axvline(float(TIMEOUT), color="black", linewidth=1.0, alpha=0.6)
        axis.set_xscale("log")
        axis.set_title(panel.title, fontsize=default_fontsize)
        if n == 1:
            axis.set_xlabel("Time (s)", fontsize=default_fontsize)
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.tick_params(axis="both", which="major", labelsize=default_fontsize)
    axes[0].set_ylabel("Fraction proven optimal", fontsize=default_fontsize)
    if legend_handles:
        ordered_methods = [
            method for method in PAPER_METHODS if method in legend_handles
        ]
        axes[0].legend(
            [legend_handles[method] for method in ordered_methods],
            [METHOD_LABELS[method] for method in ordered_methods],
            loc="center",
            bbox_to_anchor=(0.67, 0.62),
            ncol=1,
            **legend_style,
        )
    figure.tight_layout()
    figure.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(figure)


def plot_anytime(
    instances: list[InstanceResult],
    output_dir: Path,
    only_both_optimal: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    legend_style = {
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "#d9d9d9",
        "framealpha": 0.94,
        "fontsize": default_fontsize,
    }
    time_grid = np.geomspace(0.05, float(TIMEOUT), 220)
    figure, axes = plt.subplots(1, len(PANELS), figsize=(15.4, 4.0), sharey=True)
    legend_handles: dict[str, Any] = {}

    for n, (axis, panel) in enumerate(zip(axes, PANELS)):
        panel_methods = tuple(
            method
            for method in ["cp", "mip"]
            if method in methods_for_model(panel.model_type)
        )
        records: list[tuple[InstanceResult, float]] = []
        for instance in default_records(instances, panel):
            optimum = (
                jointly_optimal_objective(instance, panel_methods)
                if only_both_optimal
                else anytime_optimum(instance, panel_methods)
            )
            if optimum is not None:
                records.append((instance, optimum))

        for method in panel_methods:
            curve: list[float] = []
            for time_value in time_grid:
                gaps: list[float] = []
                for instance, optimum in records:
                    all_objectives = [
                        objective
                        for other_method in panel_methods
                        for _, objective in instance.methods[other_method].callbacks
                        if math.isfinite(objective)
                    ]
                    initial_scale = max(all_objectives) if all_objectives else optimum
                    denominator = max(initial_scale - optimum, 1e-9)
                    current_best = best_at(
                        instance.methods[method].callbacks, float(time_value)
                    )
                    gap = (
                        1.0
                        if current_best is None
                        else max(0.0, min(1.0, (current_best - optimum) / denominator))
                    )
                    gaps.append(gap)
                curve.append(float(np.mean(gaps)) if gaps else float("nan"))

            (line,) = axis.plot(
                time_grid,
                curve,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                linewidth=3.0,
            )
            if any(math.isfinite(value) for value in curve):
                legend_handles.setdefault(method, line)

        axis.set_xscale("log")
        axis.set_ylim(-0.02, 1.02)
        axis.set_yscale("log")
        axis.set_title(panel.title, fontsize=default_fontsize)
        if n == 1:
            axis.set_xlabel("Time (s)", fontsize=default_fontsize)
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.tick_params(axis="both", which="major", labelsize=default_fontsize)
    axes[0].set_ylabel("Mean normalized error", fontsize=default_fontsize)
    if legend_handles:
        ordered_methods = [
            method for method in PAPER_METHODS if method in legend_handles
        ]
        axes[0].legend(
            [legend_handles[method] for method in ordered_methods],
            [METHOD_LABELS[method] for method in ordered_methods],
            loc="upper right",
            ncol=1,
            **legend_style,
        )
    figure.tight_layout()
    figure.savefig(
        output_dir / "default_anytime_cp_ocean_logscale.pdf", bbox_inches="tight"
    )
    plt.close(figure)


def plot_scaling(
    instances: list[InstanceResult],
    values: list[int],
    record_filter: Callable[[InstanceResult, int], bool],
    output_dir: Path,
    filename: str,
    x_label: str,
    median_policy: str,
    *,
    error_bars: str | None = None,
    legend_loc: str,
    legend_bbox: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    legend_style = {
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "#d9d9d9",
        "framealpha": 0.94,
        "fontsize": default_fontsize,
    }
    figure, axes = plt.subplots(1, len(PANELS), figsize=(15.4, 4.0), sharey=True)
    legend_handles: dict[str, Any] = {}
    for n, (axis, panel) in enumerate(zip(axes, PANELS)):
        for method in methods_for_model(panel.model_type):
            y_values: list[float] = []
            lower_errors: list[float] = []
            upper_errors: list[float] = []
            for value in values:
                records = [
                    instance
                    for instance in instances
                    if instance.model_type == panel.model_type
                    and instance.voting == panel.voting
                    and record_filter(instance, value)
                ]
                center, y_error = scaling_summary(
                    scaling_times(records, method, policy=median_policy),
                    error_bars,
                )
                y_values.append(center)
                if error_bars is not None:
                    if y_error is None:
                        lower_errors.append(float("nan"))
                        upper_errors.append(float("nan"))
                    else:
                        lower_errors.append(y_error[0])
                        upper_errors.append(y_error[1])
            if any(math.isfinite(value) for value in y_values):
                if error_bars is None:
                    (line,) = axis.plot(
                        values,
                        y_values,
                        marker="o",
                        label=METHOD_LABELS[method],
                        color=METHOD_COLORS[method],
                        linestyle=METHOD_LINESTYLES[method],
                        linewidth=3.0,
                        markersize=4.5,
                    )
                else:
                    error_color = METHOD_COLORS[method]
                    if error_bars == "quartiles":
                        error_color = mcolors.to_rgba(error_color, 0.7)
                    container = axis.errorbar(
                        values,
                        y_values,
                        yerr=[lower_errors, upper_errors],
                        marker="o",
                        label=METHOD_LABELS[method],
                        color=METHOD_COLORS[method],
                        linestyle=METHOD_LINESTYLES[method],
                        linewidth=3.0,
                        markersize=4.5,
                        ecolor=error_color,
                        elinewidth=1.3,
                        capsize=3.0,
                        capthick=1.3,
                    )
                    line = container.lines[0]
                legend_handles.setdefault(method, line)
        axis.set_yscale("log")
        axis.tick_params(axis="both", which="major", labelsize=default_fontsize)
        axis.set_title(panel.title, fontsize=default_fontsize)
        if n == 1:
            axis.set_xlabel(x_label, fontsize=default_fontsize)
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Median time (s)", fontsize=default_fontsize)
    if legend_handles:
        ordered_methods = [
            method for method in PAPER_METHODS if method in legend_handles
        ]
        legend_kwargs: dict[str, Any] = {
            "loc": legend_loc,
            "ncol": 1,
            **legend_style,
        }
        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox
        axes[0].legend(
            [legend_handles[method] for method in ordered_methods],
            [METHOD_LABELS[method] for method in ordered_methods],
            **legend_kwargs,
        )
    if legend_loc == "center left":
        axes[0].set_ylim(bottom=0.1)
    figure.tight_layout()
    figure.savefig(
        output_dir / scaling_filename(filename, error_bars), bbox_inches="tight"
    )
    plt.close(figure)


def load_norm_results() -> list[NormResult]:
    rows: list[NormResult] = []
    multinorm_datasets: set[str] = set()
    for path in sorted(MULTINORM_DIR.glob("multinorms_*.json")):
        with path.open("r") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not payload:
            continue
        entry = payload[0]
        if not isinstance(entry, dict):
            continue

        dataset = entry.get("dataset")
        if not isinstance(dataset, str):
            continue
        multinorm_datasets.add(dataset)
        explanations = entry.get("explanations") or []
        norms = entry.get("norms") or []
        if not isinstance(explanations, list) or not isinstance(norms, list):
            continue

        for explanation in explanations:
            if not isinstance(explanation, dict):
                continue
            for raw_norm in norms:
                norm = int(raw_norm)
                suffix = str(raw_norm).replace(".", "_")
                for method in ["cp", "mip"]:
                    if explanation.get(f"{method}_{suffix}_status") != "OPTIMAL":
                        continue
                    solve_time = finite_float(
                        explanation.get(f"{method}_{suffix}_time")
                    )
                    if solve_time is None:
                        continue
                    build_time = finite_float(entry.get(f"{method}_build_time")) or 0.0
                    rows.append(
                        NormResult(
                            dataset=dataset,
                            norm=norm,
                            method=method,
                            total_time=solve_time + build_time,
                        )
                    )

    # The L1 values are the default objective values from the corresponding
    # standard soft-voting runs for the same multinorm datasets.
    for dataset in sorted(multinorm_datasets):
        path = (
            RESULTS_DIR
            / f"exp_{dataset}_{DEFAULT_N_ESTIMATORS}_{DEFAULT_MAX_DEPTH}_2.json"
        )
        if not path.exists():
            continue
        with path.open("r") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not payload:
            continue
        entry = payload[0]
        if not isinstance(entry, dict):
            continue
        explanations = entry.get("explanations") or []
        if not isinstance(explanations, list):
            continue
        for explanation in explanations:
            if not isinstance(explanation, dict):
                continue
            for method in ["cp", "mip"]:
                if explanation.get(f"{method}_status") != "OPTIMAL":
                    continue
                solve_time = finite_float(explanation.get(f"{method}_time"))
                if solve_time is None:
                    continue
                build_time = finite_float(entry.get(f"{method}_build_time")) or 0.0
                rows.append(
                    NormResult(
                        dataset=dataset,
                        norm=1,
                        method=method,
                        total_time=solve_time + build_time,
                    )
                )
    return rows


def dataset_acronym(dataset: str) -> str:
    return DATASET_ACRONYMS.get(dataset, dataset[:2].upper())


def add_boxplot_legend_table(
    axis: Any, methods: tuple[str, ...], *, dual_method_plot: bool
) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyBboxPatch, Rectangle

    row_count = len(methods)
    if row_count == 0:
        return

    x0 = 0.022
    y_top = 0.965
    width = 0.27 if dual_method_plot else 0.22
    height = 0.182 if dual_method_plot else 0.132
    label_col = 0.085 if dual_method_plot else 0.072
    col_gap = 0.058 if dual_method_plot else 0.056
    swatch_size = 0.024
    row_gap = 0.07
    header_y = y_top - 0.042
    first_row_y = y_top - 0.1
    swatch_x0 = x0 + label_col

    background = FancyBboxPatch(
        (x0, y_top - height),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=axis.transAxes,
        facecolor="white",
        edgecolor="#d9d9d9",
        linewidth=0.8,
        alpha=0.96,
        zorder=4,
    )
    axis.add_patch(background)

    for column_index, norm in enumerate([0, 1, 2]):
        axis.text(
            swatch_x0 + column_index * col_gap + swatch_size / 2.0,
            header_y,
            rf"$L_{norm}$",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=default_fontsize,
            fontweight="semibold",
            zorder=5,
        )

    for row_index, method in enumerate(methods):
        y_center = first_row_y - row_index * row_gap
        axis.text(
            x0 + 0.016,
            y_center,
            METHOD_LABELS[method],
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=default_fontsize,
            zorder=5,
        )
        for column_index, norm in enumerate([0, 1, 2]):
            facecolor = (
                mcolors.to_rgba(METHOD_COLORS[method], NORM_ALPHAS[norm])
                if dual_method_plot
                else NORM_GRAYS[norm]
            )
            edgecolor = METHOD_COLORS[method] if dual_method_plot else "#3f3f3f"
            axis.add_patch(
                Rectangle(
                    (
                        swatch_x0 + column_index * col_gap,
                        y_center - swatch_size / 2.0,
                    ),
                    swatch_size,
                    swatch_size,
                    transform=axis.transAxes,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.0,
                    zorder=5,
                )
            )


def plot_norm_boxplot(
    norm_rows: list[NormResult],
    output_dir: Path,
    *,
    methods: tuple[str, ...],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    valid_methods = tuple(
        method for method in methods if method in VALID_NORM_BOXPLOT_METHODS
    )
    if not valid_methods:
        raise ValueError(
            f"Norm boxplot methods must be drawn from {VALID_NORM_BOXPLOT_METHODS}, got {methods!r}."
        )

    norms = [0, 1, 2]
    grouped_values: dict[str, dict[int, dict[str, list[float]]]] = {}
    for row in norm_rows:
        if (
            row.method not in valid_methods
            or row.norm not in norms
            or not math.isfinite(row.total_time)
        ):
            continue
        grouped_values.setdefault(row.dataset, {}).setdefault(row.norm, {}).setdefault(
            row.method, []
        ).append(row.total_time)

    datasets = [
        dataset
        for dataset in DATASETS
        if dataset in grouped_values
        and all(
            grouped_values[dataset].get(norm, {}).get(method)
            for norm in norms
            for method in valid_methods
        )
    ]
    if not datasets:
        raise ValueError(
            "No complete multinorm results found for the requested boxplot methods."
        )

    dual_method_plot = len(valid_methods) > 1
    dataset_step = 1.72 if not dual_method_plot else 1.96
    norm_gap = 0.39 if not dual_method_plot else 0.56
    method_gap = 0.0 if not dual_method_plot else 0.18
    box_width = 0.24 if not dual_method_plot else 0.12
    norm_offsets = {norm: (index - 1) * norm_gap for index, norm in enumerate(norms)}
    method_offsets = {
        method: (index - (len(valid_methods) - 1) / 2.0) * method_gap
        for index, method in enumerate(valid_methods)
    }

    with plt.rc_context(
        {
            "axes.labelsize": default_fontsize,
            "axes.titlesize": default_fontsize,
            "xtick.labelsize": default_fontsize,
            "ytick.labelsize": default_fontsize,
            "axes.linewidth": 1.0,
        }
    ):
        figure, axis = plt.subplots(figsize=(15.4, 4.15))

        centers = [1.0 + index * dataset_step for index in range(len(datasets))]
        positions: list[float] = []
        data: list[list[float]] = []
        box_styles: list[tuple[int, str]] = []

        for dataset_index, dataset in enumerate(datasets):
            center = centers[dataset_index]
            if dataset_index % 2 == 0:
                axis.axvspan(
                    center - dataset_step / 2.0,
                    center + dataset_step / 2.0,
                    color="#f6f6f6",
                    zorder=0,
                )
            for norm in norms:
                for method in valid_methods:
                    values = grouped_values[dataset][norm][method]
                    positions.append(
                        center + norm_offsets[norm] + method_offsets[method]
                    )
                    data.append(values)
                    box_styles.append((norm, method))

        boxplot = axis.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "black", "linewidth": 1.35},
            whiskerprops={"color": "#525252", "linewidth": 0.5},
            capprops={"color": "#525252", "linewidth": 1.0},
        )
        for patch, (norm, method) in zip(boxplot["boxes"], box_styles):
            if dual_method_plot:
                patch.set_facecolor(
                    mcolors.to_rgba(METHOD_COLORS[method], NORM_ALPHAS[norm])
                )
                patch.set_edgecolor(METHOD_COLORS[method])
                patch.set_linewidth(1.15)
            else:
                patch.set_facecolor(NORM_GRAYS[norm])
                patch.set_edgecolor("#262626")
                patch.set_linewidth(1.0)

        for center in centers[:-1]:
            axis.axvline(
                center + dataset_step / 2.0,
                color="#d0d0d0",
                linewidth=0.5,
                zorder=0,
            )

        axis.set_xlim(centers[0] - dataset_step / 2.0, centers[-1] + dataset_step / 2.0)
        axis.set_xticks(centers, [dataset_acronym(dataset) for dataset in datasets])
        axis.set_yscale("log")
        axis.tick_params(axis="both", which="major", labelsize=default_fontsize)
        axis.set_ylabel("Runtime (s)", fontsize=default_fontsize)
        # axis.set_xlabel("Dataset")
        axis.grid(
            True, which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.35
        )
        axis.grid(
            True, which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.16
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#404040")
        axis.spines["bottom"].set_color("#404040")
        axis.tick_params(axis="x", pad=8)

        add_boxplot_legend_table(axis, valid_methods, dual_method_plot=dual_method_plot)

        figure.tight_layout()
    figure.savefig(output_dir / "norm_boxplot.pdf", bbox_inches="tight")
    plt.close(figure)


def print_coverage(
    instances: list[InstanceResult],
    norm_rows: list[NormResult],
    *,
    cactus_scope: str,
    cactus_filename: str,
    scaling_median_policy: str,
    scaling_error_bars: str | None,
) -> None:
    default_datasets = sorted(
        {
            instance.dataset
            for instance in instances
            if instance.n_estimators == DEFAULT_N_ESTIMATORS
            and instance.max_depth == DEFAULT_MAX_DEPTH
            and instance.voting in {"SOFT", "HARD"}
        }
    )
    depth_datasets = sorted(
        {
            instance.dataset
            for instance in instances
            if instance.n_estimators == DEFAULT_N_ESTIMATORS
            and instance.max_depth in set(MAX_DEPTHS)
            and instance.voting in {"SOFT", "HARD"}
        }
    )
    estimator_datasets = sorted(
        {
            instance.dataset
            for instance in instances
            if instance.max_depth == DEFAULT_MAX_DEPTH
            and instance.n_estimators in set(N_ESTIMATORS)
            and instance.voting in {"SOFT", "HARD"}
        }
    )
    norm_datasets = sorted({row.dataset for row in norm_rows})
    cactus_instances = [
        instance
        for instance in instances
        if instance.voting in {"SOFT", "HARD"}
        and (
            cactus_scope == "all"
            or (
                instance.n_estimators == DEFAULT_N_ESTIMATORS
                and instance.max_depth == DEFAULT_MAX_DEPTH
            )
        )
    ]
    cactus_datasets = sorted({instance.dataset for instance in cactus_instances})
    cactus_configurations = sorted(
        {(instance.n_estimators, instance.max_depth) for instance in cactus_instances}
    )

    print("Dataset coverage for main-paper aggregate figures:")
    print(
        f"- {cactus_filename}: {cactus_scope} cactus scope, "
        f"three panels (RF soft, RF hard, XGB), "
        f"{len(cactus_datasets)} datasets, {len(cactus_configurations)} configurations: "
        f"{', '.join(cactus_datasets)}"
    )
    print(
        f"- default_anytime_cp_ocean_logscale.pdf: three panels (RF soft, RF hard, XGB), "
        f"all {len(default_datasets)} datasets, restricted to CPCF/OCEAN instances "
        f"where at least one method proved optimality within each panel"
    )
    print(
        f"- {scaling_filename('depth_scaling.pdf', scaling_error_bars)}: "
        f"three panels (RF soft, RF hard, XGB), all {len(depth_datasets)} datasets, "
        f"median policy={scaling_median_policy} "
        f"(missing/non-optimal counted as timeout): {', '.join(depth_datasets)}"
    )
    print(
        f"- {scaling_filename('estimator_scaling.pdf', scaling_error_bars)}: "
        f"three panels (RF soft, RF hard, XGB), all {len(estimator_datasets)} datasets, "
        f"median policy={scaling_median_policy} "
        f"(missing/non-optimal counted as timeout): {', '.join(estimator_datasets)}"
    )
    print(
        f"- norm_boxplot.pdf: multinorm subset only ({len(norm_datasets)} datasets): {', '.join(norm_datasets)}"
    )
    if sorted(DATASETS) != default_datasets:
        print(
            "Warning: default plot dataset coverage differs from parameters.DATASETS."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cactus-scope",
        choices=sorted(VALID_CACTUS_SCOPES),
        default=CACTUS_SCOPE,
        help="Scope for the cactus plot. Defaults to the CACTUS_SCOPE value near the top of this file.",
    )
    parser.add_argument(
        "--cactus-output",
        default=CACTUS_OUTPUT_FILENAME,
        help="Filename for the cactus PDF inside the output directory.",
    )
    parser.add_argument(
        "--scaling-median-policy",
        choices=sorted(VALID_SCALING_MEDIAN_POLICIES),
        default=SCALING_MEDIAN_POLICY,
        help="Median policy for depth and estimator scaling plots.",
    )
    parser.add_argument(
        "--error-bars",
        "--error-bar",
        dest="error_bars",
        type=scaling_error_bars_arg,
        default=None,
        metavar="{quartiles,mad}",
        help=(
            "Add error bars to depth and estimator scaling plots. "
            "Use 'quartiles' for 25th/75th percentiles or 'mad' for median absolute deviation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the paper figures are written.",
    )
    parser.add_argument(
        "--norm-boxplot-methods",
        nargs="+",
        choices=sorted(VALID_NORM_BOXPLOT_METHODS),
        default=list(DEFAULT_NORM_BOXPLOT_METHODS),
        help=(
            "Methods shown in the norm boxplot. "
            "Defaults to CPCF only; add 'mip' to include OCEAN as paired boxes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")

    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = load_standard_results()
    norm_rows = load_norm_results()

    print_coverage(
        instances,
        norm_rows,
        cactus_scope=args.cactus_scope,
        cactus_filename=args.cactus_output,
        scaling_median_policy=args.scaling_median_policy,
        scaling_error_bars=args.error_bars,
    )
    plot_cactus(
        instances,
        output_dir,
        scope=args.cactus_scope,
        filename=args.cactus_output,
    )
    plot_anytime(instances, output_dir)
    plot_scaling(
        instances,
        list(MAX_DEPTHS),
        lambda instance, value: (
            instance.n_estimators == DEFAULT_N_ESTIMATORS
            and instance.max_depth == value
        ),
        output_dir,
        "depth_scaling.pdf",
        "Max depth",
        args.scaling_median_policy,
        error_bars=args.error_bars,
        legend_loc="center left",
        legend_bbox=(0.02, 0.6),
    )
    plot_scaling(
        instances,
        list(N_ESTIMATORS),
        lambda instance, value: (
            instance.max_depth == DEFAULT_MAX_DEPTH and instance.n_estimators == value
        ),
        output_dir,
        "estimator_scaling.pdf",
        "Number of trees",
        args.scaling_median_policy,
        error_bars=args.error_bars,
        legend_loc="lower right",
        legend_bbox=(0.985, 0.04),
    )
    plot_norm_boxplot(
        norm_rows,
        output_dir,
        methods=tuple(args.norm_boxplot_methods),
    )

    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
