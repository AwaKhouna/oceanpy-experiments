#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import signal
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MACE_DIR = Path(__file__).resolve().parent / "mace"

_MACE_MODULES: tuple[Any, Any] | None = None


def _ensure_pysmt_solver() -> None:
    try:
        from pysmt.shortcuts import get_env
    except ModuleNotFoundError as exc:
        msg = (
            "MACEExplainer requires PySMT and a configured SMT solver in the "
            "same Python environment. Install `pysmt` and `z3-solver`, then "
            "register Z3 with `pysmt-install --z3 --confirm-agreement`."
        )
        raise RuntimeError(msg) from exc

    solvers = get_env().factory.all_solvers()
    if not solvers or "z3" not in solvers:
        msg = (
            "MACEExplainer requires a PySMT backend solver, but none is "
            "available. Install `z3-solver` in this environment and run "
            "`pysmt-install --z3 --confirm-agreement`."
        )
        raise RuntimeError(msg)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _get_mace_modules() -> tuple[Any, Any]:
    global _MACE_MODULES
    if _MACE_MODULES is not None:
        return _MACE_MODULES

    module_names = (
        "utils",
        "debug",
        "modelConversion",
        "normalizedDistance",
        "loadCausalConstraints",
        "loadData",
        "generateSATExplanations",
    )
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    previous_path = list(sys.path)
    sys.path.insert(0, str(MACE_DIR))
    try:
        _load_module("utils", MACE_DIR / "utils.py")
        _load_module("debug", MACE_DIR / "debug.py")
        _load_module("modelConversion", MACE_DIR / "modelConversion.py")
        _load_module("normalizedDistance", MACE_DIR / "normalizedDistance.py")
        _load_module(
            "loadCausalConstraints",
            MACE_DIR / "_data_main" / "loadCausalConstraints.py",
        )
        load_data = _load_module("loadData", MACE_DIR / "loadData.py")
        generate_sat = _load_module(
            "generateSATExplanations",
            MACE_DIR / "generateSATExplanations.py",
        )
    finally:
        sys.path[:] = previous_path
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    _MACE_MODULES = load_data, generate_sat
    return _MACE_MODULES


def _timeout_handler(signum: Any, frame: Any) -> None:  # noqa: ANN401, ARG001
    raise TimeoutError("Timeout for mace!")


class MACEExplanation:
    def __init__(
        self,
        x: np.ndarray,
        query: np.ndarray,
        names: list[str],
    ) -> None:
        self._x = x.astype(float)
        self._query = query.astype(float)
        self._names = tuple(names)

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def query(self) -> np.ndarray:
        return self._query

    @property
    def value(self) -> dict[str, float]:
        return dict(zip(self._names, self._x.tolist(), strict=False))

    def to_numpy(self) -> np.ndarray:
        return self._x


class MACEExplainer:
    Type = "mace"
    Status: str = "UNKNOWN"

    def __init__(
        self,
        ensemble: RandomForestClassifier,
        *,
        mapper: Any,
        data: pd.DataFrame,
        target: pd.Series,
        dataset_name: str = "dataset",
        epsilon: float = 1e-3,
    ) -> None:
        if not isinstance(ensemble, RandomForestClassifier):
            msg = "MACEExplainer currently supports sklearn RandomForestClassifier only."
            raise ValueError(msg)
        if target.nunique() != 2:
            msg = "MACEExplainer currently supports binary classification only."
            raise ValueError(msg)

        _ensure_pysmt_solver()
        load_data, generate_sat = _get_mace_modules()
        self._mace_load_data = load_data
        self._generate_sat = generate_sat
        self.model = ensemble
        self.mapper = mapper
        self.epsilon = epsilon
        self.callback: Any | None = None
        self._warned_num_workers = False
        self.dataset = self._build_dataset(data, target, dataset_name)
        self.input_names = list(self.dataset.getInputAttributeNames("kurz"))
        self._distance: float | None = None
        self._native_distance: float | None = None
        self.explanation = MACEExplanation(
            np.array([], dtype=float),
            np.array([], dtype=float),
            self.input_names,
        )

    def _build_dataset(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        dataset_name: str,
    ) -> Any:
        flat: dict[str, pd.Series] = {}
        attributes: dict[str, Any] = {}
        has_one_hot = False
        feature_positions = {
            name: feature_idx for feature_idx, name in enumerate(self.mapper.keys())
        }
        one_hot_positions: dict[Any, int] = {}

        for col_idx, column in enumerate(data.columns):
            if isinstance(column, tuple):
                name = column[0]
            else:
                name = column
            feature = self.mapper[name]
            base_long = str(name)
            base_kurz = f"x{feature_positions[name]}"
            series = data.iloc[:, col_idx]

            if feature.is_one_hot_encoded:
                has_one_hot = True
                code_idx = one_hot_positions.get(name, 0)
                one_hot_positions[name] = code_idx + 1
                attr_name_long = f"{base_long}_cat_{code_idx}"
                attr_name_kurz = f"{base_kurz}_cat_{code_idx}"
                cast_series = series.astype(int)
                flat[attr_name_long] = cast_series
                attributes[attr_name_long] = self._mace_load_data.DatasetAttribute(
                    attr_name_long=attr_name_long,
                    attr_name_kurz=attr_name_kurz,
                    attr_type="sub-categorical",
                    node_type="input",
                    actionability="any",
                    mutability=True,
                    parent_name_long=base_long,
                    parent_name_kurz=base_kurz,
                    lower_bound=int(cast_series.min()),
                    upper_bound=int(cast_series.max()),
                )
                continue

            if feature.is_binary:
                attr_type = "binary"
                cast_series = series.astype(int)
                lower_bound = int(cast_series.min())
                upper_bound = int(cast_series.max())
            elif feature.is_discrete:
                attr_type = "numeric-int"
                cast_series = series.round().astype(int)
                lower_bound = int(cast_series.min())
                upper_bound = int(cast_series.max())
            else:
                attr_type = "numeric-real"
                cast_series = series.astype(float)
                lower_bound = float(cast_series.min())
                upper_bound = float(cast_series.max())

            flat[base_long] = cast_series
            attributes[base_long] = self._mace_load_data.DatasetAttribute(
                attr_name_long=base_long,
                attr_name_kurz=base_kurz,
                attr_type=attr_type,
                node_type="input",
                actionability="any",
                mutability=True,
                parent_name_long=-1,
                parent_name_kurz=-1,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        label_name = "Label"
        flat[label_name] = target.astype(int)
        attributes[label_name] = self._mace_load_data.DatasetAttribute(
            attr_name_long=label_name,
            attr_name_kurz="y",
            attr_type="binary",
            node_type="output",
            actionability="none",
            mutability=False,
            parent_name_long=-1,
            parent_name_kurz=-1,
            lower_bound=int(target.min()),
            upper_bound=int(target.max()),
        )

        data_frame = pd.DataFrame(flat, index=data.index)
        return self._mace_load_data.Dataset(
            data_frame=data_frame,
            attributes=attributes,
            is_one_hot=has_one_hot,
            dataset_name=dataset_name,
        )

    def _norm_to_string(self, norm: int) -> str:
        mapping = {
            0: "zero_norm",
            1: "one_norm",
            2: "two_norm",
        }
        if norm not in mapping:
            raise ValueError(f"Unsupported norm for MACEExplainer: {norm}")
        return mapping[norm]

    def _cast_factual_value(self, name: str, value: float) -> int | float:
        attr_type = self.dataset.attributes_kurz[name].attr_type
        if attr_type == "numeric-real":
            return float(value)
        return int(round(float(value)))

    def _build_factual_sample(self, query: np.ndarray) -> dict[str, Any]:
        factual_sample = {
            name: self._cast_factual_value(name, value)
            for name, value in zip(self.input_names, query, strict=False)
        }
        factual_sample["y"] = bool(int(self.model.predict([query])[0]))
        return factual_sample

    def get_distance(self) -> float:
        if self._distance is None:
            raise RuntimeError("No explanation has been computed yet.")
        return self._distance

    def get_native_distance(self) -> float:
        if self._native_distance is None:
            raise RuntimeError("No explanation has been computed yet.")
        return self._native_distance

    def _compute_ocean_distance(
        self,
        query: np.ndarray,
        counterfactual: np.ndarray,
        *,
        norm: int,
    ) -> float:
        distance = 0.0
        for name, feature in self.mapper.items():
            if feature.is_one_hot_encoded:
                feature_distance = 0.0
                for code in feature.codes:
                    idx = self.mapper.idx.get(name, code)
                    delta = float(counterfactual[idx]) - float(query[idx])
                    feature_distance += abs(delta) ** norm
                distance += feature_distance / 2.0
            else:
                idx = self.mapper.idx.get(name)
                delta = float(counterfactual[idx]) - float(query[idx])
                distance += abs(delta) ** norm
        if norm != 1:
            distance **= 1.0 / norm
        return float(distance)

    def _compute_search_upper_bound(self, query: np.ndarray, *, norm: int) -> float:
        feature_bounds: list[float] = []
        handled_ohe: set[Any] = set()

        for name, feature in self.mapper.items():
            if feature.is_one_hot_encoded:
                if name in handled_ohe:
                    continue
                feature_bounds.append(1.0)
                handled_ohe.add(name)
                continue

            idx = self.mapper.idx.get(name)
            attr = self.dataset.attributes_kurz[f"x{list(self.mapper.keys()).index(name)}"]
            lower = float(attr.lower_bound)
            upper = float(attr.upper_bound)
            q = float(query[idx])
            feature_bounds.append(max(abs(q - lower), abs(upper - q)))

        if not feature_bounds:
            return 0.0
        if norm == 0:
            return float(len(feature_bounds))
        if norm == 1:
            return float(sum(feature_bounds))
        if norm == 2:
            return float(np.sqrt(sum(bound * bound for bound in feature_bounds)))
        if norm == np.inf:
            return float(max(feature_bounds))
        raise ValueError(f"Unsupported norm for MACEExplainer: {norm}")

    def _get_absolute_difference(self, symbol_1: Any, symbol_2: Any) -> Any:
        from pysmt.shortcuts import GE, Ite, Minus, Real, ToReal

        return Ite(
            GE(Minus(ToReal(symbol_1), ToReal(symbol_2)), Real(0)),
            Minus(ToReal(symbol_1), ToReal(symbol_2)),
            Minus(ToReal(symbol_2), ToReal(symbol_1)),
        )

    def _get_ocean_distance_formula(
        self,
        model_symbols: dict[str, Any],
        factual_sample: dict[str, Any],
        *,
        norm: int,
        norm_threshold: float,
    ) -> Any:
        from pysmt.shortcuts import And, EqualsOrIff, Ite, LE, Max, Plus, Pow, Real, ToReal

        if norm not in {0, 1, 2}:
            raise ValueError(f"Unsupported norm for MACEExplainer: {norm}")

        absolute_distances = []
        squared_distances = []

        mutable_attributes = self.dataset.getMutableAttributeNames("kurz")
        one_hot_attributes = self.dataset.getOneHotAttributesNames("kurz")
        non_hot_attributes = self.dataset.getNonHotAttributesNames("kurz")

        for attr_name_kurz in np.intersect1d(mutable_attributes, non_hot_attributes):
            diff = ToReal(
                self._get_absolute_difference(
                    model_symbols["counterfactual"][attr_name_kurz]["symbol"],
                    factual_sample[attr_name_kurz],
                )
            )
            absolute_distances.append(diff)
            squared_distances.append(Pow(diff, Real(2)))

        already_considered: set[str] = set()
        for attr_name_kurz in np.intersect1d(mutable_attributes, one_hot_attributes):
            if attr_name_kurz in already_considered:
                continue
            siblings_kurz = self.dataset.getSiblingsFor(attr_name_kurz)
            sibling_abs = [
                ToReal(
                    self._get_absolute_difference(
                        model_symbols["counterfactual"][sib]["symbol"],
                        factual_sample[sib],
                    )
                )
                for sib in siblings_kurz
            ]
            group_distance = Plus(sibling_abs) / Real(2)
            absolute_distances.append(group_distance)
            squared_distances.append(Pow(group_distance, Real(2)))
            already_considered.update(siblings_kurz)

        if not absolute_distances:
            return Real(0)

        if norm == 0:
            return LE(
                Plus(
                    [
                        Ite(EqualsOrIff(elem, Real(0)), Real(0), Real(1))
                        for elem in absolute_distances
                    ]
                ),
                Real(norm_threshold),
            )
        if norm == 1:
            return LE(Plus(absolute_distances), Real(norm_threshold))
        return LE(Plus(squared_distances), Pow(Real(norm_threshold), Real(2)))

    def _extract_samples_from_model(
        self,
        model: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        counterfactual_pysmt_sample = {}
        interventional_pysmt_sample = {}
        for symbol_key, symbol_value in model:
            tmp = str(symbol_key)
            if "counterfactual" in tmp:
                tmp = tmp[:-15]
                if tmp in self.dataset.getInputOutputAttributeNames("kurz"):
                    counterfactual_pysmt_sample[tmp] = symbol_value
            elif "interventional" in tmp:
                tmp = tmp[:-15]
                if tmp in self.dataset.getInputOutputAttributeNames("kurz"):
                    interventional_pysmt_sample[tmp] = symbol_value
            elif tmp in self.dataset.getInputOutputAttributeNames("kurz"):
                counterfactual_pysmt_sample[tmp] = symbol_value
                interventional_pysmt_sample[tmp] = symbol_value
        return (
            self._generate_sat.getDictSampleFromPySMTSample(
                counterfactual_pysmt_sample, self.dataset
            ),
            self._generate_sat.getDictSampleFromPySMTSample(
                interventional_pysmt_sample, self.dataset
            ),
        )

    def _validate_counterfactual(
        self,
        counterfactual_sample: dict[str, Any],
        factual_sample: dict[str, Any],
    ) -> tuple[bool, int]:
        vectorized_sample = [
            counterfactual_sample[attr_name_kurz]
            for attr_name_kurz in self.dataset.getInputAttributeNames("kurz")
        ]
        sklearn_prediction = int(self.model.predict([vectorized_sample])[0])
        factual_prediction = int(factual_sample["y"])
        if sklearn_prediction == factual_prediction:
            return False, sklearn_prediction
        return True, sklearn_prediction

    def _run_mace_search(
        self,
        query: np.ndarray,
        *,
        target_class: int,
        norm: int,
    ) -> dict[str, Any]:
        from pysmt.shortcuts import And, Solver, TRUE

        start_time = time.time()
        factual_sample = self._build_factual_sample(query)
        factual_pysmt_sample = self._generate_sat.getPySMTSampleFromDictSample(
            factual_sample, self.dataset
        )
        model_symbols = {
            "counterfactual": {},
            "interventional": {},
            "output": {"y": {"symbol": self._generate_sat.Symbol("y", self._generate_sat.BOOL)}},
        }

        for attr_name_kurz in self.dataset.getInputAttributeNames("kurz"):
            attr_obj = self.dataset.attributes_kurz[attr_name_kurz]
            lower_bound = attr_obj.lower_bound
            upper_bound = attr_obj.upper_bound
            if attr_obj.attr_type == "numeric-real":
                model_symbols["counterfactual"][attr_name_kurz] = {
                    "symbol": self._generate_sat.Symbol(
                        attr_name_kurz + "_counterfactual", self._generate_sat.REAL
                    ),
                    "lower_bound": self._generate_sat.Real(float(lower_bound)),
                    "upper_bound": self._generate_sat.Real(float(upper_bound)),
                }
                model_symbols["interventional"][attr_name_kurz] = {
                    "symbol": self._generate_sat.Symbol(
                        attr_name_kurz + "_interventional", self._generate_sat.REAL
                    ),
                    "lower_bound": self._generate_sat.Real(float(lower_bound)),
                    "upper_bound": self._generate_sat.Real(float(upper_bound)),
                }
            else:
                model_symbols["counterfactual"][attr_name_kurz] = {
                    "symbol": self._generate_sat.Symbol(
                        attr_name_kurz + "_counterfactual", self._generate_sat.INT
                    ),
                    "lower_bound": self._generate_sat.Int(int(lower_bound)),
                    "upper_bound": self._generate_sat.Int(int(upper_bound)),
                }
                model_symbols["interventional"][attr_name_kurz] = {
                    "symbol": self._generate_sat.Symbol(
                        attr_name_kurz + "_interventional", self._generate_sat.INT
                    ),
                    "lower_bound": self._generate_sat.Int(int(lower_bound)),
                    "upper_bound": self._generate_sat.Int(int(upper_bound)),
                }

        model_formula = self._generate_sat.getModelFormula(model_symbols, self.model)
        counterfactual_formula = self._generate_sat.EqualsOrIff(
            model_symbols["output"]["y"]["symbol"],
            self._generate_sat.Bool(bool(target_class)),
        )
        plausibility_formula = self._generate_sat.getPlausibilityFormula(
            model_symbols, self.dataset, factual_pysmt_sample, "mace"
        )

        norm_lower_bound = 0.0
        norm_upper_bound = self._compute_search_upper_bound(query, norm=norm)
        curr_norm_threshold = (norm_lower_bound + norm_upper_bound) / 2.0
        distance_formula = self._get_ocean_distance_formula(
            model_symbols,
            factual_pysmt_sample,
            norm=norm,
            norm_threshold=curr_norm_threshold,
        )
        diversity_formula = TRUE()

        counterfactuals = [
            {
                "counterfactual_sample": {},
                "counterfactual_distance": np.inf,
                "time": np.inf,
            }
        ]

        iters = 1
        max_iters = 100
        while iters < max_iters and norm_upper_bound - norm_lower_bound >= self.epsilon:
            iters += 1
            formula = And(
                model_formula,
                counterfactual_formula,
                plausibility_formula,
                distance_formula,
                diversity_formula,
            )
            with Solver(name="z3") as solver:
                solver.add_assertion(formula)
                iter_start = time.time()
                solved = solver.solve()
                iter_end = time.time()
                if solved:
                    model = solver.get_model()
                    counterfactual_sample, _ = self._extract_samples_from_model(model)
                    is_valid, sklearn_prediction = self._validate_counterfactual(
                        counterfactual_sample, factual_sample
                    )
                    if not is_valid:
                        diversity_formula = And(
                            diversity_formula,
                            self._generate_sat.getDiversityFormulaUpdate(model),
                        )
                        continue

                    counterfactual_sample["y"] = bool(sklearn_prediction)
                    cf = np.array(
                        [
                            float(counterfactual_sample[name])
                            for name in self.input_names
                        ],
                        dtype=float,
                    )
                    counterfactual_distance = self._compute_ocean_distance(
                        query, cf, norm=norm
                    )
                    counterfactuals.append(
                        {
                            "counterfactual_sample": counterfactual_sample,
                            "counterfactual_distance": counterfactual_distance,
                            "time": iter_end - iter_start,
                        }
                    )
                    norm_upper_bound = float(counterfactual_distance + self.epsilon / 100)
                else:
                    norm_lower_bound = curr_norm_threshold

                curr_norm_threshold = (norm_lower_bound + norm_upper_bound) / 2.0
                distance_formula = self._get_ocean_distance_formula(
                    model_symbols,
                    factual_pysmt_sample,
                    norm=norm,
                    norm_threshold=curr_norm_threshold,
                )

        closest_counterfactual_sample = min(
            counterfactuals, key=lambda x: x["counterfactual_distance"]
        )
        end_time = time.time()
        return {
            "cfe_found": np.isfinite(
                float(closest_counterfactual_sample["counterfactual_distance"])
            ),
            "cfe_plausible": np.isfinite(
                float(closest_counterfactual_sample["counterfactual_distance"])
            ),
            "cfe_time": end_time - start_time,
            "cfe_sample": closest_counterfactual_sample["counterfactual_sample"],
            "cfe_distance": closest_counterfactual_sample["counterfactual_distance"],
        }

    def get_solving_status(self) -> str:
        return self.Status

    def cleanup(self) -> None:
        return None

    def explain(
        self,
        x: np.ndarray,
        *,
        y: int,
        norm: int,
        return_callback: bool = False,
        verbose: bool = False,  # noqa: ARG002
        max_time: int = 60,
        num_workers: int | None = None,
        random_seed: int = 42,  # noqa: ARG002
        clean_up: bool = True,
    ) -> MACEExplanation | None:
        if num_workers not in (None, 1) and not self._warned_num_workers:
            warnings.warn(
                "MACEExplainer does not currently expose thread control; ignoring num_workers.",
                category=UserWarning,
                stacklevel=2,
            )
            self._warned_num_workers = True

        query = np.asarray(x, dtype=float).flatten()
        if self.model.n_classes_ != 2:
            raise ValueError("MACEExplainer currently supports binary classification only.")

        current_class = int(self.model.predict([query])[0])
        if int(y) == current_class:
            self.Status = "INVALID_TARGET"
            if clean_up:
                self.cleanup()
            return None

        self.callback = SimpleNamespace(sollist=[]) if return_callback else None
        factual_sample = self._build_factual_sample(query)
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(max(1, int(np.ceil(max_time))))
        try:
            result = self._run_mace_search(
                query,
                target_class=int(y),
                norm=norm,
            )
        except TimeoutError as exc:
            warnings.warn(str(exc), category=UserWarning, stacklevel=2)
            self.Status = "TIME_LIMIT"
            self._distance = None
            self._native_distance = None
            if clean_up:
                self.cleanup()
            return None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

        cf_sample = result.get("cfe_sample", {})
        distance = float(result.get("cfe_distance", np.inf))
        if not cf_sample or not np.isfinite(distance):
            self.Status = "INFEASIBLE"
            self._distance = None
            self._native_distance = None
            if clean_up:
                self.cleanup()
            return None

        cf = np.array([float(cf_sample[name]) for name in self.input_names], dtype=float)
        self.explanation = MACEExplanation(cf, query, self.input_names)
        self._native_distance = distance
        self._distance = self._compute_ocean_distance(query, cf, norm=norm)
        self.Status = "OPTIMAL"
        if self.callback is not None:
            self.callback.sollist.append(
                {
                    "objective_value": self._distance,
                    "time": float(result.get("cfe_time", 0.0)),
                }
            )
        if clean_up:
            self.cleanup()
        return self.explanation
