"""Shared experiment policy and paired-distance metrics (no solver dependencies)."""
from __future__ import annotations

import math
from typing import Any

# Original mapper names, including full one-hot groups rather than dummy columns.
ACTIONABILITY = {
    "Adult": {"Age": "increase", "Sex": "fixed"},
    "COMPAS": {"AgeGroup": "increase", "Sex": "fixed"},
    "CreditCard": {"AgeGroup": "increase", "isMale": "fixed"},
    "GermanCredit": {"Age": "increase", "Sex": "fixed"},
    "StudentsPerformance": {"age": "increase", "sex": "fixed"},
}
SCHEMA_VERSION = 2
VALIDATION_TOLERANCE = 1e-6


def paired_ratio(explanation: dict[str, Any], method: str) -> tuple[float | None, str]:
    """Use the same method/query/target; never turn a failed solve into a ratio."""
    for suffix in ("plain", "actionable"):
        prefix = f"{method}_{suffix}"
        if explanation.get(f"{prefix}_cf") is None:
            return None, f"{suffix}_missing_cf"
        if explanation.get(f"{prefix}_valid") is not True:
            return None, f"{suffix}_invalid_target"
    if explanation.get(f"{method}_actionable_actionability_valid") is not True:
        return None, "actionability_violation"
    try:
        baseline = float(explanation[f"{method}_plain_objective"])
        constrained = float(explanation[f"{method}_actionable_objective"])
    except (KeyError, TypeError, ValueError):
        return None, "missing_distance"
    if not (math.isfinite(baseline) and math.isfinite(constrained)):
        return None, "nonfinite_distance"
    if baseline <= 0:
        return None, "nonpositive_baseline_distance"
    if constrained < 0:
        return None, "negative_actionable_distance"
    ratio = constrained / baseline
    if not math.isfinite(ratio):
        return None, "nonfinite_ratio"
    return ratio, "included"


def pair_is_optimal(explanation: dict[str, Any], method: str) -> bool:
    return all(explanation.get(f"{method}_{suffix}_status") == "OPTIMAL"
               for suffix in ("plain", "actionable"))
