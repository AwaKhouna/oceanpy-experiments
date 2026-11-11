"""Regression tests for paired aggregation, rule validation, and saved run identity."""
import copy
import json

import numpy as np
import pytest

from actionability_metrics import paired_ratio
from actionability_experiment import (
    ACTIONABILITY, ACTIONABILITY_DATASETS, EXPERIMENT_IDS, SCHEMA_VERSION, add_summaries,
    save_entry, same_run, validate_actionability,
)
from make_actionability_table import build_table


def pair(plain=2, actionable=4, status="OPTIMAL"):
    row = {}
    for method in ("cp", "mip"):
        for suffix, objective in (("plain", plain), ("actionable", actionable)):
            prefix = f"{method}_{suffix}"
            row.update({f"{prefix}_cf": [1], f"{prefix}_objective": objective,
                        f"{prefix}_valid": True, f"{prefix}_status": status,
                        f"{prefix}_time": 2, f"{prefix}_actionability_valid": True})
    return row


def entry(rows):
    return dict(schema_version=SCHEMA_VERSION, complete=True, dataset="Adult", seed=2,
                voting="SOFT", explanations=rows,
                **{f"{m}_{s}_build_time": 3 for m in ("cp", "mip") for s in ("plain", "actionable")})


def test_average_of_ratios_and_time():
    data = entry([pair(1, 2), pair(10, 30)])
    table = build_table([data])
    assert table.loc["AD", ("CP-AC", "D ratio")] == 2.5  # Not 32/11.
    assert table.loc["AD", ("CP-noAC", "D ratio")] == 1
    assert table.loc["AD", ("MIP-AC", "T (s)")] == 5
    add_summaries(data)
    assert data["cp_paired_summary"]["mean_distance_ratio"] == 2.5


@pytest.mark.parametrize("key,value,reason", [
    ("cp_actionable_cf", None, "actionable_missing_cf"),
    ("cp_plain_valid", False, "plain_invalid_target"),
    ("cp_actionable_actionability_valid", False, "actionability_violation"),
    ("cp_plain_objective", 0, "nonpositive_baseline_distance"),
    ("cp_plain_objective", float("nan"), "nonfinite_distance"),
])
def test_exclusions(key, value, reason):
    row = pair()
    row[key] = value
    assert paired_ratio(row, "cp") == (None, reason)


def test_optimal_filter_and_missing_times():
    data = entry([pair(), pair(1, 8, "TIME_LIMIT")])
    data["explanations"][0]["cp_actionable_time"] = None
    assert build_table([data]).loc["AD", ("CP-AC", "D ratio")] == 5
    assert build_table([data], optimal_only=True).loc["AD", ("CP-AC", "D ratio")] == 2


def test_full_one_hot_group_and_age_validation():
    query = np.array([40., 1., 0.])
    indices = {"Age": [0], "Sex": [1, 2]}
    rules = ACTIONABILITY["GermanCredit"]
    assert validate_actionability(query, np.array([41., 1., 0.]), indices, rules) == []
    assert validate_actionability(query, np.array([39., 0., 1.]), indices, rules) == ["Age", "Sex"]
    assert EXPERIMENT_IDS == (1, 2, 3, 4, 5)
    assert ACTIONABILITY_DATASETS == (
        "Adult", "COMPAS", "CreditCard", "GermanCredit", "StudentsPerformance",
    )


def test_save_replaces_matching_run_and_preserves_other_budgets(tmp_path):
    data = entry([pair()])
    data["timeout"] = 10
    path = tmp_path / "actionability_1.json"
    save_entry(path, data)
    replacement = copy.deepcopy(data)
    replacement["accuracy"] = .9
    save_entry(path, replacement)
    changed = dict(replacement, timeout=20)
    assert not same_run(replacement, changed)
    save_entry(path, changed)
    saved = json.loads(path.read_text())
    assert len(saved) == 2
    assert saved[0]["accuracy"] == .9
    with pytest.raises(ValueError, match="Mixed configurations"):
        build_table(saved)


def test_table_rejects_duplicate_seeds_and_ignores_incomplete():
    data = entry([pair()])
    assert build_table([dict(data, complete=False)]).empty
    with pytest.raises(ValueError, match="Duplicate"):
        build_table([data, data])
