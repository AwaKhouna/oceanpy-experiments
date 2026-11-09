from pathlib import Path

import pandas as pd

import meta_classifier as mc


TYPE_COUNTS = {
    "n_features_F": 2,
    "n_features_B": 2,
    "n_features_E": 2,
    "n_features_D": 2,
}


def test_collect_query_rows_prefers_fastest_optimal_with_build_time():
    entry = {
        "explanations": [
            {
                "cp_status": "OPTIMAL",
                "cp_time": 1.0,
                "cp_valid": True,
                "cp_cf": [1.0],
                "mip_status": "OPTIMAL",
                "mip_time": 0.5,
                "mip_valid": True,
                "mip_cf": [2.0],
            }
        ]
    }

    rows = mc.collect_query_rows(
        entry,
        base_row={"seed": 127},
        method_prefixes={"cp": "cp", "mip": "mip"},
        build_times={"cp": 2.0, "mip": 3.0},
        overall_best_explainer="MIP",
        expand_per_cf=True,
    )

    assert len(rows) == 1
    assert rows[0]["best_explainer"] == "CP"
    assert rows[0]["overall_best_explainer"] == "MIP"
    assert rows[0]["preferred_metric"] == mc.PER_COUNTERFACTUAL_METRIC
    assert rows[0]["query_index"] == 0


def test_collect_query_rows_uses_best_anytime_objective_without_optimal():
    entry = {
        "explanations": [
            {
                "cp_status": "TIME_LIMIT",
                "cp_time": 5.0,
                "cp_objective": 2.0,
                "cp_callback": [{"time": 1.0, "objective_value": 2.0}],
                "mip_status": "TIME_LIMIT",
                "mip_time": 3.0,
                "mip_objective": 1.5,
                "mip_callback": [
                    {"time": 2.0, "objective_value": 1.8},
                    {"time": 3.0, "objective_value": 1.5},
                ],
            }
        ]
    }

    rows = mc.collect_query_rows(
        entry,
        base_row={},
        method_prefixes={"cp": "cp", "mip": "mip"},
        build_times={"cp": 0.0, "mip": 0.0},
        overall_best_explainer="CP",
        expand_per_cf=True,
    )

    assert len(rows) == 1
    assert rows[0]["best_explainer"] == "MIP"


def test_collect_standard_entry_rows_default_emits_three_metric_rows():
    entry = {
        "dataset": "Adult",
        "seed": 127,
        "n_features": 8,
        "model_type": "rf",
        "voting": "SOFT",
        "n_estimators": 100,
        "max_depth": 5,
        "cp_build_time": 1.0,
        "mip_build_time": 2.0,
        "nodes": [15, 15],
        "split_levels": {"0": 2, "1": 4},
        "explanations": [
            {
                "cp_status": "OPTIMAL",
                "cp_time": 3.0,
                "cp_valid": True,
                "cp_cf": [1.0],
                "mip_status": "OPTIMAL",
                "mip_time": 5.0,
                "mip_valid": True,
                "mip_cf": [2.0],
            },
            {
                "cp_status": "OPTIMAL",
                "cp_time": 4.0,
                "cp_valid": True,
                "cp_cf": [3.0],
                "mip_status": "TIME_LIMIT",
                "mip_time": 6.0,
                "mip_valid": False,
                "mip_cf": None,
                "mip_callback": [{"time": 6.0, "objective_value": 9.0}],
            },
        ],
    }

    rows = mc.collect_standard_entry_rows(
        path=Path("results/rf/exp_Adult_100_5_127.json"),
        entry=entry,
        dataset="Adult",
        type_counts=TYPE_COUNTS,
        expand_per_cf=False,
    )

    assert len(rows) == len(mc.PREFERRED_METRICS)
    assert {row["preferred_metric"] for row in rows} == set(mc.PREFERRED_METRICS)
    assert all(row["query_index"] == -1 for row in rows)
    assert all(row["best_explainer"] == "CP" for row in rows)
    assert all(row["overall_best_explainer"] == "CP" for row in rows)
    assert "hard_voting_isolation" not in rows[0]
    assert "n_samples" not in rows[0]


def test_collect_standard_entry_rows_expand_per_cf_emits_one_row_per_query():
    entry = {
        "dataset": "Adult",
        "seed": 127,
        "n_features": 8,
        "model_type": "rf",
        "voting": "SOFT",
        "n_estimators": 100,
        "max_depth": 5,
        "cp_build_time": 1.0,
        "mip_build_time": 2.0,
        "nodes": [15, 15],
        "split_levels": {"0": 2, "1": 4},
        "explanations": [
            {
                "cp_status": "OPTIMAL",
                "cp_time": 3.0,
                "cp_valid": True,
                "cp_cf": [1.0],
                "mip_status": "OPTIMAL",
                "mip_time": 5.0,
                "mip_valid": True,
                "mip_cf": [2.0],
            },
            {
                "cp_status": "OPTIMAL",
                "cp_time": 4.0,
                "cp_valid": True,
                "cp_cf": [3.0],
                "mip_status": "TIME_LIMIT",
                "mip_time": 6.0,
                "mip_valid": False,
                "mip_cf": None,
                "mip_callback": [{"time": 6.0, "objective_value": 9.0}],
            },
        ],
    }

    rows = mc.collect_standard_entry_rows(
        path=Path("results/rf/exp_Adult_100_5_127.json"),
        entry=entry,
        dataset="Adult",
        type_counts=TYPE_COUNTS,
        expand_per_cf=True,
    )

    assert len(rows) == 2
    assert [row["query_index"] for row in rows] == [0, 1]
    assert all(row["preferred_metric"] == mc.PER_COUNTERFACTUAL_METRIC for row in rows)
    assert all(row["best_explainer"] == "CP" for row in rows)
    assert all(row["overall_best_explainer"] == "CP" for row in rows)


def test_collect_standard_entry_rows_metric_variants_change_winner():
    entry = {
        "dataset": "Adult",
        "seed": 127,
        "n_features": 8,
        "model_type": "rf",
        "voting": "SOFT",
        "n_estimators": 100,
        "max_depth": 5,
        "cp_build_time": 100.0,
        "mip_build_time": 0.0,
        "nodes": [15, 15],
        "split_levels": {"0": 2, "1": 4},
        "explanations": [
            {
                "cp_status": "OPTIMAL",
                "cp_time": 0.0,
                "cp_valid": True,
                "cp_cf": [1.0],
                "mip_status": "OPTIMAL",
                "mip_time": 60.0,
                "mip_valid": True,
                "mip_cf": [2.0],
            },
            {
                "cp_status": "OPTIMAL",
                "cp_time": 0.0,
                "cp_valid": True,
                "cp_cf": [3.0],
                "mip_status": "OPTIMAL",
                "mip_time": 60.0,
                "mip_valid": True,
                "mip_cf": [4.0],
            },
        ],
    }

    rows = mc.collect_standard_entry_rows(
        path=Path("results/rf/exp_Adult_100_5_127.json"),
        entry=entry,
        dataset="Adult",
        type_counts=TYPE_COUNTS,
        expand_per_cf=False,
    )

    winners = {row["preferred_metric"]: row["best_explainer"] for row in rows}
    assert winners == {
        "median+build": "MIP",
        "median+buildAmortized": "CP",
        "median": "CP",
    }


def test_prepare_decision_diagram_training_data_uses_metric_feature_and_target(
    tmp_path,
):
    data = pd.DataFrame(
        [
            {
                "dataset": "Adult",
                "best_explainer": "CP",
                "overall_best_explainer": "MIP",
                "preferred_metric": "median+build",
                "n_features": 8,
                "n_features_F": 2,
                "n_features_B": 2,
                "n_features_E": 2,
                "n_features_D": 2,
                "seed": 127,
                "query_index": -1,
                "n_estimators": 100,
                "max_depth": 5,
                "time_limit": 900.0,
                "norm": 1.0,
                "isolation": 0,
                "hard_voting": 0,
                "total_tree_nodes": 30.0,
                "mean_tree_nodes": 15.0,
                "max_tree_nodes": 15.0,
                "total_split_levels": 6.0,
                "mean_split_levels_per_feature": 3.0,
                "max_split_levels_per_feature": 4.0,
                "model_type": "rf",
                "voting_type": "SOFT",
            }
        ]
    )

    processed, feature_names, categorical_encoders, class_decoder = (
        mc.prepare_decision_diagram_training_data(
            data,
            processed_csv=tmp_path / "processed.csv",
        )
    )

    assert "overall_best_explainer" not in processed.columns
    assert "voting_type" not in feature_names
    assert "preferred_metric" in feature_names
    assert "seed" in feature_names
    assert "query_index" in feature_names
    assert feature_names == mc.DD_FEATURE_COLUMNS
    assert "preferred_metric" in categorical_encoders
    assert categorical_encoders["model_type"]["rf"] == 0
    assert class_decoder[0] == "CP"
