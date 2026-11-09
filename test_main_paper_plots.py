import main_paper_plots as mpp


def make_method(
    *,
    status: str,
    objective: float | None,
    has_cf: bool = True,
) -> mpp.MethodResult:
    return mpp.MethodResult(
        status=status,
        total_time=1.0,
        objective=objective,
        has_cf=has_cf,
        callbacks=[],
        evaluated=True,
    )


def test_shared_terminal_objective_uses_max_across_methods():
    explanation = {
        "cp_status": "OPTIMAL",
        "cp_cf": [1.0],
        "cp_objective": 1.5,
        "cp_callback": [{"time": 1.0, "objective_value": 1.0}],
        "mip_status": "OPTIMAL",
        "mip_cf": [2.0],
        "mip_objective": 1.8,
        "mip_callback": [{"time": 2.0, "objective_value": 2.0}],
    }

    assert mpp.shared_terminal_objective(explanation, ("cp", "mip")) == 2.0

    callbacks = mpp.normalize_callback(
        explanation,
        "cp",
        build_time=0.0,
        final_time=3.0,
        final_objective=2.0,
    )

    assert callbacks == [(1.0, 2.0), (3.0, 2.0)]


def test_jointly_optimal_objective_requires_matching_final_objectives():
    matching = mpp.InstanceResult(
        dataset="Adult",
        model_type="rf",
        voting="SOFT",
        n_estimators=100,
        max_depth=5,
        seed=127,
        query_index=0,
        methods={
            "cp": make_method(status="OPTIMAL", objective=2.0),
            "mip": make_method(status="OPTIMAL", objective=2.0),
        },
    )
    mismatched = mpp.InstanceResult(
        dataset="Adult",
        model_type="rf",
        voting="SOFT",
        n_estimators=100,
        max_depth=5,
        seed=127,
        query_index=0,
        methods={
            "cp": make_method(status="OPTIMAL", objective=2.0),
            "mip": make_method(status="OPTIMAL", objective=2.001),
        },
    )

    assert mpp.jointly_optimal_objective(matching, ("cp", "mip")) == 2.0
    assert mpp.jointly_optimal_objective(mismatched, ("cp", "mip")) is None


def test_scaling_summary_quartiles():
    center, errors = mpp.scaling_summary([1.0, 2.0, 3.0, 4.0], "quartiles")

    assert center == 2.5
    assert errors == (0.75, 0.75)


def test_scaling_summary_mad():
    center, errors = mpp.scaling_summary([1.0, 2.0, 4.0, 8.0, 16.0], "mad")

    assert center == 4.0
    assert errors == (3.0, 3.0)


def test_scaling_error_bar_alias_and_filename():
    assert mpp.scaling_error_bars_arg("quratiles") == "quartiles"
    assert (
        mpp.scaling_filename("depth_scaling.pdf", "quartiles")
        == "depth_scaling_quartiles.pdf"
    )
