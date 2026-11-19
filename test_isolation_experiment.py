"""Regression coverage for feature-state contamination between variants."""
import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from ocean.feature import parse_features
from isolation_experiment import close_explainer, make_explainer


def fingerprint(explainer, method):
    if method == 'mip':
        explainer.update()
        return explainer.Fingerprint
    return hashlib.sha256(explainer.Proto().SerializeToString(deterministic=True)).hexdigest()


@pytest.mark.parametrize('method', ['cp', 'mip'])
def test_isolation_build_does_not_change_later_plain_model(method):
    data = pd.DataFrame({'amount': np.arange(20, dtype=float)})
    X, mapper = parse_features(data, scale=False)
    forest = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=2).fit(X, (data.amount >= 10).astype(int))
    isolation = IsolationForest(n_estimators=3, max_samples=10, random_state=2).fit(X)
    initial_levels = mapper['amount'].levels.copy()
    explainers = []
    try:
        before, _ = make_explainer(method, model=forest, mapper=mapper,
                                  isolation=None, isolation_threshold=None)
        explainers.append(before)
        expected = fingerprint(before, method)
        with_if, _ = make_explainer('cp', model=forest, mapper=mapper,
                                   isolation=isolation, isolation_threshold=0.5)
        explainers.append(with_if)
        after, _ = make_explainer(method, model=forest, mapper=mapper,
                                 isolation=None, isolation_threshold=None)
        explainers.append(after)
        assert len(with_if.mapper['amount'].levels) > len(initial_levels)
        np.testing.assert_array_equal(mapper['amount'].levels, initial_levels)
        assert fingerprint(after, method) == expected
    finally:
        for explainer in explainers:
            close_explainer(explainer)


def test_plain_runner_builds_once_and_handles_both_targets(monkeypatch, tmp_path):
    import isolation_experiment as experiment

    data = pd.DataFrame({'amount': np.arange(20, dtype=float)})
    X, mapper = parse_features(data, scale=False)
    y = (data.amount >= 10).astype(int)
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'results_path', lambda _: tmp_path / 'smoke.json')
    monkeypatch.setattr(experiment, 'load_dataset', lambda _: ((X, y), mapper))
    monkeypatch.setattr(experiment, 'RF_N_ESTIMATORS', 3)
    monkeypatch.setattr(experiment, 'RF_MAX_DEPTH', 2)
    monkeypatch.setattr(experiment, 'N_ISOLATORS', 3)
    monkeypatch.setattr(experiment, 'sample_queries', lambda *a, **k: [
        (np.array([0.]), 1), (np.array([19.]), 0)])
    builds = []
    make = experiment.make_explainer

    def record_build(method, **kwargs):
        builds.append((method, kwargs['isolation'] is not None))
        return make(method, **kwargs)

    monkeypatch.setattr(experiment, 'make_explainer', record_build)
    saved = []
    monkeypatch.setattr(experiment, 'save_entry', lambda path, result, **k: saved.append(result))
    experiment.run_dataset_experiment(1, seed=2, threads=1, timeout=10,
                                     voting='SOFT', query_count=2, overwrite=True)
    assert builds == [('cp', False), ('cp', True), ('cp', True),
                      ('mip', False), ('mip', True), ('mip', True)]
    for method in ('cp', 'mip'):
        variant = method + '_plain'
        assert list(saved[0][variant + '_target_build_times']) == ['shared']
        assert saved[0][variant + '_build_time'] == saved[0][variant + '_target_build_times']['shared']
        assert all(row[variant + '_valid'] for row in saved[0]['explanations'])
