import copy
import pytest
from make_isolation_table import build_table, mean_distance_ratio
from actionability_metrics import SCHEMA_VERSION


def result(dataset='Adult'):
    rows = []
    for base, constrained in [(1, 2), (10, 30)]:
        r = {}
        for m in ('cp', 'mip'):
            for suffix, distance in [('plain', base), ('iso', constrained)]:
                r.update({f'{m}_{suffix}_cf': [1], f'{m}_{suffix}_valid': True,
                          f'{m}_{suffix}_plausible': suffix == 'iso',
                          f'{m}_{suffix}_objective': distance, f'{m}_{suffix}_time': 2})
        rows.append(r)
    return dict(dataset=dataset, seed=2, voting='SOFT', explanations=rows,
                **{f'{m}_{s}_build_time': 3 for m in ('cp', 'mip') for s in ('plain', 'iso')})


def test_ac_times_replace_only_shared_baselines():
    iso = result()
    other = result('Seeds')
    ac = copy.deepcopy(iso)
    ac.update(complete=True, schema_version=SCHEMA_VERSION)
    for m in ('cp', 'mip'):
        ac[f'{m}_plain_build_time'] = 7
        for r in ac['explanations']:
            r[f'{m}_plain_time'] = 10
            r[f'{m}_plain_objective'] = 999
    table = build_table([iso, other], False, actionability_entries=[ac])
    for m in ('CP', 'MIP'):
        assert table.loc['AD', (m + '-noIF', 'T (s)')] == 17
        assert table.loc['AD', (m + '-IF', 'T (s)')] == 5
        assert table.loc['SE', (m + '-noIF', 'T (s)')] == 5
        assert table.loc['AD', (m + '-IF', 'D ratio')] == 2.5
        assert table.loc['AD', (m + '-noIF', 'D ratio')] == 1
    with pytest.raises(ValueError, match='Expected one matching actionability'):
        build_table([iso], False, actionability_entries=[])


@pytest.mark.parametrize('key,value', [('cp_plain_objective', 0),
    ('cp_plain_cf', None), ('cp_iso_valid', False), ('cp_iso_plausible', False),
    ('cp_iso_objective', float('nan'))])
def test_ratio_excludes_invalid_pairs(key, value):
    data = result()
    data['explanations'][0][key] = value
    assert mean_distance_ratio([data], 'cp_iso') == 3
    assert mean_distance_ratio([data], 'cp_plain') == 1
