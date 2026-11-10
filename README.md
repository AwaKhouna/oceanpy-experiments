# oceanpy-experiments

This repository contains the supplementary code and results for the paper "Optimal Counterfactual Search in Tree Ensembles: A Study Across Modeling and Solution Paradigms". It
compares exact methods for counterfactual explanations on tree ensembles.

## Code locations

- `OCEAN/ocean/cp/`: CPCF, our constraint programming method.
- `OCEAN/ocean/mip/`: OCEAN, the mixed-integer programming method.
- `OCEAN/ocean/maxsat/`: MaxSAT baseline.
- `mace/`: MACE baseline.
- `datasets/`: preprocessed CSV datasets used by the experiments.
- `run_experiment.py`: main RF and XGBoost experiment runner.
- `norms_experiment.py`: experiments with L1, L2, and L-infinity norms.
- `isolation_experiment.py`: experiments with and without Isolation Forest
  plausibility constraints.
- `main_paper_plots.py`, `plot_results.py`, and `make_isolation_table.py`:
  scripts that rebuild figures and tables from saved JSON results.

## Environment

Use Python 3.12 or 3.13. A full reproduction needs the external solvers used by
the methods: Gurobi for OCEAN/MIP, OR-Tools for CPCF, PySAT for MaxSAT, and
PySMT with Z3 for MACE.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
SETUPTOOLS_SCM_PRETEND_VERSION=2.0.7 python -m pip install -e OCEAN
pysmt-install --z3 --confirm-agreement
```

Install and activate a Gurobi license before running the MIP/OCEAN experiments.
The editable `OCEAN` install is important because it uses the local CP, MIP, and
MaxSAT code shipped in this repository.

## Rebuild the paper figures from saved results

The repository already contains the JSON results under `results/`. To rebuild
the main paper plots, run:

```bash
python main_paper_plots.py --error-bar=quartiles --norm-boxplot-methods cp mip
```

This writes the aggregate figures to `plots/rf/paper/`, including the cactus
plot, anytime plot, depth scaling plot, estimator scaling plot, and norm
boxplot.

To rebuild the per-dataset diagnostic plots for RF and XGBoost, run:

```bash
python plot_results.py
```

To rebuild the Isolation Forest table, run:

```bash
python make_isolation_table.py
```

This writes `plots/isolation_table.csv` and `plots/isolation_table.tex`.

## Rerun the main experiments

The main runner uses one 1-indexed experiment id per independent job. Each id
fixes one dataset, seed, method, voting mode, and scaling setup.

RF experiments:

```bash
for i in $(seq 1 800); do
  python run_experiment.py --model-type rf --experiment "$i" --threads 8
done
```

XGBoost experiments:

```bash
for i in $(seq 1 300); do
  python run_experiment.py --model-type xgb --experiment "$i" --threads 8
done
```

These jobs write JSON files to `results/rf/` and `results/xgb/`. Existing
complete method results are skipped. Use `--force` to rerun an experiment, or
`--overwrite` to replace the saved results for that method.

The experiment grid is defined in `parameters.py`:

- datasets: the 10 names in `DATASETS`;
- seeds: `2, 31, 73, 127, 179`;
- methods: `cp`, `mip`, `mace`, `maxsat` for RF;
- methods: `cp`, `mip`, `maxsat` for XGBoost;
- RF voting: `SOFT` and `HARD`;
- XGBoost voting: `SOFT` only;
- setup 1: `n_estimators = 100` and `max_depth` in `3, 4, 5, 6, 7, 8`;
- setup 2: `max_depth = 5` and `n_estimators` in
  `10, 20, 50, 100, 200, 500`.

Each result file is named as
`results/<model_type>/exp_<Dataset>_<n_estimators>_<max_depth>_<seed>.json`.
RF hard-voting files end with `_HARD.json`.

## Rerun the extra experiments

Norm experiments:

```bash
python norms_experiment.py --threads 8
```

This writes `results/rf/multinorms/multinorms_<id>.json`.

Isolation Forest experiments:

```bash
python isolation_experiment.py --threads 8
python make_isolation_table.py
```

This writes `results/rf/isolation/isolation_<id>.json` and rebuilds the table.

## Notes

The default timeout is 900 seconds per explanation and each standard experiment
uses 50 sampled queries. Full reproduction can take a long time, so the numeric
experiment ids are designed for cluster or batch execution. Runtime values can
change across machines, but the saved objectives, statuses, counterfactuals,
and plots are produced by the commands above.
