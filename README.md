# oceanpy-experiments

This repository contains the supplementary code and results for the paper "Optimal Counterfactual Search in Tree Ensembles: A Study Across Modeling and Solution Paradigms". It compares exact methods for counterfactual explanations on tree ensembles.

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
- `actionability_experiment.py`: paired CP/MIP experiments with and without non-decreasing age and fixed sex.
- `main_paper_plots.py`, `plot_results.py`, and `make_isolation_table.py`:
  scripts that rebuild figures and tables from saved JSON results.

## Environment

Use Python 3.12 or 3.13. A full reproduction needs the external solvers used by the methods: Gurobi for OCEAN/MIP, OR-Tools for CPCF, PySAT for MaxSAT, and
PySMT with Z3 for MACE.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
SETUPTOOLS_SCM_PRETEND_VERSION=2.0.7 python -m pip install -e OCEAN
pysmt-install --z3 --confirm-agreement
```

Install and activate a Gurobi license before running the MIP/OCEAN experiments. The editable `OCEAN` install is important because it uses the local CP, MIP, and MaxSAT code shipped in this repository.

## Rebuild the paper figures from saved results

The repository already contains the JSON results under `results/`. To rebuild the main paper plots, run:

```bash
python main_paper_plots.py --error-bar=quartiles --norm-boxplot-methods cp mip
```

This writes the aggregate figures to `plots/rf/paper/`, including the cactus plot, anytime plot, depth scaling plot, estimator scaling plot, and norm boxplot.

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

The main runner uses one 1-indexed experiment id per independent job. Each id fixes one dataset, seed, method, voting mode, and scaling setup.

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

These jobs write JSON files to `results/rf/` and `results/xgb/`. Existing complete method results are skipped. Use `--force` to rerun an experiment, or `--overwrite` to replace the saved results for that method.

The experiment grid is defined in `parameters.py`:

- datasets: the 10 names in `DATASETS`;
- seeds: `2, 31, 73, 127, 179`;
- methods: `cp`, `mip`, `mace`, `maxsat` for RF;
- methods: `cp`, `mip`, `maxsat` for XGBoost;
- RF voting: `SOFT` and `HARD`;
- XGBoost voting: `SOFT` only;
- setup 1: `n_estimators = 100` and `max_depth` in `3, 4, 5, 6, 7, 8`;
- setup 2: `max_depth = 5` and `n_estimators` in `10, 20, 50, 100, 200, 500`.

Each result file is named as `results/<model_type>/exp_<Dataset>_<n_estimators>_<max_depth>_<seed>.json`. RF hard-voting files end with `_HARD.json`.

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

Actionability experiments compare CP and MIP with and without feature constraints using the same queries and targets. The setup uses 100 trees, maximum depth 5, L1 distance, and the same sampling defaults as isolation.

```bash
python actionability_experiment.py --threads 8
python make_actionability_table.py
```

Following [Parmentier and Vidal (2021), Section 4.1](https://arxiv.org/pdf/2106.06631), age cannot decrease and sex stays fixed. These rules are passed to each solver through `explain(..., actionability=...)`. Only datasets with these features are used:

| ID | Dataset | Non-decreasing feature | Fixed feature |
| --- | --- | --- | --- |
| 1 | Adult | Age | Sex |
| 2 | COMPAS | AgeGroup | Sex |
| 3 | CreditCard | AgeGroup | isMale |
| 4 | GermanCredit | Age | Sex (entire one-hot group) |
| 5 | StudentsPerformance | age | sex |

Both actionability variants use `scale=True`: continuous features are scaled to `[-0.5, 0.5]` using the dataset minimum and maximum. Discrete features, including age and age groups, and categorical encodings remain unchanged. The main, norms, and isolation experiments use `scale=False`.

Use `--experiment` and `--seed` to select a dataset and seed:

```bash
python actionability_experiment.py --experiment=4 --seed 31 --threads 8
```

Each ID runs one dataset and writes its own result file, so IDs 1–5 can run as
independent parallel jobs. For example, launch all five locally with one solver
thread per job:

```bash
for i in 1 2 3 4 5; do
  python actionability_experiment.py --experiment="$i" --threads 1 &
done
wait
python make_actionability_table.py
```

Results are saved to `results/rf/actionability/actionability_<id>.json`. They include settings and scaling, queries, counterfactuals, distances, timings, solver statuses, bounds and gaps, callbacks, constraint checks, and paired ratios with exclusion counts. Matching completed runs are skipped; use `--overwrite` to replace them.

The table is saved to `plots/actionability_table.csv` and `.tex`, with CP/MIP columns for runs with and without actionability:

- **D ratio:** mean of per-query `d_with / d_without` for valid pairs with a positive baseline distance. Unconstrained columns equal 1. Missing or invalid pairs are excluded. Use `--optimal-only` to include only proven-optimal pairs; otherwise time-limited feasible solutions are included.
- **T (s):** median of solve time plus explainer build time across all queries, following the isolation table convention.

The table defaults to soft voting; use `--voting HARD` for hard voting. Seeds are pooled. Keep runs with different solver budgets or configurations in separate `--results-dir` directories.

## Notes

The default timeout is 900 seconds per explanation and each standard experiment uses 50 sampled queries. Full reproduction can take a long time, so the numeric experiment ids are designed for cluster or batch execution. Runtime values can change across machines, but the saved objectives, statuses, counterfactuals, and plots are produced by the commands above.
