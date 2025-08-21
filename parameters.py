DATASETS = ["COMPAS", "Adult", "Credit"]
SEEDS = [2]  # , 42, 123, 456, 789]
MODELS = ["cp", "mip"]
N_ESTIMATORS = [10, 50, 100, 200, 500, 1000]
MAX_DEPTHS = [3, 5, 7, 9, None]
TIMEOUT = 1200  # seconds allowed for each explainer by explanation
N_SAMPLES = 50  # number of queries to sample per dataset
