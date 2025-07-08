DATASETS = ["COMPAS", "Adult", "Credit"]
SEEDS = [2, 42, 123, 456, 789]
MODELS = ["cp", "mip"]
N_ESTIMATORS = [10, 50, 100, 200, 500, 1000]
MAX_DEPTHS = [5, 10, 20, None]
TIMEOUT = 3000  # seconds allowed for each explainer by explanation
N_SAMPLES = 100  # number of queries to sample per dataset
