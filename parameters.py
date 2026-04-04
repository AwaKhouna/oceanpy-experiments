DATASETS = ["Adult", "Breast-Cancer-Wisconsin", "COMPAS", "Credit", "Seeds", "Spambase"]
SEEDS = [2, 31, 73, 127, 179]
MODELS = ["cp", "mip", "maxsat"]
N_ESTIMATORS = [10, 50, 100, 200, 500, 1000]
MAX_DEPTHS = [3, 5, 7, 9, None]
TIMEOUT = 900  # seconds allowed for each explainer by explanation
N_SAMPLES = 50  # number of queries to sample per dataset
N_THREADS = 8  # number of threads per job
