DATASETS = [
    "Adult",
    "BreastCancerWisconsin",
    "COMPAS",
    "CreditCard",
    "GermanCredit",
    "OnlineNewsPopularity",
    "Phishing",
    "Seeds",
    "Spambase",
    "StudentsPerformance",
]
SEEDS = [2, 31, 73, 127, 179]
MODELS = ["cp", "mip", "mace", "maxsat"]
N_ESTIMATORS = [10, 20, 50, 100, 200, 500]
MAX_DEPTHS = [3, 4, 5, 6, 7, 8]
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 5
VOTING = ["SOFT", "HARD"]
TIMEOUT = 900  # seconds allowed for each explainer by explanation
N_SAMPLES = 50  # number of queries to sample per dataset
N_THREADS = 8  # number of threads per job
