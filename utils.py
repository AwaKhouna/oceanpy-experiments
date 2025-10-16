from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from ocean.feature import parse_features
from ocean.typing import BaseExplainableEnsemble
from typing import Dict, List, Literal
from xgboost import XGBClassifier

URL = "https://github.com/eminyous/ocean-datasets/blob/main"


def parse_dataset(
    dataset: str,
    scale: bool = True,
    return_mapper: bool = False,
) -> pd.DataFrame:
    """
    Parse the dataset from the given path and return it as a pandas DataFrame.

    Args:
        dataset_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Parsed dataset.
    """
    # path = "datasets"
    dataset_path = f"{URL}/{dataset}/{dataset}.csv?raw=true"
    # dataset_path = f"{path}/{dataset}.csv"
    data = pd.read_csv(dataset_path, header=[0, 1])
    types: pd.Index[str] = data.columns.get_level_values(1)
    columns: pd.Index[str] = data.columns.get_level_values(0)
    data.columns = columns
    targets = data.columns[types == "T"].to_list()
    y = data[targets].iloc[:, 0].astype(int)
    discretes = tuple(data.columns[types == "D"].to_list())
    encoded = tuple(data.columns[types == "E"].to_list())
    data = data.drop(columns=targets)
    data, mapper = parse_features(
        data,
        discretes=discretes,
        encoded=encoded,
        scale=scale,
    )

    if return_mapper:
        return (data, y), mapper
    return data, y


def train_model(
    data: pd.DataFrame,
    y: pd.Series,
    model_type: Literal["rf", "xgb"] = "rf",
    n_estimators: int = 100,
    max_depth: int = None,
    seed: int = 42,
    return_accuracy: bool = False,
) -> BaseExplainableEnsemble | tuple[BaseExplainableEnsemble, float]:
    """
    Train an ensemble on the given dataset.

    Args:
        data (pd.DataFrame): Dataset to train the model on.
        y (pd.Series): Target values.
        seed (int): Random seed for reproducibility.

    Returns:
        BaseExplainableEnsemble: Trained model.
    """
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        )
    elif model_type == "xgb":
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        )
    model.fit(data, y)

    if return_accuracy:
        predictions = model.predict(data)
        accuracy = accuracy_score(y, predictions)
        return model, accuracy

    return model


def get_split_levels(model: BaseExplainableEnsemble) -> Dict[str, int]:
    """
    Get the split levels of the trees in the ensemble model.
    Args:
        model (BaseExplainableEnsemble): Trained ensemble model.
    Returns:
        Dict[str, int]: Dictionary of split levels for each feature.
    """
    if isinstance(model, RandomForestClassifier):
        return get_split_levels_rf(model)
    elif isinstance(model, XGBClassifier):
        return get_split_levels_xgb(model)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_split_levels_rf(model: RandomForestClassifier) -> Dict[str, int]:
    """
    Get the split levels of the trees in the random forest model.

    Args:
        model (RandomForestClassifier): Trained random forest model.

    Returns:
        List[]: List of split levels for each tree.
    """
    splits = {}
    for clf in model.estimators_:
        n_nodes = clf.tree_.node_count
        feature = clf.tree_.feature  # Array of feature indices for each node
        threshold = clf.tree_.threshold  # Array of thresholds for each node

        for node_idx in range(n_nodes):
            # Ignore leaf nodes (feature index == -2)
            if feature[node_idx] != -2:
                feat_idx = feature[node_idx]
                thresh = threshold[node_idx]
                if feat_idx not in splits:
                    splits[feat_idx] = set()  # Use a set to store unique thresholds
                splits[feat_idx].add(thresh)

    # Convert sets to counts
    splits = {str(feat): int(len(thresh_set)) for feat, thresh_set in splits.items()}
    return splits


def get_split_levels_xgb(model: XGBClassifier) -> Dict[str, int]:
    """
    Count the number of unique split conditions per feature across all trees
    in an XGBClassifier. For numeric splits we count unique thresholds; for
    categorical splits (if present) we count distinct split nodes.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost classifier.

    Returns:
        Dict[str, int]: {feature_name: count_of_unique_splits}
    """
    booster = model.get_booster()
    df = booster.trees_to_dataframe()

    if df.empty:
        return {}

    # Keep only decision (non-leaf) nodes
    df_nonleaf = df[df["Feature"] != "Leaf"].copy()
    if df_nonleaf.empty:
        return {}

    counts: Dict[str, int] = {}

    # Newer XGBoost adds "Decision Type" (e.g., "<=" for numeric, "==" for categorical)
    if "Decision Type" in df_nonleaf.columns:
        # Numeric splits: count unique thresholds per feature
        num_mask = df_nonleaf["Decision Type"].astype(str).str.contains("<=")
        if num_mask.any():
            num_counts = df_nonleaf[num_mask].groupby("Feature")["Split"].nunique()
            counts.update({str(k): int(v) for k, v in num_counts.to_dict().items()})

        # Categorical splits (if used): count unique nodes per feature
        cat_mask = df_nonleaf["Decision Type"].astype(str).str.contains("==")
        if cat_mask.any():
            cat_counts = df_nonleaf[cat_mask].groupby("Feature")["Node"].nunique()
            for k, v in cat_counts.to_dict().items():
                counts[str(k)] = int(counts.get(str(k), 0) + int(v))
    else:
        # Fallback for older versions: use presence/absence of "Split" values
        if "Split" in df_nonleaf.columns:
            num_counts = (
                df_nonleaf[df_nonleaf["Split"].notna()]
                .groupby("Feature")["Split"]
                .nunique()
                .to_dict()
            )
            counts.update({str(k): int(v) for k, v in num_counts.items()})

        other_counts = (
            df_nonleaf[df_nonleaf["Split"].isna()]
            .groupby("Feature")["Node"]
            .nunique()
            .to_dict()
        )
        for k, v in other_counts.items():
            counts[str(k)] = int(counts.get(str(k), 0) + int(v))

    return counts


def get_node_count(model: BaseExplainableEnsemble) -> List[int]:
    """
    Get the number of nodes in each tree of the ensemble model.

    Args:
        model (BaseExplainableEnsemble): Trained ensemble model.

    Returns:
        List[int]: List of node counts for each tree.
    """
    if isinstance(model, RandomForestClassifier):
        return get_node_count_rf(model)
    elif isinstance(model, XGBClassifier):
        return get_node_count_xgb(model)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_node_count_rf(model: RandomForestClassifier) -> List[int]:
    """
    Get the number of nodes in each tree of the random forest model.

    Args:
        model (RandomForestClassifier): Trained random forest model.

    Returns:
        List[int]: List of node counts for each tree.
    """
    return [clf.tree_.node_count for clf in model.estimators_]


def get_node_count_xgb(model: XGBClassifier) -> List[int]:
    """
    Get the total number of nodes in each tree of an XGBClassifier.

    Args:
        model (xgb.XGBClassifier): Trained XGBoost classifier.

    Returns:
        List[int]: Node counts for each tree (includes decision + leaf nodes).
    """
    booster = model.get_booster()
    df = booster.trees_to_dataframe()
    if df.empty:
        return []
    return df.groupby("Tree").size().astype(int).tolist()


def test_functions():
    """
    Test the train_model function with a sample dataset.
    """
    # Sample dataset
    dataset = "COMPAS"
    (data, y), mapper = parse_dataset(dataset, return_mapper=True)
    print("Parsed DataFrame:")
    print(data.head())
    print("Target values:")
    print(y.head())
    print("Feature Mapper:")
    print(mapper)

    # Train the model
    model, accuracy = train_model(
        data, y, n_estimators=10, max_depth=3, return_accuracy=True, seed=42
    )

    # Print the model
    print("Accuracy:", accuracy)
    print("Trained Model:")
    print(model)


# if __name__ == "__main__":
#    test_functions()
