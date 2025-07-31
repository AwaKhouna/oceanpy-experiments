from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from ocean.feature import parse_features


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
    path = "datasets"
    # dataset_path = f"{URL}/{dataset}/{dataset}.csv?raw=true"
    dataset_path = f"{path}/{dataset}.csv"
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
    n_estimators: int = 100,
    max_depth: int = None,
    seed: int = 42,
    n_jobs: int = -1,
    return_accuracy: bool = False,
) -> RandomForestClassifier | tuple[RandomForestClassifier, float]:
    """
    Train a random forest model on the given dataset.

    Args:
        data (pd.DataFrame): Dataset to train the model on.
        y (pd.Series): Target values.
        seed (int): Random seed for reproducibility.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(data, y)

    if return_accuracy:
        predictions = model.predict(data)
        accuracy = accuracy_score(y, predictions)
        return model, accuracy

    return model


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
