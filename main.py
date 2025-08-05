"""Entry point for data preparation, model training and evaluation."""

from __future__ import annotations

import pickle
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.prepare_data.preprocessing import data_preprocessing_pipeline
from src.train.config import Data
from src.train.model import BusynessEstimation


def prepare_data(data_path: str, prepared_path: str) -> pd.DataFrame:
    """Load raw data, preprocess it and persist the result."""

    train_df = pd.read_csv(data_path)
    train_df_preprocessed = data_preprocessing_pipeline(train_df)
    train_df_preprocessed.to_csv(prepared_path, index=False)
    return train_df_preprocessed


def split_data(
    df: pd.DataFrame,
    train_path: str,
    test_path: str,
    test_size: float = Data.test_size,
    random_state: int = Data.split_random_state,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train and test datasets and save them.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split.
    train_path : str
        Destination file path for the train split.
    test_path : str
        Destination file path for the test split.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Seed used by the random number generator.
    """

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    return df_train, df_test


def train_model(train_path: str, test_path: str) -> BusynessEstimation:
    """Train the busyness estimation model."""

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    busyness_model = BusynessEstimation(test_data.copy())
    X_train = train_data.drop(Data.target, axis=1)
    y_train = train_data[Data.target]
    busyness_model.fit(X_train, y_train)
    return busyness_model


def save_model(model: BusynessEstimation, model_path: str) -> None:
    """Persist model artifact to ``model_path``."""

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": model.model_pipeline,
                "target": model.target,
                "scores_dict": model.scores,
            },
            f,
        )


def load_model(model_path: str) -> dict:
    """Load model artifact from ``model_path``."""

    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate(model_artifact: dict, test_path: str) -> float:
    """Evaluate loaded model on the test dataset."""

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(Data.target, axis=1)
    y_test = test_df[Data.target[0]]
    return model_artifact["pipeline"].score(X_test, y_test)


def main() -> None:
    """Orchestrate data preparation, training, and evaluation."""

    data_path = "assignment/final_dataset.csv"
    prepared_path = "assignment/model_data.csv"
    train_path = "assignment/train.csv"
    test_path = "assignment/test.csv"
    model_path = "assignment/rf.sav"

    df_prepared = prepare_data(data_path, prepared_path)
    split_data(
        df_prepared,
        train_path,
        test_path,
        test_size=Data.test_size,
        random_state=Data.split_random_state,
    )

    model = train_model(train_path, test_path)
    print(model.best_params)
    save_model(model, model_path)

    artifact = load_model(model_path)
    result = evaluate(artifact, test_path)
    print(artifact["target"])
    print(artifact["scores_dict"])
    print(artifact["pipeline"])
    print(result)


if __name__ == "__main__":
    main()
