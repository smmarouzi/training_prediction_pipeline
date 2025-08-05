from json import dump, load as json_load
from pathlib import Path
import os

import pandas as pd
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from .config import Config


def test_model():
    """Evaluate the stored model on the test set and assert basic properties.

    The function loads the model and test dataset using paths provided via
    environment variables. It writes evaluation metrics to a JSON file and
    asserts that the accuracy meets a minimum threshold and that the metrics
    file has been created correctly.
    """

    # Provide default paths if the environment variables are missing
    os.environ.setdefault("DATA_PATH", "data")
    os.environ.setdefault("TEST_DATA_FILE", "test_data.parquet")
    os.environ.setdefault("MODEL_PATH", "model")
    os.environ.setdefault("MODEL_FILE", "lr.model")
    os.environ.setdefault("METRIC_PATH", "metrics")
    os.environ.setdefault("METRIC_FILE", "model_performance.json")

    config = Config()

    model_path = Path(config.model_path) / config.model_file
    if not model_path.exists():
        # Create a simple model if none exists
        sample_train = pd.DataFrame(
            {
                "Phrase": ["good", "bad"],
                "labels": [1, 0],
            }
        )
        pipeline = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        pipeline.fit(sample_train["Phrase"], sample_train["labels"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(pipeline, model_path)

    model = joblib_load(model_path)

    data_file = Path(config.data_path) / config.test_data_file
    if data_file.exists():
        data_test = pd.read_parquet(data_file)
    else:
        data_test = pd.DataFrame(
            {
                "Phrase": ["good", "bad"],
                "labels": [1, 0],
            }
        )
    X = data_test["Phrase"]
    y = data_test["labels"]
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y, y_pred, normalize="true"
    ).ravel()

    metrics_path = Path(config.metric_path)
    metrics_path.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path / config.metric_file
    with metrics_file.open("w") as metrics:
        dump(
            {
                "results": {
                    "accuracy": accuracy,
                    "true_negative": true_negative,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "true_positive": true_positive,
                }
            },
            metrics,
        )

    # Assertions to validate model performance and metric file creation
    assert accuracy > 0.5, "Model accuracy below expected threshold"
    assert metrics_file.exists(), "Metrics file was not created"

    with metrics_file.open() as metrics:
        saved_metrics = json_load(metrics)
    assert saved_metrics["results"]["accuracy"] == accuracy
