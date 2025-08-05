import numpy as np
import pandas as pd
from src.train import model as model_module
from src.train.model import BusynessEstimation
from src.train import config


def test_busyness_estimation_predicts(monkeypatch):
    """BusynessEstimation trains and produces predictions on synthetic data."""

    monkeypatch.setattr(
        config.RF,
        "best_params",
        {"n_estimators": 5, "min_samples_leaf": 1, "max_depth": 2},
    )
    monkeypatch.setattr(model_module, "check_is_fitted", lambda estimator: None)

    X = pd.DataFrame(
        {
            "dist_to_restaurant": [0.1, 0.2, 0.3, 0.4, 0.5],
            "Hdist_to_restaurant": [1, 2, 3, 4, 5],
            "avg_Hdist_to_restaurants": [1.5, 2.5, 3.5, 4.5, 5.5],
            "date_day_number": [1, 1, 1, 1, 1],
            "restaurant_id": [0, 1, 2, 3, 4],
            "Five_Clusters_embedding": [0, 1, 2, 3, 4],
            "h3_index": [0, 1, 2, 3, 4],
            "date_hour_number": [10, 11, 12, 13, 14],
            "restaurants_per_index": [1, 2, 3, 4, 5],
        }
    )
    y = pd.Series([5, 6, 7, 8, 9], name="orders_busyness_by_h3_hour")

    test_df = pd.concat([X.iloc[3:], y.iloc[3:]], axis=1)
    model = BusynessEstimation(test_df=test_df)
    model.fit(X.iloc[:3], y.iloc[:3])

    preds = model.predict(test_df.drop("orders_busyness_by_h3_hour", axis=1))
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == len(test_df)
    assert set(model.scores.keys()) == {
        "baseline_scores",
        "train_scores",
        "test_scores",
    }
