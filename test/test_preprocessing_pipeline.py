import numpy as np
import pandas as pd
from src.prepare_data import preprocessing, utils
from src.prepare_data.config import FinalFeatures


def test_data_preprocessing_pipeline(monkeypatch):
    """Data preprocessing pipeline produces expected features on synthetic data."""

    def euclidean(p1x, p1y, p2x, p2y):
        p1x_arr = np.asarray(p1x, dtype=float)
        p1y_arr = np.asarray(p1y, dtype=float)
        p2x_arr = np.asarray(p2x, dtype=float)
        p2y_arr = np.asarray(p2y, dtype=float)
        dist = np.sqrt((p2x_arr - p1x_arr) ** 2 + (p2y_arr - p1y_arr) ** 2)
        if dist.ndim == 0:
            return float(dist)
        return dist.tolist()

    monkeypatch.setattr(utils, "calculate_euclidean_dist", euclidean)
    monkeypatch.setattr(preprocessing, "calculate_euclidean_dist", euclidean)

    n = 5
    df = pd.DataFrame(
        {
            "courier_lat": [0.0] * n,
            "courier_lon": [0.0] * n,
            "restaurant_lat": [0, 0.01, 0.02, 0.03, 0.04],
            "restaurant_lon": [0.01, 0.02, 0.03, 0.04, 0.05],
            "courier_location_timestamp": pd.date_range(
                "2024-01-01 10:00:00", periods=n, freq="min"
            ),
            "order_created_timestamp": pd.date_range(
                "2024-01-01 09:55:00", periods=n, freq="min"
            ),
        }
    )

    processed = preprocessing.data_preprocessing_pipeline(df)

    expected_cols = FinalFeatures.features + FinalFeatures.target
    assert list(processed.columns) == expected_cols
    assert processed["orders_busyness_by_h3_hour"].nunique() == 1
    assert processed["orders_busyness_by_h3_hour"].iloc[0] == n
    assert processed["restaurants_per_index"].nunique() == 1
    assert processed["restaurants_per_index"].iloc[0] == n
    assert processed.isna().sum().sum() == 0
