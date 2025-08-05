import pandas as pd
import numpy as np
import collections
from typing import Sequence, Union

from src.prepare_data.config import Radius


def copy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a copy of the DataFrame for further use."""

    return df.copy()


def calculate_euclidean_dist(
    p1x: float, p1y: float, p2x: float, p2y: float
) -> Union[float, list[float]]:
    """Compute Euclidean distance between two points."""

    p1 = (p2x - p1x) ** 2
    p2 = (p2y - p1y) ** 2
    dist = np.sqrt(p1 + p2)
    return dist.tolist() if isinstance(p1x, collections.abc.Sequence) else float(dist)


def calculate_avg_distance_to_restaurants(
    courier_lat: float, courier_lon: float, restaurants_ids: dict[str, dict[str, float]]
) -> float:
    """Average Euclidean distance from courier to restaurants."""

    return float(
        np.mean(
            [
                calculate_euclidean_dist(
                    v["lat"], v["lon"], courier_lat, courier_lon
                )
                for v in restaurants_ids.values()
            ]
        )
    )


def calculate_haversine_dist(
    lat1: Union[float, Sequence[float]],
    lon1: Union[float, Sequence[float]],
    lat2: Union[float, Sequence[float]],
    lon2: Union[float, Sequence[float]],
    radians_metric: str = "kilometers",
) -> Union[float, list[float]]:
    """Calculate the haversine distance between two coordinates."""

    radius = Radius(radians_metric)

    lat1_arr = np.asarray(lat1, dtype=float)
    lon1_arr = np.asarray(lon1, dtype=float)
    lat2_arr = np.asarray(lat2, dtype=float)
    lon2_arr = np.asarray(lon2, dtype=float)

    dLat = np.radians(lat2_arr - lat1_arr)
    dLon = np.radians(lon2_arr - lon1_arr)
    lat1_rad = np.radians(lat1_arr)
    lat2_rad = np.radians(lat2_arr)

    a = np.sin(dLat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dLon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = radius.R * c

    return dist.tolist() if isinstance(lat1, collections.abc.Sequence) else float(dist)


def calculate_avg_Hdist_to_restaurants(
    courier_lat: float, courier_lon: float, restaurants_ids: dict[str, dict[str, float]]
) -> float:
    """Average haversine distance from courier to restaurants."""

    return float(
        np.mean(
            [
                calculate_haversine_dist(v["lat"], v["lon"], courier_lat, courier_lon)
                for v in restaurants_ids.values()
            ]
        )
    )

  
  
