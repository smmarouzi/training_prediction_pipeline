"""Configuration objects used throughout the data-preparation pipeline."""

from __future__ import annotations

from typing import List


class Radius:
    """Store the Earth radius for a given metric."""

    R: float

    def __init__(self, metric: str = "kilometers") -> None:
        if metric == "kilometers":
            self.R = 6372.8
        elif metric == "mile":
            self.R = 3959.87433


class Centroids:
    """Number of centroids used for clustering."""

    K: int = 5


class Random:
    """Seed used for any pseudo-random operations."""

    seed: int = 1


class H3:
    """Configuration for H3 geospatial indexing."""

    resolution: int = 7


class FinalFeatures:
    """Names of features used for the final model and the target column."""

    features: List[str] = [
        "dist_to_restaurant",
        "Hdist_to_restaurant",
        "avg_Hdist_to_restaurants",
        "date_day_number",
        "restaurant_id",
        "Five_Clusters_embedding",
        "h3_index",
        "date_hour_number",
        "restaurants_per_index",
    ]
    target: List[str] = ["orders_busyness_by_h3_hour"]
