"""Utilities for running grid search during model training."""

from typing import Any

from sklearn.model_selection import GridSearchCV

# Alias the configuration class to avoid naming conflicts with this module's
# ``GridSearch`` class. Without the alias, the class below would shadow the
# imported configuration and attempting to access attributes such as
# ``params`` or ``cv`` would raise an ``AttributeError``.
from src.train.config import GridSearch as GridSearchConfig


class GridSearch:
    """Wrapper around :class:`sklearn.model_selection.GridSearchCV`.

    Parameters for the grid search are taken from
    :class:`src.train.config.GridSearch`.
    """

    grid_search: GridSearchCV
    rf_best: Any

    def __init__(self, regr: Any) -> None:
        self.grid_search = GridSearchCV(
            estimator=regr,
            param_grid=GridSearchConfig.params,
            cv=GridSearchConfig.cv,
            n_jobs=GridSearchConfig.n_jobs,
            verbose=GridSearchConfig.verbose,
            scoring=GridSearchConfig.scoring,
        )

    def fit(self, X_train: Any, y_train: Any) -> None:
        """Fit the grid search on training data."""
        self.grid_search.fit(X_train, y_train)
        self.rf_best = self.grid_search.best_estimator_

    def best_score(self) -> float:
        """Return the best score obtained during grid search."""
        return float(self.grid_search.best_score_)

    def evaluate(self, X_test: Any, y_test: Any) -> float:
        """Evaluate the best model on the test data."""
        return float(self.rf_best.score(X_test, y_test))
