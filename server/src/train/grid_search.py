from typing import Any

from sklearn.model_selection import GridSearchCV

from src.train.config import GridSearch as GridSearchConfig


class GridSearch:
    """Wrapper around ``sklearn.model_selection.GridSearchCV``."""

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
        self.grid_search.fit(X_train, y_train)
        self.rf_best = self.grid_search.best_estimator_

    def best_score(self) -> float:
        return float(self.grid_search.best_score_)

    def evaluate(self, X_test: Any, y_test: Any) -> float:
        return float(self.rf_best.score(X_test, y_test))
