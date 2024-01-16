from sklearn.model_selection import GridSearchCV
from src.train.config import GridSearch

class GridSearch:
    def __init__(self, regr) -> None:
        self.grid_search = GridSearchCV(estimator=regr,
                                param_grid=GridSearch.params,
                                cv = GridSearch.cv,
                                n_jobs=GridSearch.n_jobs, verbose=1, scoring=GridSearch.scoring)
    def fit(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)
        self.rf_best = self.grid_search.best_estimator_
        
    def best_score(self):
        return self.grid_search.best_score_
    
    def evaluate(self, X_test, y_test):
        self.rf_best.score(X_test, y_test)
