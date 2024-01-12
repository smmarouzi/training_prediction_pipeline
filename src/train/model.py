import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
import shap

import logging 
import sys

from src.train.config import RF, Data, GridSearch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


class BusynessEstimation(object):
    def __init__(self,
                 # train_df:pd.DataFrame=None,
                 test_df:pd.DataFrame=None,
                 target:str=Data.target,
                 random_state:int=RF.regr_random_state          
                ):

        # self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.random_state  =  random_state

    def create_final_training_dataset(self, train_df, test_df):
        self.df = pd.concat([train_df, test_df], ignore_index=True)


    def make_pipeline(self, **params):
        self.cat_features = self.X_train.select_dtypes(include=['category', object, bool]).columns.tolist()
        
        return Pipeline([
                ('rf', RandomForestRegressor(**params))
                ])

    def fit_and_evaluate_model(self):
        self.best_params = RF.best_params

        # self.best_params.update(self.study.best_params)
        self.model_pipeline = self.make_pipeline(**self.best_params)
        
        logger.info(f'\nTRAINING THE MODEL...')
        
        self.model_pipeline.fit(self.X_train, self.y_train)
        """
        logger.info(f"\nGETTING THE SHAP VALUES")
        self.X_test_transformed = Pipeline(self.model_pipeline.steps[:-1]).transform(self.test_df.drop(self.target, axis=1))
        self.explainer = shap.TreeExplainer(self.model_pipeline.named_steps["rf"])
        self.shap_values = self.explainer.shap_values(self.X_test_transformed)
        """
        baseline_scores = {
                           "r2": r2_score(self.y_train, [self.y_train.mean()]*len(self.y_train)) 
                           } 

        def evaluate(dataset):
            return {
                    "r2": r2_score(dataset[self.target], self.model_pipeline.predict(dataset.drop(self.target, axis=1)))
                    }

        train_scores = evaluate(pd.concat([self.X_train, self.y_train], axis=1))
        test_scores = evaluate(self.test_df)

        self.scores = {"baseline_scores": baseline_scores,
                       "train_scores": train_scores,
                       "test_scores": test_scores}

    
    def fit(self, X, y=None):

        self.X_train = X.copy()
        self.y_train = y.copy()
        
        self.fit_and_evaluate_model()
        
        logger.info(f'CREATING THE FINAL TRAINING DATASET...')
        self.create_final_training_dataset(pd.concat([self.X_train, self.y_train], axis=1), self.test_df)

        logger.info(f'\nTRAINING THE FINAL MODEL...')
    
        self.model_pipeline.fit(self.df.drop(self.target, axis=1), self.df[self.target])
 
        return self


    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        return self.model_pipeline.predict(X)