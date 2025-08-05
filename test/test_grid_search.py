from sklearn.ensemble import RandomForestRegressor

from src.train.grid_search import GridSearch
from src.train.config import GridSearch as GridSearchConfig


def test_grid_search_instantiation_uses_config_params():
    gs = GridSearch(RandomForestRegressor())
    assert gs.grid_search.cv == GridSearchConfig.cv

