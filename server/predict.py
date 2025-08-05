# This module is intended to be used in the serving container
import os
import pickle
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.prepare_data.preprocessing import data_preprocessing_pipeline

import logging
import sys

logger = logging.getLogger("App")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


def load_model(path: str) -> Dict[str, Any]:
    """Load a model artifact from ``path``."""

    with open(path, "rb") as f:
        model: Dict[str, Any] = pickle.load(f)

    return model


def model_predict(model: Dict[str, Any], data: pd.DataFrame) -> Tuple[List[Any], str]:
    """Use ``model`` to obtain predictions for ``data``."""

    pipeline = model["pipeline"]
    target = model["target"]

    logger.info("PREPROCESSING THE DATA...")
    data_preprocessed = data_preprocessing_pipeline(data)

    logger.info(
        "STARTING PREDICT ON DATAFRAME WITH SHAPE: %s and dtypes: %s",
        data_preprocessed.shape,
        data_preprocessed.dtypes,
    )
    model_output = pipeline.predict(data_preprocessed)

    return model_output.tolist(), target