# This module is intended to be used in the serving container
import os
from typing import Dict, List, Tuple
import pickle
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


def load_model(path: str):
    """Loads a model artifact.

    Raises a ``FileNotFoundError`` if the model file is missing and logs other
    issues encountered while deserializing the object.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except (OSError, pickle.UnpicklingError) as exc:
        logger.error("Failed to load model from %s: %s", path, exc)
        raise

    return model


def model_predict(model: Dict, data: pd.DataFrame) -> Tuple[List, str]:
    """
    Use the input model to get predictions on the input data.

    model: A dictionary of objects used to make prediction.
           In its simplest case the dictionary has one item e.g. a scikit.learn Estimator
    data: A generic pandas DataFrame

    Returns: A list where each element is the predictions of the model for a single instance
             of input data
    """

    try:
        pipeline = model["pipeline"]
        target = model["target"]
    except KeyError as exc:
        logger.error("Model dictionary missing required key: %s", exc)
        raise

    logger.info("PREPROCESSING THE DATA...")
    try:
        data_preprocessed = data_preprocessing_pipeline(data)
    except Exception as exc:  # noqa: BLE001 - propagate unexpected errors
        logger.error("Error during data preprocessing: %s", exc)
        raise

    logger.info(
        "STARTING PREDICT ON DATAFRAME WITH SHAPE: %s and dtypes: %s",
        data_preprocessed.shape,
        data_preprocessed.dtypes,
    )
    try:
        model_output = pipeline.predict(data_preprocessed)
    except Exception as exc:  # noqa: BLE001 - propagate unexpected errors
        logger.error("Error during model prediction: %s", exc)
        raise

    return model_output, target
