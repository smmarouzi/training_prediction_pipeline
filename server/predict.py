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
    """Load a model artifact from ``path``.

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


def model_predict(model: Dict[str, Any], data: pd.DataFrame) -> Tuple[List[Any], str]:
    """Use ``model`` to obtain predictions for ``data``."""

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
