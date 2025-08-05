import os
import sys
import json
import re
import logging
from threading import Lock
from typing import Any, Dict, Tuple

import pandas as pd
from flask import Flask, request, abort
from google.cloud import storage

from server.predict import load_model, model_predict

logger = logging.getLogger("App")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

app = Flask(__name__)

PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
# Vertex AI sets this env with path to the model artifact
AIP_STORAGE_URI = os.environ["AIP_STORAGE_URI"]
logger.info(f"MODEL PATH: {AIP_STORAGE_URI}")

MODEL_PATH = "model/model.pickle"

# Cache model in memory to avoid reloading on every request.
_model: Dict[str, Any] | None = None
_model_lock = Lock()


def decode_gcs_url(url: str) -> Tuple[str, str]:
    """Split a Google Cloud Storage path into bucket and blob."""

    bucket = re.findall(r"gs://([^/]+)", url)[0]
    blob = url.split("/", 3)[-1]
    return bucket, blob


def download_artifacts(artifacts_uri: str, local_path: str) -> None:
    logger.info("Downloading %s to %s", artifacts_uri, local_path)
    storage_client = storage.Client()
    src_bucket, src_blob = decode_gcs_url(artifacts_uri)
    source_bucket = storage_client.bucket(src_bucket)
    source_blob = source_bucket.blob(src_blob)
    source_blob.download_to_filename(local_path)
    logger.info("Downloaded.")


def load_artifacts(artifacts_uri: str = AIP_STORAGE_URI) -> None:
    model_uri = os.path.join(artifacts_uri, "model")
    logger.info("Loading artifacts from %s", model_uri)
    download_artifacts(model_uri, MODEL_PATH)


def get_model() -> Dict[str, Any]:
    """Lazy-load and cache the model artifact.

    When running in production, the model is loaded once and reused for
    subsequent requests, which significantly improves throughput under heavy
    load.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                load_artifacts()
                _model = load_model(MODEL_PATH)
                logger.info("MODEL LOADED")
    return _model

# Flask route for Liveness checks


@app.route(HEALTH_ROUTE, methods=["GET"])
def health_check() -> str:
    return "I am alive, 200"

# Flask route for predictions


@app.route(PREDICT_ROUTE, methods=["POST"])
def prediction() -> Dict[str, Any]:
    logger.info("SERVING ENDPOINT: Received predict request.")
    model = get_model()
    payload = json.loads(request.data)

    instances = payload["instances"]

    try:
        df_str = "\n".join(instances)
        instances = pd.read_json(df_str, lines=True)
    except Exception as e:  # pragma: no cover - best effort
        logger.error("Failed to process payload:\n %s", e)
        abort(500, "Failed to score request.")

    logger.info("Running MODEL_PREDICT for request.")
    model_output, target_name = model_predict(model, instances)
    logger.info("MODEL_PREDICT completed. Target: %s", target_name)

    response: Dict[str, Any] = {"predictions": model_output}
    logger.info("SERVING ENDPOINT: Finished processing.")

    return response


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
