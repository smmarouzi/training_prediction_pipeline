# Training Prediction Pipeline

This project provides an end–to–end pipeline for preparing data, training a
busyness estimation model, and evaluating its performance.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the pipeline

Execute the entry point script to preprocess data, train the model, and output
evaluation results:

```bash
python main.py
```

## Running tests

Run the unit tests with:

```bash
pytest
```

## Secrets

The CD workflow (`.github/workflows/cd.yml`) requires the following secrets:

- `GCP_CREDENTIALS`: Service account key JSON used for authentication.
- `GCP_PROJECT_ID`: Google Cloud project ID.
- `GCP_REGION`: Deployment region (e.g., `us-central1`).
- `GCP_SERVICE_NAME`: Cloud Run service name to deploy.
