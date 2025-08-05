# Training Prediction Pipeline

## Secrets

The CD workflow (`.github/workflows/cd.yml`) requires the following secrets:

- `GCP_CREDENTIALS`: Service account key JSON used for authentication.
- `GCP_PROJECT_ID`: Google Cloud project ID.
- `GCP_REGION`: Deployment region (e.g., `us-central1`).
- `GCP_SERVICE_NAME`: Cloud Run service name to deploy.
