#!/bin/bash

# Allow PROJECT_ID to be provided via the environment. Default to a placeholder
# value so the script can be run locally without modification.
PROJECT_ID=${PROJECT_ID:-your-project-id}
REGION="us-central1"
REPOSITORY="regionbusyness"
IMAGE_TAG='serving:latest'

# Create repository in the artifact registry
gcloud beta artifacts repositories create "$REPOSITORY" \
  --repository-format=docker \
  --location="$REGION"

# Configure Docker
gcloud auth configure-docker "$REGION-docker.pkg.dev"

# Push
docker push "$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG"

