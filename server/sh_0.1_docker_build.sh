#!/bin/bash

# Allow PROJECT_ID to be provided via the environment. Default to a placeholder
# value so the script can be run locally without modification.
PROJECT_ID=${PROJECT_ID:-your-project-id}
REGION="us-central1"
REPOSITORY="regionbusyness"
IMAGE='serving'
IMAGE_TAG='serving:latest'

docker build -f Dockerfile -t "$IMAGE" ..
docker tag "$IMAGE" "$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG"

