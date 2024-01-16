#!/bin/bash     
PROJECT_ID="skip-the-dishes-410816"
REGION="us-central1"
REPOSITORY="regionbusyness"
IMAGE_TAG='serving:latest'

#Create repository in the artifact registry
gcloud beta artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$REGION
 
# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
 
 # Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG