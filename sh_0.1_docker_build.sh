#!/bin/bash     
PROJECT_ID="id"
REGION="us-central1"
REPOSITORY="rb"
IMAGE='training'
IMAGE_TAG='training:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
