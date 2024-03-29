#!/bin/bash     
PROJECT_ID="skip-the-dishes-410816"
REGION="us-central1"
REPOSITORY="regionbusyness"
IMAGE='training'
IMAGE_TAG='training:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG