#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

NAME=v2c_docker

docker build -t $NAME .
docker tag $NAME ghcr.io/ai-med/$NAME:latest
docker push ghcr.io/ai-med/$NAME:latest

