#!/bin/bash

CONTAINER_NAME="data_gen"

# Check if the container exists
if docker ps -a --format '{{.Names}}' | grep -Eq "^$CONTAINER_NAME$"; then
    # Container exists, start it
    echo "Container Already Exist. Starting it"
    docker start $CONTAINER_NAME  
else
    # Container does not exist, run it
    echo "Container dosent exist. Running it"
    docker run -it \
        -v "$(pwd)/..:/home/noc_latency_predictor" \
        --name $CONTAINER_NAME -d systemc
fi
