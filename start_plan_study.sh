#!/usr/bin/env bash

docker build -t plan-study .
docker run -it \
  -v "$(pwd)/raw_trajectories:/artifact/raw_trajectories" \
  -v "$(pwd)/output:/artifact/output" \
  plan-study bash
