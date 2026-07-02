#!/usr/bin/env bash

docker build -t plan-study .
docker run -it \
  -v "$(pwd)/raw_trajectories:/plan_study/raw_trajectories" \
  -v "$(pwd)/artifacts/:/plan_study/artifacts/" \
  plan-study bash
