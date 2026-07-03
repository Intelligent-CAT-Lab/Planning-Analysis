#!/usr/bin/env bash
set -euo pipefail

docker build -t plan-study .

docker run --rm -v "$(pwd)/artifacts:/plan_study/artifacts" plan-study scripts/plot_all_heatmaps.sh
docker run --rm -v "$(pwd)/artifacts:/plan_study/artifacts" plan-study scripts/plot_all_upsets.sh
docker run --rm -v "$(pwd)/artifacts:/plan_study/artifacts" plan-study scripts/plot_all_sankey.sh


echo "[plan-study] All figures regenerated under artifacts/ — PASS"
