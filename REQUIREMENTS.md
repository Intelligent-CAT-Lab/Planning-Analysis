# REQUIREMENTS

## Hardware

- Architecture: x86-64 or ARM64
  (packaged and tested on Ubuntu 24.04 x86-64)
- RAM: 8 GB recommended
- Storage: ~4 GB total
  (repository including 1.3 GB of bundled raw trajectories, plus the
  Docker image and regenerated figures)
- No GPU required.
- No non-commodity peripherals required.

## Software

Recommended path (Option A in README):

- Docker >= 24
- bash (to run `./start_plan_study.sh`)

Alternative local path (Option B in README):

- Python >= 3.10 with pip
- All Python dependencies are declared in `pyproject.toml`
  (bashlex, gsppy, matplotlib, networkx, numpy, pandas, PyYAML, scipy)
  and are installed automatically by `pip install .`

Machine-readable dependency specifications included in the artifact:

- `Dockerfile` (container build)
- `pyproject.toml` (Python package and dependencies)

## Network

No network access, API keys, or external services are required at
runtime. All experiments are offline analyses of the pre-recorded agent
trajectories bundled with the artifact. Network access is only needed
once to build the Docker image (base image + pip packages); a pre-built
image can alternatively be loaded offline with `docker load`.

## Estimated Evaluation Time

- Docker / Option A (`./start_plan_study.sh`): approximately 2 minutes in
  the artifact reviewer's environment; an initial image download may add
  time depending on network speed.
- Local / Option B: approximately 1 minute in the artifact reviewer's
  environment after dependencies are installed.
