FROM python:3.12-slim

WORKDIR /plan_study

COPY pyproject.toml README.md LICENSE ./
COPY graph_construction/ graph_construction/
COPY lang_construction/ lang_construction/
COPY lang_analysis/ lang_analysis/
RUN pip install --no-cache-dir .

COPY plan-settings/ plan-settings/
COPY scripts/ scripts/
RUN chmod +x scripts/*.sh
COPY artifacts/ artifacts/
COPY raw_trajectories/ raw_trajectories/

ENV MPLBACKEND=Agg

CMD ["/bin/bash"]
