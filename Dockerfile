FROM python:3.12-slim

WORKDIR /artifact

COPY pyproject.toml ./
COPY graph_construction/ graph_construction/
COPY lang_construction/ lang_construction/
COPY lang_analysis/ lang_analysis/
RUN pip install --no-cache-dir .

COPY artifacts/ artifacts/
#COPY sample_trajectories/ sample_trajectories/

RUN mkdir -p raw_trajectories

ENV MPLBACKEND=Agg

CMD ["/bin/bash"]
