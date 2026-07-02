FROM python:3.12-slim

WORKDIR /plan_study

COPY pyproject.toml ./
COPY graph_construction/ graph_construction/
COPY lang_construction/ lang_construction/
COPY lang_analysis/ lang_analysis/
RUN pip install --no-cache-dir .

RUN mkdir -p raw_trajectories
RUN mkdir -p artifacts/

ENV MPLBACKEND=Agg

CMD ["/bin/bash"]
