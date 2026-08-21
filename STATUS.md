# STATUS

We apply for all three ACM artifact badges: **Artifacts Functional**,
**Artifacts Reusable**, and **Artifacts Available**.

## Artifacts Functional

The artifact is documented, consistent, complete, exercisable, and
includes evidence of verification and validation:

- **Documented.** The README follows the ASE two-part structure
  (Getting Started with a <30-minute smoke test; step-by-step
  instructions mapping each paper figure/finding to a command), and
  REQUIREMENTS.md lists the exact OS, software, and storage needed to
  run it.

- **Consistent.** The repository structure, plan-setting names,
  benchmark names, and model names used throughout the code
  (plan-settings/, raw_trajectories/, artifacts/) match the
  terminology and notation used in the paper (e.g., the plan alphabet
  N/R/P/V and Equations 1-4 for PPC/POC/PPF/PC).

- **Complete.** All code needed to go from raw trajectories to every
  figure and statistical test in the paper is included:
  graph_construction/ (trajectory to Graphectory), lang_construction/
  (Graphectory to Langutory), and lang_analysis/ (metrics, hypothesis
  tests, and all plotting scripts).

- **Exercisable.** A single command
  (`./start_plan_study.sh`) builds a self-contained Docker image and
  regenerates the paper figures from the bundled data in minutes,
  requiring no GPU or API keys. After the image and its dependencies
  are available, the analysis runs without network access. This script
  serves as both the smoke test and the figure reproduction, so reviewers
  can exercise the artifact with one command.

- **Verification and validation.** All pre-computed results reported
  in the paper (compliance heatmaps, UpSet plots, Sankey diagrams,
  and per-model/per-setting statistics) ship under artifacts/,
  allowing reviewers to directly diff freshly regenerated outputs
  against the shipped results to confirm the pipeline reproduces the
  paper's claims.

## Artifacts Reusable

Beyond reproducing the paper's results, the artifact is documented and
structured to facilitate reuse and repurposing:

- **One-command reproduction.** A single script
  (`./start_plan_study.sh`) builds a self-contained Docker image and
  regenerates every figure in the paper (compliance heatmaps, UpSet
  plots, and phase-flow Sankey diagrams) from the bundled raw
  trajectories in minutes on a commodity laptop. This doubles as the
  smoke test. No GPU or API keys are required; network access is needed
  only for an initial Docker build that must download its base image and
  Python packages.

- **Reusable metric implementation.** The plan compliance metrics (PPC,
  POC, PPF, PC; Equations 1-4 in the paper) are implemented as an
  installable Python package (pip install .) that operates on any
  SWE-agent trajectory, not only our dataset. All scripts expose a
  documented command-line interface (--help).

- **Reusable dataset.** The 21,120 raw trajectories are shipped in a
  documented JSON format (see README, "Data Schema"), together with
  their intermediate Graphectory and Langutory representations,
  enabling new analyses beyond plan compliance (e.g., trajectory
  mining, behavioral studies of agents).

- **Extensible experiment configurations.** The eight plan-setting
  system prompts (plan-settings/) are standard SWE-agent YAML configs
  pinned to a specific scaffold commit, so researchers can rerun the
  study with new models or design new plan variations. The README's
  "Extending the Artifact" section documents exactly how to add a new
  model or setting (raw_trajectories/{BENCHMARK}/{SETTING}/{MODEL}/traj.7z
  plus the corresponding evaluation report under
  artifacts/{BENCHMARK}/{SETTING}/{MODEL}/report/) and rerun the same
  scripts to fold it into the analysis.

- **Documentation.** The README follows the ASE two-part structure
  (Getting Started with a <30-minute smoke test; step-by-step
  instructions mapping each paper figure to a command), and explicitly
  states which paper claims are and are not supported by the artifact.

## Artifacts Available

The artifact is permanently archived on Zenodo with DOI
10.5281/zenodo.19339901, publicly accessible without registration, and
released under the MIT license (see LICENSE). The Zenodo record contains
the complete artifact: analysis code, SWE-agent plan configurations, all
21,120 raw agent trajectories (~1.3 GB), and all pre-computed results
reported in the paper. Zenodo provides a declared plan for permanent
accessibility of archived records. The DOI is included in the data
availability statement at the end of the paper.
