# STATUS

We apply for the following ACM badges: **Artifacts Available** and
**Artifacts Reusable**.

## Artifacts Available

The artifact is permanently archived on Zenodo with DOI
10.5281/zenodo.19339901, publicly accessible without registration, and
released under the MIT license (see LICENSE). The Zenodo record contains
the complete artifact: analysis code, SWE-agent plan configurations, all
16,991 raw agent trajectories (~1.3 GB), and all pre-computed results
reported in the paper.

## Artifacts Reusable

Beyond reproducing the paper's results, the artifact is documented and
structured to facilitate reuse and repurposing:

- **One-command reproduction.** A single script
  (scripts/start_plan_study.sh) builds a self-contained Docker image and
  regenerates every figure in the paper (compliance heatmaps, UpSet
  plots, and phase-flow Sankey diagrams) from the bundled raw
  trajectories in minutes on a commodity laptop. This doubles as the
  smoke test. No GPU, API keys, or network access is required at
  runtime.

- **Reusable metric implementation.** The plan compliance metrics (PPC,
  POC, PPF, PC; Equations 1-4 in the paper) are implemented as an
  installable Python package (pip install .) that operates on any
  SWE-agent trajectory, not only our dataset. All scripts expose a
  documented command-line interface (--help).

- **Reusable dataset.** The 16,991 raw trajectories are shipped in a
  documented JSON format (see README, "Data Schema"), together with
  their intermediate Graphectory and Langutory representations,
  enabling new analyses beyond plan compliance (e.g., trajectory
  mining, behavioral studies of agents).

- **Extensible experiment configurations.** The eight plan-setting
  system prompts (plan-settings/) are standard SWE-agent YAML configs
  pinned to a specific scaffold commit, so researchers can rerun the
  study with new models or design new plan variations.

- **Documentation.** The README follows the ASE two-part structure
  (Getting Started with a <30-minute smoke test; step-by-step
  instructions mapping each paper figure to a command), and explicitly
  states which paper claims are and are not supported by the artifact.
