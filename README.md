# Evaluating Plan Compliance in Autonomous Programming Agents

**ASE 2026 Artifact** — A large-scale empirical study of [**plan compliance in programming agents**](https://arxiv.org/abs/2604.12147), analyzing 21,120 SWE-agent trajectories across four LLMs, two benchmarks, and eight plan settings.

This README follows the ASE 2026 artifact guidelines: **Part 1 — Getting Started** (installation + smoke test, ≤30 minutes) and **Part 2 — Step-by-Step Reproduction** (mapping paper claims to commands).

---

## Overview

LLM-based programming agents are commonly instructed to follow a task-specific plan (e.g., navigate → reproduce → patch → validate) via their system prompt. But do they actually follow it?

This artifact provides the data and analysis pipeline for the first extensive, systematic analysis of plan compliance in programming agents. We introduce novel **plan compliance metrics**, evaluate agent behavior under diverse plan configurations, and study how plan adherence relates to task success.

Our analysis builds on **Graphectory** and **Langutory**, two process-centric representations introduced in [Process-Centric Analysis of Agentic Software Systems](https://arxiv.org/abs/2512.02393).

### Plan Compliance Metrics

| Metric | Description |
|---|---|
| **Plan Phase Compliance (PPC)** | Fraction of expected plan phases that appear in the trajectory |
| **Plan Order Compliance (POC)** | Fraction of phases appearing in the correct relative order (via longest increasing subsequence) |
| **Plan Phase Fidelity (PPF)** | Penalizes phases outside the specified plan alphabet |

Overall score: **PC = (PPC · POC · PPF)^(1/3)**

---

# Part 1 — Getting Started (≈15 minutes)

## Requirements

- **OS/Arch:** Linux or macOS, x86-64 or ARM64 (tested on Ubuntu 24.04 x86-64)
- **Software:** Docker ≥ 24 (recommended) *or* Python ≥ 3.10 with pip
- **Storage:** ~4 GB total (repository incl. 1.3 GB of raw trajectories + Docker image)
- **No GPU or API keys required.** All experiments are offline analysis of pre-recorded trajectories, fully bundled in this artifact.

See [REQUIREMENTS](REQUIREMENTS.md) for details.

## Option A — Docker (recommended): One-Command Smoke Test + Full Reproduction

Install and start Docker once:

- **Ubuntu/Linux:** follow the official [Docker Engine installation guide](https://docs.docker.com/engine/install/ubuntu/), then start the Docker service if it is not already running.
- **macOS:** follow the official [Docker Desktop installation guide](https://docs.docker.com/desktop/setup/install/mac-install/), then open Docker Desktop.

Verify the installation with `docker --version` and `docker run --rm hello-world`. Then, from the repository root, build the image and run the artifact with:

```bash
./start_plan_study.sh
```

The image is fully self-contained: code, configs, all 21,120 raw trajectories, and pre-computed results are baked in. The command builds the `plan-study` image and runs three containerized jobs that regenerate all compliance heatmaps, UpSet plots, and phase-flow (Sankey) diagrams. No Docker knowledge beyond installing and starting it is required.

**Expected output:** the script exits without error and figures are (re)written on the host under `artifacts/{BENCHMARK}/{SETTING}/`, e.g.:

```
artifacts/SWE-Bench-Verified/no_reproduce/compliance_heatmap.pdf
artifacts/SWE-Bench-Verified/no_reproduce/upset_plan_vs_no_reproduce.png
artifacts/SWE-Bench-Verified/plan/deepseek_v3/lang/sankey_deepseek_v3.pdf
```

Regenerated figures should match the pre-computed versions shipped in `artifacts/`.

For interactive exploration inside the container:

```bash
docker run -it -v "$(pwd)/artifacts:/plan_study/artifacts" plan-study bash
```

The `artifacts/` mount makes generated figures appear on your host. Running without the mount also works; copy results out with `docker cp` afterwards.

## Option B — Local install (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install .

scripts/plot_all_heatmaps.sh
scripts/plot_all_upsets.sh
scripts/plot_all_sankey.sh
```

---

# Part 2 — Step-by-Step Reproduction

## What This Artifact Does and Does Not Reproduce

**Supported:** All plan compliance metrics, statistical tests, and figures in the paper are reproducible from the bundled trajectories using the commands below.

**Not supported:** Re-generating the 21,120 trajectories themselves. This requires paid model APIs (GPT-5 mini, DeepSeek, Devstral), substantial inference cost, and days of wall-clock time; results would also differ due to inherent nondeterminism (RQ7, §6.2 of the paper). Instead, we ship the exact trajectories analyzed in the paper (`raw_trajectories/`, 1.3 GB), plus the SWE-agent configuration files (`plan-settings/`) and scaffold version (commit `8089c8b`) needed to re-run generation independently.

## One-Shot Reproduction Scripts

Inside the container (or a local venv):

```bash
# All compliance heatmaps (Figures 2, 8, 10, 13, 15, 18, 20, 22)
scripts/plot_all_heatmaps.sh

# All UpSet plots comparing resolved-instance sets across plan settings
# (Figures 5, 7, 9, 12, 14, 17, 19)
scripts/plot_all_upsets.sh

# All phase-flow (Sankey) diagrams (Figures 3, 4, 6, 11, 16, 21)
scripts/plot_all_sankey.sh
```

Or as one-shot Docker jobs from the host (no interactive shell needed):

```bash
docker run --rm -v "$(pwd)/artifacts:/plan_study/artifacts" plan-study scripts/plot_all_heatmaps.sh
docker run --rm -v "$(pwd)/artifacts:/plan_study/artifacts" plan-study scripts/plot_all_upsets.sh
docker run --rm -v "$(pwd)/artifacts:/plan_study/artifacts" plan-study scripts/plot_all_sankey.sh
```

All scripts run in minutes on a commodity laptop and write into `artifacts/{BENCHMARK}/{SETTING}/`.

## Claim → Command Mapping

Substitute `MODEL ∈ {gpt5-mini, deepseek-v3, deepseek-r1, devstral-small}`, `BENCHMARK ∈ {SWE-Bench-Verified, SWE-Bench_Pro}`, `SETTING ∈ {plan, no_plan, no_reproduce, no_validation, plan_and_regression, plan_and_summary, plan_reordered, plan_reminded}`.

| Paper claim | Command |
|---|---|
| **Compliance metric heatmaps** (RQ1–RQ6; Figures 2, 8, 10, 13, 15, 18, 20, 22) | `python lang_analysis/heatmap_plot.py --benchmark BENCHMARK --plan SETTING .` |
| **Success-rate / resolved-set comparisons** (Findings 6, 7, 9; Figures 5, 7, 9, 12, 14, 17, 19) | `python lang_analysis/updset_plot.py --benchmark BENCHMARK plan SETTING` |
| **Phase flow (Sankey) diagrams** (Figures 3, 4, 6, 11, 16, 21) | `python lang_analysis/sankey_lang_plot.py --lang-path artifacts/BENCHMARK/SETTING/MODEL/lang/languatory.json` |
| **Plan compliance scores** (PPC/POC/PPF/PC, Eqs. 1–4) | `python lang_analysis/compute_plan_compliance_scores.py --dataset BENCHMARK --setting SETTING --model MODEL` |

Run any script with `--help` for all options.

## Pre-computed Results

All results reported in the paper ship under `artifacts/`, so reviewers can verify outputs without recomputation and diff freshly generated figures against them:

- **Compliance heatmaps:** `artifacts/{BENCHMARK}/{SETTING}/compliance_heatmap.pdf`
- **UpSet plots:** `artifacts/{BENCHMARK}/{SETTING}/upset_plan_vs_{SETTING}.png`
- **Compliance metrics:** `artifacts/{BENCHMARK}/{SETTING}/{MODEL}/stats/continuous_plan_test/`
- **Phase flow (Sankey) diagrams:** `artifacts/{BENCHMARK}/{SETTING}/{MODEL}/lang/`

---

## Extending the Artifact: Adding a New Model or Setting

The pipeline is not limited to the four models and eight settings analyzed in the paper. To evaluate an additional model or plan setting, add data in the two locations below and rerun the scripts — no code changes required.

1. **Add the raw trajectories.** Place the archived trajectory data at:

   ```
   raw_trajectories/{BENCHMARK}/{SETTING}/{MODEL}/traj.7z
   ```

   where `BENCHMARK ∈ {SWE-Bench-Verified, SWE-Bench_Pro}`, `SETTING` is either one of the existing plan settings or a new one you define (see `plan-settings/` for the YAML config format), and `MODEL` is the new or existing model name.

2. **Add the corresponding evaluation report.** Place the SWE-bench evaluation report JSON at:

   ```
   artifacts/{BENCHMARK}/{SETTING}/{MODEL}/report/
   ```

   This report supplies the resolution status used by the UpSet plots and success-rate comparisons.

3. **Rerun the pipeline.** Re-run `./start_plan_study.sh` (Docker) or the individual `scripts/plot_all_*.sh` scripts (local venv) as described in Part 1/Part 2 above. The graph/langutory construction and metric computation steps will pick up the new `{BENCHMARK}/{SETTING}/{MODEL}` combination automatically, and the corresponding heatmaps, UpSet plots, and Sankey diagrams will be (re)written under `artifacts/{BENCHMARK}/{SETTING}/`.

> **Note:** A new `SETTING` should also have a matching config directory under `plan-settings/` describing its plan alphabet, since `compute_plan_compliance_scores.py` uses it to determine expected phases (PPC/POC/PPF).

---

## Experimental Setup (as in the paper)

### Models

| Model | Type |
|---|---|
| GPT-5 mini | Closed-source frontier reasoning model |
| DeepSeek-R1 | Open-source reasoning model |
| DeepSeek-V3 | Open-source general-purpose model |
| Devstral-small (24B) | Distilled model specialized in coding |

### Scaffold

All trajectories were generated with [**SWE-agent**](https://github.com/SWE-agent/SWE-agent) at commit `8089c8b`, default configuration, varying only the plan section of the system prompt (`plan-settings/`).

### Benchmarks

- **SWE-bench Verified** — 500 real-world GitHub issues (Easy / Medium / Hard)
- **SWE-bench Pro** — 31 python instances resolved by Claude Opus 4.1, Claude Sonnet 4, and Gemini 2.5 Pro according to their [official trajectories](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/traj). We use [SWE-bench_Pro-os](https://github.com/scaleapi/SWE-bench_Pro-os) at commit `0c64e26`.

### Plan Settings

| Setting | Plan Formulation | Variation Type | Config |
|---|---|---|---|
| Standard (Default) | ⟨N, R, P, V⟩ | Baseline | `plan-settings/plan/` |
| No Plan | — | Reduction | `plan-settings/no_plan/` |
| No Reproduction | ⟨N, ¬R, P, V⟩ | Reduction | `plan-settings/no_reproduce/` |
| No Validation | ⟨N, R, P, ¬V⟩ | Reduction | `plan-settings/no_validation/` |
| + Regression Testing | ⟨R_G, N, R, P, V, V_G⟩ | Augmentation | `plan-settings/plan_and_regression/` |
| + Summary of Changes | ⟨N, R, P, V, S⟩ | Augmentation | `plan-settings/plan_and_summary/` |
| Reordered | ⟨N, P, R, V⟩ | Reordering | `plan-settings/plan_reordered/` |
| Periodic Reminder | ⟨N, R, P, V⟩ every 5 steps | Repeating | `plan-settings/plan_reminded/` |

**Plan phases:** Navigation (N) · Reproduction (R) · Patch (P) · Validation (V)

---

## Repository Structure

```
.
├── plan-settings/        SWE-agent YAML system-prompt configs, one dir per plan setting
├── raw_trajectories/     Raw trajectory data, 21,120 runs (~1.3 GB, fully bundled)
│   ├── SWE-Bench-Verified/
│   └── SWE-Bench_Pro/
├── graph_construction/   Trajectory → Graphectory (buildGraph.py)
├── lang_construction/    Graphectory → Langutory phase sequences (mapLang.py, get_Lang.py)
├── lang_analysis/        Metrics, statistical tests, and plotting
│   ├── compute_plan_compliance_scores.py   Compute PPC / POC / PPF / PC (Eqs. 1–4)
│   ├── heatmap_plot.py                     Compliance metric heatmaps
│   ├── updset_plot.py                      UpSet plots of resolved-instance sets
│   ├── sankey_lang_plot.py                 Phase flow diagrams
│   └── plan_hypothesis_test.py             Mann–Whitney / McNemar tests
├── artifacts/            Pre-computed results: all figures, stats, Langutory files
├── scripts/
│   ├── plot_all_heatmaps.sh    Run INSIDE the container (or venv): all compliance heatmaps
│   ├── plot_all_upsets.sh      Run INSIDE the container (or venv): all UpSet plots
│   └── plot_all_sankey.sh      Run INSIDE the container (or venv): all phase-flow diagrams
├── start_plan_study.sh     Run on the HOST: builds the image and regenerates ALL figures
├── Dockerfile
├── REQUIREMENTS.md          Hardware/software requirements
├── STATUS.md                Badges requested + justification
└── LICENSE               MIT
```

### Data Schema

Each raw trajectory is a JSON file (one per benchmark instance) containing the SWE-agent run: an ordered list of steps, each with the model's `thought`, the executed `action` (tool call), and the environment `observation`. Intermediate representations:

- **Graphectory** (`graphs/*/{instance_id}.json`): nodes = distinct agent actions; edges = chronological execution order.
- **Langutory** (`languatory.json`): per-instance phase sequence over the alphabet Φ = {N, R, P, V, …}, e.g. `NRRPVVVPV`, used by all compliance metrics.

---

## License

MIT — see `LICENSE`.
