# Evaluating Plan Compliance in Autonomous Programming Agents

A large-scale empirical study of **plan compliance in programming agents**, analyzing 16,991 trajectories across four LLMs, two benchmarks, and eight plan settings.

---

## Overview

LLM-based programming agents are commonly instructed to follow a task-specific plan (e.g., navigate → reproduce → patch → validate) via their system prompt. But do they actually follow it?

This repository provides the artifacts for the first extensive, systematic analysis of plan compliance in programming agents. We introduce novel **plan compliance metrics**, evaluate agent behavior under diverse plan configurations, and study how plan adherence relates to task success.

---

## Built Upon: Process-Centric Analysis Tools

Our analysis uses **Graphectory** and **Langutory**, two process-centric representations introduced by Shuyang Liu in [Process-Centric Analysis of Agentic Software Systems](https://arxiv.org/abs/2512.02393)

---

## Plan Compliance Metrics

We propose **Plan Compliance (PC)**, measured across three dimensions:

| Metric | Description |
|---|---|
| **Plan Phase Compliance (PPC)** | Fraction of expected plan phases that appear in the trajectory |
| **Plan Order Compliance (POC)** | Fraction of phases appearing in the correct relative order (via longest increasing subsequence) |
| **Plan Phase Fidelity (PPF)** | Penalizes the appearance of phases outside the specified plan alphabet |

The overall score is the geometric mean: **PC = (PPC · POC · PPF)^(1/3)**

---

## Experimental Setup

### Models

| Model | Type |
|---|---|
| GPT-5 mini | Closed-source frontier model |
| DeepSeek-R1 | Open-source reasoning model |
| DeepSeek-V3 | Open-source general-purpose model |
| Devstral-small (24GB) | Distilled model specialized in coding |

### Scaffold

All experiments use [**SWE-agent**](https://github.com/SWE-agent/SWE-agent) at commit `8089c8b`.

### Benchmarks

- **SWE-bench Verified** — 500 real-world GitHub issues (Easy / Medium / Hard)
- **SWE-bench Pro** — 31 Python instances

### Plan Settings

| Setting | Plan Formulation | Variation Type |
|---|---|---|
| Standard (Default) | ⟨N, R, P, V⟩ | Baseline |
| No Plan | — | Removal |
| No Reproduction | ⟨N, ¬R, P, V⟩ | Removal |
| No Validation | ⟨N, R, P, ¬V⟩ | Removal |
| + Regression Testing | ⟨RG, N, R, P, V, VG⟩ | Addition |
| + Summary of Changes | ⟨N, R, P, V, S⟩ | Addition |
| Reordered | ⟨N, P, R, V⟩ | Reordering |
| Periodic Reminder | ⟨N, R, P, V⟩ (re-injected every 5 steps) | Reminder |

**Plan phases:** Navigation (N) · Reproduction (R) · Patch (P) · Validation (V)

---

## Repository Structure

- **[`raw_trajectories/`](raw_trajectories/)** - Raw trajectory data from all 16,991 runs across four models, two benchmarks, and eight plan settings.
- **[`artifacts/`](artifacts/)** - All plots, figures, and analysis reported in the paper.
- **[`plan-settings`](plan-settings)** - SWE-agent YAML configuration files corresponding to each plan settings.
- **[`lang_analysis/`](lang_analysis/compute_plan_compliance_scores.py)** - Implementation of plan compliance metrics (PPC, POC, PPF, PC).

---

## Usage

### Compute Plan Compliance

```bash
python lang_analysis/compute_plan_compliance_score.py \
    --dataset BENCHMARK --setting SETTING --model MODEL
```


### Generate Phase Flow Diagrams

```bash
# outputs to same folder as languatory.json
python lang_analysis/sankey_lang_plot.py \
    --lang-path artifacts/DATASET/SETTING/MODEL/lang/languatory.json
```

Run with `--help` to see all available options.

### Pre-computed Results
 
- **Compliance rates:** `artifacts/BENCHMARK/SETTING/MODEL/stats/continuous_plan_test/`
- **Phase flow (Sankey) diagrams:** `artifacts/BENCHMARK/SETTING/MODEL/lang/`

---
