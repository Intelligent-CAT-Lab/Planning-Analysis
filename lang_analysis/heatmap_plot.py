#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BENCHMARKS = ["SWE-Bench-Verified", "SWE-Bench_Pro"]
PLANS = [
    "no_plan", "no_reproduce", "no_validation", "plan",
    "plan_and_regression", "plan_and_summary", "plan_reminded", "plan_reordered",
]

METRICS = ["ppc", "poc", "ppf", "pc"]


def collect_rows(folder):
    """Return list of [model, status, ppc, poc, ppf, pc]."""
    rows = []
    for fp in sorted(folder.rglob("*_scores.json")):
        if "deterministic_stats" in str(fp):
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {fp}: {e}", file=sys.stderr)
            continue

        model = data.get("model", "")
        stats = data.get("summary_stats", {})
        for resolution in ("resolved", "unresolved"):
            s = stats.get(resolution)
            if not s:
                continue
            rows.append([
                model,
                resolution,
                s.get("mean_s2", float("nan")),
                s.get("mean_s3", float("nan")),
                s.get("mean_s1", float("nan")),
                s.get("mean_final", float("nan")),
            ])
    return rows


def plot_heatmap(rows, bench, plan, out_pdf):
    df = pd.DataFrame(rows, columns=["Model", "Status"] + METRICS)
    df["Row_Label"] = df["Model"] + "_" + df["Status"]

    mat = df.set_index("Row_Label")[METRICS]
    if mat.empty:
        return

    height = max(2.0, 0.6 * len(mat))
    fig, ax = plt.subplots(figsize=(7, height))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.8,
                cbar_kws={"shrink": 0.8}, ax=ax, vmin=0.4, vmax=1)
    ax.set_title(f"{bench} / {plan}", fontsize=14, pad=10)
    ax.set_ylabel("")
    ax.set_xlabel("Metrics")
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Base path containing artifact/")
    ap.add_argument("--benchmark", choices=BENCHMARKS, default=None)
    ap.add_argument("--plan", choices=PLANS, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    benchmarks = [args.benchmark] if args.benchmark else BENCHMARKS
    plans = [args.plan] if args.plan else PLANS

    for bench in benchmarks:
        for plan in plans:
            folder = root / "artifacts" / bench / plan
            if not folder.is_dir():
                print(f"Skipping missing folder: {folder}", file=sys.stderr)
                continue
            rows = collect_rows(folder)
            if not rows:
                print(f"No scores found in: {folder}", file=sys.stderr)
                continue
            plot_heatmap(rows, bench, plan, folder / "compliance_heatmap.pdf")


if __name__ == "__main__":
    main()
