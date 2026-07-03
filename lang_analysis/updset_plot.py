#!/usr/bin/env python3
"""
UpSet plot: compare two settings (Setting A vs Setting B) per instance,
across 4 models, with stacked bars by difficulty.

For each model, reads all trajectory_metrics.csv under:
    artifacts/SWE-Bench-verified/{setting}/{model}/analysis/
concatenates them, then joins Setting A and Setting B on `instance`.
Each instance's resolution in each setting becomes a boolean category
(resolved=True). Stacked bars show difficulty (Easy/Medium/Hard).

Usage:
    python upset_plot.py SETTING_A SETTING_B [--root artifacts] [--out output/upset.png]
"""

import sys
import argparse
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from upsetplot import UpSet
import upsetplot.plotting as _up_plotting


def _patch_upsetplot_cow():
    """
    upsetplot 0.9.0's plot_matrix uses Series.fillna(..., inplace=True) to fill
    default dot colors. Under pandas >= 3.0 (copy-on-write always on) those
    inplace writes are silently dropped, leaving NaN facecolors that crash
    matplotlib. Reimplement plot_matrix using non-inplace fillna.
    """
    try:
        if int(pd.__version__.split(".")[0]) < 3:
            return  # older pandas: inplace works, no patch needed
    except Exception:
        pass

    def plot_matrix(self, ax):
        ax = self._reorient(ax)
        data = self.intersections
        n_cats = data.index.nlevels
        inclusion = data.index.to_frame().values

        styles = [
            [
                self.subset_styles[i]
                if inclusion[i, j]
                else {"facecolor": self._other_dots_color, "linewidth": 0}
                for j in range(n_cats)
            ]
            for i in range(len(data))
        ]
        styles = sum(styles, [])
        style_columns = {
            "facecolor": "facecolors",
            "edgecolor": "edgecolors",
            "linewidth": "linewidths",
            "linestyle": "linestyles",
            "hatch": "hatch",
        }
        styles = (
            pd.DataFrame(styles)
            .reindex(columns=style_columns.keys())
            .astype(
                {
                    "facecolor": "O",
                    "edgecolor": "O",
                    "linewidth": float,
                    "linestyle": "O",
                    "hatch": "O",
                }
            )
        )
        # non-inplace fills (CoW-safe)
        styles["linewidth"] = styles["linewidth"].fillna(1)
        styles["facecolor"] = styles["facecolor"].fillna(self._facecolor)
        styles["edgecolor"] = styles["edgecolor"].fillna(styles["facecolor"])
        styles["linestyle"] = styles["linestyle"].fillna("solid")
        del styles["hatch"]

        x = np.repeat(np.arange(len(data)), n_cats)
        y = np.tile(np.arange(n_cats), len(data))

        s = (self._element_size * 0.35) ** 2 if self._element_size is not None else 200
        ax.scatter(
            *self._swapaxes(x, y),
            s=s,
            zorder=10,
            **styles.rename(columns=style_columns),
        )

        if self._with_lines:
            idx = np.flatnonzero(inclusion)
            line_data = (
                pd.Series(y[idx], index=x[idx]).groupby(level=0).aggregate(["min", "max"])
            )
            colors = pd.Series(
                [
                    style.get("edgecolor", style.get("facecolor", self._facecolor))
                    for style in self.subset_styles
                ],
                name="color",
            )
            line_data = line_data.join(colors)
            ax.vlines(
                line_data.index.values,
                line_data["min"],
                line_data["max"],
                lw=2,
                colors=line_data["color"],
                zorder=5,
            )

        tick_axis = ax.yaxis
        tick_axis.set_ticks(np.arange(n_cats))
        tick_axis.set_ticklabels(
            data.index.names, rotation=0 if self._horizontal else -90
        )
        ax.xaxis.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        if not self._horizontal:
            ax.yaxis.set_ticks_position("top")
        ax.set_frame_on(False)
        ax.set_xlim(-0.5, x[-1] + 0.5, auto=False)
        ax.grid(False)

    _up_plotting.UpSet.plot_matrix = plot_matrix


_patch_upsetplot_cow()

MODELS = ["gpt5_mini", "devstral", "deepseek_r1", "deepseek_v3"]
TITLES = ["GPT5-mini", "Devstral-small", "DeepSeek-R1", "DeepSeek-V3"]

BENCHMARK = "SWE-Bench-Verified"

DIFFICULTY_MAP = {
    "under15min": "Easy",
    "under1h": "Medium",
    "under4h": "Hard",
}
DIFF_ORDER = ["Easy", "Medium", "Hard"]
COLORS = ["#88CCEE", "#DDCC77", "#CC6677"]  # Easy, Medium, Hard


def load_setting(root, benchmark, setting, model):
    """Concat all trajectory_metrics.csv under the model/setting analysis folder."""
    folder = Path(root) / benchmark / setting / model / "analysis"
    print("FOLDER:", folder)
    files = sorted(folder.rglob("trajectory_metrics.csv"))
    if not files:
        print(f"  [warn] no trajectory_metrics.csv under {folder}", file=sys.stderr)
        return None
    frames = []
    for fp in files:
        try:
            frames.append(pd.read_csv(fp))
        except Exception as e:
            print(f"  [warn] skipping {fp}: {e}", file=sys.stderr)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[["instance", "resolution", "debug_difficulty"]].copy()
    # Treat anything that isn't 'resolved' (e.g. 'unsubmitted') as 'unresolved'
    df["resolution"] = df["resolution"].where(df["resolution"] == "resolved", "unresolved")
    return df


def build_model_df(root, benchmark, setting_a, setting_b, model, name_a, name_b):
    """
    Join setting A and B per instance and return a tidy frame:
        instance_id | difficulty | {name_a} | {name_b}
    where the two setting columns hold 'resolved'/'unresolved' strings and
    difficulty is Easy/Medium/Hard.
    """
    a = load_setting(root, benchmark, setting_a, model)
    b = load_setting(root, benchmark, setting_b, model)
    if a is None or b is None:
        return None

    a = a.rename(columns={"resolution": "res_a", "debug_difficulty": "diff_a"})
    b = b.rename(columns={"resolution": "res_b", "debug_difficulty": "diff_b"})

    merged = a.merge(b, on="instance", how="outer")
    if merged.empty:
        print(f"  [warn] no instances for {model}", file=sys.stderr)
        return None

    # An instance missing from a setting means it wasn't resolved there.
    merged["res_a"] = merged["res_a"].fillna("unresolved")
    merged["res_b"] = merged["res_b"].fillna("unresolved")

    # difficulty: prefer setting A's, fall back to B's
    merged["difficulty"] = (
        merged["diff_a"].map(DIFFICULTY_MAP)
        .fillna(merged["diff_b"].map(DIFFICULTY_MAP))
    )
    unmapped = merged["difficulty"].isna().sum()
    if unmapped:
        bad_vals = sorted(
            set(merged.loc[merged["difficulty"].isna(), "diff_a"].dropna())
            | set(merged.loc[merged["difficulty"].isna(), "diff_b"].dropna())
        )
        print(f"  [warn] {model}: dropping {unmapped} rows with unmapped difficulty "
              f"(values: {bad_vals})", file=sys.stderr)
        merged = merged[merged["difficulty"].notna()]
    if merged.empty:
        return None

    out = merged.rename(columns={
        "instance": "instance_id",
        "res_a": name_a,
        "res_b": name_b,
    })[["instance_id", "difficulty", name_a, name_b]].reset_index(drop=True)
    return out


def print_resolution_breakdown(model_df, title, name_a, name_b):
    """Print counts: resolved by both, by each only, and by neither."""
    a = model_df[name_a] == "resolved"
    b = model_df[name_b] == "resolved"
    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    neither = int((~a & ~b).sum())
    total = len(model_df)
    print(f"\n[{title}] resolution breakdown (n={total}):")
    print(f"  Resolved by BOTH ({name_a} & {name_b}): {both}")
    print(f"  Resolved by {name_a} only:              {only_a}")
    print(f"  Resolved by {name_b} only:              {only_b}")
    print(f"  Resolved by NEITHER:                    {neither}")


def make_upset_image(model_df, name_a, name_b):
    """
    Render one UpSet plot from a tidy frame with columns:
        instance_id | difficulty | {name_a} | {name_b}
    The two setting columns hold 'resolved'/'unresolved' strings.
    """
    df_bool = model_df[["difficulty", name_a, name_b]].copy()
    df_bool[[name_a, name_b]] = df_bool[[name_a, name_b]].apply(lambda c: c == "resolved")
    df_bool = df_bool.rename(columns={"difficulty": "Difficulty"})
    df_bool["Difficulty"] = pd.Categorical(
        df_bool["Difficulty"], categories=DIFF_ORDER, ordered=True
    )
    df_bool = df_bool.set_index([name_a, name_b])

    upset = UpSet(
        df_bool,
        intersection_plot_elements=0,
        show_counts=True,
        sort_categories_by=None,
        facecolor="black",
    )
    upset.add_stacked_bars(
        by="Difficulty", colors=COLORS, title="Instances by Difficulty", elements=10
    )
    upset.plot()
    fig = plt.gcf()

    for ax in fig.get_axes():
        leg = ax.get_legend()
        ax.grid(False, axis="y")
        if leg:
            leg.remove()

    for ax in fig.get_axes():
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=12)
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=14, fontweight="bold")

    i = 0
    for ax in fig.get_axes():
        i += 1
        for text in ax.texts:
            # Normalize any array-valued positions to scalars (avoids draw crash)
            x, y = text.get_position()
            try:
                x = float(np.asarray(x).item())
                y = float(np.asarray(y).item())
            except (ValueError, TypeError):
                x = float(np.asarray(x).ravel()[0])
                y = float(np.asarray(y).ravel()[0])
            if i < 4:
                text.set_position((x, y))
                text.set_fontsize(18)
                continue
            text.set_fontsize(26)
            text.set_rotation(90)
            text.set_position((x, y + 5))
            text.set_va("bottom")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = mpimg.imread(buf)
    buf.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setting_a", help="First setting (Plan axis)")
    ap.add_argument("setting_b", help="Second setting (other axis)")
    ap.add_argument("--root", default="artifacts", help="Root dir (default: artifacts)")
    ap.add_argument("--benchmark", default=BENCHMARK)
    ap.add_argument("--out", default="output")
    ap.add_argument("--name-a", default=None, help="Label for setting A axis")
    ap.add_argument("--name-b", default=None, help="Label for setting B axis")
    args = ap.parse_args()

    name_a = args.name_a or args.setting_a
    name_b = args.name_b or args.setting_b

    images = []
    for model, title in zip(MODELS, TITLES):
        print(f"Processing {model} ({title})...")
        mdf = build_model_df(args.root, args.benchmark, args.setting_a, args.setting_b, model, name_a, name_b)
        if mdf is None:
            print(f"  [skip] {model}: no data", file=sys.stderr)
            images.append(None)
            continue
        print_resolution_breakdown(mdf, title, name_a, name_b)
        images.append(make_upset_image(mdf, name_a, name_b))

    if all(img is None for img in images):
        print("No data for any model; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, 4, figsize=(10, 8))
    for ax, img, title in zip(axes, images, TITLES):
        if img is not None:
            ax.imshow(img)
        ax.set_xlabel(title, fontsize=15, fontweight="bold", labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_elements = [
        Patch(facecolor="#88CCEE", label="Easy"),
        Patch(facecolor="#DDCC77", label="Medium"),
        Patch(facecolor="#CC6677", label="Hard"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=14,
        frameon=True,
        ncol=3,
        bbox_to_anchor=(0.5, 0.74),
        handleheight=1.0,
        handlelength=1.0,
    )

    plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99, bottom=0.08)

    out = Path(f"{args.out}/upset_{args.setting_b}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"\nSaved combined UpSet plot to {out}")


if __name__ == "__main__":
    main()
