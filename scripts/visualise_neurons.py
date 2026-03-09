"""
visualise_neurons.py

Generates three figures summarising which layers and neurons differ
between code-switching and language confusion conditions.

Outputs (written to visuals/neuron_analysis/):
  figure1_selectivity_scatter.png  -- per-neuron specificity by layer
  figure2_activation_heatmap.png   -- top candidate neurons × all conditions
  figure3_layer_distribution.png   -- count of condition-specific neurons per layer

Usage:
    python scripts/visualise_neurons.py \
        --source_run outputs/serious_run_french_002 \
        --top_k 20 \
        --output_dir visuals/neuron_analysis
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── colour palette ──────────────────────────────────────────────────────────
CS_COLOUR   = "#4C9BE8"   # blue  – code-switched
CONF_COLOUR = "#E8654C"   # red   – confused
COND_PALETTE = {
    "english":         "#888888",
    "target_language": "#5BAD6F",
    "code_switched":   CS_COLOUR,
    "confused":        CONF_COLOUR,
}
COND_LABELS = {
    "english":         "English",
    "target_language": "French",
    "code_switched":   "Code-switched",
    "confused":        "Confused",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_neuron_tendency(source_run: Path) -> pd.DataFrame:
    path = source_run / "tables" / "neuron_tendency.csv"
    df = pd.read_csv(path)
    return df


def compute_specificity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (layer, neuron), compute a specificity score per condition:
        specificity(cond) = activation_mean(cond) - max(activation_mean(all other conds))

    A high positive score means the neuron fires strongly for that condition
    and weakly for every other condition.
    """
    pivot = (
        df.groupby(["layer", "neuron", "condition"])["activation_mean"]
        .mean()
        .reset_index()
        .pivot_table(index=["layer", "neuron"], columns="condition", values="activation_mean")
        .fillna(0)
        .reset_index()
    )

    conditions = [c for c in ["english", "target_language", "code_switched", "confused"]
                  if c in pivot.columns]

    for cond in conditions:
        others = [c for c in conditions if c != cond]
        pivot[f"specificity_{cond}"] = pivot[cond] - pivot[others].max(axis=1)

    return pivot


def select_candidates(pivot: pd.DataFrame, top_k: int):
    """Return top_k CS-specific and top_k confusion-specific neurons."""
    cs   = (pivot.sort_values("specificity_code_switched", ascending=False)
                 .head(top_k)
                 .copy())
    conf = (pivot.sort_values("specificity_confused", ascending=False)
                 .head(top_k)
                 .copy())
    cs["category"]   = "code_switched"
    conf["category"] = "confused"
    return cs, conf


def neuron_label(layer: int, neuron: int) -> str:
    return f"L{layer}:N{neuron}"


# ── Figure 1 – specificity scatter ──────────────────────────────────────────

def plot_selectivity_scatter(cs: pd.DataFrame, conf: pd.DataFrame,
                             pivot: pd.DataFrame, out_path: Path):
    """
    Each dot is a neuron. X = layer, Y = specificity score for its winning
    condition. Blue = CS-specific, red = confusion-specific.
    Size scales with the neuron's mean activation in its winning condition.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for df_sub, cond, colour, label in [
        (cs,   "code_switched", CS_COLOUR,   "Code-switched specific"),
        (conf, "confused",      CONF_COLOUR, "Confusion specific"),
    ]:
        spec_col = f"specificity_{cond}"
        act_col  = cond if cond in df_sub.columns else "activation_mean"
        sizes    = (df_sub[act_col].clip(lower=0) * 300 + 40) if act_col in df_sub.columns else 80

        ax.scatter(
            df_sub["layer"],
            df_sub[spec_col],
            s=sizes,
            c=colour,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.6,
            label=label,
            zorder=3,
        )

        # label the top-5 per category
        for _, row in df_sub.nlargest(5, spec_col).iterrows():
            ax.annotate(
                neuron_label(int(row["layer"]), int(row["neuron"])),
                xy=(row["layer"], row[spec_col]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=colour,
            )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Specificity score\n(target condition − max of others)", fontsize=11)
    ax.set_title("Neuron Selectivity by Layer: Code-switching vs Language Confusion",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure 1 saved → {out_path}")


# ── Figure 2 – activation heatmap ────────────────────────────────────────────

def plot_activation_heatmap(cs: pd.DataFrame, conf: pd.DataFrame,
                            pivot: pd.DataFrame, out_path: Path):
    """
    Rows = top candidate neurons (CS-specific first, then confusion-specific).
    Columns = conditions.
    Colour = mean activation across all samples.
    """
    conditions = [c for c in ["english", "target_language", "code_switched", "confused"]
                  if c in pivot.columns]

    # build combined candidate list (deduplicated)
    cs_labels   = [neuron_label(int(r.layer), int(r.neuron)) for _, r in cs.iterrows()]
    conf_labels = [neuron_label(int(r.layer), int(r.neuron)) for _, r in conf.iterrows()]
    seen, ordered = set(), []
    for lbl in cs_labels + conf_labels:
        if lbl not in seen:
            seen.add(lbl)
            ordered.append(lbl)

    # map label → activation row
    rows = []
    for lbl in ordered:
        layer_n, neuron_n = lbl.split(":N")
        layer_n = int(layer_n[1:])
        neuron_n = int(neuron_n)
        row_data = pivot[(pivot["layer"] == layer_n) & (pivot["neuron"] == neuron_n)]
        if row_data.empty:
            rows.append({c: 0.0 for c in conditions})
        else:
            rows.append({c: float(row_data[c].iloc[0]) for c in conditions if c in row_data.columns})

    heat_df = pd.DataFrame(rows, index=ordered)[conditions]
    heat_df.columns = [COND_LABELS.get(c, c) for c in conditions]

    # row colours to indicate category
    row_colours = []
    cs_set = set(cs_labels)
    for lbl in ordered:
        row_colours.append(CS_COLOUR if lbl in cs_set else CONF_COLOUR)

    fig, ax = plt.subplots(figsize=(8, max(6, len(ordered) * 0.38)))

    sns.heatmap(
        heat_df,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        linewidths=0.4,
        linecolor="#e0e0e0",
        cbar_kws={"label": "Mean activation", "shrink": 0.6},
        annot_kws={"size": 8},
    )

    # colour the y-tick labels
    for ytick, colour in zip(ax.get_yticklabels(), row_colours):
        ytick.set_color(colour)
        ytick.set_fontweight("bold")
        ytick.set_fontsize(8)

    ax.set_title("Top Candidate Neuron Activations Across Conditions\n"
                 "(blue labels = CS-specific, red labels = confusion-specific)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_ylabel("Neuron", fontsize=11)
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure 2 saved → {out_path}")


# ── Figure 3 – layer distribution bar chart ──────────────────────────────────

def plot_layer_distribution(cs: pd.DataFrame, conf: pd.DataFrame,
                            n_layers: int, out_path: Path):
    """
    Grouped bar chart: for each layer, how many CS-specific and
    confusion-specific neurons were found.
    """
    cs_counts   = cs["layer"].value_counts().reindex(range(n_layers), fill_value=0)
    conf_counts = conf["layer"].value_counts().reindex(range(n_layers), fill_value=0)

    # Only show layers that have at least one candidate neuron
    active = sorted(set(cs["layer"].tolist() + conf["layer"].tolist()))

    x   = np.arange(len(active))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(active) * 0.7), 5))

    bars_cs   = ax.bar(x - w/2, [cs_counts[l]   for l in active],
                       width=w, color=CS_COLOUR,   label="Code-switched specific",
                       edgecolor="white", linewidth=0.5)
    bars_conf = ax.bar(x + w/2, [conf_counts[l] for l in active],
                       width=w, color=CONF_COLOUR, label="Confusion specific",
                       edgecolor="white", linewidth=0.5)

    # value labels on bars
    for bar in list(bars_cs) + list(bars_conf):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                    str(int(h)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in active], fontsize=9)
    ax.set_ylabel("Number of condition-specific neurons", fontsize=11)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_title("Distribution of Condition-Specific Neurons Across Layers",
                 fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure 3 saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise condition-specific neurons")
    parser.add_argument("--source_run", default="outputs/serious_run_french_002",
                        help="Path to a completed pipeline run folder")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top neurons to highlight per condition")
    parser.add_argument("--output_dir", default="visuals/neuron_analysis",
                        help="Where to save the figures")
    args = parser.parse_args()

    source_run = Path(args.source_run)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data from: {source_run}")
    df    = load_neuron_tendency(source_run)
    pivot = compute_specificity(df)
    cs, conf = select_candidates(pivot, top_k=args.top_k)

    # infer number of layers from data
    n_layers = int(df["layer"].max()) + 1

    print(f"\nTop-{args.top_k} CS-specific neurons:")
    print(cs[["layer", "neuron", "specificity_code_switched"]].to_string(index=False))
    print(f"\nTop-{args.top_k} confusion-specific neurons:")
    print(conf[["layer", "neuron", "specificity_confused"]].to_string(index=False))

    print(f"\nGenerating figures → {out_dir}/")
    plot_selectivity_scatter(cs, conf, pivot, out_dir / "figure1_selectivity_scatter.png")
    plot_activation_heatmap(cs, conf, pivot,  out_dir / "figure2_activation_heatmap.png")
    plot_layer_distribution(cs, conf, n_layers, out_dir / "figure3_layer_distribution.png")

    print("\nDone. All figures saved.")


if __name__ == "__main__":
    main()
