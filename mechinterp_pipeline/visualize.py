from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METRICS = [
    "hidden_norm",
    "delta_norm",
    "cosine_to_final",
    "lens_entropy",
    "lens_top1_prob",
    "lens_to_final_kl",
    "next_token_nll",
]


def plot_layer_metrics(layer_diff: pd.DataFrame, out_path: Path, ref: str) -> None:
    fig = px.line(
        layer_diff,
        x="layer",
        y="delta_vs_reference",
        color="condition",
        facet_row="metric",
        title=f"Layer-wise Metric Delta vs {ref}",
        markers=True,
        height=2400,
    )
    fig.update_layout(template="plotly_white")
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_layer_metrics_absolute(layer_df: pd.DataFrame, out_path: Path, condition: str) -> None:
    subset = layer_df[layer_df["condition"] == condition]
    long_df = subset.melt(
        id_vars=["id", "domain", "condition", "layer"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    agg = long_df.groupby(["layer", "metric"], as_index=False)["value"].mean()
    fig = px.line(
        agg,
        x="layer",
        y="value",
        facet_row="metric",
        title=f"Layer-wise Absolute Metrics ({condition})",
        markers=True,
        height=2400,
    )
    fig.update_layout(template="plotly_white")
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_attention_heatmap(
    attn_diff: pd.DataFrame,
    out_path: Path,
    ref: str,
    condition: str,
) -> None:
    subset = attn_diff[attn_diff["condition"] == condition]
    pivot = subset.pivot(index="layer", columns="head", values="delta_vs_reference")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="Entropy Delta"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Attention Entropy Difference ({condition} - {ref})",
        xaxis_title="Head",
        yaxis_title="Layer",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_attention_heatmap_absolute(
    attn_df: pd.DataFrame,
    out_path: Path,
    condition: str,
) -> None:
    subset = attn_df[attn_df["condition"] == condition]
    if subset.empty:
        return
    agg = subset.groupby(["layer", "head"], as_index=False)["attention_entropy"].mean()
    pivot = agg.pivot(index="layer", columns="head", values="attention_entropy")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            colorbar=dict(title="Entropy"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Attention Entropy (Absolute, {condition})",
        xaxis_title="Head",
        yaxis_title="Layer",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_neuron_deltas(
    neuron_diff: pd.DataFrame,
    out_path: Path,
    ref: str,
    condition: str,
    topn: int = 100,
) -> None:
    subset = neuron_diff[neuron_diff["condition"] == condition]
    top = subset.reindex(
        subset["delta_vs_reference"].abs().sort_values(ascending=False).index
    ).head(topn)
    top = top.copy()
    top["layer_neuron"] = top.apply(lambda r: f"L{int(r['layer'])}:N{int(r['neuron'])}", axis=1)

    fig = px.bar(
        top,
        x="layer_neuron",
        y="delta_vs_reference",
        title=f"Top {topn} Neuron Proxy Activation Shifts ({condition} - {ref})",
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=75)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_neuron_absolute(
    neuron_df: pd.DataFrame,
    out_path: Path,
    condition: str,
    topn: int = 100,
) -> None:
    subset = neuron_df[neuron_df["condition"] == condition]
    agg = subset.groupby(["layer", "neuron"], as_index=False)["activation"].mean()
    top = agg.nlargest(topn, "activation").copy()
    top["layer_neuron"] = top.apply(lambda r: f"L{int(r['layer'])}:N{int(r['neuron'])}", axis=1)
    fig = px.bar(
        top,
        x="layer_neuron",
        y="activation",
        title=f"Top {topn} Neuron Proxy Activations (Absolute, {condition})",
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=75)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_neuron_layer_heatmap_absolute(
    neuron_df: pd.DataFrame,
    out_path: Path,
    condition: str,
    max_neurons: int = 256,
) -> None:
    subset = neuron_df[neuron_df["condition"] == condition]
    if subset.empty:
        return
    agg = subset.groupby(["layer", "neuron"], as_index=False)["activation"].mean()
    top_neurons = (
        agg.groupby("neuron", as_index=False)["activation"]
        .mean()
        .nlargest(max_neurons, "activation")["neuron"]
        .tolist()
    )
    filt = agg[agg["neuron"].isin(top_neurons)]
    pivot = filt.pivot(index="layer", columns="neuron", values="activation").fillna(0.0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            colorbar=dict(title="Activation"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Neuron Activation Map by Layer (Absolute, {condition})",
        xaxis_title="Neuron Index (Top Mean Activation)",
        yaxis_title="Layer",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_neuron_layer_3d_absolute(
    neuron_df: pd.DataFrame,
    out_path: Path,
    condition: str,
    max_points: int = 4000,
) -> None:
    subset = neuron_df[neuron_df["condition"] == condition]
    if subset.empty:
        return
    agg = subset.groupby(["layer", "neuron"], as_index=False)["activation"].mean()
    view = agg.nlargest(max_points, "activation").copy()
    fig = px.scatter_3d(
        view,
        x="layer",
        y="neuron",
        z="activation",
        color="activation",
        title=f"3D Neuron Activation Landscape (Absolute, {condition})",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        template="plotly_white",
        scene=dict(
            xaxis_title="Layer",
            yaxis_title="Neuron",
            zaxis_title="Activation",
        ),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_neuron_layer_heatmap_delta(
    neuron_diff: pd.DataFrame,
    out_path: Path,
    ref: str,
    condition: str,
    max_neurons: int = 256,
) -> None:
    subset = neuron_diff[neuron_diff["condition"] == condition]
    if subset.empty:
        return
    top_neurons = (
        subset.groupby("neuron", as_index=False)["delta_vs_reference"]
        .mean()
        .assign(abs_delta=lambda d: d["delta_vs_reference"].abs())
        .nlargest(max_neurons, "abs_delta")["neuron"]
        .tolist()
    )
    filt = subset[subset["neuron"].isin(top_neurons)]
    agg = filt.groupby(["layer", "neuron"], as_index=False)["delta_vs_reference"].mean()
    pivot = agg.pivot(index="layer", columns="neuron", values="delta_vs_reference").fillna(0.0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title=f"{condition}-{ref}"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Neuron Delta Map by Layer ({condition} - {ref})",
        xaxis_title="Neuron Index (Top Mean |Delta|)",
        yaxis_title="Layer",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_neuron_layer_3d_delta(
    neuron_diff: pd.DataFrame,
    out_path: Path,
    ref: str,
    condition: str,
    max_points: int = 4000,
) -> None:
    subset = neuron_diff[neuron_diff["condition"] == condition]
    if subset.empty:
        return
    agg = subset.groupby(["layer", "neuron"], as_index=False)["delta_vs_reference"].mean()
    view = agg.assign(abs_delta=lambda d: d["delta_vs_reference"].abs()).nlargest(max_points, "abs_delta")
    fig = px.scatter_3d(
        view,
        x="layer",
        y="neuron",
        z="delta_vs_reference",
        color="delta_vs_reference",
        title=f"3D Neuron Delta Landscape ({condition} - {ref})",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    )
    fig.update_layout(
        template="plotly_white",
        scene=dict(
            xaxis_title="Layer",
            yaxis_title="Neuron",
            zaxis_title=f"Delta ({condition}-{ref})",
        ),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_domain_metric_heatmap(
    layer_df: pd.DataFrame,
    out_path: Path,
    ref: str,
    condition: str,
) -> None:
    rows = []
    for domain, g in layer_df.groupby("domain"):
        g_ref = g[g["condition"] == ref]
        g_cond = g[g["condition"] == condition]
        for metric in METRICS:
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "delta": g_cond[metric].mean() - g_ref[metric].mean(),
                }
            )

    ddf = pd.DataFrame(rows)
    pivot = ddf.pivot(index="metric", columns="domain", values="delta")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title=f"{condition}-{ref}"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Domain-level Metric Shifts ({condition} - {ref})",
        xaxis_title="Domain",
        yaxis_title="Metric",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_domain_metric_heatmap_absolute(
    layer_df: pd.DataFrame,
    out_path: Path,
    condition: str,
) -> None:
    subset = layer_df[layer_df["condition"] == condition]
    rows = []
    for domain, g in subset.groupby("domain"):
        for metric in METRICS:
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "value": g[metric].mean(),
                }
            )
    ddf = pd.DataFrame(rows)
    pivot = ddf.pivot(index="metric", columns="domain", values="value")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Viridis",
            colorbar=dict(title="Value"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Domain-level Absolute Metrics ({condition})",
        xaxis_title="Domain",
        yaxis_title="Metric",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def render_summary_markdown(
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    out_path: Path,
    ref: str,
) -> None:
    lines = [
        "# LLM Internal Comparison Summary",
        "",
        f"Reference condition: `{ref}`",
        "",
        "## Global Metric Deltas vs Reference",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['metric']} | {row['condition']} - {ref} = {row['delta_vs_reference']:.4f}"
        )

    lines.extend(["", "## Pairwise Metric Deltas (All Conditions)"])
    for _, row in pairwise_df.iterrows():
        lines.append(
            f"- {row['metric']} | {row['condition_b']} - {row['condition_a']} = {row['delta_b_minus_a']:.4f}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
