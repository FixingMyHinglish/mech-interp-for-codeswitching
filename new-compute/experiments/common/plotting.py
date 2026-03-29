from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def format_offset(relative_offset: int) -> str:
    return f"{relative_offset:+d}".replace("+", "plus_").replace("-", "minus_")


def write_neuron_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: str | Path,
) -> None:
    if df.empty:
        return

    max_layer = int(df["layer"].max())
    max_neuron = int(df["neuron"].max())
    matrix = np.full((max_layer + 1, max_neuron + 1), np.nan, dtype=np.float64)
    for row in df.itertuples(index=False):
        matrix[int(row.layer), int(row.neuron)] = float(getattr(row, value_col))

    finite = matrix[np.isfinite(matrix)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    zmax = max(max_abs, 1e-9)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                colorscale="RdBu",
                zmid=0.0,
                zmin=-zmax,
                zmax=zmax,
                colorbar={"title": value_col},
                hovertemplate="layer=%{y}<br>neuron=%{x}<br>value=%{z:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Neuron",
        yaxis_title="Layer",
        template="plotly_white",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")

