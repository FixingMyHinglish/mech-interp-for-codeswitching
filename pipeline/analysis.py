from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GroupComparisons:
    layer_metrics: pd.DataFrame
    attention_diff: pd.DataFrame
    neuron_diff: pd.DataFrame
    summary: pd.DataFrame
    pairwise_summary: pd.DataFrame
    conditions: list[str]
    reference_condition: str


METRIC_COLS = [
    "hidden_norm",
    "delta_norm",
    "cosine_to_final",
    "lens_entropy",
    "lens_top1_prob",
    "lens_to_final_kl",
    "next_token_nll",
]



def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))



def _pick_reference(conditions: list[str], preferred: str | None) -> str:
    if preferred is not None:
        if preferred not in conditions:
            raise ValueError(
                f"Configured reference_condition '{preferred}' not found in dataset conditions: {conditions}"
            )
        return preferred

    for c in conditions:
        if "english" in c.lower():
            return c
    return conditions[0]



def aggregate_layer_metrics(per_sample_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in per_sample_records:
        cond = rec["condition"]
        sid = rec["id"]
        source_id = rec.get("source_id", sid)
        domain = rec["domain"]
        for layer_idx, vals in rec["layer_metrics"].items():
            rows.append(
                {
                    "id": sid,
                    "source_id": source_id,
                    "domain": domain,
                    "condition": cond,
                    "layer": layer_idx,
                    "hidden_norm": vals["hidden_norm"],
                    "delta_norm": vals["delta_norm"],
                    "cosine_to_final": vals["cosine_to_final"],
                    "lens_entropy": vals["lens_entropy"],
                    "lens_top1_prob": vals["lens_top1_prob"],
                    "lens_to_final_kl": vals["lens_to_final_kl"],
                    "next_token_nll": vals["next_token_nll"],
                }
            )
    return pd.DataFrame(rows)



def aggregate_attention_metrics(per_sample_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in per_sample_records:
        sid = rec["id"]
        source_id = rec.get("source_id", sid)
        cond = rec["condition"]
        domain = rec["domain"]
        for layer_idx, entropies in rec["attention_entropy"].items():
            for head_idx, h_val in enumerate(entropies):
                rows.append(
                    {
                        "id": sid,
                        "source_id": source_id,
                        "domain": domain,
                        "condition": cond,
                        "layer": layer_idx,
                        "head": head_idx,
                        "attention_entropy": h_val,
                    }
                )
    cols = ["id", "source_id", "domain", "condition", "layer", "head", "attention_entropy"]
    return pd.DataFrame(rows, columns=cols)



def aggregate_neuron_metrics(per_sample_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in per_sample_records:
        sid = rec["id"]
        source_id = rec.get("source_id", sid)
        cond = rec["condition"]
        domain = rec["domain"]
        for layer_idx, top_neurons in rec["top_neurons"].items():
            for neuron_idx, score in top_neurons:
                rows.append(
                    {
                        "id": sid,
                        "source_id": source_id,
                        "domain": domain,
                        "condition": cond,
                        "layer": layer_idx,
                        "neuron": neuron_idx,
                        "activation": score,
                    }
                )
    cols = ["id", "source_id", "domain", "condition", "layer", "neuron", "activation"]
    return pd.DataFrame(rows, columns=cols)



def build_neuron_event_table(per_sample_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in per_sample_records:
        sid = rec["id"]
        source_id = rec.get("source_id", sid)
        cond = rec["condition"]
        domain = rec["domain"]
        concept_label = rec.get("concept_label", domain)
        n_tokens = rec.get("n_tokens", None)
        for layer_idx, top_neurons in rec["top_neurons"].items():
            for rank, (neuron_idx, score) in enumerate(top_neurons, start=1):
                rows.append(
                    {
                        "id": sid,
                        "source_id": source_id,
                        "domain": domain,
                        "concept_label": concept_label,
                        "condition": cond,
                        "n_tokens": n_tokens,
                        "layer": layer_idx,
                        "neuron": neuron_idx,
                        "rank_in_sample_layer": rank,
                        "activation": score,
                    }
                )
    cols = [
        "id",
        "source_id",
        "domain",
        "concept_label",
        "condition",
        "n_tokens",
        "layer",
        "neuron",
        "rank_in_sample_layer",
        "activation",
    ]
    return pd.DataFrame(rows, columns=cols)



def aggregate_neuron_tendency(neuron_events_df: pd.DataFrame) -> pd.DataFrame:
    if neuron_events_df.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "domain",
                "layer",
                "neuron",
                "event_count",
                "activation_mean",
                "activation_max",
                "activation_min",
                "avg_rank_in_sample_layer",
            ]
        )

    grouped = (
        neuron_events_df.groupby(["condition", "domain", "layer", "neuron"], as_index=False)
        .agg(
            event_count=("activation", "size"),
            activation_mean=("activation", "mean"),
            activation_max=("activation", "max"),
            activation_min=("activation", "min"),
            avg_rank_in_sample_layer=("rank_in_sample_layer", "mean"),
        )
        .sort_values(
            ["condition", "domain", "layer", "event_count", "activation_mean"],
            ascending=[True, True, True, False, False],
        )
    )
    return grouped



def build_sample_neuron_contrasts(
    neuron_events_df: pd.DataFrame,
    reference_condition: str = "english",
    topn_for_jaccard: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if neuron_events_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "source_id",
                    "layer",
                    "neuron",
                    "dominant_condition",
                    "dominant_margin",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "source_id",
                    "layer",
                    "condition_a",
                    "condition_b",
                    "cosine_similarity",
                    "cosine_distance",
                    "topn_jaccard",
                ]
            ),
        )

    grouped = (
        neuron_events_df.groupby(["source_id", "condition", "layer", "neuron"], as_index=False)["activation"]
        .mean()
    )
    conditions = sorted(grouped["condition"].unique().tolist())

    wide = (
        grouped.pivot(
            index=["source_id", "layer", "neuron"],
            columns="condition",
            values="activation",
        )
        .fillna(0.0)
        .reset_index()
    )

    for cond in conditions:
        if cond not in wide.columns:
            wide[cond] = 0.0

    cond_values = wide[conditions].to_numpy(dtype=np.float64)
    max_idx = cond_values.argmax(axis=1)
    sorted_vals = np.sort(cond_values, axis=1)
    margins = sorted_vals[:, -1] - sorted_vals[:, -2] if sorted_vals.shape[1] > 1 else sorted_vals[:, -1]
    wide["dominant_condition"] = [conditions[i] for i in max_idx]
    wide["dominant_margin"] = margins

    if reference_condition in conditions:
        for cond in conditions:
            if cond == reference_condition:
                continue
            wide[f"delta_{cond}_minus_{reference_condition}"] = wide[cond] - wide[reference_condition]

    distance_rows = []
    for (source_id, layer), g in grouped.groupby(["source_id", "layer"]):
        mat = (
            g.pivot(index="neuron", columns="condition", values="activation")
            .fillna(0.0)
        )
        for cond in conditions:
            if cond not in mat.columns:
                mat[cond] = 0.0
        mat = mat[conditions]

        top_sets: dict[str, set[int]] = {}
        for cond in conditions:
            s = mat[cond]
            k = min(topn_for_jaccard, len(s))
            top_sets[cond] = set(s.nlargest(k).index.tolist())

        for cond_a, cond_b in combinations(conditions, 2):
            va = mat[cond_a].to_numpy(dtype=np.float64)
            vb = mat[cond_b].to_numpy(dtype=np.float64)
            denom = (np.linalg.norm(va) * np.linalg.norm(vb))
            cos = float(np.dot(va, vb) / denom) if denom > 0 else 0.0
            inter = len(top_sets[cond_a].intersection(top_sets[cond_b]))
            union = len(top_sets[cond_a].union(top_sets[cond_b]))
            jacc = float(inter / union) if union else 0.0

            distance_rows.append(
                {
                    "source_id": source_id,
                    "layer": int(layer),
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "cosine_similarity": cos,
                    "cosine_distance": 1.0 - cos,
                    "topn_jaccard": jacc,
                }
            )

    layer_distance_df = pd.DataFrame(distance_rows)
    return wide, layer_distance_df



def compare_conditions(
    layer_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    neuron_df: pd.DataFrame,
    reference_condition: str | None = None,
) -> GroupComparisons:
    conditions = sorted(layer_df["condition"].unique().tolist())
    if len(conditions) < 2:
        raise ValueError(f"Need at least 2 conditions, found {conditions}")

    ref = _pick_reference(conditions, reference_condition)
    others = [c for c in conditions if c != ref]

    grouped = (
        layer_df.groupby(["condition", "layer"])[METRIC_COLS]
        .mean()
        .reset_index()
        .pivot(index="layer", columns="condition", values=METRIC_COLS)
    )

    layer_rows = []
    for layer in grouped.index:
        for metric in METRIC_COLS:
            ref_val = float(grouped.loc[layer, (metric, ref)])
            for cond in others:
                cond_val = float(grouped.loc[layer, (metric, cond)])
                layer_rows.append(
                    {
                        "layer": int(layer),
                        "metric": metric,
                        "reference_condition": ref,
                        "condition": cond,
                        "reference_mean": ref_val,
                        "condition_mean": cond_val,
                        "delta_vs_reference": cond_val - ref_val,
                    }
                )
    layer_diff = pd.DataFrame(layer_rows)

    if attention_df.empty:
        attention_diff = pd.DataFrame(
            columns=[
                "layer",
                "head",
                "reference_condition",
                "condition",
                "reference_mean",
                "condition_mean",
                "delta_vs_reference",
            ]
        )
    else:
        attn_grouped = (
            attention_df.groupby(["condition", "layer", "head"])["attention_entropy"]
            .mean()
            .reset_index()
            .pivot(index=["layer", "head"], columns="condition", values="attention_entropy")
            .reset_index()
        )
        attn_rows = []
        for _, r in attn_grouped.iterrows():
            for cond in others:
                if ref not in r or cond not in r:
                    continue
                attn_rows.append(
                    {
                        "layer": int(r["layer"]),
                        "head": int(r["head"]),
                        "reference_condition": ref,
                        "condition": cond,
                        "reference_mean": float(r[ref]),
                        "condition_mean": float(r[cond]),
                        "delta_vs_reference": float(r[cond] - r[ref]),
                    }
                )
        attention_diff = pd.DataFrame(attn_rows)

    neuron_means = (
        neuron_df.groupby(["condition", "layer", "neuron"])["activation"]
        .mean()
        .reset_index()
        .pivot(index=["layer", "neuron"], columns="condition", values="activation")
        .fillna(0)
        .reset_index()
    )
    neuron_rows = []
    for _, r in neuron_means.iterrows():
        for cond in others:
            neuron_rows.append(
                {
                    "layer": int(r["layer"]),
                    "neuron": int(r["neuron"]),
                    "reference_condition": ref,
                    "condition": cond,
                    "reference_mean": float(r[ref]),
                    "condition_mean": float(r[cond]),
                    "delta_vs_reference": float(r[cond] - r[ref]),
                }
            )
    neuron_diff = pd.DataFrame(neuron_rows)

    summary_rows = []
    for metric in METRIC_COLS:
        ref_mean = _safe_mean(layer_df[layer_df["condition"] == ref][metric].tolist())
        for cond in others:
            cond_mean = _safe_mean(layer_df[layer_df["condition"] == cond][metric].tolist())
            summary_rows.append(
                {
                    "metric": metric,
                    "reference_condition": ref,
                    "condition": cond,
                    "reference_overall": ref_mean,
                    "condition_overall": cond_mean,
                    "delta_vs_reference": cond_mean - ref_mean,
                }
            )
    summary = pd.DataFrame(summary_rows)

    pair_rows = []
    for metric in METRIC_COLS:
        means = layer_df.groupby("condition")[metric].mean().to_dict()
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i + 1 :]:
                pair_rows.append(
                    {
                        "metric": metric,
                        "condition_a": c1,
                        "condition_b": c2,
                        "mean_a": float(means[c1]),
                        "mean_b": float(means[c2]),
                        "delta_b_minus_a": float(means[c2] - means[c1]),
                    }
                )
    pairwise_summary = pd.DataFrame(pair_rows)

    return GroupComparisons(
        layer_metrics=layer_diff,
        attention_diff=attention_diff,
        neuron_diff=neuron_diff,
        summary=summary,
        pairwise_summary=pairwise_summary,
        conditions=conditions,
        reference_condition=ref,
    )
