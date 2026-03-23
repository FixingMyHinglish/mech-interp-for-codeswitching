#!/usr/bin/env python3
"""Shared neuron selection logic for the new-compute pipeline.

Both ``compute_condition_specific_neurons.py`` and
``ab_compare_hf_vs_tl_subset.py`` import from this module so the selection
algorithm stays in one place.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


NeuronKey = Tuple[int, int]


def to_keys(df: pd.DataFrame) -> set[NeuronKey]:
    """Return a set of (layer, neuron) tuples from a DataFrame slice."""
    if df.empty:
        return set()
    return set(zip(df["layer"].astype(int), df["neuron"].astype(int)))


def jaccard(a: set[NeuronKey], b: set[NeuronKey]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def run_selection(
    proxy_df: pd.DataFrame,
    activation_cutoff: float,
    importance_quantile: float,
    importance_min: float | None,
    min_domain_consistency: float,
    cs_label: str,
    confused_label: str,
    english_label: str,
    target_label: str,
) -> dict[str, pd.DataFrame]:
    """Core neuron selection pipeline shared across all compute scripts.

    Steps
    -----
    1. Filter to rows where ``activation > activation_cutoff``.
    2. Compute per-condition/domain importance:
       ``importance_score = activation_mean * coverage``
       where ``coverage = fire_count / n_samples_in_condition_domain``.
    3. Keep neurons above the per-condition/domain quantile (or absolute min).
    4. Remove English + target-language neurons from CS and confused sets.
    5. Aggregate across domains to compute cross-domain consistency.

    Returns
    -------
    dict with keys: ``fired``, ``importance``, ``important``, ``filtered``,
    ``consistent``.
    """
    required = {"source_id", "domain", "condition", "layer", "neuron", "activation"}
    missing = required.difference(proxy_df.columns)
    if missing:
        raise ValueError(f"Proxy input missing columns: {sorted(missing)}")

    # Step 1 – threshold
    fired = proxy_df[proxy_df["activation"] > activation_cutoff].copy()

    sample_counts = (
        fired.groupby(["condition", "domain"])["source_id"]
        .nunique()
        .rename("samples_in_condition_domain")
        .reset_index()
    )

    # Step 2 – importance score
    imp = (
        fired.groupby(["condition", "domain", "layer", "neuron"])
        .agg(
            fire_count=("source_id", "nunique"),
            event_count=("activation", "size"),
            activation_mean=("activation", "mean"),
            activation_median=("activation", "median"),
            activation_max=("activation", "max"),
        )
        .reset_index()
        .merge(sample_counts, on=["condition", "domain"], how="left")
    )
    imp["coverage"] = imp["fire_count"] / imp["samples_in_condition_domain"].clip(lower=1)
    imp["importance_score"] = imp["activation_mean"] * imp["coverage"]

    # Step 3 – cutoff
    if importance_min is not None:
        imp["importance_cutoff"] = float(importance_min)
        important = imp[imp["importance_score"] >= imp["importance_cutoff"]].copy()
    else:
        q = (
            imp.groupby(["condition", "domain"])["importance_score"]
            .quantile(importance_quantile)
            .rename("importance_cutoff")
            .reset_index()
        )
        imp = imp.merge(q, on=["condition", "domain"], how="left")
        important = imp[imp["importance_score"] >= imp["importance_cutoff"]].copy()

    # Step 4 – remove base-language neurons
    base_df = important[important["condition"].isin([english_label, target_label])]
    base_set = to_keys(base_df[["layer", "neuron"]].drop_duplicates())

    cs_df = important[important["condition"] == cs_label].copy()
    conf_df = important[important["condition"] == confused_label].copy()
    cs_df["_key"] = list(zip(cs_df["layer"], cs_df["neuron"]))
    conf_df["_key"] = list(zip(conf_df["layer"], conf_df["neuron"]))
    cs_filtered = cs_df[~cs_df["_key"].isin(base_set)].drop(columns=["_key"])
    conf_filtered = conf_df[~conf_df["_key"].isin(base_set)].drop(columns=["_key"])
    filtered = pd.concat([cs_filtered, conf_filtered], ignore_index=True)

    # Step 5 – cross-domain consistency
    domains_per_cond = (
        filtered.groupby("condition")["domain"]
        .nunique()
        .rename("n_domains_total")
        .reset_index()
    )
    consistent = (
        filtered.groupby(["condition", "layer", "neuron"])
        .agg(
            domains_present=("domain", "nunique"),
            mean_importance=("importance_score", "mean"),
            mean_activation=("activation_mean", "mean"),
            mean_coverage=("coverage", "mean"),
            max_importance=("importance_score", "max"),
        )
        .reset_index()
        .merge(domains_per_cond, on="condition", how="left")
    )
    consistent["domain_consistency"] = (
        consistent["domains_present"] / consistent["n_domains_total"].clip(lower=1)
    )
    consistent["passes_consistency"] = consistent["domain_consistency"] >= min_domain_consistency
    consistent = consistent.sort_values(
        ["condition", "passes_consistency", "mean_importance", "domain_consistency"],
        ascending=[True, False, False, False],
    )

    return {
        "fired": fired,
        "importance": imp,
        "important": important,
        "filtered": filtered,
        "consistent": consistent,
    }
