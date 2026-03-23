from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover
    KMeans = None
    LogisticRegression = None
    accuracy_score = None
    f1_score = None
    roc_auc_score = None
    silhouette_score = None
    train_test_split = None


@dataclass
class ConceptMetricOutputs:
    selectivity: pd.DataFrame
    purity: pd.DataFrame
    layer_density: pd.DataFrame
    classifier_summary: pd.DataFrame
    classifier_per_class: pd.DataFrame
    clustering_summary: pd.DataFrame
    hierarchy_consistency: pd.DataFrame
    functional_effects: pd.DataFrame


def _safe_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _kl_hist(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    mn = min(float(a.min()), float(b.min()))
    mx = max(float(a.max()), float(b.max()))
    if mx <= mn:
        return 0.0
    ha, edges = np.histogram(a, bins=bins, range=(mn, mx), density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    ha = ha + 1e-9
    hb = hb + 1e-9
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return float(np.sum(ha * np.log(ha / hb)))


def compute_selectivity(
    neuron_events_df: pd.DataFrame,
    concept_col: str,
    bins_for_kl: int = 30,
) -> pd.DataFrame:
    req = {"layer", "neuron", "activation", concept_col}
    if not req.issubset(neuron_events_df.columns):
        return pd.DataFrame()

    rows = []
    concepts = sorted(neuron_events_df[concept_col].dropna().astype(str).unique().tolist())

    for (layer, neuron), g in neuron_events_df.groupby(["layer", "neuron"]):
        all_vals = g["activation"].to_numpy(dtype=np.float64)
        for concept in concepts:
            in_vals = g[g[concept_col].astype(str) == concept]["activation"].to_numpy(dtype=np.float64)
            out_vals = g[g[concept_col].astype(str) != concept]["activation"].to_numpy(dtype=np.float64)
            if in_vals.size == 0:
                continue
            mean_in = float(in_vals.mean())
            mean_out = float(out_vals.mean()) if out_vals.size else 0.0
            diff = mean_in - mean_out
            ratio = mean_in / (mean_out + 1e-9)
            var_in = float(in_vals.var())
            var_out = float(out_vals.var()) if out_vals.size else 0.0
            pooled = np.sqrt((var_in + var_out) / 2 + 1e-9)
            effect_size = float(diff / pooled)
            kl = _kl_hist(in_vals, all_vals, bins=bins_for_kl)
            rows.append(
                {
                    "layer": int(layer),
                    "neuron": int(neuron),
                    "concept": concept,
                    "concept_count": int(in_vals.size),
                    "non_concept_count": int(out_vals.size),
                    "mean_in": mean_in,
                    "mean_out": mean_out,
                    "selectivity_diff": diff,
                    "selectivity_ratio": ratio,
                    "selectivity_effect_size": effect_size,
                    "selectivity_kl_to_global": kl,
                }
            )

    return pd.DataFrame(rows)


def compute_purity(
    neuron_events_df: pd.DataFrame,
    concept_col: str,
    top_n: int = 50,
) -> pd.DataFrame:
    req = {"layer", "neuron", "activation", concept_col}
    if not req.issubset(neuron_events_df.columns):
        return pd.DataFrame()

    rows = []
    for (layer, neuron), g in neuron_events_df.groupby(["layer", "neuron"]):
        top = g.nlargest(min(top_n, len(g)), "activation")
        if top.empty:
            continue
        counts = top[concept_col].astype(str).value_counts(normalize=True)
        purity = float(counts.iloc[0])
        entropy = _safe_entropy(counts.to_numpy(dtype=np.float64))
        rows.append(
            {
                "layer": int(layer),
                "neuron": int(neuron),
                "top_concept": str(counts.index[0]),
                "top_n": int(len(top)),
                "purity": purity,
                "label_entropy_topn": entropy,
            }
        )
    return pd.DataFrame(rows)


def compute_layer_density(
    selectivity_df: pd.DataFrame,
    diff_threshold: float = 0.5,
    min_concept_count: int = 5,
) -> pd.DataFrame:
    if selectivity_df.empty:
        return pd.DataFrame()

    assoc = selectivity_df[
        (selectivity_df["selectivity_diff"] >= diff_threshold)
        & (selectivity_df["concept_count"] >= min_concept_count)
    ].copy()

    total_neurons_per_layer = (
        selectivity_df[["layer", "neuron"]].drop_duplicates().groupby("layer").size().rename("total_neurons")
    )

    assoc_counts = (
        assoc[["layer", "concept", "neuron"]]
        .drop_duplicates()
        .groupby(["layer", "concept"])
        .size()
        .rename("associated_neurons")
        .reset_index()
    )
    assoc_counts = assoc_counts.merge(total_neurons_per_layer.reset_index(), on="layer", how="left")
    assoc_counts["concept_density"] = assoc_counts["associated_neurons"] / assoc_counts["total_neurons"].clip(lower=1)
    return assoc_counts.sort_values(["concept_density", "associated_neurons"], ascending=False)


def compute_classifier_metrics(
    neuron_events_df: pd.DataFrame,
    concept_col: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if LogisticRegression is None or train_test_split is None:
        return (
            pd.DataFrame([{"status": "skipped", "reason": "scikit-learn not installed"}]),
            pd.DataFrame(),
        )

    req = {"id", "layer", "neuron", "activation", concept_col}
    if not req.issubset(neuron_events_df.columns):
        return (pd.DataFrame([{"status": "skipped", "reason": "missing required columns"}]), pd.DataFrame())

    feats = neuron_events_df.copy()
    feats["feature"] = feats["layer"].astype(str) + ":" + feats["neuron"].astype(str)

    X = feats.pivot_table(index="id", columns="feature", values="activation", aggfunc="mean", fill_value=0.0)
    y = feats.groupby("id")[concept_col].first().astype(str)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    if y.nunique() < 2 or len(y) < 10:
        return (
            pd.DataFrame([{"status": "skipped", "reason": "insufficient samples/classes"}]),
            pd.DataFrame(),
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Keep constructor compatible across sklearn versions.
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    auc = float("nan")
    if roc_auc_score is not None and hasattr(clf, "predict_proba") and y.nunique() > 2:
        proba = clf.predict_proba(X_test)
        auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))

    summary = pd.DataFrame(
        [
            {
                "status": "ok",
                "n_samples": int(len(y)),
                "n_features": int(X.shape[1]),
                "n_classes": int(y.nunique()),
                "accuracy": acc,
                "f1_macro": f1m,
                "auc_macro_ovr": auc,
            }
        ]
    )

    per = []
    for cls in sorted(y.unique()):
        mask = (y_test == cls)
        per.append(
            {
                "concept": cls,
                "support": int(mask.sum()),
                "accuracy_one_vs_rest": float((y_pred[mask] == cls).mean()) if mask.sum() else float("nan"),
            }
        )
    return summary, pd.DataFrame(per)


def compute_clustering_metrics(selectivity_df: pd.DataFrame) -> pd.DataFrame:
    if KMeans is None or silhouette_score is None:
        return pd.DataFrame([{"status": "skipped", "reason": "scikit-learn not installed"}])
    if selectivity_df.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "empty selectivity table"}])

    mat = (
        selectivity_df.pivot_table(
            index=["layer", "neuron"],
            columns="concept",
            values="selectivity_diff",
            aggfunc="mean",
            fill_value=0.0,
        )
    )
    if mat.shape[0] < 10 or mat.shape[1] < 2:
        return pd.DataFrame([{"status": "skipped", "reason": "insufficient neurons/concepts"}])

    n_clusters = min(max(2, mat.shape[1]), 10)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(mat.values)
    sil = float(silhouette_score(mat.values, labels))

    return pd.DataFrame(
        [
            {
                "status": "ok",
                "n_neurons": int(mat.shape[0]),
                "n_concepts": int(mat.shape[1]),
                "n_clusters": int(n_clusters),
                "silhouette": sil,
            }
        ]
    )


def compute_hierarchy_consistency(selectivity_df: pd.DataFrame, hierarchy_path: str | None) -> pd.DataFrame:
    if not hierarchy_path:
        return pd.DataFrame([{"status": "skipped", "reason": "no hierarchy file provided"}])
    p = Path(hierarchy_path)
    if not p.exists():
        return pd.DataFrame([{"status": "skipped", "reason": f"hierarchy path not found: {hierarchy_path}"}])

    h = pd.read_csv(p)
    if not {"child", "parent"}.issubset(h.columns):
        return pd.DataFrame([{"status": "skipped", "reason": "hierarchy CSV must have child,parent columns"}])

    if selectivity_df.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "empty selectivity table"}])

    best = (
        selectivity_df.sort_values("selectivity_diff", ascending=False)
        .groupby(["layer", "neuron"], as_index=False)
        .first()[["layer", "neuron", "concept", "selectivity_diff"]]
        .rename(columns={"concept": "best_concept", "selectivity_diff": "best_diff"})
    )

    parent_map = dict(zip(h["child"].astype(str), h["parent"].astype(str)))
    best["parent_concept"] = best["best_concept"].astype(str).map(parent_map)

    parent_strength = (
        selectivity_df[["layer", "neuron", "concept", "selectivity_diff"]]
        .rename(columns={"concept": "parent_concept", "selectivity_diff": "parent_diff"})
    )
    merged = best.merge(parent_strength, on=["layer", "neuron", "parent_concept"], how="left")
    merged["consistent"] = (merged["parent_concept"].notna()) & (merged["parent_diff"].fillna(0.0) > 0)

    return pd.DataFrame(
        [
            {
                "status": "ok",
                "n_neurons": int(len(merged)),
                "n_with_parent": int(merged["parent_concept"].notna().sum()),
                "hierarchy_consistency_rate": float(merged["consistent"].mean()),
            }
        ]
    )


def _find_down_proj(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module | None:
    named = dict(model.named_modules())
    for prefix in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        key = f"{prefix}.{layer_idx}.mlp.down_proj"
        if key in named:
            return named[key]
    return None


def _compute_nll(model: torch.nn.Module, tokenizer, text: str, max_length: int, device: torch.device) -> float | None:
    encoded = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)
    if encoded["input_ids"].shape[1] < 3:
        return None
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits.squeeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    labels = encoded["input_ids"].squeeze(0)
    next_labels = labels[1:]
    n = min(log_probs.shape[0] - 1, next_labels.shape[0])
    nll = float((-log_probs[:n].gather(1, next_labels[:n].unsqueeze(-1)).squeeze(-1)).mean())
    return nll


def _run_with_ablation(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    max_length: int,
    device: torch.device,
    layer_neurons: list[tuple[int, int]],
    mode: str,
    scale: float,
) -> float | None:
    hooks = []
    by_layer: dict[int, list[int]] = {}
    for layer, neuron in layer_neurons:
        by_layer.setdefault(int(layer), []).append(int(neuron))

    for layer_idx, neurons in by_layer.items():
        down_proj = _find_down_proj(model, layer_idx)
        if down_proj is None:
            continue

        def make_hook(ns: list[int]):
            def hook(module, args):
                inp = list(args)
                inp[0] = inp[0].clone()
                if mode == "zero":
                    inp[0][:, :, ns] = 0.0
                elif mode == "scale":
                    inp[0][:, :, ns] = inp[0][:, :, ns] * scale
                return tuple(inp)

            return hook

        hooks.append(down_proj.register_forward_pre_hook(make_hook(neurons)))

    try:
        return _compute_nll(model, tokenizer, text, max_length=max_length, device=device)
    finally:
        for h in hooks:
            h.remove()


def compute_functional_effects(
    selectivity_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    concept_col: str,
    prepared,
    max_length: int,
    topk_neurons_per_concept: int = 20,
    max_samples: int = 200,
    random_seed: int = 42,
    boost_scale: float = 1.25,
    inhibit_scale: float = 0.75,
) -> pd.DataFrame:
    req = {"id", "text", concept_col}
    if not req.issubset(dataset_df.columns):
        return pd.DataFrame([{"status": "skipped", "reason": "dataset missing required columns"}])
    if selectivity_df.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "empty selectivity table"}])

    model = prepared.model
    tokenizer = prepared.tokenizer
    device = prepared.device

    rng = np.random.default_rng(random_seed)
    rows = []
    concepts = sorted(dataset_df[concept_col].astype(str).dropna().unique().tolist())

    full_df = dataset_df.sample(n=min(len(dataset_df), max_samples), random_state=random_seed)

    for concept in concepts:
        sdf = selectivity_df[selectivity_df["concept"].astype(str) == concept].sort_values(
            "selectivity_diff", ascending=False
        )
        top = sdf.head(topk_neurons_per_concept)
        if top.empty:
            continue
        layer_neurons = [(int(r.layer), int(r.neuron)) for r in top.itertuples()]

        subset = dataset_df[dataset_df[concept_col].astype(str) == concept]
        if len(subset) > max_samples:
            subset = subset.sample(n=max_samples, random_state=random_seed)

        for split_name, split_df in [("concept_subset", subset), ("full_sample", full_df)]:
            base, zeroed, boosted, inhibited = [], [], [], []
            for _, row in split_df.iterrows():
                txt = str(row["text"])
                b = _compute_nll(model, tokenizer, txt, max_length=max_length, device=device)
                if b is None:
                    continue
                z = _run_with_ablation(
                    model, tokenizer, txt, max_length=max_length, device=device,
                    layer_neurons=layer_neurons, mode="zero", scale=1.0
                )
                up = _run_with_ablation(
                    model, tokenizer, txt, max_length=max_length, device=device,
                    layer_neurons=layer_neurons, mode="scale", scale=boost_scale
                )
                dn = _run_with_ablation(
                    model, tokenizer, txt, max_length=max_length, device=device,
                    layer_neurons=layer_neurons, mode="scale", scale=inhibit_scale
                )
                if z is None or up is None or dn is None:
                    continue
                base.append(b)
                zeroed.append(z)
                boosted.append(up)
                inhibited.append(dn)

            if not base:
                continue

            base_arr = np.asarray(base, dtype=np.float64)
            zero_arr = np.asarray(zeroed, dtype=np.float64)
            up_arr = np.asarray(boosted, dtype=np.float64)
            dn_arr = np.asarray(inhibited, dtype=np.float64)

            rows.append(
                {
                    "status": "ok",
                    "concept": concept,
                    "split": split_name,
                    "n_samples_eval": int(len(base_arr)),
                    "topk_neurons": int(len(layer_neurons)),
                    "baseline_nll": float(base_arr.mean()),
                    "delta_nll_zero_ablation": float((zero_arr - base_arr).mean()),
                    "delta_nll_boost": float((up_arr - base_arr).mean()),
                    "delta_nll_inhibit": float((dn_arr - base_arr).mean()),
                }
            )

    if not rows:
        return pd.DataFrame([{"status": "skipped", "reason": "no functional rows computed"}])
    return pd.DataFrame(rows)


def compute_all_concept_metrics(
    neuron_events_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    concept_col: str,
    prepared,
    max_length: int,
    hierarchy_path: str | None,
    top_n_purity: int,
    classifier_test_size: float,
    random_seed: int,
    compute_functional_tests: bool,
    functional_topk_neurons: int,
    functional_max_samples: int,
) -> ConceptMetricOutputs:
    selectivity = compute_selectivity(neuron_events_df, concept_col=concept_col)
    purity = compute_purity(neuron_events_df, concept_col=concept_col, top_n=top_n_purity)
    layer_density = compute_layer_density(selectivity)
    clf_summary, clf_per_class = compute_classifier_metrics(
        neuron_events_df,
        concept_col=concept_col,
        test_size=classifier_test_size,
        random_seed=random_seed,
    )
    clustering = compute_clustering_metrics(selectivity)
    hierarchy = compute_hierarchy_consistency(selectivity, hierarchy_path=hierarchy_path)

    if compute_functional_tests:
        functional = compute_functional_effects(
            selectivity,
            dataset_df,
            concept_col=concept_col,
            prepared=prepared,
            max_length=max_length,
            topk_neurons_per_concept=functional_topk_neurons,
            max_samples=functional_max_samples,
            random_seed=random_seed,
        )
    else:
        functional = pd.DataFrame([{"status": "skipped", "reason": "disabled by config"}])

    return ConceptMetricOutputs(
        selectivity=selectivity,
        purity=purity,
        layer_density=layer_density,
        classifier_summary=clf_summary,
        classifier_per_class=clf_per_class,
        clustering_summary=clustering,
        hierarchy_consistency=hierarchy,
        functional_effects=functional,
    )
