#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common import (  # noqa: E402
    configure_gpu_runtime,
    encode_text,
    extract_post_activations,
    infer_target_script,
    label_token_language,
    load_dataset,
    load_tl_model,
    resolve_device,
)

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency in runtime
    KMeans = None
    silhouette_score = None
    StandardScaler = None

import plotly.graph_objects as go


EN_FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "am",
    "it",
    "that",
    "this",
    "for",
    "with",
    "as",
    "by",
    "from",
    "but",
    "if",
    "then",
    "than",
    "so",
    "not",
}

PUNCT_RE = re.compile(r"^\W+$", flags=re.UNICODE)
NUM_RE = re.compile(r"^\d+([.,:]\d+)?$")
CAP_RE = re.compile(r"^[A-Z][a-zA-Z]+$")
WORD_RE = re.compile(r"^[A-Za-z]+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 7: cluster neurons by code-switched vs monolingual activation "
            "patterns and generate interpretable cluster descriptions."
        )
    )
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--out_dir", default="new-compute/experiments/exp7_neuron_clustering_interpretation/results")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_rows_per_condition", type=int, default=200)
    p.add_argument("--code_switched_label", default="code_switched")
    p.add_argument("--english_label", default="english")
    p.add_argument("--target_label", default="target_language")
    p.add_argument("--switch_window", nargs="+", type=int, default=[-1, 0, 1])
    p.add_argument("--topk_neurons_per_token", type=int, default=6)
    p.add_argument("--top_tokens_per_cluster", type=int, default=20)
    p.add_argument("--n_clusters", type=int, default=0, help="0 means auto-select with silhouette.")
    p.add_argument("--min_clusters", type=int, default=4)
    p.add_argument("--max_clusters", type=int, default=12)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument(
        "--gpu_friendly",
        action="store_true",
        help=(
            "Enable GPU-friendly settings: TF32, cuDNN benchmark, and mixed-precision "
            "model loading (bfloat16/float16 fallback) when running on CUDA."
        ),
    )
    return p.parse_args()


def _clean_token(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").strip()


def _is_function_word(token: str) -> bool:
    t = _clean_token(token).lower()
    return t in EN_FUNCTION_WORDS


def _is_punctuation(token: str) -> bool:
    t = _clean_token(token)
    return bool(t) and bool(PUNCT_RE.match(t))


def _is_number_like(token: str) -> bool:
    t = _clean_token(token)
    return bool(t) and bool(NUM_RE.match(t))


def _is_named_entity_like(token: str) -> bool:
    t = _clean_token(token)
    return bool(t) and bool(CAP_RE.match(t))


def _is_content_word_like(token: str) -> bool:
    t = _clean_token(token)
    return bool(t) and bool(WORD_RE.match(t)) and not _is_function_word(t)


def _switch_points(token_lang_labels: list[str]) -> list[int]:
    out = []
    for i in range(1, len(token_lang_labels)):
        a = token_lang_labels[i - 1]
        b = token_lang_labels[i]
        if a in {"english", "target"} and b in {"english", "target"} and a != b:
            out.append(i)
    return out


def _arg_topk(values: np.ndarray, k: int) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=int)
    k = max(1, min(int(k), int(values.size)))
    idx = np.argpartition(values, -k)[-k:]
    return idx[np.argsort(values[idx])[::-1]]


def _safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return float(a / b)


def _choose_n_clusters(
    x_scaled: np.ndarray,
    requested_k: int,
    min_k: int,
    max_k: int,
    seed: int,
) -> tuple[int, dict[int, float]]:
    if requested_k > 1:
        return int(requested_k), {}

    n = int(x_scaled.shape[0])
    lo = max(2, int(min_k))
    hi = min(int(max_k), max(2, n - 1))
    if hi < lo:
        return 2, {}

    rng = np.random.default_rng(seed)
    sample_n = min(3500, n)
    if sample_n < n:
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        x_eval = x_scaled[sample_idx]
    else:
        x_eval = x_scaled

    scores: dict[int, float] = {}
    best_k = lo
    best_score = -float("inf")
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(x_scaled)
        labels_eval = km.predict(x_eval)
        if len(np.unique(labels_eval)) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(x_eval, labels_eval))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k), scores


def _cluster_label(mean_row: pd.Series) -> tuple[str, str]:
    sw = float(mean_row.get("switch_event_ratio", 0.0))
    cs_en = float(mean_row.get("cs_minus_english", 0.0))
    cs_tgt = float(mean_row.get("cs_minus_target", 0.0))
    tgt_en = float(mean_row.get("target_minus_english", 0.0))
    tgt_token = float(mean_row.get("target_token_ratio", 0.0))
    eng_token = float(mean_row.get("english_token_ratio", 0.0))
    punct = float(mean_row.get("punct_ratio", 0.0))
    ne_like = float(mean_row.get("named_entity_like_ratio", 0.0))
    func = float(mean_row.get("function_word_ratio", 0.0))

    if sw >= 0.35 and cs_en > 0.0 and cs_tgt > 0.0:
        return (
            "code-switch detection neurons",
            "High switch-event ratio and positive CS-vs-monolingual activation deltas.",
        )
    if tgt_token > eng_token * 1.25 and tgt_en > 0.0:
        return (
            "language identity neurons (target-biased)",
            "Target-language tokens dominate activation patterns with positive target-vs-english effect.",
        )
    if eng_token > tgt_token * 1.25 and tgt_en < 0.0:
        return (
            "language identity neurons (english-biased)",
            "English tokens dominate activation patterns with negative target-vs-english effect.",
        )
    if punct >= 0.30:
        return (
            "boundary/punctuation marker neurons",
            "Activation is concentrated on punctuation and boundary-like tokens.",
        )
    if ne_like >= 0.18:
        return (
            "named-entity leaning neurons",
            "High fraction of capitalized token events suggests named-entity sensitivity.",
        )
    if func >= 0.35:
        return (
            "function-word structure neurons",
            "Activation events are enriched for high-frequency function words.",
        )
    return (
        "mixed context neurons",
        "No single dominant linguistic marker; cluster appears context-distributed.",
    )


def main() -> None:
    args = parse_args()
    if KMeans is None or StandardScaler is None or silhouette_score is None:
        raise ImportError("scikit-learn is required for exp7 clustering.")

    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset_csv)
    cond_keep = {
        str(args.code_switched_label),
        str(args.english_label),
        str(args.target_label),
    }
    df = df[df["condition"].isin(cond_keep)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering for requested conditions.")

    sampled_parts = []
    for cond in sorted(cond_keep):
        part = df[df["condition"] == cond].copy().head(int(args.max_rows_per_condition))
        sampled_parts.append(part)
    run_df = pd.concat(sampled_parts, ignore_index=True)

    target_script = infer_target_script(run_df, target_label=str(args.target_label))
    device = resolve_device(args.device)
    configure_gpu_runtime(device=device, gpu_friendly=bool(args.gpu_friendly))
    model = load_tl_model(
        args.model_name,
        device=device,
        gpu_friendly=bool(args.gpu_friendly),
    )
    tokenizer = model.tokenizer

    cond_sums: dict[tuple[str, int], np.ndarray] = {}
    cond_counts: dict[tuple[str, int], int] = defaultdict(int)

    cs_switch_sums: dict[int, np.ndarray] = {}
    cs_switch_counts: dict[int, int] = defaultdict(int)
    cs_nonswitch_sums: dict[int, np.ndarray] = {}
    cs_nonswitch_counts: dict[int, int] = defaultdict(int)

    event_weights: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    token_weights: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    n_rows_done = 0
    n_cs_with_switch = 0
    total_switch_events = 0

    for idx, row in enumerate(tqdm(run_df.itertuples(index=False), total=len(run_df), desc="Exp7 extraction"), start=1):
        input_ids = encode_text(
            tokenizer=tokenizer,
            text=row.text,
            max_length=int(args.max_length),
            device=device,
        )
        ids = input_ids[0].detach().cpu().tolist()
        if len(ids) < 2:
            continue

        acts_by_layer = extract_post_activations(model, input_ids)
        token_texts = tokenizer.convert_ids_to_tokens(ids)
        lang_labels = [label_token_language(tok, target_script=target_script) for tok in token_texts]
        cond = str(row.condition)

        for layer, arr in acts_by_layer.items():
            vec = np.abs(arr).mean(axis=0).astype(np.float64, copy=False)
            key = (cond, int(layer))
            if key not in cond_sums:
                cond_sums[key] = np.zeros_like(vec, dtype=np.float64)
            cond_sums[key] += vec
            cond_counts[key] += 1

        if cond == str(args.code_switched_label):
            switch_idx = _switch_points(lang_labels)
            if switch_idx:
                n_cs_with_switch += 1
                total_switch_events += len(switch_idx)
            switch_pos = set()
            for sw in switch_idx:
                for off in args.switch_window:
                    pos = int(sw + off)
                    if 0 <= pos < len(ids):
                        switch_pos.add(pos)

            for layer, arr in acts_by_layer.items():
                abs_arr = np.abs(arr).astype(np.float64, copy=False)
                if switch_pos:
                    sw_ix = np.array(sorted(switch_pos), dtype=int)
                    ns_ix = np.array([i for i in range(abs_arr.shape[0]) if i not in switch_pos], dtype=int)
                    sw_vec = abs_arr[sw_ix].mean(axis=0)
                    key_layer = int(layer)
                    if key_layer not in cs_switch_sums:
                        cs_switch_sums[key_layer] = np.zeros_like(sw_vec, dtype=np.float64)
                    cs_switch_sums[key_layer] += sw_vec
                    cs_switch_counts[key_layer] += 1

                    if ns_ix.size > 0:
                        ns_vec = abs_arr[ns_ix].mean(axis=0)
                        if key_layer not in cs_nonswitch_sums:
                            cs_nonswitch_sums[key_layer] = np.zeros_like(ns_vec, dtype=np.float64)
                        cs_nonswitch_sums[key_layer] += ns_vec
                        cs_nonswitch_counts[key_layer] += 1

                    for pos in sw_ix.tolist():
                        tok = _clean_token(token_texts[pos]) or "<blank>"
                        tok_lang = lang_labels[pos]
                        is_sw = pos in switch_pos
                        is_punct = _is_punctuation(tok)
                        is_num = _is_number_like(tok)
                        is_ne = _is_named_entity_like(tok)
                        is_func = _is_function_word(tok)
                        is_content = _is_content_word_like(tok)

                        top_idx = _arg_topk(abs_arr[pos], int(args.topk_neurons_per_token))
                        for neuron in top_idx.tolist():
                            val = float(abs_arr[pos, neuron])
                            nkey = (int(layer), int(neuron))
                            event_weights[nkey]["total"] += val
                            event_weights[nkey][f"lang::{tok_lang}"] += val
                            if is_sw:
                                event_weights[nkey]["switch_event"] += val
                            if is_punct:
                                event_weights[nkey]["punct"] += val
                            if is_num:
                                event_weights[nkey]["number"] += val
                            if is_ne:
                                event_weights[nkey]["named_entity_like"] += val
                            if is_func:
                                event_weights[nkey]["function_word"] += val
                            if is_content:
                                event_weights[nkey]["content_word"] += val
                            token_weights[nkey][tok.lower()] += val

        n_rows_done += 1
        if bool(args.gpu_friendly) and device.type == "cuda" and idx % 64 == 0:
            torch.cuda.empty_cache()

    if n_rows_done == 0:
        raise RuntimeError("No valid rows were processed.")

    available_layers = sorted({layer for _, layer in cond_sums.keys()})
    features = []
    for layer in available_layers:
        cs_key = (str(args.code_switched_label), layer)
        en_key = (str(args.english_label), layer)
        tg_key = (str(args.target_label), layer)
        if cs_key not in cond_sums or en_key not in cond_sums or tg_key not in cond_sums:
            continue

        cs_mean = cond_sums[cs_key] / max(cond_counts[cs_key], 1)
        en_mean = cond_sums[en_key] / max(cond_counts[en_key], 1)
        tg_mean = cond_sums[tg_key] / max(cond_counts[tg_key], 1)

        sw_mean = None
        ns_mean = None
        if layer in cs_switch_sums and cs_switch_counts[layer] > 0:
            sw_mean = cs_switch_sums[layer] / cs_switch_counts[layer]
        if layer in cs_nonswitch_sums and cs_nonswitch_counts[layer] > 0:
            ns_mean = cs_nonswitch_sums[layer] / cs_nonswitch_counts[layer]

        d = int(cs_mean.shape[0])
        for neuron in range(d):
            nkey = (int(layer), int(neuron))
            ew = event_weights.get(nkey, Counter())
            total_w = float(ew.get("total", 0.0))

            target_token_ratio = _safe_div(ew.get("lang::target", 0.0), total_w)
            english_token_ratio = _safe_div(ew.get("lang::english", 0.0), total_w)
            mixed_token_ratio = _safe_div(ew.get("lang::mixed", 0.0), total_w)
            switch_event_ratio = _safe_div(ew.get("switch_event", 0.0), total_w)
            punct_ratio = _safe_div(ew.get("punct", 0.0), total_w)
            number_ratio = _safe_div(ew.get("number", 0.0), total_w)
            ne_ratio = _safe_div(ew.get("named_entity_like", 0.0), total_w)
            function_ratio = _safe_div(ew.get("function_word", 0.0), total_w)
            content_ratio = _safe_div(ew.get("content_word", 0.0), total_w)

            switch_gain = 0.0
            if sw_mean is not None and ns_mean is not None:
                switch_gain = float(sw_mean[neuron] - ns_mean[neuron])

            features.append(
                {
                    "layer": int(layer),
                    "neuron": int(neuron),
                    "mean_code_switched": float(cs_mean[neuron]),
                    "mean_english": float(en_mean[neuron]),
                    "mean_target": float(tg_mean[neuron]),
                    "cs_minus_english": float(cs_mean[neuron] - en_mean[neuron]),
                    "cs_minus_target": float(cs_mean[neuron] - tg_mean[neuron]),
                    "target_minus_english": float(tg_mean[neuron] - en_mean[neuron]),
                    "switch_gain": float(switch_gain),
                    "target_token_ratio": float(target_token_ratio),
                    "english_token_ratio": float(english_token_ratio),
                    "mixed_token_ratio": float(mixed_token_ratio),
                    "switch_event_ratio": float(switch_event_ratio),
                    "punct_ratio": float(punct_ratio),
                    "number_ratio": float(number_ratio),
                    "named_entity_like_ratio": float(ne_ratio),
                    "function_word_ratio": float(function_ratio),
                    "content_word_ratio": float(content_ratio),
                    "event_weight_total": float(total_w),
                }
            )

    feat_df = pd.DataFrame(features)
    if feat_df.empty:
        raise RuntimeError("No neuron feature rows were built.")

    feature_cols = [
        "mean_code_switched",
        "mean_english",
        "mean_target",
        "cs_minus_english",
        "cs_minus_target",
        "target_minus_english",
        "switch_gain",
        "target_token_ratio",
        "english_token_ratio",
        "mixed_token_ratio",
        "switch_event_ratio",
        "punct_ratio",
        "number_ratio",
        "named_entity_like_ratio",
        "function_word_ratio",
        "content_word_ratio",
        "event_weight_total",
    ]
    x = feat_df[feature_cols].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    n_clusters, sil_scores = _choose_n_clusters(
        x_scaled=x_scaled,
        requested_k=int(args.n_clusters),
        min_k=int(args.min_clusters),
        max_k=int(args.max_clusters),
        seed=int(args.random_seed),
    )
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(args.random_seed), n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    feat_df["cluster_id"] = labels.astype(int)
    feat_df.to_csv(tables_dir / "neuron_cluster_assignments.csv.gz", index=False, compression="gzip")

    cluster_summary = (
        feat_df.groupby("cluster_id", as_index=False)
        .agg(
            n_neurons=("neuron", "size"),
            mean_layer=("layer", "mean"),
            mean_cs_minus_english=("cs_minus_english", "mean"),
            mean_cs_minus_target=("cs_minus_target", "mean"),
            mean_target_minus_english=("target_minus_english", "mean"),
            mean_switch_gain=("switch_gain", "mean"),
            switch_event_ratio=("switch_event_ratio", "mean"),
            target_token_ratio=("target_token_ratio", "mean"),
            english_token_ratio=("english_token_ratio", "mean"),
            mixed_token_ratio=("mixed_token_ratio", "mean"),
            punct_ratio=("punct_ratio", "mean"),
            number_ratio=("number_ratio", "mean"),
            named_entity_like_ratio=("named_entity_like_ratio", "mean"),
            function_word_ratio=("function_word_ratio", "mean"),
            content_word_ratio=("content_word_ratio", "mean"),
            mean_event_weight=("event_weight_total", "mean"),
        )
        .sort_values("n_neurons", ascending=False)
    )

    layer_dist_rows = []
    for cid, sub in feat_df.groupby("cluster_id"):
        counts = sub["layer"].value_counts().sort_index()
        total = float(counts.sum())
        for layer, count in counts.items():
            layer_dist_rows.append(
                {
                    "cluster_id": int(cid),
                    "layer": int(layer),
                    "count": int(count),
                    "fraction": float(count / max(total, 1.0)),
                }
            )
    layer_dist_df = pd.DataFrame(layer_dist_rows).sort_values(["cluster_id", "layer"])
    layer_dist_df.to_csv(tables_dir / "cluster_layer_distribution.csv", index=False)

    cluster_token_rows = []
    cluster_interp_rows = []
    for cid, sub in feat_df.groupby("cluster_id"):
        member_keys = set(zip(sub["layer"].astype(int), sub["neuron"].astype(int)))
        agg = Counter()
        for key in member_keys:
            agg.update(token_weights.get(key, Counter()))
        top_tokens = agg.most_common(int(args.top_tokens_per_cluster))
        for rank, (token, weight) in enumerate(top_tokens, start=1):
            cluster_token_rows.append(
                {
                    "cluster_id": int(cid),
                    "rank": int(rank),
                    "token": str(token),
                    "weight": float(weight),
                    "log_weight": float(math.log1p(weight)),
                }
            )

        csum = cluster_summary[cluster_summary["cluster_id"] == cid].iloc[0]
        label, rationale = _cluster_label(csum)
        cluster_interp_rows.append(
            {
                "cluster_id": int(cid),
                "semantic_label": label,
                "rationale": rationale,
                "top_tokens_preview": ", ".join(tok for tok, _ in top_tokens[:8]),
            }
        )

    cluster_tokens_df = pd.DataFrame(cluster_token_rows)
    cluster_interp_df = pd.DataFrame(cluster_interp_rows)
    cluster_summary = cluster_summary.merge(cluster_interp_df, on="cluster_id", how="left")
    cluster_summary.to_csv(tables_dir / "cluster_summary.csv", index=False)
    cluster_tokens_df.to_csv(tables_dir / "cluster_token_patterns.csv", index=False)
    cluster_interp_df.to_csv(tables_dir / "cluster_interpretations.csv", index=False)

    centroid = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
    centroid.insert(0, "cluster_id", np.arange(n_clusters))
    centroid.to_csv(tables_dir / "cluster_centroids_scaled.csv", index=False)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=centroid[feature_cols].to_numpy(dtype=float),
                x=feature_cols,
                y=[f"cluster_{c}" for c in centroid["cluster_id"].tolist()],
                colorscale="RdBu",
                zmid=0.0,
                colorbar={"title": "scaled centroid"},
                hovertemplate="cluster=%{y}<br>feature=%{x}<br>value=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Exp7 Cluster Centroids (scaled feature space)",
        xaxis_title="Feature",
        yaxis_title="Cluster",
        template="plotly_white",
    )
    fig.write_html(str(figures_dir / "cluster_centroids_heatmap.html"), include_plotlyjs="cdn")

    summary = {
        "experiment": "exp7_neuron_clustering_interpretation",
        "dataset_csv": str(args.dataset_csv),
        "model_name": str(args.model_name),
        "target_script_inferred": target_script,
        "conditions": {
            "code_switched_label": str(args.code_switched_label),
            "english_label": str(args.english_label),
            "target_label": str(args.target_label),
        },
        "max_rows_per_condition": int(args.max_rows_per_condition),
        "rows_processed": int(n_rows_done),
        "n_cs_with_switch": int(n_cs_with_switch),
        "total_switch_events": int(total_switch_events),
        "n_neuron_rows": int(len(feat_df)),
        "n_clusters": int(n_clusters),
        "silhouette_scores_by_k": {str(k): float(v) for k, v in sil_scores.items()},
        "gpu_friendly": bool(args.gpu_friendly),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

