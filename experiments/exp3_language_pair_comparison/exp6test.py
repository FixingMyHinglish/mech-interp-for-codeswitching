#!/usr/bin/env python3
"""
exp6.py ? Language-Selective Neurons.

Identifies neurons in the MLP layers that activate more strongly for one language
than another, using a selectivity index. Connects directly to Exp 21 findings by
asking: which specific neurons are being hijacked when English tokens are processed
in a matrix-language context?

Reads:
  - combined_dataset_preprocessed.csv  (for token_lang_labels)
  - mlp/{id}_{condition}.pt            (MLP activations from extract_activations.py)

Computes per neuron per layer:
  selectivity(n, A, B) = (mean_act_A - mean_act_B) / (mean_act_A + mean_act_B + eps)
  Range: [-1, +1]. +1 = fully A-selective, -1 = fully B-selective, 0 = no preference.

Key comparisons:
  - HI vs EN  (monolingual Hindi vs monolingual English)
  - FR vs EN  (monolingual French vs monolingual English)
  - cs_hi vs EN  (CS Hindi-English vs monolingual English)
  - cs_fr vs EN  (CS French-English vs monolingual English)

Key analyses:
  1. Fraction of neurons with |selectivity| > threshold per layer       ? how many neurons are language-specialized?
  2. Layer-wise selectivity distribution                                 ? where do specialized neurons live?
  3. Overlap between mono-HI-selective and cs_hi-selective neurons      ? are the same neurons hijacked in CS?
  4. For EN tokens inside cs_hi: which neurons fire ? HI-selective or EN-selective ones?  ? the hijacking analysis

Outputs:
  out_dir/
    selectivity_per_neuron.npz        ? raw selectivity arrays [n_layers, n_neurons] per comparison
    layer_stats.csv                   ? fraction selective, mean selectivity per layer per comparison
    neuron_overlap.csv                ? overlap between mono and CS selective neurons per layer
    hijacking_analysis.csv            ? for EN tokens in CS: activation in HI-selective vs EN-selective neurons
    summary.json

Usage:
  python exp6.py \\
    --activations_dir /scratch0/jabraham/qwen_activations \\
    --dataset_csv combined_dataset_preprocessed_qwen.csv \\
    --out_dir /scratch0/jabraham/exp6_qwen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ?? Args ??????????????????????????????????????????????????????????????????????

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 6: Language-selective neurons.")
    p.add_argument("--activations_dir",   required=True)
    p.add_argument("--dataset_csv",       required=True)
    p.add_argument("--out_dir",           default="results/exp6")
    p.add_argument("--selectivity_thresh", type=float, default=0.5,
                   help="Threshold for calling a neuron language-selective (default 0.5)")
    p.add_argument("--max_sentences",     type=int, default=None)
    return p.parse_args()


# ?? Helpers ???????????????????????????????????????????????????????????????????

def safe_id(row_id: str, condition: str) -> str:
    return f"{row_id.replace(':', '_').replace('/', '_')}_{condition}"


def load_tensor(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu").float()


# ?? Step 1: Accumulate mean activations per condition ?????????????????????????

def accumulate_mean_activations(
    ids: list[str],
    mlp_dir: Path,
    conditions: list[str],
) -> dict[str, np.ndarray]:
    """
    For each condition, compute mean MLP activation per neuron per layer
    across all sentences.

    Returns {condition: array of shape [n_layers, n_neurons]}
    where each value is the mean activation of that neuron across all tokens
    and all sentences.
    """
    print("\n[Step 1] Accumulating mean activations per condition ...")

    # First pass: determine n_layers and n_neurons from first available file
    n_layers = n_neurons = None
    for sid in ids:
        for cond in conditions:
            t = load_tensor(mlp_dir / f"{safe_id(sid, cond)}.pt")
            if t is not None:
                n_layers, _, n_neurons = t.shape  # [n_layers, n_tokens, n_neurons]
                break
        if n_layers is not None:
            break

    if n_layers is None:
        raise RuntimeError("Could not determine model dimensions ? no activation files found.")

    print(f"  Model dimensions: {n_layers} layers, {n_neurons} neurons per layer")

    # Accumulators: sum of activations and count of tokens
    sums   = {c: np.zeros((n_layers, n_neurons), dtype=np.float64) for c in conditions}
    counts = {c: np.zeros((n_layers, n_neurons), dtype=np.float64) for c in conditions}

    for sid in tqdm(ids, desc="Accumulating activations"):
        for cond in conditions:
            t = load_tensor(mlp_dir / f"{safe_id(sid, cond)}.pt")
            if t is None:
                continue
            arr = t.numpy()  # [n_layers, n_tokens, n_neurons]
            # Sum over token dimension
            sums[cond]   += arr.sum(axis=1)    # [n_layers, n_neurons]
            counts[cond] += arr.shape[1]        # broadcast: each neuron gets n_tokens added

    means = {}
    for cond in conditions:
        denom = np.where(counts[cond] > 0, counts[cond], 1.0)
        means[cond] = sums[cond] / denom

    return means


# ?? Step 1b: Accumulate mean activations for EN tokens inside CS sentences ????

def accumulate_embedded_en_activations(
    ids: list[str],
    mlp_dir: Path,
    dataset_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    For each CS condition, compute mean MLP activation using ONLY the
    English-labelled token positions within each CS sentence.

    Returns {cs_cond: array [n_layers, n_neurons]}

    This is used for the hijacking analysis: which neurons fire when the model
    processes an English word in a matrix-language context?
    """
    print("\n[Step 1b] Accumulating embedded EN token activations in CS sentences ...")

    # Build lookup: (sid, cond) ? list of EN token positions
    en_positions: dict[tuple[str, str], list[int]] = {}
    cs_df = dataset_df[dataset_df["condition"].isin(["cs_fr", "cs_hi"])]
    for _, row in cs_df.iterrows():
        sid    = str(row["id"])
        cond   = str(row["condition"])
        labels = json.loads(row["token_lang_labels"]) if isinstance(row["token_lang_labels"], str) else []
        en_pos = [i for i, lbl in enumerate(labels) if lbl == "EN"]
        if en_pos:
            en_positions[(sid, cond)] = en_pos

    # Determine dimensions
    n_layers = n_neurons = None
    for sid in ids:
        t = load_tensor(mlp_dir / f"{safe_id(sid, 'cs_hi')}.pt")
        if t is not None:
            n_layers, _, n_neurons = t.shape
            break

    if n_layers is None:
        raise RuntimeError("Could not determine model dimensions.")

    sums   = {c: np.zeros((n_layers, n_neurons), dtype=np.float64) for c in ["cs_hi", "cs_fr"]}
    counts = {c: np.zeros((n_layers,  1),        dtype=np.float64) for c in ["cs_hi", "cs_fr"]}

    for sid in tqdm(ids, desc="Embedded EN activations"):
        for cond in ["cs_hi", "cs_fr"]:
            en_pos = en_positions.get((sid, cond), [])
            if not en_pos:
                continue
            t = load_tensor(mlp_dir / f"{safe_id(sid, cond)}.pt")
            if t is None:
                continue
            arr        = t.numpy()           # [n_layers, n_tokens, n_neurons]
            n_tokens   = arr.shape[1]
            valid_pos  = [p for p in en_pos if p < n_tokens]
            if not valid_pos:
                continue
            en_acts    = arr[:, valid_pos, :]      # [n_layers, n_en_tokens, n_neurons]
            sums[cond]   += en_acts.sum(axis=1)    # [n_layers, n_neurons]
            counts[cond] += len(valid_pos)

    means = {}
    for cond in ["cs_hi", "cs_fr"]:
        denom = np.where(counts[cond] > 0, counts[cond], 1.0)
        means[cond] = sums[cond] / denom

    return means


# ?? Step 2: Compute selectivity index ?????????????????????????????????????????

def compute_selectivity(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Selectivity index = (mean_A - mean_B) / (|mean_A| + |mean_B| + eps)

    Shape: [n_layers, n_neurons]
    Range: [-1, +1]
    +1 = fully A-selective, -1 = fully B-selective, 0 = no preference.

    Uses absolute values in denominator to handle negative activations (e.g. after
    non-linearities that can produce negative values).
    """
    num   = mean_a - mean_b
    denom = np.abs(mean_a) + np.abs(mean_b) + eps
    return num / denom


# ?? Step 3: Layer statistics ???????????????????????????????????????????????????

def compute_layer_stats(
    selectivity_dict: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    """
    Per layer per comparison:
      - fraction of neurons with selectivity > +threshold  (A-selective)
      - fraction of neurons with selectivity < -threshold  (B-selective)
      - fraction with |selectivity| > threshold            (any specialization)
      - mean and std of selectivity distribution
    """
    print("\n[Step 3] Computing layer-wise selectivity statistics ...")
    rows = []
    for name, sel in selectivity_dict.items():
        n_layers, n_neurons = sel.shape
        for l in range(n_layers):
            s = sel[l]
            rows.append({
                "comparison":          name,
                "layer":               l,
                "frac_pos_selective":  float((s >  threshold).mean()),
                "frac_neg_selective":  float((s < -threshold).mean()),
                "frac_any_selective":  float((np.abs(s) > threshold).mean()),
                "mean_selectivity":    float(s.mean()),
                "std_selectivity":     float(s.std()),
                "max_selectivity":     float(s.max()),
                "min_selectivity":     float(s.min()),
                "n_neurons":           n_neurons,
            })
    return pd.DataFrame(rows)


# ?? Step 4: Neuron overlap ? mono vs CS ???????????????????????????????????????

def compute_neuron_overlap(
    sel_mono_hi: np.ndarray,
    sel_cs_hi:   np.ndarray,
    sel_mono_fr: np.ndarray,
    sel_cs_fr:   np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    For each layer, compute overlap between:
      - neurons selective for monolingual Hindi  vs  neurons selective for cs_hi
      - neurons selective for monolingual French vs  neurons selective for cs_fr

    Overlap metric: Jaccard = |A ? B| / |A ? B|

    High overlap means the same neurons that are activated by monolingual Hindi
    are also activated when processing CS Hindi-English text ? i.e. the model
    is routing CS through the same language-specific circuitry as monolingual.

    This is the key connection to Exp 21: if the hijacked neurons in cs_hi are
    the same ones that are Hindi-selective in monolingual Hindi, we know exactly
    which neurons are doing the hijacking.
    """
    print("\n[Step 4] Computing neuron overlap (mono vs CS selective neurons) ...")
    n_layers = sel_mono_hi.shape[0]
    rows = []
    for l in range(n_layers):
        for lang, sel_mono, sel_cs in [
            ("hindi",  sel_mono_hi[l], sel_cs_hi[l]),
            ("french", sel_mono_fr[l], sel_cs_fr[l]),
        ]:
            # Positive selective = favours this language over English
            mono_selective = sel_mono >  threshold
            cs_selective   = sel_cs   >  threshold

            intersection = (mono_selective & cs_selective).sum()
            union        = (mono_selective | cs_selective).sum()
            jaccard      = float(intersection / union) if union > 0 else 0.0

            # Also compute: of the CS-selective neurons, what fraction were
            # already mono-selective? (precision: how many CS-selective neurons
            # are "genuine" language neurons vs newly recruited?)
            n_cs_sel = cs_selective.sum()
            precision = float(intersection / n_cs_sel) if n_cs_sel > 0 else 0.0

            rows.append({
                "layer":                    l,
                "language":                 lang,
                "jaccard_overlap":          jaccard,
                "precision_cs_in_mono":     precision,
                "n_mono_selective":         int(mono_selective.sum()),
                "n_cs_selective":           int(cs_selective.sum()),
                "n_intersection":           int(intersection),
            })
    return pd.DataFrame(rows)


# ?? Step 5: Hijacking analysis ????????????????????????????????????????????????

def compute_hijacking_analysis(
    embedded_en_means: dict[str, np.ndarray],
    selectivity_dict:  dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    """
    The core hijacking analysis connecting Exp 6 to Exp 21.

    For English tokens embedded in CS sentences, measure their mean activation
    separately within:
      - neurons that are Hindi-selective  (HI > EN selectivity > threshold)
      - neurons that are English-selective (EN > HI selectivity < -threshold)
      - neutral neurons

    If the English words are being processed as Hindi, their activation should
    be higher in Hindi-selective neurons than in English-selective neurons.

    This directly answers: are Hindi-favoring neurons being activated by English
    words in a Hindi context?
    """
    print("\n[Step 5] Computing hijacking analysis ...")

    rows = []
    for cs_cond, mono_lang in [("cs_hi", "hindi_vs_english"), ("cs_fr", "french_vs_english")]:
        if cs_cond not in embedded_en_means:
            continue
        if mono_lang not in selectivity_dict:
            continue

        en_acts = embedded_en_means[cs_cond]   # [n_layers, n_neurons]
        sel     = selectivity_dict[mono_lang]   # [n_layers, n_neurons]
        n_layers = en_acts.shape[0]

        for l in range(n_layers):
            acts_l = en_acts[l]   # [n_neurons]
            sel_l  = sel[l]       # [n_neurons]

            # Neuron masks
            matrix_selective  = sel_l >  threshold   # fire more for matrix lang
            english_selective = sel_l < -threshold   # fire more for English
            neutral           = ~matrix_selective & ~english_selective

            # Mean activation of embedded EN tokens in each neuron group
            mean_in_matrix_neurons  = float(acts_l[matrix_selective].mean())  if matrix_selective.sum()  > 0 else float("nan")
            mean_in_english_neurons = float(acts_l[english_selective].mean()) if english_selective.sum() > 0 else float("nan")
            mean_in_neutral_neurons = float(acts_l[neutral].mean())           if neutral.sum()           > 0 else float("nan")

            rows.append({
                "layer":                       l,
                "cs_condition":                cs_cond,
                "mean_act_in_matrix_neurons":  mean_in_matrix_neurons,
                "mean_act_in_english_neurons": mean_in_english_neurons,
                "mean_act_in_neutral_neurons": mean_in_neutral_neurons,
                "n_matrix_selective":          int(matrix_selective.sum()),
                "n_english_selective":         int(english_selective.sum()),
                "n_neutral":                   int(neutral.sum()),
                # Ratio: >1 means matrix neurons more active than english neurons
                "matrix_vs_english_ratio":     float(mean_in_matrix_neurons / mean_in_english_neurons)
                                               if (mean_in_english_neurons and mean_in_english_neurons != 0)
                                               else float("nan"),
            })

    return pd.DataFrame(rows)


# ?? Summary ???????????????????????????????????????????????????????????????????

def build_summary(
    layer_stats_df:    pd.DataFrame,
    overlap_df:        pd.DataFrame,
    hijacking_df:      pd.DataFrame,
    threshold:         float,
    n_sentences:       int,
) -> dict:

    def layer_summary(df: pd.DataFrame, comparison: str) -> dict:
        sub = df[df["comparison"] == comparison]
        if sub.empty:
            return {}
        max_layer = int(sub["layer"].max())
        early = sub[sub["layer"] <= 3]
        late  = sub[sub["layer"] >= max_layer - 3]
        peak  = sub.loc[sub["frac_any_selective"].idxmax()]
        return {
            "frac_selective_mean":  float(sub["frac_any_selective"].mean()),
            "frac_selective_early": float(early["frac_any_selective"].mean()),
            "frac_selective_late":  float(late["frac_any_selective"].mean()),
            "peak_selective_layer": int(peak["layer"]),
            "peak_selective_frac":  float(peak["frac_any_selective"]),
        }

    def overlap_summary(df: pd.DataFrame, lang: str) -> dict:
        sub = df[df["language"] == lang]
        if sub.empty:
            return {}
        max_layer = int(sub["layer"].max())
        early = sub[sub["layer"] <= 3]
        late  = sub[sub["layer"] >= max_layer - 3]
        return {
            "jaccard_mean":         float(sub["jaccard_overlap"].mean()),
            "jaccard_early":        float(early["jaccard_overlap"].mean()),
            "jaccard_late":         float(late["jaccard_overlap"].mean()),
            "precision_mean":       float(sub["precision_cs_in_mono"].mean()),
            "precision_late":       float(late["precision_cs_in_mono"].mean()),
        }

    def hijacking_summary(df: pd.DataFrame, cs_cond: str) -> dict:
        sub = df[df["cs_condition"] == cs_cond]
        if sub.empty:
            return {}
        max_layer = int(sub["layer"].max())
        early = sub[sub["layer"] <= 3]
        late  = sub[sub["layer"] >= max_layer - 3]
        return {
            "mean_act_matrix_neurons_early":  float(early["mean_act_in_matrix_neurons"].mean()),
            "mean_act_english_neurons_early": float(early["mean_act_in_english_neurons"].mean()),
            "mean_act_matrix_neurons_late":   float(late["mean_act_in_matrix_neurons"].mean()),
            "mean_act_english_neurons_late":  float(late["mean_act_in_english_neurons"].mean()),
            "matrix_vs_english_ratio_early":  float(early["matrix_vs_english_ratio"].mean()),
            "matrix_vs_english_ratio_late":   float(late["matrix_vs_english_ratio"].mean()),
            "interpretation": "ratio > 1 means matrix-selective neurons more active than english-selective "
                              "neurons when processing embedded English tokens ? hijacking confirmed.",
        }

    return {
        "n_sentences":         n_sentences,
        "selectivity_threshold": threshold,

        "selectivity_distribution": {
            "hindi_vs_english":  layer_summary(layer_stats_df, "hindi_vs_english"),
            "french_vs_english": layer_summary(layer_stats_df, "french_vs_english"),
            "cs_hi_vs_english":  layer_summary(layer_stats_df, "cs_hi_vs_english"),
            "cs_fr_vs_english":  layer_summary(layer_stats_df, "cs_fr_vs_english"),
            "note": "Fraction of neurons with |selectivity| > threshold per layer. "
                    "Shows how many neurons are language-specialized and where they live.",
        },

        "neuron_overlap_mono_vs_cs": {
            "hindi":  overlap_summary(overlap_df, "hindi"),
            "french": overlap_summary(overlap_df, "french"),
            "note": "Jaccard overlap between neurons selective in monolingual condition "
                    "and neurons selective in CS condition. High overlap = same circuitry used. "
                    "Precision = fraction of CS-selective neurons that were already mono-selective.",
        },

        "hijacking_analysis": {
            "cs_hi": hijacking_summary(hijacking_df, "cs_hi"),
            "cs_fr": hijacking_summary(hijacking_df, "cs_fr"),
            "note": "Mean activation of embedded English tokens inside matrix-selective neurons "
                    "vs English-selective neurons. ratio > 1 = matrix neurons more active = hijacking.",
        },
    }


# ?? Main ??????????????????????????????????????????????????????????????????????

def main() -> None:
    args    = parse_args()
    mlp_dir = Path(args.activations_dir) / "mlp"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset_csv} ...")
    dataset_df = pd.read_csv(args.dataset_csv)
    ids = dataset_df["id"].unique().tolist()
    if args.max_sentences:
        ids = ids[:args.max_sentences]
    print(f"  {len(ids)} unique sentence ids")

    CONDITIONS = ["english", "french", "hindi", "cs_fr", "cs_hi"]

    # Step 1: mean activations per condition
    means = accumulate_mean_activations(ids, mlp_dir, CONDITIONS)

    # Step 1b: mean activations of embedded EN tokens in CS sentences
    embedded_en_means = accumulate_embedded_en_activations(ids, mlp_dir, dataset_df)

    # Step 2: selectivity indices
    print("\n[Step 2] Computing selectivity indices ...")
    selectivity = {
        "hindi_vs_english":  compute_selectivity(means["hindi"],  means["english"]),
        "french_vs_english": compute_selectivity(means["french"], means["english"]),
        "cs_hi_vs_english":  compute_selectivity(means["cs_hi"],  means["english"]),
        "cs_fr_vs_english":  compute_selectivity(means["cs_fr"],  means["english"]),
        # embedded EN tokens in CS vs monolingual English
        "embedded_en_cs_hi_vs_english": compute_selectivity(embedded_en_means["cs_hi"], means["english"]),
        "embedded_en_cs_fr_vs_english": compute_selectivity(embedded_en_means["cs_fr"], means["english"]),
    }

    # Save raw selectivity arrays
    print("  Saving raw selectivity arrays ...")
    np.savez(
        out_dir / "selectivity_per_neuron.npz",
        **{k: v.astype(np.float32) for k, v in selectivity.items()}
    )

    # Step 3: layer stats
    layer_stats_df = compute_layer_stats(selectivity, args.selectivity_thresh)

    # Step 4: neuron overlap mono vs CS
    overlap_df = compute_neuron_overlap(
        sel_mono_hi = selectivity["hindi_vs_english"],
        sel_cs_hi   = selectivity["cs_hi_vs_english"],
        sel_mono_fr = selectivity["french_vs_english"],
        sel_cs_fr   = selectivity["cs_fr_vs_english"],
        threshold   = args.selectivity_thresh,
    )

    # Step 5: hijacking analysis
    hijacking_df = compute_hijacking_analysis(
        embedded_en_means = embedded_en_means,
        selectivity_dict  = selectivity,
        threshold         = args.selectivity_thresh,
    )

    # Save CSVs
    layer_stats_df.to_csv(out_dir / "layer_stats.csv",        index=False)
    overlap_df.to_csv(out_dir     / "neuron_overlap.csv",     index=False)
    hijacking_df.to_csv(out_dir   / "hijacking_analysis.csv", index=False)

    # Summary
    summary = build_summary(
        layer_stats_df = layer_stats_df,
        overlap_df     = overlap_df,
        hijacking_df   = hijacking_df,
        threshold      = args.selectivity_thresh,
        n_sentences    = len(ids),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone. Results written to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
