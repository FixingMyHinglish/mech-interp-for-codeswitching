#!/usr/bin/env python3
"""
exp21.py ? Language Pair Comparison (Hi-En vs Fr-En).

Measures two mechanistic metrics:
  1. Neuron activation similarity  ? Jaccard + Cosine on MLP activations
  2. Layer-wise error emergence    ? residual stream divergence at switch points

Plus frequency-artifact control checks:
  Check 2: Repeat metric 1 on balanced sentences only (45?55% matrix language tokens)
           ? If dominance persists at equal token ratios, it's not a frequency artifact.
  Check 3: For English tokens embedded inside CS sentences, measure whether they activate
           more like the matrix language or like monolingual English.
           ? If even embedded EN tokens look like the matrix language, dominance is structural.
  Check 3b: Position-matched version of Check 3 ? same token string in CS vs pure EN.

Question answered: Are code-switching mechanisms language-pair-agnostic or language-specific?

Outputs:
  out_dir/
    neuron_similarity.csv          ? metric 1: per layer per condition pair
    neuron_similarity_balanced.csv ? check 2:  same but filtered to balanced sentences
    embedded_en_similarity.csv     ? check 3:  EN tokens inside CS vs whole-sentence mono baselines
    aligned_en_similarity.csv      ? check 3b: same EN token in CS vs same token in pure EN (position-matched)
    error_emergence.csv            ? metric 2: per layer divergence at switch points
    token_ratio_stats.csv          ? distribution of matrix-language token ratios
    summary.json

Usage:
  python exp21.py \\
    --activations_dir /scratch0/jabraham/qwen_activations \\
    --dataset_csv combined_dataset_preprocessed_qwen.csv \\
    --out_dir /scratch0/jabraham/exp21_qwen
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
    p = argparse.ArgumentParser(description="Exp 21: Language pair comparison.")
    p.add_argument("--activations_dir",      required=True)
    p.add_argument("--dataset_csv",          required=True)
    p.add_argument("--out_dir",              default="results/exp21")
    p.add_argument("--activation_threshold", type=float, default=0.0)
    p.add_argument("--max_sentences",        type=int,   default=None)
    p.add_argument("--balance_ratio_min",    type=float, default=0.45,
                   help="Min matrix-language token ratio for balanced check (default 0.45)")
    p.add_argument("--balance_ratio_max",    type=float, default=0.55,
                   help="Max matrix-language token ratio for balanced check (default 0.55)")
    return p.parse_args()


# ?? Helpers ???????????????????????????????????????????????????????????????????

def safe_id(row_id: str, condition: str) -> str:
    return f"{row_id.replace(':', '_').replace('/', '_')}_{condition}"


def load_tensor(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu").float()


def jaccard(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    fa = a > threshold
    fb = b > threshold
    both   = np.logical_and(fa, fb).sum()
    either = np.logical_or(fa, fb).sum()
    return float(both / either) if either > 0 else 1.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ?? Condition pairs for metric 1 ??????????????????????????????????????????????

PAIRS = [
    ("cs_fr",  "french",  "fr_group"),
    ("cs_fr",  "english", "fr_group"),
    ("french", "english", "fr_group"),
    ("cs_hi",  "hindi",   "hi_group"),
    ("cs_hi",  "english", "hi_group"),
    ("hindi",  "english", "hi_group"),
    ("cs_fr",  "cs_hi",   "cross_cs"),
]


# ?? Token ratio utilities ?????????????????????????????????????????????????????

def compute_token_ratios(dataset_df: pd.DataFrame) -> dict[str, float]:
    """
    Returns {'{sid}_{cond}': matrix_lang_token_ratio} for cs_fr and cs_hi rows.

    ratio = matrix_language_tokens / total_tokens
    Matrix language is FR for cs_fr, HI for cs_hi.
    Used for Check 2 (filter to balanced sentences) and ratio-correlation analysis.
    """
    ratios: dict[str, float] = {}
    cs_df = dataset_df[dataset_df["condition"].isin(["cs_fr", "cs_hi"])]
    for _, row in cs_df.iterrows():
        sid    = str(row["id"])
        cond   = str(row["condition"])
        labels = json.loads(row["token_lang_labels"]) if isinstance(row["token_lang_labels"], str) else []
        total  = len(labels)
        if total == 0:
            continue
        matrix_lang  = "HI" if cond == "cs_hi" else "FR"
        matrix_count = sum(1 for lbl in labels if lbl == matrix_lang)
        ratios[f"{sid}_{cond}"] = matrix_count / total
    return ratios


def save_ratio_stats(token_ratios: dict[str, float], out_dir: Path) -> None:
    """Save distribution of matrix-language token ratios so reviewers can inspect it."""
    rows = []
    for key, ratio in token_ratios.items():
        # key format: '{sid}_cs_fr' or '{sid}_cs_hi'
        cond = "cs_fr" if key.endswith("_cs_fr") else "cs_hi"
        rows.append({"key": key, "condition": cond, "matrix_ratio": ratio})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "token_ratio_stats.csv", index=False)

    for cond in ["cs_fr", "cs_hi"]:
        sub = df[df["condition"] == cond]["matrix_ratio"]
        print(f"  Token ratio [{cond}]: mean={sub.mean():.3f}  std={sub.std():.3f}  "
              f"min={sub.min():.3f}  max={sub.max():.3f}  "
              f"balanced (0.45?0.55): {((sub >= 0.45) & (sub <= 0.55)).sum()} sentences")


# ?? Metric 1 + Check 2: Neuron activation similarity ?????????????????????????

def compute_pairwise_similarity(
    ids: list[str],
    mlp_dir: Path,
    threshold: float,
    token_ratios: dict[str, float] | None = None,
    ratio_min: float = 0.0,
    ratio_max: float = 1.0,
    label: str = "Neuron similarity",
) -> pd.DataFrame:
    """
    Jaccard + Cosine on MLP activations for all condition pairs, per layer.

    If token_ratios is provided, only sentences whose cs_fr AND cs_hi ratio
    fall within [ratio_min, ratio_max] are included. This is Check 2.
    """
    print(f"\n[Metric 1] {label} ...")
    CONDITIONS = ["cs_fr", "cs_hi", "english", "french", "hindi"]
    jaccard_scores: dict[tuple, dict[int, list[float]]] = {(a, b): {} for a, b, _ in PAIRS}
    cosine_scores:  dict[tuple, dict[int, list[float]]] = {(a, b): {} for a, b, _ in PAIRS}

    n_filtered = 0
    for sid in tqdm(ids, desc=label):

        # Check 2 filter ? skip if either CS condition is outside the ratio window
        if token_ratios is not None:
            fr_ratio = token_ratios.get(f"{sid}_cs_fr")
            hi_ratio = token_ratios.get(f"{sid}_cs_hi")
            fr_ok = fr_ratio is None or (ratio_min <= fr_ratio <= ratio_max)
            hi_ok = hi_ratio is None or (ratio_min <= hi_ratio <= ratio_max)
            if not (fr_ok and hi_ok):
                n_filtered += 1
                continue

        mlps = {c: load_tensor(mlp_dir / f"{safe_id(sid, c)}.pt") for c in CONDITIONS}
        if any(v is None for v in mlps.values()):
            continue

        n_layers = min(m.shape[0] for m in mlps.values())
        for l in range(n_layers):
            # Mean-pool over token dimension ? single vector per layer per condition
            vecs = {c: mlps[c][l].mean(dim=0).numpy() for c in mlps}
            for cond_a, cond_b, _ in PAIRS:
                a, b = vecs[cond_a], vecs[cond_b]
                jaccard_scores[(cond_a, cond_b)].setdefault(l, []).append(jaccard(a, b, threshold))
                cosine_scores[(cond_a, cond_b)].setdefault(l, []).append(cosine(a, b))

    if token_ratios is not None:
        print(f"  Filtered out {n_filtered} sentences outside ratio [{ratio_min}, {ratio_max}]")

    all_layers = sorted(set().union(*[set(d.keys()) for d in list(jaccard_scores.values()) + list(cosine_scores.values())]))
    rows = []
    for l in all_layers:
        for cond_a, cond_b, group in PAIRS:
            key = (cond_a, cond_b)
            j = jaccard_scores[key].get(l, [])
            c = cosine_scores[key].get(l, [])
            rows.append({
                "layer":        l,
                "group":        group,
                "pair":         f"{cond_a}_vs_{cond_b}",
                "jaccard_mean": float(np.mean(j)) if j else float("nan"),
                "jaccard_std":  float(np.std(j))  if j else float("nan"),
                "cosine_mean":  float(np.mean(c)) if c else float("nan"),
                "cosine_std":   float(np.std(c))  if c else float("nan"),
                "n_sentences":  len(j),
            })
    return pd.DataFrame(rows)


# ?? Check 3b: Aligned embedded English token similarity ??????????????????????

def compute_aligned_embedded_similarity(
    ids: list[str],
    dataset_df: pd.DataFrame,
    mlp_dir: Path,
    threshold: float,
) -> pd.DataFrame:
    """
    Check 3b (position-matched): For each English token embedded in a CS sentence,
    find the SAME token string at its position in the monolingual English sentence,
    then compare activations directly ? same word, different surrounding context.

    This eliminates Limitation 2 (whole-sentence mean baseline) by comparing
    activation of e.g. 'especially' in CS context vs 'especially' in pure EN context.

    Three comparisons per CS condition per layer:
      cs_hi:  EN-token-in-CS-context  vs  same-token-in-EN-context   (context shift)
      cs_hi:  EN-token-in-CS-context  vs  HI-token-mean-at-same-pos  (matrix pull)
      cs_fr:  same two comparisons with FR

    Outputs per layer per comparison:
      cosine_mean, cosine_std, jaccard_mean, n_matched_sentences, n_matched_tokens_mean
    """
    print("\n[Check 3b] Aligned embedded English token similarity ...")

    # Build lookup: (sid, cond) ? {token_string: [positions_in_cs]}
    # and           (sid, 'english') ? {token_string: [positions_in_en]}
    cs_token_lookup:  dict[tuple[str,str], dict[str, list[int]]] = {}
    en_token_lookup:  dict[str, dict[str, list[int]]]            = {}

    for _, row in dataset_df.iterrows():
        sid    = str(row["id"])
        cond   = str(row["condition"])
        labels = json.loads(row["token_lang_labels"]) if isinstance(row["token_lang_labels"], str) else []
        toks   = json.loads(row["token_strings"])     if isinstance(row["token_strings"],     str) else []

        if cond == "english":
            pos_map: dict[str, list[int]] = {}
            for i, tok in enumerate(toks):
                pos_map.setdefault(tok, []).append(i)
            en_token_lookup[sid] = pos_map

        elif cond in ("cs_fr", "cs_hi"):
            # Only keep positions labelled EN
            en_pos_map: dict[str, list[int]] = {}
            for i, (tok, lbl) in enumerate(zip(toks, labels)):
                if lbl == "EN":
                    en_pos_map.setdefault(tok, []).append(i)
            cs_token_lookup[(sid, cond)] = en_pos_map

    # Accumulators: (cs_cond, comparison) ? {layer: [cosine_scores]}
    # comparisons: 'vs_en_context'  ? EN token in CS  vs  same token in EN sentence
    #              'vs_matrix_mean' ? EN token in CS  vs  matrix-lang mean at same layer
    acc_cosine:  dict[tuple, dict[int, list[float]]] = {}
    acc_jaccard: dict[tuple, dict[int, list[float]]] = {}
    acc_ntokens: dict[tuple, dict[int, list[int]]]   = {}

    comparison_keys = [
        ("cs_hi", "vs_en_context"),
        ("cs_hi", "vs_matrix_mean"),
        ("cs_fr", "vs_en_context"),
        ("cs_fr", "vs_matrix_mean"),
    ]
    for key in comparison_keys:
        acc_cosine[key]  = {}
        acc_jaccard[key] = {}
        acc_ntokens[key] = {}

    for sid in tqdm(ids, desc="Aligned EN tokens"):
        en_pos_map = en_token_lookup.get(sid)
        if en_pos_map is None:
            continue

        en_mlp = load_tensor(mlp_dir / f"{safe_id(sid, 'english')}.pt")
        if en_mlp is None:
            continue

        for cs_cond in ("cs_hi", "cs_fr"):
            matrix_cond = "hindi" if cs_cond == "cs_hi" else "french"
            cs_en_map   = cs_token_lookup.get((sid, cs_cond), {})
            if not cs_en_map:
                continue

            cs_mlp     = load_tensor(mlp_dir / f"{safe_id(sid, cs_cond)}.pt")
            matrix_mlp = load_tensor(mlp_dir / f"{safe_id(sid, matrix_cond)}.pt")
            if cs_mlp is None or matrix_mlp is None:
                continue

            n_layers     = min(en_mlp.shape[0], cs_mlp.shape[0], matrix_mlp.shape[0])
            n_tokens_cs  = cs_mlp.shape[1]
            n_tokens_en  = en_mlp.shape[1]
            n_tokens_mat = matrix_mlp.shape[1]

            # Find matched token positions: tokens that appear in BOTH cs and en
            matched: list[tuple[int, int]] = []  # (cs_pos, en_pos)
            for tok, cs_positions in cs_en_map.items():
                en_positions = en_pos_map.get(tok, [])
                if not en_positions:
                    continue
                # Pair up positions (zip stops at shorter list)
                for cs_p, en_p in zip(cs_positions, en_positions):
                    if cs_p < n_tokens_cs and en_p < n_tokens_en:
                        matched.append((cs_p, en_p))

            if not matched:
                continue

            cs_positions_valid  = [m[0] for m in matched]
            en_positions_valid  = [m[1] for m in matched]
            n_matched           = len(matched)

            for l in range(n_layers):
                # Mean activation of matched EN tokens in CS context
                cs_vec = cs_mlp[l, cs_positions_valid, :].mean(dim=0).numpy()

                # comparison A: same tokens in pure EN context
                en_vec = en_mlp[l, en_positions_valid, :].mean(dim=0).numpy()

                # comparison B: matrix language mean at this layer
                mat_vec = matrix_mlp[l, :n_tokens_mat, :].mean(dim=0).numpy()

                key_en  = (cs_cond, "vs_en_context")
                key_mat = (cs_cond, "vs_matrix_mean")

                acc_cosine[key_en].setdefault(l,  []).append(cosine(cs_vec, en_vec))
                acc_cosine[key_mat].setdefault(l, []).append(cosine(cs_vec, mat_vec))

                acc_jaccard[key_en].setdefault(l,  []).append(jaccard(cs_vec, en_vec,  threshold))
                acc_jaccard[key_mat].setdefault(l, []).append(jaccard(cs_vec, mat_vec, threshold))

                acc_ntokens[key_en].setdefault(l,  []).append(n_matched)
                acc_ntokens[key_mat].setdefault(l, []).append(n_matched)

    all_layers = sorted(set().union(*[set(d.keys()) for d in acc_cosine.values()]))
    rows = []
    for l in all_layers:
        for cs_cond, comparison in comparison_keys:
            key = (cs_cond, comparison)
            c   = acc_cosine[key].get(l,  [])
            j   = acc_jaccard[key].get(l, [])
            nt  = acc_ntokens[key].get(l, [])
            rows.append({
                "layer":              l,
                "cs_condition":       cs_cond,
                "comparison":         comparison,
                "pair":               f"EN_in_{cs_cond}_{comparison}",
                "cosine_mean":        float(np.mean(c))  if c  else float("nan"),
                "cosine_std":         float(np.std(c))   if c  else float("nan"),
                "jaccard_mean":       float(np.mean(j))  if j  else float("nan"),
                "jaccard_std":        float(np.std(j))   if j  else float("nan"),
                "n_sentences":        len(c),
                "n_matched_tok_mean": float(np.mean(nt)) if nt else float("nan"),
            })
    return pd.DataFrame(rows)


# ?? Check 3: Embedded English token similarity ????????????????????????????????

def compute_embedded_token_similarity(
    ids: list[str],
    dataset_df: pd.DataFrame,
    mlp_dir: Path,
    threshold: float,
) -> pd.DataFrame:
    """
    Check 3: For English tokens that appear INSIDE a CS sentence, do their
    MLP activations look more like monolingual English or the matrix language?

    Compares per layer:
      EN-tokens-in-cs_hi  vs  monolingual_english  ? high = language-agnostic
      EN-tokens-in-cs_hi  vs  monolingual_hindi    ? high = matrix dominates even EN tokens
      EN-tokens-in-cs_fr  vs  monolingual_english
      EN-tokens-in-cs_fr  vs  monolingual_french

    This is the strongest frequency-artifact control: if even individual English
    tokens activate like Hindi/French, the dominance cannot be explained by the
    number of matrix-language tokens.
    """
    print("\n[Check 3] Embedded English token similarity ...")

    # Build lookup: (sid, cond) ? list of token indices labelled EN
    en_positions: dict[tuple[str, str], list[int]] = {}
    cs_df = dataset_df[dataset_df["condition"].isin(["cs_fr", "cs_hi"])]
    for _, row in cs_df.iterrows():
        sid    = str(row["id"])
        cond   = str(row["condition"])
        labels = json.loads(row["token_lang_labels"]) if isinstance(row["token_lang_labels"], str) else []
        en_pos = [i for i, lbl in enumerate(labels) if lbl == "EN"]
        if en_pos:
            en_positions[(sid, cond)] = en_pos

    CONDITIONS = ["english", "french", "hindi"]

    # Accumulators: (cs_cond, baseline_cond) ? {layer: [scores]}
    jaccard_acc: dict[tuple, dict[int, list[float]]] = {}
    cosine_acc:  dict[tuple, dict[int, list[float]]] = {}

    comparison_pairs = [
        ("cs_hi", "english"),  # embedded EN vs mono EN ? should be high if agnostic
        ("cs_hi", "hindi"),    # embedded EN vs mono HI ? high = matrix dominates
        ("cs_fr", "english"),
        ("cs_fr", "french"),
    ]
    for pair in comparison_pairs:
        jaccard_acc[pair] = {}
        cosine_acc[pair]  = {}

    for sid in tqdm(ids, desc="Embedded EN tokens"):
        # Load monolingual baselines for this sentence
        mono = {c: load_tensor(mlp_dir / f"{safe_id(sid, c)}.pt") for c in CONDITIONS}
        if any(v is None for v in mono.values()):
            continue

        for cs_cond in ["cs_hi", "cs_fr"]:
            en_pos = en_positions.get((sid, cs_cond))
            if not en_pos:
                continue  # no embedded EN tokens in this sentence

            cs_mlp = load_tensor(mlp_dir / f"{safe_id(sid, cs_cond)}.pt")
            if cs_mlp is None:
                continue

            n_layers   = min(cs_mlp.shape[0], min(m.shape[0] for m in mono.values()))
            n_tokens   = cs_mlp.shape[1]
            valid_pos  = [p for p in en_pos if p < n_tokens]
            if not valid_pos:
                continue

            for l in range(n_layers):
                # Mean activation of embedded EN tokens at this layer
                embedded_vec = cs_mlp[l, valid_pos, :].mean(dim=0).numpy()

                for _, baseline_cond in [(p, q) for p, q in comparison_pairs if p == cs_cond]:
                    # Mean activation of monolingual baseline at this layer
                    baseline_vec = mono[baseline_cond][l].mean(dim=0).numpy()

                    pair = (cs_cond, baseline_cond)
                    jaccard_acc[pair].setdefault(l, []).append(jaccard(embedded_vec, baseline_vec, threshold))
                    cosine_acc[pair].setdefault(l, []).append(cosine(embedded_vec, baseline_vec))

    all_layers = sorted(set().union(*[set(d.keys()) for d in list(jaccard_acc.values()) + list(cosine_acc.values())]))
    rows = []
    for l in all_layers:
        for cs_cond, baseline_cond in comparison_pairs:
            pair = (cs_cond, baseline_cond)
            j = jaccard_acc[pair].get(l, [])
            c = cosine_acc[pair].get(l, [])
            rows.append({
                "layer":           l,
                "cs_condition":    cs_cond,
                "baseline":        baseline_cond,
                "pair":            f"EN_in_{cs_cond}_vs_{baseline_cond}",
                "jaccard_mean":    float(np.mean(j)) if j else float("nan"),
                "jaccard_std":     float(np.std(j))  if j else float("nan"),
                "cosine_mean":     float(np.mean(c)) if c else float("nan"),
                "cosine_std":      float(np.std(c))  if c else float("nan"),
                "n_sentences":     len(j),
            })
    return pd.DataFrame(rows)


# ?? Metric 2: Layer-wise error emergence ??????????????????????????????????????

def compute_error_emergence(
    ids: list[str],
    switch_positions: dict[str, dict[str, list[int]]],
    residual_dir: Path,
) -> pd.DataFrame:
    """
    At each layer, compute L2 distance between cs and english residual stream
    specifically at switch token positions ? not averaged across whole sentence.

    This shows where in the network the model's internal state diverges from
    monolingual English specifically at the moment of switching.
    """
    print("\n[Metric 3] Layer-wise error emergence at switch points ...")

    fr_dists: dict[int, list[float]] = {}
    hi_dists: dict[int, list[float]] = {}

    for sid in tqdm(ids, desc="Error emergence"):
        en_resid = load_tensor(residual_dir / f"{safe_id(sid, 'english')}.pt")
        if en_resid is None:
            continue

        for cond, dists_dict in [("cs_fr", fr_dists), ("cs_hi", hi_dists)]:
            sw_pos = switch_positions.get(sid, {}).get(cond, [])
            if not sw_pos:
                continue

            cs_resid = load_tensor(residual_dir / f"{safe_id(sid, cond)}.pt")
            if cs_resid is None:
                continue

            n_layers    = min(en_resid.shape[0], cs_resid.shape[0])
            n_tokens_cs = cs_resid.shape[1]
            n_tokens_en = en_resid.shape[1]

            valid_sw = [s for s in sw_pos if s < n_tokens_cs]
            if not valid_sw:
                continue

            for l in range(n_layers):
                cs_at_switch = cs_resid[l, valid_sw, :].mean(dim=0).numpy()
                en_mean      = en_resid[l, :n_tokens_en, :].mean(dim=0).numpy()
                dist         = float(np.linalg.norm(cs_at_switch - en_mean))
                dists_dict.setdefault(l, []).append(dist)

    all_layers = sorted(set(fr_dists.keys()) | set(hi_dists.keys()))
    rows = []
    for l in all_layers:
        fr = fr_dists.get(l, [])
        hi = hi_dists.get(l, [])
        rows.append({
            "layer":            l,
            "cs_fr_error_mean": float(np.mean(fr)) if fr else float("nan"),
            "cs_fr_error_std":  float(np.std(fr))  if fr else float("nan"),
            "cs_hi_error_mean": float(np.mean(hi)) if hi else float("nan"),
            "cs_hi_error_std":  float(np.std(hi))  if hi else float("nan"),
            "n_fr":             len(fr),
            "n_hi":             len(hi),
        })
    return pd.DataFrame(rows)


# ?? Load switch positions ?????????????????????????????????????????????????????

def load_switch_positions(dataset_df: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    cs_df = dataset_df[dataset_df["condition"].isin(["cs_fr", "cs_hi"])]
    for _, row in cs_df.iterrows():
        sid  = str(row["id"])
        cond = str(row["condition"])
        sw   = json.loads(row["switch_positions"]) if isinstance(row["switch_positions"], str) else []
        result.setdefault(sid, {})[cond] = sw
    return result


# ?? Summary ???????????????????????????????????????????????????????????????????

def pair_stats(sim_df: pd.DataFrame, cond_a: str, cond_b: str) -> dict:
    sub = sim_df[sim_df["pair"] == f"{cond_a}_vs_{cond_b}"]
    if sub.empty:
        return {}
    max_layer = int(sub["layer"].max())
    early = sub[sub["layer"] <= 3]
    late  = sub[sub["layer"] >= max_layer - 3]
    return {
        "jaccard_mean":       float(sub["jaccard_mean"].mean()),
        "cosine_mean":        float(sub["cosine_mean"].mean()),
        "jaccard_early":      float(early["jaccard_mean"].mean()),
        "cosine_early":       float(early["cosine_mean"].mean()),
        "jaccard_late":       float(late["jaccard_mean"].mean()),
        "cosine_late":        float(late["cosine_mean"].mean()),
        "best_jaccard_layer": int(sub.loc[sub["jaccard_mean"].idxmax(), "layer"]),
        "best_cosine_layer":  int(sub.loc[sub["cosine_mean"].idxmax(),  "layer"]),
        "n_sentences":        int(sub["n_sentences"].max()),
    }


def embedded_stats(emb_df: pd.DataFrame, cs_cond: str, baseline: str) -> dict:
    sub = emb_df[emb_df["pair"] == f"EN_in_{cs_cond}_vs_{baseline}"]
    if sub.empty:
        return {}
    max_layer = int(sub["layer"].max())
    early = sub[sub["layer"] <= 3]
    late  = sub[sub["layer"] >= max_layer - 3]
    return {
        "cosine_mean":  float(sub["cosine_mean"].mean()),
        "cosine_early": float(early["cosine_mean"].mean()),
        "cosine_late":  float(late["cosine_mean"].mean()),
        "n_sentences":  int(sub["n_sentences"].max()),
    }


def aligned_stats(aln_df: pd.DataFrame, cs_cond: str, comparison: str) -> dict:
    sub = aln_df[
        (aln_df["cs_condition"] == cs_cond) &
        (aln_df["comparison"]   == comparison)
    ]
    if sub.empty:
        return {}
    max_layer = int(sub["layer"].max())
    early = sub[sub["layer"] <= 3]
    late  = sub[sub["layer"] >= max_layer - 3]
    return {
        "cosine_mean":        float(sub["cosine_mean"].mean()),
        "cosine_early":       float(early["cosine_mean"].mean()),
        "cosine_late":        float(late["cosine_mean"].mean()),
        "jaccard_mean":       float(sub["jaccard_mean"].mean()),
        "n_sentences":        int(sub["n_sentences"].max()),
        "n_matched_tok_mean": float(sub["n_matched_tok_mean"].mean()),
    }


def build_summary(
    sim_df: pd.DataFrame,
    sim_df_balanced: pd.DataFrame,
    emb_df: pd.DataFrame,
    aln_df: pd.DataFrame,
    err_df: pd.DataFrame,
    n_sentences: int,
    threshold: float,
    ratio_min: float,
    ratio_max: float,
) -> dict:

    max_layer = int(err_df["layer"].max()) if not err_df.empty else 0
    early_err = err_df[err_df["layer"] <= 3]
    late_err  = err_df[err_df["layer"] >= max_layer - 3]

    return {
        "n_sentences":          n_sentences,
        "activation_threshold": threshold,

        "metric1_neuron_similarity": {
            "fr_en_group": {
                "cs_fr_vs_french":   pair_stats(sim_df, "cs_fr",  "french"),
                "cs_fr_vs_english":  pair_stats(sim_df, "cs_fr",  "english"),
                "french_vs_english": pair_stats(sim_df, "french", "english"),
            },
            "hi_en_group": {
                "cs_hi_vs_hindi":    pair_stats(sim_df, "cs_hi",  "hindi"),
                "cs_hi_vs_english":  pair_stats(sim_df, "cs_hi",  "english"),
                "hindi_vs_english":  pair_stats(sim_df, "hindi",  "english"),
            },
            "cross_cs": {
                "cs_fr_vs_cs_hi": pair_stats(sim_df, "cs_fr", "cs_hi"),
            },
        },

        "check2_balanced_sentences": {
            "ratio_window": [ratio_min, ratio_max],
            "note": "Same as metric1 but only sentences with matrix-lang token ratio in window. "
                    "If dominance persists here, it cannot be explained by token frequency.",
            "fr_en_group": {
                "cs_fr_vs_french":   pair_stats(sim_df_balanced, "cs_fr",  "french"),
                "cs_fr_vs_english":  pair_stats(sim_df_balanced, "cs_fr",  "english"),
            },
            "hi_en_group": {
                "cs_hi_vs_hindi":    pair_stats(sim_df_balanced, "cs_hi",  "hindi"),
                "cs_hi_vs_english":  pair_stats(sim_df_balanced, "cs_hi",  "english"),
            },
            "cross_cs": {
                "cs_fr_vs_cs_hi": pair_stats(sim_df_balanced, "cs_fr", "cs_hi"),
            },
        },

        "check3b_aligned_embedded_english": {
            "note": "Position-matched comparison: same EN token string extracted from CS context "
                    "vs the identical token in the monolingual English sentence. "
                    "Eliminates whole-sentence mean baseline confound. "
                    "vs_en_context: how much does CS context shift EN token activations away from pure EN? "
                    "vs_matrix_mean: how much do shifted EN tokens resemble the matrix language?",
            "cs_hi": {
                "EN_tok_CS_vs_EN_context":    aligned_stats(aln_df, "cs_hi", "vs_en_context"),
                "EN_tok_CS_vs_matrix_mean":   aligned_stats(aln_df, "cs_hi", "vs_matrix_mean"),
                "interpretation": "If vs_matrix_mean > vs_en_context in late layers: "
                                  "Hindi context pulls EN token activations toward Hindi ? structural matrix dominance.",
            },
            "cs_fr": {
                "EN_tok_CS_vs_EN_context":    aligned_stats(aln_df, "cs_fr", "vs_en_context"),
                "EN_tok_CS_vs_matrix_mean":   aligned_stats(aln_df, "cs_fr", "vs_matrix_mean"),
                "interpretation": "Same as above for French-English.",
            },
        },

        "check3_embedded_english_tokens": {
            "note": "Activations of EN tokens inside CS sentences vs monolingual baselines. "
                    "If EN-in-cs_hi is more similar to Hindi than English, "
                    "the matrix language dominates even individual embedded tokens ? "
                    "ruling out frequency as an explanation.",
            "cs_hi": {
                "EN_tokens_vs_english": embedded_stats(emb_df, "cs_hi", "english"),
                "EN_tokens_vs_hindi":   embedded_stats(emb_df, "cs_hi", "hindi"),
                "interpretation":       "If hindi > english: matrix dominance is token-level, not frequency artifact",
            },
            "cs_fr": {
                "EN_tokens_vs_english": embedded_stats(emb_df, "cs_fr", "english"),
                "EN_tokens_vs_french":  embedded_stats(emb_df, "cs_fr", "french"),
                "interpretation":       "If french > english: matrix dominance is token-level, not frequency artifact",
            },
        },

        "metric2_error_emergence": {
            "cs_fr_error_mean":  float(err_df["cs_fr_error_mean"].mean()) if not err_df.empty else None,
            "cs_hi_error_mean":  float(err_df["cs_hi_error_mean"].mean()) if not err_df.empty else None,
            "cs_fr_error_early": float(early_err["cs_fr_error_mean"].mean()) if not early_err.empty else None,
            "cs_hi_error_early": float(early_err["cs_hi_error_mean"].mean()) if not early_err.empty else None,
            "cs_fr_error_late":  float(late_err["cs_fr_error_mean"].mean()) if not late_err.empty else None,
            "cs_hi_error_late":  float(late_err["cs_hi_error_mean"].mean()) if not late_err.empty else None,
            "peak_layer_cs_fr":  int(err_df.loc[err_df["cs_fr_error_mean"].idxmax(), "layer"]) if not err_df.empty else None,
            "peak_layer_cs_hi":  int(err_df.loc[err_df["cs_hi_error_mean"].idxmax(), "layer"]) if not err_df.empty else None,
            "note": "L2 distance from English baseline measured at switch token positions per layer.",
        },
    }


# ?? Main ??????????????????????????????????????????????????????????????????????

def main() -> None:
    args         = parse_args()
    mlp_dir      = Path(args.activations_dir) / "mlp"
    residual_dir = Path(args.activations_dir) / "residual"
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset_csv} ...")
    dataset_df = pd.read_csv(args.dataset_csv)
    ids = dataset_df["id"].unique().tolist()
    if args.max_sentences:
        ids = ids[:args.max_sentences]
    print(f"  {len(ids)} unique sentence ids")

    # Pre-compute token ratios for Check 2
    print("\nComputing matrix-language token ratios ...")
    token_ratios = compute_token_ratios(dataset_df)
    save_ratio_stats(token_ratios, out_dir)

    switch_pos = load_switch_positions(dataset_df)

    # Metric 1: full dataset
    sim_df = compute_pairwise_similarity(
        ids, mlp_dir, args.activation_threshold,
        label="Full neuron similarity",
    )

    # Check 2: balanced sentences only
    print(f"\n[Check 2] Re-running on balanced sentences "
          f"(matrix ratio {args.balance_ratio_min}?{args.balance_ratio_max}) ...")
    sim_df_balanced = compute_pairwise_similarity(
        ids, mlp_dir, args.activation_threshold,
        token_ratios=token_ratios,
        ratio_min=args.balance_ratio_min,
        ratio_max=args.balance_ratio_max,
        label="Balanced neuron similarity",
    )

    # Check 3 (original): embedded EN tokens vs whole-sentence baselines
    emb_df = compute_embedded_token_similarity(
        ids, dataset_df, mlp_dir, args.activation_threshold,
    )

    # Check 3b (aligned): same EN token in CS context vs same token in pure EN sentence
    aln_df = compute_aligned_embedded_similarity(
        ids, dataset_df, mlp_dir, args.activation_threshold,
    )

    # Metric 2: error emergence
    err_df = compute_error_emergence(ids, switch_pos, residual_dir)

    # Save CSVs
    sim_df.to_csv(out_dir          / "neuron_similarity.csv",          index=False)
    sim_df_balanced.to_csv(out_dir / "neuron_similarity_balanced.csv", index=False)
    emb_df.to_csv(out_dir          / "embedded_en_similarity.csv",     index=False)
    aln_df.to_csv(out_dir          / "aligned_en_similarity.csv",      index=False)
    err_df.to_csv(out_dir          / "error_emergence.csv",            index=False)

    summary = build_summary(
        sim_df, sim_df_balanced, emb_df, aln_df, err_df,
        n_sentences=len(ids),
        threshold=args.activation_threshold,
        ratio_min=args.balance_ratio_min,
        ratio_max=args.balance_ratio_max,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone. Results written to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
