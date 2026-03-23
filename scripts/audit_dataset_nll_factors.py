#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit dataset/tokenization factors that can invert NLL expectations across conditions"
    )
    p.add_argument("--dataset", required=True, help="Path to prepared dataset CSV/JSON/JSONL")
    p.add_argument("--model_name", default="gpt2", help="Tokenizer source model")
    p.add_argument("--run_dir", default="", help="Optional completed run dir to join NLL metrics")
    p.add_argument("--output_dir", default="outputs/audit_nll_factors", help="Output directory")
    return p.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raise ValueError("Unsupported dataset format")

    required = {"id", "text", "condition", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df.copy()


def tokenize_stats(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    token_lists = []
    for t in df["text"].astype(str).tolist():
        enc = tokenizer(t, add_special_tokens=False, return_attention_mask=False)
        token_lists.append(enc["input_ids"])

    df = df.copy()
    df["token_ids"] = token_lists
    df["n_chars"] = df["text"].astype(str).str.len()
    df["n_words"] = df["text"].astype(str).str.split().str.len()
    df["n_tokens"] = df["token_ids"].apply(len)
    df["chars_per_token"] = df["n_chars"] / df["n_tokens"].replace(0, np.nan)

    all_ids = [tid for row in token_lists for tid in row]
    if all_ids:
        vc = pd.Series(all_ids).value_counts()
        total = float(vc.sum())

        def rarity(ids: list[int]) -> float:
            if not ids:
                return float("nan")
            probs = np.array([float(vc.get(i, 1)) / total for i in ids], dtype=np.float64)
            return float((-np.log(probs + 1e-12)).mean())

        def uniq_ratio(ids: list[int]) -> float:
            if not ids:
                return float("nan")
            return float(len(set(ids)) / len(ids))

        df["token_rarity"] = df["token_ids"].apply(rarity)
        df["token_unique_ratio"] = df["token_ids"].apply(uniq_ratio)
    else:
        df["token_rarity"] = np.nan
        df["token_unique_ratio"] = np.nan

    return df


def load_run_nll(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "tables" / "layer_metrics_raw.csv"
    if not p.exists():
        return pd.DataFrame()
    ldf = pd.read_csv(p)
    req = {"id", "condition", "layer", "next_token_nll"}
    if not req.issubset(ldf.columns):
        return pd.DataFrame()

    # Use final layer per sample as closest proxy to model output behavior.
    idx = ldf.groupby(["id", "condition"]) ["layer"].idxmax()
    fin = ldf.loc[idx, ["id", "condition", "next_token_nll"]].copy()
    fin = fin.rename(columns={"next_token_nll": "final_layer_nll"})
    return fin


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_root = Path(args.output_dir)
    tables = out_root / "tables"
    out_root.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tok_df = tokenize_stats(df, tokenizer)

    dup_rate = (
        tok_df.groupby("condition")["text"].apply(lambda s: 1.0 - (s.nunique() / len(s))).rename("duplicate_rate")
    )

    cols = [
        "n_chars",
        "n_words",
        "n_tokens",
        "chars_per_token",
        "token_rarity",
        "token_unique_ratio",
    ]
    cond_stats = tok_df.groupby("condition")[cols].agg(["mean", "median", "std"]) 
    cond_stats.columns = [f"{a}_{b}" for a, b in cond_stats.columns]
    cond_stats = cond_stats.reset_index().merge(dup_rate.reset_index(), on="condition", how="left")

    run_nll_df = pd.DataFrame()
    if args.run_dir:
        run_nll_df = load_run_nll(Path(args.run_dir))
        if not run_nll_df.empty:
            tok_df = tok_df.merge(run_nll_df, on=["id", "condition"], how="left")
            tok_df["nll_per_token"] = tok_df["final_layer_nll"] / tok_df["n_tokens"].replace(0, np.nan)

    # Pairwise matched-source analysis when source_id exists
    paired_summary = pd.DataFrame()
    if "source_id" in tok_df.columns:
        pair_rows = []
        piv = tok_df.pivot_table(
            index="source_id",
            columns="condition",
            values=["n_tokens", "token_rarity", "final_layer_nll", "nll_per_token"],
            aggfunc="first",
        )
        if isinstance(piv.columns, pd.MultiIndex):
            metrics = sorted(set(m for m, _ in piv.columns))
            conds = sorted(set(c for _, c in piv.columns))
            for m in metrics:
                for i, a in enumerate(conds):
                    for b in conds[i + 1 :]:
                        if (m, a) not in piv.columns or (m, b) not in piv.columns:
                            continue
                        d = (piv[(m, a)] - piv[(m, b)]).dropna()
                        if len(d) == 0:
                            continue
                        pair_rows.append(
                            {
                                "metric": m,
                                "condition_a": a,
                                "condition_b": b,
                                "mean_a_minus_b": float(d.mean()),
                                "median_a_minus_b": float(d.median()),
                                "n_pairs": int(len(d)),
                            }
                        )
            paired_summary = pd.DataFrame(pair_rows)

    tok_df.drop(columns=["token_ids"], errors="ignore").to_csv(tables / "sample_tokenization_audit.csv", index=False)
    cond_stats.to_csv(tables / "condition_tokenization_summary.csv", index=False)
    if not run_nll_df.empty:
        run_nll_df.to_csv(tables / "run_final_layer_nll.csv", index=False)
    if not paired_summary.empty:
        paired_summary.to_csv(tables / "paired_condition_differences.csv", index=False)

    lines = [
        "# Dataset/NLL Audit",
        "",
        f"- Dataset: {dataset_path}",
        f"- Tokenizer model: {args.model_name}",
        f"- Samples: {len(df)}",
        f"- Conditions: {sorted(df['condition'].unique().tolist())}",
        "",
        "## High-level checks",
        "- Compare `n_tokens_mean` across conditions (length effects).",
        "- Compare `token_rarity_mean` and `token_unique_ratio_mean` (rarity/repetition effects).",
        "- Compare `duplicate_rate` (repeated text can lower NLL).",
    ]
    if args.run_dir:
        lines += [
            "- Compare `nll_per_token` by condition (normalize for length).",
            "- Use paired differences table if `source_id` exists.",
        ]

    (out_root / "AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "dataset": str(dataset_path),
        "model_name": args.model_name,
        "run_dir": args.run_dir,
        "n_samples": int(len(df)),
        "conditions": sorted(df["condition"].astype(str).unique().tolist()),
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Audit written to: {out_root}")


if __name__ == "__main__":
    main()
