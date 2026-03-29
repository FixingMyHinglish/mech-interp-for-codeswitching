#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 21: compare mechanistic patterns between language pairs "
            "(Hi-En vs Fr-En) using outputs from exp4 and exp6."
        )
    )
    p.add_argument("--hindi_exp4_dir", required=True)
    p.add_argument("--french_exp4_dir", required=True)
    p.add_argument("--hindi_exp6_dir", default=None)
    p.add_argument("--french_exp6_dir", default=None)
    p.add_argument("--out_dir", default="new-compute/experiments/exp21_language_pair_comparison/results")
    p.add_argument("--z_threshold", type=float, default=2.0)
    p.add_argument("--selectivity_threshold", type=float, default=0.5)
    return p.parse_args()


def _jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_exp4_switch_set(exp4_dir: Path, z_threshold: float) -> tuple[pd.DataFrame, dict[int, set[tuple[int, int]]]]:
    consensus_path = exp4_dir / "tables" / "switch_consensus.csv.gz"
    if consensus_path.exists():
        df = pd.read_csv(consensus_path)
        df = df[df["passes_consistency"].astype(bool)].copy()
        if df.empty:
            return df, {}
        by_offset = {
            int(offset): set(zip(sub["layer"].astype(int), sub["neuron"].astype(int)))
            for offset, sub in df.groupby("relative_offset")
        }
        return df, by_offset

    stats_path = exp4_dir / "tables" / "activation_pair_stats.csv.gz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing exp4 output file: {consensus_path} or {stats_path}")
    stats = pd.read_csv(stats_path)
    if stats.empty:
        return stats, {}

    pivot = stats.pivot_table(
        index=["relative_offset", "layer", "neuron"],
        columns="baseline_condition",
        values="z_score",
        aggfunc="first",
    )
    if pivot.empty:
        return stats, {}
    mask = (pivot >= float(z_threshold)).all(axis=1)
    kept = pivot[mask].reset_index()[["relative_offset", "layer", "neuron"]]
    by_offset = {
        int(offset): set(zip(sub["layer"].astype(int), sub["neuron"].astype(int)))
        for offset, sub in kept.groupby("relative_offset")
    }
    return kept, by_offset


def _layer_vector(neuron_set: set[tuple[int, int]]) -> np.ndarray:
    if not neuron_set:
        return np.zeros(1, dtype=np.float64)
    max_layer = max(layer for layer, _ in neuron_set)
    arr = np.zeros(max_layer + 1, dtype=np.float64)
    for layer, _ in neuron_set:
        arr[layer] += 1.0
    return arr


def _compare_sets_by_offset(
    hindi_by_offset: dict[int, set[tuple[int, int]]],
    french_by_offset: dict[int, set[tuple[int, int]]],
    label: str,
) -> pd.DataFrame:
    rows = []
    for offset in sorted(set(hindi_by_offset) | set(french_by_offset)):
        hi_set = hindi_by_offset.get(offset, set())
        fr_set = french_by_offset.get(offset, set())
        hi_vec = _layer_vector(hi_set)
        fr_vec = _layer_vector(fr_set)
        max_len = max(len(hi_vec), len(fr_vec))
        hi_pad = np.pad(hi_vec, (0, max_len - len(hi_vec)))
        fr_pad = np.pad(fr_vec, (0, max_len - len(fr_vec)))
        rows.append(
            {
                "metric_group": label,
                "relative_offset": int(offset),
                "hindi_count": int(len(hi_set)),
                "french_count": int(len(fr_set)),
                "overlap_count": int(len(hi_set & fr_set)),
                "hindi_only_count": int(len(hi_set - fr_set)),
                "french_only_count": int(len(fr_set - hi_set)),
                "jaccard": float(_jaccard(hi_set, fr_set)),
                "layer_profile_cosine": float(_cosine(hi_pad, fr_pad)),
            }
        )
    return pd.DataFrame(rows)


def _load_exp6_selective_set(
    exp6_dir: Path,
    threshold: float,
) -> tuple[pd.DataFrame, dict[int, set[tuple[int, int]]]]:
    path = exp6_dir / "tables" / "language_selectivity.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing exp6 output file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df, {}
    kept = df[np.abs(df["selectivity_index"]) >= float(threshold)].copy()
    by_offset = {
        int(offset): set(zip(sub["layer"].astype(int), sub["neuron"].astype(int)))
        for offset, sub in kept.groupby("relative_offset")
    }
    return kept, by_offset


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hindi_exp4_dir = Path(args.hindi_exp4_dir)
    french_exp4_dir = Path(args.french_exp4_dir)
    _, hi_exp4 = _load_exp4_switch_set(hindi_exp4_dir, z_threshold=float(args.z_threshold))
    _, fr_exp4 = _load_exp4_switch_set(french_exp4_dir, z_threshold=float(args.z_threshold))
    exp4_cmp = _compare_sets_by_offset(hi_exp4, fr_exp4, label="exp4_switch_specific")
    exp4_cmp.to_csv(out_dir / "exp4_overlap_by_offset.csv", index=False)

    exp6_cmp = pd.DataFrame()
    exp6_used = bool(args.hindi_exp6_dir and args.french_exp6_dir)
    if exp6_used:
        hindi_exp6_dir = Path(args.hindi_exp6_dir)
        french_exp6_dir = Path(args.french_exp6_dir)
        _, hi_exp6 = _load_exp6_selective_set(
            hindi_exp6_dir, threshold=float(args.selectivity_threshold)
        )
        _, fr_exp6 = _load_exp6_selective_set(
            french_exp6_dir, threshold=float(args.selectivity_threshold)
        )
        exp6_cmp = _compare_sets_by_offset(hi_exp6, fr_exp6, label="exp6_language_selective")
        exp6_cmp.to_csv(out_dir / "exp6_overlap_by_offset.csv", index=False)

    combined = pd.concat(
        [df for df in [exp4_cmp, exp6_cmp] if not df.empty],
        ignore_index=True,
    ) if (not exp4_cmp.empty or not exp6_cmp.empty) else pd.DataFrame()
    combined.to_csv(out_dir / "language_pair_comparison.csv", index=False)

    summary = {
        "experiment": "exp21_language_pair_comparison",
        "hindi_exp4_dir": str(hindi_exp4_dir),
        "french_exp4_dir": str(french_exp4_dir),
        "hindi_exp6_dir": str(args.hindi_exp6_dir) if args.hindi_exp6_dir else None,
        "french_exp6_dir": str(args.french_exp6_dir) if args.french_exp6_dir else None,
        "z_threshold": float(args.z_threshold),
        "selectivity_threshold": float(args.selectivity_threshold),
        "exp6_used": exp6_used,
        "rows_exp4_comparison": int(len(exp4_cmp)),
        "rows_exp6_comparison": int(len(exp6_cmp)),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

