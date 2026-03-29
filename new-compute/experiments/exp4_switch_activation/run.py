#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
    format_offset,
    load_dataset,
    load_tl_model,
    longest_common_prefix,
    resolve_device,
    write_neuron_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 4: switch-point neuron activation patterns "
            "(code-switched vs monolingual baselines)."
        )
    )
    p.add_argument("--dataset_csv", required=True, help="Dataset with id/text/condition/domain/source_id.")
    p.add_argument("--model_name", required=True, help="Hugging Face model name for TransformerLens.")
    p.add_argument("--out_dir", default="new-compute/experiments/exp4_switch_activation/results")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--focus_condition", default="code_switched")
    p.add_argument("--baseline_conditions", nargs="+", default=["english", "target_language"])
    p.add_argument("--token_offsets", nargs="+", type=int, default=[-1, 0, 1])
    p.add_argument("--z_threshold", type=float, default=2.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_groups", type=int, default=None)
    p.add_argument(
        "--gpu_friendly",
        action="store_true",
        help=(
            "Enable GPU-friendly settings: TF32, cuDNN benchmark, and mixed-precision "
            "model loading (bfloat16/float16 fallback) when running on CUDA."
        ),
    )
    return p.parse_args()


def _pair_name(focus_condition: str, baseline_condition: str) -> str:
    return f"{focus_condition}_vs_{baseline_condition}"


def _ensure_slot(slot_map: dict, key: tuple[str, str, int, int], vector: np.ndarray) -> dict:
    if key not in slot_map:
        zeros = np.zeros_like(vector, dtype=np.float64)
        slot_map[key] = {
            "n": 0,
            "focus_sum": zeros.copy(),
            "focus_sumsq": zeros.copy(),
            "base_sum": zeros.copy(),
            "base_sumsq": zeros.copy(),
            "delta_sum": zeros.copy(),
        }
    return slot_map[key]


def _positions_for_offsets(anchor: int, offsets: list[int], seq_len: int) -> set[int]:
    return {anchor + off for off in offsets if 0 <= anchor + off < seq_len}


def _safe_std(mean: np.ndarray, sumsq: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.full_like(mean, np.nan, dtype=np.float64)
    var = np.maximum((sumsq - (mean * mean * n)) / max(n - 1, 1), 0.0)
    return np.sqrt(var)


def _build_consensus(stats_df: pd.DataFrame, focus_condition: str, z_threshold: float) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()
    pivot = stats_df.pivot_table(
        index=["focus_condition", "relative_offset", "layer", "neuron"],
        columns="baseline_condition",
        values=["z_score", "mean_delta", "n_pairs"],
        aggfunc="first",
    )
    baseline_cols = list(pivot.columns.get_level_values(1).unique())
    if len(baseline_cols) < 2:
        return pd.DataFrame()

    z_cols = [("z_score", b) for b in baseline_cols]
    d_cols = [("mean_delta", b) for b in baseline_cols]
    n_cols = [("n_pairs", b) for b in baseline_cols]
    out = pivot.copy()
    out["min_z_score"] = out[z_cols].min(axis=1)
    out["min_mean_delta"] = out[d_cols].min(axis=1)
    out["min_pairs"] = out[n_cols].min(axis=1).astype(int)
    out["passes_consistency"] = (
        (out["min_z_score"] >= z_threshold)
        & (out["min_mean_delta"] > 0.0)
    )
    out = out.reset_index()
    out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    out = out[out["focus_condition"] == focus_condition].copy()
    out["required_baselines"] = ",".join(str(x) for x in baseline_cols)
    cols = [
        "focus_condition",
        "relative_offset",
        "layer",
        "neuron",
        "min_z_score",
        "min_mean_delta",
        "min_pairs",
        "passes_consistency",
        "required_baselines",
    ]
    return out[cols]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset_csv)
    device = resolve_device(args.device)
    configure_gpu_runtime(device=device, gpu_friendly=bool(args.gpu_friendly))
    model = load_tl_model(
        args.model_name,
        device=device,
        gpu_friendly=bool(args.gpu_friendly),
    )
    tokenizer = model.tokenizer

    by_source: dict[str, dict[str, dict[str, str]]] = {}
    for row in df.itertuples(index=False):
        by_source.setdefault(row.source_id, {})[row.condition] = {
            "id": row.id,
            "text": row.text,
            "domain": row.domain,
            "condition": row.condition,
        }

    source_items = list(by_source.items())
    if args.max_groups is not None:
        source_items = source_items[: max(1, int(args.max_groups))]

    stats_map: dict[tuple[str, str, int, int], dict[str, np.ndarray | int]] = {}
    event_rows: list[dict[str, object]] = []
    pair_counts: dict[tuple[str, str, int], int] = {}

    for idx, (source_id, cond_map) in enumerate(tqdm(source_items, desc="Exp4 matched groups"), start=1):
        if args.focus_condition not in cond_map:
            continue

        valid_baselines = [b for b in args.baseline_conditions if b in cond_map]
        if not valid_baselines:
            continue

        needed_conditions = [args.focus_condition] + valid_baselines
        token_id_cache: dict[str, list[int]] = {}
        act_cache: dict[str, dict[int, np.ndarray]] = {}
        seq_len_cache: dict[str, int] = {}

        for cond in needed_conditions:
            input_ids = encode_text(
                tokenizer=tokenizer,
                text=cond_map[cond]["text"],
                max_length=int(args.max_length),
                device=device,
            )
            ids = input_ids[0].detach().cpu().tolist()
            if len(ids) < 2:
                continue
            token_id_cache[cond] = ids
            seq_len_cache[cond] = len(ids)
            act_cache[cond] = extract_post_activations(model, input_ids)

        if args.focus_condition not in token_id_cache:
            continue

        for baseline_condition in valid_baselines:
            if baseline_condition not in token_id_cache:
                continue
            focus_ids = token_id_cache[args.focus_condition]
            base_ids = token_id_cache[baseline_condition]
            prefix_len = longest_common_prefix(focus_ids, base_ids)
            if prefix_len >= min(len(focus_ids), len(base_ids)):
                continue

            pair_name = _pair_name(args.focus_condition, baseline_condition)
            focus_anchor = int(prefix_len)
            base_anchor = int(prefix_len)
            event_rows.append(
                {
                    "source_id": source_id,
                    "comparison": pair_name,
                    "focus_condition": args.focus_condition,
                    "baseline_condition": baseline_condition,
                    "focus_event_token_index": focus_anchor,
                    "baseline_event_token_index": base_anchor,
                }
            )

            focus_positions = _positions_for_offsets(
                focus_anchor, args.token_offsets, seq_len_cache[args.focus_condition]
            )
            base_positions = _positions_for_offsets(
                base_anchor, args.token_offsets, seq_len_cache[baseline_condition]
            )
            common_offsets = []
            for rel_off in args.token_offsets:
                pf = focus_anchor + rel_off
                pb = base_anchor + rel_off
                if pf in focus_positions and pb in base_positions:
                    common_offsets.append(rel_off)

            if not common_offsets:
                continue

            for rel_off in common_offsets:
                focus_pos = focus_anchor + rel_off
                base_pos = base_anchor + rel_off
                key_count = (pair_name, baseline_condition, int(rel_off))
                pair_counts[key_count] = pair_counts.get(key_count, 0) + 1

                focus_layers = act_cache[args.focus_condition]
                base_layers = act_cache[baseline_condition]
                for layer in sorted(set(focus_layers).intersection(base_layers)):
                    focus_vec = focus_layers[layer][focus_pos].astype(np.float64, copy=False)
                    base_vec = base_layers[layer][base_pos].astype(np.float64, copy=False)
                    slot_key = (pair_name, baseline_condition, int(rel_off), int(layer))
                    slot = _ensure_slot(stats_map, slot_key, focus_vec)
                    slot["n"] = int(slot["n"]) + 1
                    slot["focus_sum"] += focus_vec
                    slot["focus_sumsq"] += focus_vec * focus_vec
                    slot["base_sum"] += base_vec
                    slot["base_sumsq"] += base_vec * base_vec
                    slot["delta_sum"] += focus_vec - base_vec
        if bool(args.gpu_friendly) and device.type == "cuda" and idx % 64 == 0:
            torch.cuda.empty_cache()

    if event_rows:
        pd.DataFrame(event_rows).to_csv(tables_dir / "event_alignments.csv", index=False)

    frames: list[pd.DataFrame] = []
    for (comparison, baseline_condition, rel_off, layer), slot in stats_map.items():
        n_pairs = int(slot["n"])
        if n_pairs <= 0:
            continue
        focus_mean = slot["focus_sum"] / n_pairs
        base_mean = slot["base_sum"] / n_pairs
        delta_mean = slot["delta_sum"] / n_pairs
        base_std = _safe_std(base_mean, slot["base_sumsq"], n_pairs)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.divide(delta_mean, base_std, out=np.zeros_like(delta_mean), where=base_std > 0)
        neurons = np.arange(len(delta_mean), dtype=int)
        frames.append(
            pd.DataFrame(
                {
                    "comparison": comparison,
                    "focus_condition": args.focus_condition,
                    "baseline_condition": baseline_condition,
                    "relative_offset": int(rel_off),
                    "layer": int(layer),
                    "neuron": neurons,
                    "n_pairs": n_pairs,
                    "mean_focus_activation": focus_mean,
                    "mean_baseline_activation": base_mean,
                    "std_baseline_activation": base_std,
                    "mean_delta": delta_mean,
                    "z_score": z,
                }
            )
        )

    stats_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not stats_df.empty:
        stats_df = stats_df.sort_values(
            ["comparison", "relative_offset", "layer", "neuron"]
        )
    stats_df.to_csv(
        tables_dir / "activation_pair_stats.csv.gz",
        index=False,
        compression="gzip",
    )

    summary_df = pd.DataFrame()
    if not stats_df.empty:
        summary_df = (
            stats_df.assign(
                z_gt_threshold=stats_df["z_score"] >= float(args.z_threshold),
            )
            .groupby(
                ["comparison", "baseline_condition", "relative_offset", "layer"],
                as_index=False,
            )
            .agg(
                n_pairs=("n_pairs", "max"),
                n_neurons=("neuron", "size"),
                n_switch_specific=("z_gt_threshold", "sum"),
                mean_abs_delta=("mean_delta", lambda s: float(np.mean(np.abs(s.to_numpy(dtype=float))))),
            )
        )
        summary_df["pct_switch_specific"] = (
            100.0 * summary_df["n_switch_specific"] / summary_df["n_neurons"].clip(lower=1)
        )
    summary_df.to_csv(tables_dir / "switch_specific_summary.csv", index=False)

    consensus_df = _build_consensus(
        stats_df=stats_df,
        focus_condition=args.focus_condition,
        z_threshold=float(args.z_threshold),
    )
    consensus_df.to_csv(
        tables_dir / "switch_consensus.csv.gz",
        index=False,
        compression="gzip",
    )

    if not stats_df.empty:
        for (comparison, rel_off), sub in stats_df.groupby(["comparison", "relative_offset"]):
            top = sub.sort_values(
                ["z_score", "mean_delta"],
                ascending=[False, False],
            ).head(int(args.top_k))
            top.to_csv(
                tables_dir / f"top_neurons_{comparison}_offset_{format_offset(int(rel_off))}.csv",
                index=False,
            )
            write_neuron_heatmap(
                sub,
                value_col="mean_delta",
                title=f"{comparison} offset {rel_off:+d} mean_delta",
                out_path=figures_dir / f"{comparison}_offset_{format_offset(int(rel_off))}_mean_delta_heatmap.html",
            )
            write_neuron_heatmap(
                sub,
                value_col="z_score",
                title=f"{comparison} offset {rel_off:+d} z_score",
                out_path=figures_dir / f"{comparison}_offset_{format_offset(int(rel_off))}_zscore_heatmap.html",
            )

    if not consensus_df.empty:
        for rel_off, sub in consensus_df.groupby("relative_offset"):
            write_neuron_heatmap(
                sub.rename(columns={"min_z_score": "value"}),
                value_col="value",
                title=f"switch consensus offset {int(rel_off):+d} min_z_score",
                out_path=figures_dir / f"switch_consensus_offset_{format_offset(int(rel_off))}_zscore_heatmap.html",
            )
            top_consensus = sub.sort_values(
                ["passes_consistency", "min_z_score", "min_mean_delta"],
                ascending=[False, False, False],
            ).head(int(args.top_k))
            top_consensus.to_csv(
                tables_dir / f"top_consensus_neurons_offset_{format_offset(int(rel_off))}.csv",
                index=False,
            )

    summary = {
        "experiment": "exp4_switch_activation",
        "dataset_csv": str(args.dataset_csv),
        "model_name": str(args.model_name),
        "focus_condition": str(args.focus_condition),
        "baseline_conditions": list(args.baseline_conditions),
        "token_offsets": [int(x) for x in args.token_offsets],
        "z_threshold": float(args.z_threshold),
        "top_k": int(args.top_k),
        "gpu_friendly": bool(args.gpu_friendly),
        "n_source_groups_scanned": int(len(source_items)),
        "n_event_pairs": int(len(event_rows)),
        "n_pair_offsets_with_data": int(len(pair_counts)),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
