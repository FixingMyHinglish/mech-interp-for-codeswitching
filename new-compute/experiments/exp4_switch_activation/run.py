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
            "Experiment 4: switch-point neuron activation patterns. "
            "For each sample, identifies neurons that are more active in the "
            "code-switched version than in EVERY monolingual baseline at the switch "
            "point. Reports neurons that show this pattern consistently across samples."
        )
    )
    p.add_argument("--dataset_csv", required=True, help="Dataset CSV with id/text/condition/domain/source_id.")
    p.add_argument("--model_name", required=True, help="Hugging Face model name for TransformerLens.")
    p.add_argument("--out_dir", default="new-compute/experiments/exp4_switch_activation/results")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--focus_condition", default="code_switched")
    p.add_argument("--baseline_conditions", nargs="+", default=["english", "target_language"])
    p.add_argument("--token_offsets", nargs="+", type=int, default=[-1, 0, 1])
    p.add_argument(
        "--min_consistency_fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of samples in which a neuron must show switch-specific "
            "activation to be included in the consistent output (default: 0.5)."
        ),
    )
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

    # Group samples by source_id so we can match code-switched with its monolinguals.
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

    # Per-(offset, layer) accumulation.
    # count_map[(rel_off, layer)] is a 1-D int array of length n_neurons where
    # count_map[key][n] = number of samples where neuron n was switch-specific.
    # total_map[rel_off] = number of samples that contributed to that offset.
    count_map: dict[tuple[int, int], np.ndarray] = {}
    total_map: dict[int, int] = {}

    event_rows: list[dict] = []
    skipped: list[dict] = []

    for idx, (source_id, cond_map) in enumerate(
        tqdm(source_items, desc="Exp4 matched groups"), start=1
    ):
        if args.focus_condition not in cond_map:
            continue

        # Require ALL baseline conditions to be present so the intersection
        # "active in cs but not in ANY monolingual" is meaningful.
        valid_baselines = [b for b in args.baseline_conditions if b in cond_map]
        if not valid_baselines:
            continue

        # ------------------------------------------------------------------
        # Tokenise all needed conditions and run the model.
        # ------------------------------------------------------------------
        needed_conditions = [args.focus_condition] + valid_baselines
        token_id_cache: dict[str, list[int]] = {}
        act_cache: dict[str, dict[int, np.ndarray]] = {}
        seq_len_cache: dict[str, int] = {}

        skip_sample = False
        for cond in needed_conditions:
            input_ids = encode_text(
                tokenizer=tokenizer,
                text=cond_map[cond]["text"],
                max_length=int(args.max_length),
                device=device,
            )
            ids = input_ids[0].detach().cpu().tolist()
            if len(ids) < 2:
                skip_sample = True
                break
            token_id_cache[cond] = ids
            seq_len_cache[cond] = len(ids)
            act_cache[cond] = extract_post_activations(model, input_ids)

        if skip_sample or any(c not in token_id_cache for c in needed_conditions):
            skipped.append({"source_id": source_id, "reason": "short_sequence"})
            continue

        # ------------------------------------------------------------------
        # Find the switch point (longest common prefix) between the focus
        # condition and each baseline separately.  Each comparison may have
        # its switch at a different token position.
        # anchor_map[baseline][rel_off] = (focus_pos, base_pos)
        # ------------------------------------------------------------------
        anchor_map: dict[str, dict[int, tuple[int, int]]] = {}

        all_valid = True
        for baseline in valid_baselines:
            focus_ids = token_id_cache[args.focus_condition]
            base_ids = token_id_cache[baseline]
            prefix_len = longest_common_prefix(focus_ids, base_ids)

            if prefix_len >= min(len(focus_ids), len(base_ids)):
                # Texts are identical up to the shorter one – no real switch point.
                skipped.append(
                    {"source_id": source_id, "reason": f"no_switch_point_vs_{baseline}"}
                )
                all_valid = False
                break

            focus_anchor = int(prefix_len)
            base_anchor = int(prefix_len)

            event_rows.append(
                {
                    "source_id": source_id,
                    "focus_condition": args.focus_condition,
                    "baseline_condition": baseline,
                    "focus_event_token_index": focus_anchor,
                    "baseline_event_token_index": base_anchor,
                }
            )

            offsets_for_pair: dict[int, tuple[int, int]] = {}
            for rel_off in args.token_offsets:
                fp = focus_anchor + rel_off
                bp = base_anchor + rel_off
                if (
                    0 <= fp < seq_len_cache[args.focus_condition]
                    and 0 <= bp < seq_len_cache[baseline]
                ):
                    offsets_for_pair[rel_off] = (fp, bp)

            anchor_map[baseline] = offsets_for_pair

        if not all_valid:
            continue

        # ------------------------------------------------------------------
        # For each offset, build the switch-specific boolean mask for this
        # sample, then accumulate counts.
        #
        # A neuron is "switch-specific" for this sample at this offset if
        # its activation in the code-switched text is strictly greater than
        # its activation in EVERY monolingual baseline at the corresponding
        # switch position.
        # ------------------------------------------------------------------
        focus_acts = act_cache[args.focus_condition]
        common_layers = set(focus_acts.keys())
        for baseline in valid_baselines:
            common_layers &= set(act_cache[baseline].keys())

        # An offset is usable only if every baseline has a valid position for it.
        usable_offsets = [
            rel_off
            for rel_off in args.token_offsets
            if all(rel_off in anchor_map[b] for b in valid_baselines)
        ]

        if not usable_offsets:
            skipped.append({"source_id": source_id, "reason": "no_usable_offsets"})
            continue

        for rel_off in usable_offsets:
            offset_contributed = False

            for layer in sorted(common_layers):
                # n_neurons determined from the first position of the focus activations.
                n_neurons = focus_acts[layer][0].shape[0]
                # Start with every neuron as a candidate.
                switch_mask = np.ones(n_neurons, dtype=bool)

                for baseline in valid_baselines:
                    focus_pos, base_pos = anchor_map[baseline][rel_off]
                    focus_vec = focus_acts[layer][focus_pos].astype(np.float64)
                    base_vec = act_cache[baseline][layer][base_pos].astype(np.float64)
                    # Neuron must be strictly more active in cs than this baseline.
                    switch_mask &= focus_vec > base_vec

                key = (int(rel_off), int(layer))
                if key not in count_map:
                    count_map[key] = np.zeros(n_neurons, dtype=np.int64)
                count_map[key] += switch_mask.astype(np.int64)
                offset_contributed = True

            # Count this sample once per offset once at least one layer was processed.
            if offset_contributed:
                total_map[rel_off] = total_map.get(rel_off, 0) + 1

        if bool(args.gpu_friendly) and device.type == "cuda" and idx % 64 == 0:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save event alignments and skipped log.
    # ------------------------------------------------------------------
    if event_rows:
        pd.DataFrame(event_rows).to_csv(tables_dir / "event_alignments.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(tables_dir / "skipped_samples.csv", index=False)

    # ------------------------------------------------------------------
    # Build the per-neuron count table and the consistency-filtered table.
    # ------------------------------------------------------------------
    all_rows: list[pd.DataFrame] = []
    consistent_rows: list[pd.DataFrame] = []

    for (rel_off, layer), counts in sorted(count_map.items()):
        n_total = total_map.get(rel_off, 0)
        if n_total == 0:
            continue

        n_neurons = len(counts)
        neurons = np.arange(n_neurons, dtype=int)
        consistency = counts / float(n_total)

        frame = pd.DataFrame(
            {
                "relative_offset": int(rel_off),
                "layer": int(layer),
                "neuron": neurons,
                "n_samples_switch_specific": counts,
                "n_samples_total": n_total,
                "consistency_fraction": consistency,
            }
        )
        all_rows.append(frame)

        mask = consistency >= float(args.min_consistency_fraction)
        if mask.any():
            consistent_rows.append(frame[mask].copy())

    # Full counts table (every neuron, every offset/layer).
    all_neurons_df = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(
            columns=[
                "relative_offset",
                "layer",
                "neuron",
                "n_samples_switch_specific",
                "n_samples_total",
                "consistency_fraction",
            ]
        )
    )
    if not all_neurons_df.empty:
        all_neurons_df = all_neurons_df.sort_values(
            ["relative_offset", "layer", "neuron"]
        ).reset_index(drop=True)

    all_neurons_df.to_csv(
        tables_dir / "switch_neuron_counts.csv.gz",
        index=False,
        compression="gzip",
    )

    # Consistent-neurons table (filtered, sorted by consistency desc).
    consistent_df = (
        pd.concat(consistent_rows, ignore_index=True)
        if consistent_rows
        else pd.DataFrame(
            columns=[
                "relative_offset",
                "layer",
                "neuron",
                "n_samples_switch_specific",
                "n_samples_total",
                "consistency_fraction",
            ]
        )
    )
    if not consistent_df.empty:
        consistent_df = consistent_df.sort_values(
            ["relative_offset", "consistency_fraction", "n_samples_switch_specific"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    consistent_df.to_csv(tables_dir / "consistent_switch_neurons.csv", index=False)

    # ------------------------------------------------------------------
    # Per-offset summary table.
    # ------------------------------------------------------------------
    summary_rows: list[dict] = []
    for rel_off in sorted(total_map.keys()):
        n_total = total_map[rel_off]
        sub = all_neurons_df[all_neurons_df["relative_offset"] == rel_off]
        if sub.empty:
            continue
        n_consistent = int(
            (sub["consistency_fraction"] >= float(args.min_consistency_fraction)).sum()
        )
        summary_rows.append(
            {
                "relative_offset": int(rel_off),
                "n_samples_total": n_total,
                "n_neuron_layer_pairs_total": int(len(sub)),
                "n_consistent_neuron_layer_pairs": n_consistent,
                "pct_consistent": round(100.0 * n_consistent / max(len(sub), 1), 4),
                "max_consistency_fraction": float(sub["consistency_fraction"].max()),
                "mean_consistency_fraction": float(sub["consistency_fraction"].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(tables_dir / "switch_neuron_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Heatmaps of consistency fraction and per-offset consistent-neuron tables.
    # ------------------------------------------------------------------
    if not all_neurons_df.empty:
        for rel_off, sub in all_neurons_df.groupby("relative_offset"):
            write_neuron_heatmap(
                sub,
                value_col="consistency_fraction",
                title=f"Switch-specific consistency offset {int(rel_off):+d}",
                out_path=figures_dir
                / f"switch_consistency_offset_{format_offset(int(rel_off))}_heatmap.html",
            )

    if not consistent_df.empty:
        for rel_off, sub in consistent_df.groupby("relative_offset"):
            sub.sort_values(
                ["consistency_fraction", "n_samples_switch_specific"],
                ascending=[False, False],
            ).to_csv(
                tables_dir
                / f"consistent_neurons_offset_{format_offset(int(rel_off))}.csv",
                index=False,
            )

    # ------------------------------------------------------------------
    # Run summary JSON.
    # ------------------------------------------------------------------
    run_summary = {
        "experiment": "exp4_switch_activation",
        "dataset_csv": str(args.dataset_csv),
        "model_name": str(args.model_name),
        "focus_condition": str(args.focus_condition),
        "baseline_conditions": list(args.baseline_conditions),
        "token_offsets": [int(x) for x in args.token_offsets],
        "min_consistency_fraction": float(args.min_consistency_fraction),
        "gpu_friendly": bool(args.gpu_friendly),
        "n_source_groups_scanned": int(len(source_items)),
        "n_event_pairs": int(len(event_rows)),
        "n_skipped": int(len(skipped)),
        "n_consistent_neuron_layer_pairs": int(len(consistent_df)),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
