#!/usr/bin/env python3
"""
Experiment 4 v2 : Code-switch specific neuron detection.

Idea
----
For every sample in ds (English / target-language / code-switched):

1. Extract post-MLP activations for each setting.
2. Pool each condition's activation across *all* token positions by taking
   the per-neuron mean (shape: [n_layers, n_neurons]).
3. Compute a CS-specificity z-score for each neuron in each layer:

       z = (act_CS - mean(act_EN, act_TL)) / std(act_EN, act_TL)

   std is taken over the two baseline values; when they are identical the
   std is 0 and z is set to 0 (no contrast available).

4. A neuron "qualifies" for this sample if z > z_threshold.

Across all samples:
5. Rank neurons by qualify_count (how many samples they qualified in),
   breaking ties by mean z-score.
6. Write ranked tables, per-layer heatmaps, and a JSON summary.
"""

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
    resolve_device,
    write_neuron_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 4 (rewrite): find neurons that fire specifically in "
            "code-switched context by z-scoring CS activations against the "
            "two monolingual baselines, then ranking neurons by how often "
            "they qualify across all samples."
        )
    )
    p.add_argument(
        "--dataset_csv",
        required=True,
        help="CSV with columns: id, text, condition, domain, source_id.",
    )
    p.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model name for TransformerLens.",
    )
    p.add_argument(
        "--out_dir", default="new-compute/experiments/exp4_switch_activation/results"
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=256)

    # Condition labels
    p.add_argument("--focus_condition", default="code_switched")
    p.add_argument(
        "--baseline_conditions", nargs="+", default=["english", "target_language"]
    )

    # Scoring
    p.add_argument(
        "--z_threshold",
        type=float,
        default=2.0,
        help="A neuron qualifies for a sample if its CS z-score " "exceeds this value.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top neurons to write per layer.",
    )
    p.add_argument(
        "--min_qualify_frac",
        type=float,
        default=0.0,
        help="Optional: only keep neurons that qualify in at least "
        "this fraction of samples (0.0 = keep all, use e.g. "
        "0.5 for a soft-50%% intersection).",
    )
    p.add_argument("--max_groups", type=int, default=None)
    p.add_argument("--gpu_friendly", action="store_true")
    return p.parse_args()


def _mean_pool(act_by_layer: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """
    Convert {layer: [seq_len, n_neurons]} → {layer: [n_neurons]} by averaging
    over all token positions.
    """
    return {
        layer: act.mean(axis=0).astype(np.float64)
        for layer, act in act_by_layer.items()
    }


def _cs_z_scores(
    pooled_cs: dict[int, np.ndarray],
    pooled_baselines: list[dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """
    For each layer compute per-neuron z-score:

        z = (act_CS - mean_baseline) / std_baseline

    where mean/std are taken across the N baseline conditions.
    When std == 0, z is set to 0.

    Returns {layer: z_vector}.
    """
    layers = sorted(pooled_cs.keys())
    z_by_layer: dict[int, np.ndarray] = {}

    for layer in layers:
        cs_vec = pooled_cs[layer]  # [n_neurons]
        base_stack = np.stack(  # [n_baselines, n_neurons]
            [b[layer] for b in pooled_baselines if layer in b],
            axis=0,
        )
        if base_stack.shape[0] == 0:
            z_by_layer[layer] = np.zeros_like(cs_vec)
            continue

        base_mean = base_stack.mean(axis=0)
        base_std = base_stack.std(axis=0, ddof=0)  # population std over baselines

        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(
                base_std > 0,
                (cs_vec - base_mean) / base_std,
                0.0,
            )
        z_by_layer[layer] = z

    return z_by_layer


class NeuronAccumulator:
    """
    Incrementally tracks qualify_count and z-score statistics per
    (layer, neuron) across samples.
    """

    def __init__(self, z_threshold: float) -> None:
        self.z_threshold = z_threshold
        # {layer: np.ndarray of shape [n_neurons]}
        self._qualify_count: dict[int, np.ndarray] = {}
        self._z_sum: dict[int, np.ndarray] = {}
        self._z_sumsq: dict[int, np.ndarray] = {}
        self._z_sumsq: dict[int, np.ndarray] = {}
        self._n_samples: dict[int, np.ndarray] = {}

    def update(self, z_by_layer: dict[int, np.ndarray]) -> None:
        for layer, z_vec in z_by_layer.items():
            n = len(z_vec)
            if layer not in self._qualify_count:
                self._qualify_count[layer] = np.zeros(n, dtype=np.int64)
                self._z_sum[layer] = np.zeros(n, dtype=np.float64)
                self._z_sumsq[layer] = np.zeros(n, dtype=np.float64)
                self._n_samples[layer] = np.zeros(n, dtype=np.int64)

            qualifies = z_vec >= self.z_threshold
            self._qualify_count[layer] += qualifies.astype(np.int64)
            self._z_sum[layer] += z_vec
            self._z_sumsq[layer] += z_vec * z_vec
            self._n_samples[layer] += 1

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for layer in sorted(self._qualify_count.keys()):
            n_samples = self._n_samples[layer]  # per-neuron sample count
            qc = self._qualify_count[layer]
            z_sum = self._z_sum[layer]
            z_sumsq = self._z_sumsq[layer]
            n_neurons = len(qc)

            safe_n = np.maximum(n_samples, 1)
            mean_z = z_sum / safe_n
            var_z = np.maximum(z_sumsq / safe_n - mean_z**2, 0.0)
            std_z = np.sqrt(var_z)
            qual_frac = qc / safe_n

            for neuron in range(n_neurons):
                rows.append(
                    {
                        "layer": int(layer),
                        "neuron": int(neuron),
                        "n_samples": int(n_samples[neuron]),
                        "qualify_count": int(qc[neuron]),
                        "qualify_frac": float(qual_frac[neuron]),
                        "mean_z_score": float(mean_z[neuron]),
                        "std_z_score": float(std_z[neuron]),
                    }
                )

        return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data + model
    # ------------------------------------------------------------------
    df = load_dataset(args.dataset_csv)
    device = resolve_device(args.device)
    configure_gpu_runtime(device=device, gpu_friendly=bool(args.gpu_friendly))
    model = load_tl_model(
        args.model_name, device=device, gpu_friendly=bool(args.gpu_friendly)
    )
    tokenizer = model.tokenizer

    # Group rows by source_id
    by_source: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        by_source.setdefault(row.source_id, {})[row.condition] = row.text

    source_items = list(by_source.items())
    if args.max_groups is not None:
        source_items = source_items[: max(1, int(args.max_groups))]

    # ------------------------------------------------------------------
    # Accumulate per-sample z-scores
    # ------------------------------------------------------------------
    accumulator = NeuronAccumulator(z_threshold=float(args.z_threshold))
    n_processed = 0
    n_skipped = 0

    for idx, (source_id, cond_map) in enumerate(
        tqdm(source_items, desc="Exp4 groups"), start=1
    ):
        # Need all conditions to be present
        needed = [args.focus_condition] + list(args.baseline_conditions)
        if not all(c in cond_map for c in needed):
            n_skipped += 1
            continue

        # Encode + extract activations for every needed condition
        pooled: dict[str, dict[int, np.ndarray]] = {}
        ok = True
        for cond in needed:
            input_ids = encode_text(
                tokenizer=tokenizer,
                text=cond_map[cond],
                max_length=int(args.max_length),
                device=device,
            )
            ids = input_ids[0].detach().cpu().tolist()
            if len(ids) < 2:
                ok = False
                break
            raw_acts = extract_post_activations(model, input_ids)
            # raw_acts: {layer: np.ndarray [seq_len, n_neurons]}
            pooled[cond] = _mean_pool(raw_acts)

        if not ok:
            n_skipped += 1
            continue

        # Compute per-neuron CS z-scores across baseline conditions
        pooled_baselines = [pooled[b] for b in args.baseline_conditions]
        z_by_layer = _cs_z_scores(pooled[args.focus_condition], pooled_baselines)

        accumulator.update(z_by_layer)
        n_processed += 1

        if bool(args.gpu_friendly) and device.type == "cuda" and idx % 64 == 0:
            torch.cuda.empty_cache()  # Build ranked results table
    # ------------------------------------------------------------------
    results_df = accumulator.to_dataframe()

    # Optional: drop neurons that never reach the min qualify fraction
    if args.min_qualify_frac > 0.0 and not results_df.empty:
        results_df = results_df[
            results_df["qualify_frac"] >= float(args.min_qualify_frac)
        ].copy()

    # Rank: primary = qualify_count (desc), secondary = mean_z_score (desc)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["qualify_count", "mean_z_score"], ascending=[False, False]
        ).reset_index(drop=True)
        results_df.insert(0, "rank", results_df.index + 1)

    results_df.to_csv(
        tables_dir / "neuron_rankings.csv.gz", index=False, compression="gzip"
    )

    # ------------------------------------------------------------------
    # Per-layer top-k tables + heatmaps
    # ------------------------------------------------------------------
    if not results_df.empty:
        for layer, layer_df in results_df.groupby("layer"):
            top = layer_df.head(int(args.top_k))
            top.to_csv(
                tables_dir / f"top_neurons_layer_{int(layer):03d}.csv", index=False
            )

        # Heatmap: qualify_frac across (layer × neuron)
        write_neuron_heatmap(
            results_df.rename(columns={"qualify_frac": "value"}),
            value_col="value",
            title="CS-specific neuron qualify_frac (fraction of samples)",
            out_path=figures_dir / "qualify_frac_heatmap.html",
        )

        # Heatmap: mean z-score across (layer × neuron)
        write_neuron_heatmap(
            results_df.rename(columns={"mean_z_score": "value"}),
            value_col="value",
            title="CS-specific neuron mean z-score",
            out_path=figures_dir / "mean_zscore_heatmap.html",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_qualified_any = int(
        (results_df["qualify_count"] > 0).sum() if not results_df.empty else 0
    )
    summary = {
        "experiment": "exp4_switch_activation_rewrite",
        "dataset_csv": str(args.dataset_csv),
        "model_name": str(args.model_name),
        "focus_condition": str(args.focus_condition),
        "baseline_conditions": list(args.baseline_conditions),
        "z_threshold": float(args.z_threshold),
        "min_qualify_frac": float(args.min_qualify_frac),
        "top_k": int(args.top_k),
        "gpu_friendly": bool(args.gpu_friendly),
        "n_source_groups_total": int(len(source_items)),
        "n_source_groups_processed": int(n_processed),
        "n_source_groups_skipped": int(n_skipped),
        "n_neurons_qualified_any": n_qualified_any,
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
