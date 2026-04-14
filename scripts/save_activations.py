#!/usr/bin/env python3
"""
save_activations.py
-------------------
Runs a TransformerLens model over every sample in a dataset and saves the
MLP post-activations for each neuron, broken down by domain and condition
(english / target_language / code_switched).

Outputs (written to --out_dir):
  activations.npz   – compressed numpy archive; load with np.load()
  metadata.csv      – one row per sample, maps array index → domain/condition

Loading the results
-------------------
    import numpy as np, pandas as pd

    data  = np.load("activations.npz", allow_pickle=True)
    acts  = data["activations"]   # shape: (n_samples, n_layers, n_neurons)
    meta  = pd.read_csv("metadata.csv")

    # e.g. mean activation per layer for code-switched French samples
    mask  = (meta["condition"] == "code_switched") & (meta["domain"] == "news")
    cs_acts = acts[mask.values]   # (n_matching, n_layers, n_neurons)
    print(cs_acts.mean(axis=0))   # (n_layers, n_neurons)
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

# ---------------------------------------------------------------------------
# Allow running from any working directory by adding the experiments root
# to sys.path so the shared `common` package can be imported.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import (  # noqa: E402
    configure_gpu_runtime,
    encode_text,
    extract_post_activations,
    load_dataset,
    load_tl_model,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save per-neuron MLP activations for every sample, domain, and condition."
    )
    p.add_argument(
        "--dataset_csv",
        required=True,
        help="Path to the dataset CSV (e.g. data/french.csv or data/hindi.csv).",
    )
    p.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace / TransformerLens model name, e.g. 'gpt2'.",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory where activations.npz and metadata.csv will be written.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum token length per sample (longer texts are truncated).",
    )
    p.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "max", "first", "last"],
        help=(
            "How to collapse the token dimension into a single vector per layer. "
            "'mean' averages all token positions; 'max' takes the element-wise max; "
            "'first' / 'last' take the first or last token position."
        ),
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["english", "target_language", "code_switched"],
        help="Which condition labels to include (default: all three).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on total rows processed (useful for quick tests).",
    )
    p.add_argument(
        "--gpu_friendly",
        action="store_true",
    )
    return p.parse_args()


def _pool(act: np.ndarray, mode: str) -> np.ndarray:
    """Collapse (seq_len, n_neurons) → (n_neurons,) using the chosen pooling mode."""
    if mode == "mean":
        return act.mean(axis=0)
    if mode == "max":
        return act.max(axis=0)
    if mode == "first":
        return act[0]
    if mode == "last":
        return act[-1]
    raise ValueError(f"Unknown pooling mode: {mode}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset_csv)

    # Filter to requested conditions only.
    df = df[df["condition"].isin(args.conditions)].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for conditions {args.conditions}. "
            f"Available: {df['condition'].unique().tolist()}"
        )

    if args.max_samples is not None:
        df = df.head(int(args.max_samples))

    device = resolve_device(args.device)
    configure_gpu_runtime(device=device, gpu_friendly=bool(args.gpu_friendly))
    model = load_tl_model(
        args.model_name,
        device=device,
        gpu_friendly=bool(args.gpu_friendly),
    )
    tokenizer = model.tokenizer
    n_layers = int(model.cfg.n_layers)

    # We do a first pass over the first sample to learn n_neurons, then pre-allocate.
    # (n_neurons is the MLP hidden dimension, e.g. 3072 for gpt2.)
    n_neurons: int | None = None
    activation_rows: list[np.ndarray] = []  # each: (n_layers, n_neurons)
    meta_rows: list[dict] = []

    for row_idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Saving activations")):
        input_ids = encode_text(
            tokenizer=tokenizer,
            text=row.text,
            max_length=int(args.max_length),
            device=device,
        )
        ids = input_ids[0].detach().cpu().tolist()
        if len(ids) < 1:
            continue

        layer_acts = extract_post_activations(model, input_ids)
        # layer_acts: {layer_idx: np.ndarray of shape (seq_len, n_neurons)}

        if not layer_acts:
            continue

        # Pool each layer down to a single vector.
        pooled_layers: list[np.ndarray] = []
        for layer_idx in range(n_layers):
            if layer_idx not in layer_acts:
                # Should not happen with a well-formed model, but handle gracefully.
                if n_neurons is not None:
                    pooled_layers.append(np.zeros(n_neurons, dtype=np.float32))
                continue
            vec = _pool(layer_acts[layer_idx], args.pooling).astype(np.float32)
            if n_neurons is None:
                n_neurons = len(vec)
            pooled_layers.append(vec)

        if len(pooled_layers) != n_layers:
            continue  # skip malformed samples

        # Stack into (n_layers, n_neurons).
        sample_array = np.stack(pooled_layers, axis=0)  # (n_layers, n_neurons)
        activation_rows.append(sample_array)

        meta_rows.append(
            {
                "row_index": len(activation_rows) - 1,
                "id": str(row.id),
                "source_id": str(row.source_id),
                "domain": str(row.domain),
                "condition": str(row.condition),
            }
        )

        if bool(args.gpu_friendly) and device.type == "cuda" and row_idx % 64 == 0:
            torch.cuda.empty_cache()

    if not activation_rows:
        raise RuntimeError("No activations were collected. Check your dataset and model.")

    # Stack all samples into a single 3-D array: (n_samples, n_layers, n_neurons).
    activations = np.stack(activation_rows, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)

    print(f"\nActivation array shape: {activations.shape}  (samples × layers × neurons)")
    print(f"Conditions: {sorted(meta_df['condition'].unique())}")
    print(f"Domains:    {sorted(meta_df['domain'].unique())}")

    # ------------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------------
    npz_path = out_dir / "activations.npz"
    np.savez_compressed(
        npz_path,
        activations=activations,
        # Include the key metadata arrays directly in the npz so the file is
        # self-contained even without the companion CSV.
        source_ids=meta_df["source_id"].to_numpy(dtype=str),
        domains=meta_df["domain"].to_numpy(dtype=str),
        conditions=meta_df["condition"].to_numpy(dtype=str),
    )
    print(f"\nSaved: {npz_path}")

    csv_path = out_dir / "metadata.csv"
    meta_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Run summary.
    # ------------------------------------------------------------------
    summary = {
        "model_name": str(args.model_name),
        "dataset_csv": str(args.dataset_csv),
        "pooling": str(args.pooling),
        "conditions": list(args.conditions),
        "n_samples": int(len(activation_rows)),
        "n_layers": int(n_layers),
        "n_neurons": int(n_neurons),
        "activations_shape": list(activations.shape),
        "npz_path": str(npz_path),
        "metadata_csv_path": str(csv_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
