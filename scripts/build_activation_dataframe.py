#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flatten saved full neuron activations into dataframe-friendly tables and overlap stats"
    )
    p.add_argument(
        "--activations",
        required=True,
        help="Path to full_neuron_activations.jsonl or .jsonl.gz",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for flattened tables",
    )
    p.add_argument(
        "--min_activation",
        type=float,
        default=0.0,
        help="Optional minimum activation threshold for inclusion in long table",
    )
    p.add_argument(
        "--overlap_topk",
        type=int,
        default=128,
        help="Top-k neurons per sample/layer/condition used for overlap computation",
    )
    return p.parse_args()


def _open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_layer_neurons(layer_payload):
    # Dense format: [v0, v1, ...]
    if isinstance(layer_payload, list):
        for idx, val in enumerate(layer_payload):
            yield idx, float(val)
        return

    # Sparse format: {"indices": [...], "values": [...]}.
    if isinstance(layer_payload, dict) and "indices" in layer_payload and "values" in layer_payload:
        for idx, val in zip(layer_payload["indices"], layer_payload["values"]):
            yield int(idx), float(val)
        return

    # Unknown format: ignore.
    return


def main() -> None:
    args = parse_args()
    in_path = Path(args.activations)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_path = out_dir / "activation_long.csv.gz"
    overlap_path = out_dir / "activation_overlap.csv"
    meta_path = out_dir / "metadata.json"

    # Track per (source_id, layer, condition) for overlap.
    by_group = defaultdict(list)
    n_samples = 0
    n_rows = 0

    with _open_jsonl(in_path) as f_in, gzip.open(long_path, "wt", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "id",
                "source_id",
                "condition",
                "domain",
                "n_tokens",
                "reduce_mode",
                "layer",
                "neuron",
                "activation",
            ],
        )
        writer.writeheader()

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_samples += 1

            sid = rec.get("source_id", rec.get("id", ""))
            cond = str(rec.get("condition", ""))
            domain = str(rec.get("domain", ""))
            rid = str(rec.get("id", ""))
            n_tokens = rec.get("n_tokens", "")
            reduce_mode = rec.get("reduce_mode", "")
            layers = rec.get("layers", {})

            for layer_key, layer_payload in layers.items():
                layer = int(layer_key)
                for neuron, act in _iter_layer_neurons(layer_payload):
                    if abs(act) < args.min_activation:
                        continue

                    writer.writerow(
                        {
                            "id": rid,
                            "source_id": sid,
                            "condition": cond,
                            "domain": domain,
                            "n_tokens": n_tokens,
                            "reduce_mode": reduce_mode,
                            "layer": layer,
                            "neuron": int(neuron),
                            "activation": float(act),
                        }
                    )
                    n_rows += 1
                    by_group[(sid, layer, cond)].append((int(neuron), float(act)))

    # Compute condition overlaps per (source_id, layer).
    overlap_rows = []
    grouped = defaultdict(dict)
    for (sid, layer, cond), pairs in by_group.items():
        # Top-k by absolute activation
        pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[: args.overlap_topk]
        grouped[(sid, layer)][cond] = set(n for n, _ in pairs_sorted)

    for (sid, layer), cond_map in grouped.items():
        conds = sorted(cond_map.keys())
        for a, b in combinations(conds, 2):
            sa = cond_map[a]
            sb = cond_map[b]
            inter = len(sa.intersection(sb))
            union = len(sa.union(sb))
            jacc = inter / union if union else 0.0
            overlap_rows.append(
                {
                    "source_id": sid,
                    "layer": int(layer),
                    "condition_a": a,
                    "condition_b": b,
                    "topk": args.overlap_topk,
                    "intersection": inter,
                    "union": union,
                    "jaccard": jacc,
                }
            )

    with overlap_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "layer",
                "condition_a",
                "condition_b",
                "topk",
                "intersection",
                "union",
                "jaccard",
            ],
        )
        writer.writeheader()
        writer.writerows(overlap_rows)

    meta = {
        "input": str(in_path),
        "output_long_csv_gz": str(long_path),
        "output_overlap_csv": str(overlap_path),
        "samples_processed": n_samples,
        "activation_rows": n_rows,
        "min_activation": args.min_activation,
        "overlap_topk": args.overlap_topk,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote long activation table: {long_path}")
    print(f"Wrote overlap table: {overlap_path}")
    print(f"Samples processed: {n_samples} | activation rows: {n_rows}")


if __name__ == "__main__":
    main()
