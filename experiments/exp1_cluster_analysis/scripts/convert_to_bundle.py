#!/usr/bin/env python3
"""
Convert sparse JSONL(.gz) full-neuron exports into a bundle under <work_dir>/02_bundle/.

Writes:
- 02_bundle/metadata.csv
- 02_bundle/csv/activations_wide.csv
- 02_bundle/summary.json
- 02_bundle/selected_neuron_indices_by_layer.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", type=Path, required=True)
    p.add_argument("--in_path", type=Path, default=None)
    p.add_argument("--drop_condition", default="confused")
    p.add_argument(
        "--topk_per_layer",
        type=int,
        default=256,
        help="How many neuron columns to keep per layer. Set <=0 to keep ALL neuron indices seen in the export.",
    )
    p.add_argument("--write_filtered_jsonl", action="store_true")
    return p.parse_args()


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with _open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _resolve_in_path(work_dir: Path, in_path: Path | None) -> Path:
    if in_path is not None:
        return in_path
    candidates = [
        work_dir / "01_raw" / "full_neuron_activations.jsonl.gz",
        work_dir / "01_raw" / "full_neuron_activations.jsonl",
        work_dir / "full_neuron_activations.jsonl.gz",
        work_dir / "full_neuron_activations.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No raw export found in work_dir (or 01_raw/).")


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    in_path = _resolve_in_path(work_dir, args.in_path).resolve()

    out_dir = (work_dir / "02_bundle").resolve()
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)

    layer_freq: dict[str, Counter[int]] = defaultdict(Counter)
    conds: Counter[str] = Counter()
    n_kept = 0
    example_reduce_mode = None

    for obj in _iter_jsonl(in_path):
        cond = obj.get("condition")
        conds[str(cond)] += 1
        if cond == args.drop_condition:
            continue
        n_kept += 1
        if example_reduce_mode is None:
            example_reduce_mode = obj.get("reduce_mode")
        for layer, payload in obj["layers"].items():
            layer_freq[str(layer)].update(payload["indices"])

    if n_kept == 0:
        raise SystemExit("No records kept after filtering.")

    layers_sorted = sorted(layer_freq.keys(), key=lambda x: int(x))
    topk = int(args.topk_per_layer)
    if topk <= 0:
        selected: dict[str, list[int]] = {
            layer: sorted(layer_freq[layer].keys()) for layer in layers_sorted
        }
    else:
        selected = {layer: [i for i, _ in layer_freq[layer].most_common(topk)] for layer in layers_sorted}

    (out_dir / "selected_neuron_indices_by_layer.json").write_text(
        json.dumps(
            {
                "in_path": str(in_path),
                "drop_condition": args.drop_condition,
                "topk_per_layer": topk,
                "layers": {k: selected[k] for k in layers_sorted},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_path = out_dir / "metadata.csv"
    wide_path = out_dir / "csv" / "activations_wide.csv"
    meta_cols = ["row_index", "id", "source_id", "domain", "condition"]
    feat_cols = [f"L{int(layer):02d}_n{idx}" for layer in layers_sorted for idx in selected[layer]]
    wide_cols = meta_cols + feat_cols

    filtered_fp = None
    filtered_path = work_dir / "01_raw" / f"full_neuron_activations.no_{args.drop_condition}.jsonl"
    if args.write_filtered_jsonl:
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_fp = filtered_path.open("w", encoding="utf-8")

    row_index = 0
    with metadata_path.open("w", encoding="utf-8", newline="") as mf, wide_path.open(
        "w", encoding="utf-8", newline=""
    ) as wf:
        meta_w = csv.DictWriter(mf, fieldnames=meta_cols)
        wide_w = csv.DictWriter(wf, fieldnames=wide_cols)
        meta_w.writeheader()
        wide_w.writeheader()

        for obj in _iter_jsonl(in_path):
            if obj.get("condition") == args.drop_condition:
                continue
            if filtered_fp is not None:
                filtered_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

            meta = {
                "row_index": row_index,
                "id": obj.get("id"),
                "source_id": obj.get("source_id"),
                "domain": obj.get("domain"),
                "condition": obj.get("condition"),
            }
            meta_w.writerow(meta)
            row: dict[str, object] = dict(meta)
            for layer in layers_sorted:
                payload = obj["layers"].get(layer)
                idx_to_val: dict[int, float] = {}
                if payload is not None:
                    idx_to_val = dict(zip(payload["indices"], payload["values"]))
                for idx in selected[layer]:
                    row[f"L{int(layer):02d}_n{idx}"] = float(idx_to_val.get(idx, 0.0))
            wide_w.writerow(row)
            row_index += 1

    if filtered_fp is not None:
        filtered_fp.close()

    n_features = sum(len(selected[l]) for l in layers_sorted)
    summary = {
        "pooling": example_reduce_mode or "unknown",
        "conditions_seen": dict(conds),
        "dropped_condition": args.drop_condition,
        "n_samples_total": int(sum(conds.values())),
        "n_samples_after_drop": int(n_kept),
        "layers_recorded": [int(x) for x in layers_sorted],
        "n_layers": len(layers_sorted),
        "topk_per_layer": int(topk),
        "n_features_exported": int(n_features),
        "metadata_csv_path": "metadata.csv",
        "wide_csv_path": "csv/activations_wide.csv",
        "selected_indices_path": "selected_neuron_indices_by_layer.json",
        "source_path": str(in_path),
        "filtered_jsonl_path": str(filtered_path) if args.write_filtered_jsonl else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Kept {n_kept} rows (dropped {conds[args.drop_condition]} {args.drop_condition!r})")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {out_dir / 'summary.json'}")
    if args.write_filtered_jsonl:
        print(f"Wrote {filtered_path}")


if __name__ == "__main__":
    main()

