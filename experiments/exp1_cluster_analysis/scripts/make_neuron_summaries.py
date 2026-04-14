#!/usr/bin/env python3
"""Build <work_dir>/03_cleaned/layer_XX_neuron_summary.csv from <work_dir>/02_bundle/csv/activations_wide.csv."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

COL_RE = re.compile(r"^L(?P<layer>\d+)_n(?P<neuron>\d+)$")


def layer_columns(columns: list[str]) -> dict[int, list[tuple[int, str]]]:
    out: dict[int, list[tuple[int, str]]] = {}
    for c in columns:
        m = COL_RE.match(c)
        if not m:
            continue
        layer = int(m.group("layer"))
        neuron = int(m.group("neuron"))
        out.setdefault(layer, []).append((neuron, c))
    for layer in out:
        out[layer].sort(key=lambda t: t[0])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", type=Path, required=True)
    return p.parse_args()


def _streaming_layer_means(
    wide_csv: Path,
    by_layer: dict[int, list[tuple[int, str]]],
) -> dict[int, tuple[dict[str, dict[str, float]], dict[str, int]]]:
    """
    One pass over activations_wide.csv: accumulate per-condition sums and row counts.
    Returns per layer: (sums_per_cond_col, row_counts_per_cond).
    """
    layers_sorted = sorted(by_layer)
    # sums[layer][condition][colname] = sum
    sums: dict[int, dict[str, defaultdict[str, float]]] = {
        L: defaultdict(lambda: defaultdict(float)) for L in layers_sorted
    }
    row_counts: dict[str, int] = defaultdict(int)

    needed = {"condition"}
    for L in layers_sorted:
        needed.update(c for _, c in by_layer[L])

    with wide_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = needed - set(fieldnames)
        if missing:
            raise SystemExit(f"CSV missing columns: {sorted(missing)[:10]}... ({len(missing)} total)")

        for row in reader:
            cond = row["condition"]
            row_counts[cond] += 1
            for L in layers_sorted:
                for _, col in by_layer[L]:
                    v = row.get(col, "") or "0"
                    try:
                        sums[L][cond][col] += float(v)
                    except ValueError:
                        sums[L][cond][col] += 0.0

    out: dict[int, tuple[dict[str, dict[str, float]], dict[str, int]]] = {}
    for L in layers_sorted:
        out[L] = (sums[L], dict(row_counts))
    return out


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    wide_csv = (work_dir / "02_bundle" / "csv" / "activations_wide.csv").resolve()
    out_dir = (work_dir / "03_cleaned").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(wide_csv, nrows=0).columns.tolist()
    by_layer = layer_columns(header)
    if not by_layer:
        raise SystemExit(f"No Lxx_nyy columns in {wide_csv}")

    layer_stats = _streaming_layer_means(wide_csv, by_layer)

    for layer in sorted(by_layer):
        sums_by_cond, row_counts = layer_stats[layer]
        colnames = [c for _, c in by_layer[layer]]
        neurons = [n for n, _ in by_layer[layer]]

        for need in ("english", "code_switched", "target_language"):
            if need not in sums_by_cond or need not in row_counts:
                raise SystemExit(f"Missing condition {need!r} in {wide_csv}")
            if row_counts[need] == 0:
                raise SystemExit(f"Zero rows for condition {need!r}")

        eng = {c: sums_by_cond["english"][c] / row_counts["english"] for c in colnames}
        cs = {c: sums_by_cond["code_switched"][c] / row_counts["code_switched"] for c in colnames}
        tg = {c: sums_by_cond["target_language"][c] / row_counts["target_language"] for c in colnames}

        out = pd.DataFrame(
            {
                "layer": layer,
                "neuron_id": neurons,
                "mean_english": [eng[c] for c in colnames],
                "mean_code_switched": [cs[c] for c in colnames],
                "mean_target": [tg[c] for c in colnames],
            }
        )
        out["delta_cs"] = out["mean_code_switched"] - out["mean_english"]
        out["delta_target"] = out["mean_target"] - out["mean_english"]
        path = out_dir / f"layer_{layer:02d}_neuron_summary.csv"
        out.to_csv(path, index=False)
        print(f"Wrote {path} ({len(out)} neurons)")


if __name__ == "__main__":
    main()
