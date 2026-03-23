#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.express as px


REQUIRED_FILES = [
    "tables/summary.csv",
    "tables/layer_metrics_diff.csv",
    "tables/neuron_tendency.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare language/run activation outputs from existing pipeline artifacts",
    )
    p.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Optional run specs label=/abs/or/rel/path. If omitted, auto-discovers valid outputs/* runs.",
    )
    p.add_argument("--outputs_root", default="outputs", help="Root folder for auto-discovery")
    p.add_argument("--output_dir", default="outputs/language_comparison", help="Where to write comparison results")
    p.add_argument("--topk_neurons", type=int, default=200, help="Top-K neurons per condition for overlap metrics")
    return p.parse_args()


def _has_required(run_dir: Path) -> bool:
    return all((run_dir / rel).exists() for rel in REQUIRED_FILES)


def _parse_runs(args: argparse.Namespace) -> dict[str, Path]:
    if args.runs:
        runs = {}
        for spec in args.runs:
            if "=" not in spec:
                raise ValueError(f"Invalid --runs spec '{spec}'. Use label=path")
            label, p = spec.split("=", 1)
            run_dir = Path(p)
            if not _has_required(run_dir):
                raise FileNotFoundError(f"Run {label} missing required files in {run_dir}")
            runs[label] = run_dir
        return runs

    root = Path(args.outputs_root)
    if not root.exists():
        raise FileNotFoundError(f"outputs_root does not exist: {root}")

    runs = {}
    for d in sorted([x for x in root.iterdir() if x.is_dir()]):
        if _has_required(d):
            runs[d.name] = d

    if len(runs) < 2:
        raise ValueError("Need at least 2 completed runs to compare")
    return runs


def load_tables(runs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, layer_rows, neuron_rows = [], [], []

    for label, run_dir in runs.items():
        s = pd.read_csv(run_dir / "tables" / "summary.csv")
        s["run"] = label
        summary_rows.append(s)

        l = pd.read_csv(run_dir / "tables" / "layer_metrics_diff.csv")
        l["run"] = label
        layer_rows.append(l)

        n = pd.read_csv(run_dir / "tables" / "neuron_tendency.csv")
        n["run"] = label
        neuron_rows.append(n)

    return pd.concat(summary_rows, ignore_index=True), pd.concat(layer_rows, ignore_index=True), pd.concat(neuron_rows, ignore_index=True)


def compute_summary_differences(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, condition), g in summary_df.groupby(["metric", "condition"]):
        for (run_a, va), (run_b, vb) in combinations(g[["run", "delta_vs_reference"]].itertuples(index=False, name=None), 2):
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "run_a": run_a,
                    "run_b": run_b,
                    "delta_vs_ref_a": float(va),
                    "delta_vs_ref_b": float(vb),
                    "difference_a_minus_b": float(va - vb),
                }
            )
    return pd.DataFrame(rows)


def compute_neuron_overlap(neuron_df: pd.DataFrame, topk: int) -> pd.DataFrame:
    rows = []

    for condition, g_cond in neuron_df.groupby("condition"):
        top_map: dict[str, set[tuple[int, int]]] = {}
        for run, g_run in g_cond.groupby("run"):
            top = g_run.sort_values("activation_mean", ascending=False).head(topk)
            top_map[run] = set((int(r.layer), int(r.neuron)) for r in top.itertuples())

        for run_a, run_b in combinations(sorted(top_map), 2):
            a = top_map[run_a]
            b = top_map[run_b]
            inter = len(a.intersection(b))
            union = len(a.union(b))
            jacc = inter / union if union else 0.0
            rows.append(
                {
                    "condition": condition,
                    "run_a": run_a,
                    "run_b": run_b,
                    "topk": topk,
                    "intersection": inter,
                    "union": union,
                    "jaccard": jacc,
                }
            )

    return pd.DataFrame(rows)


def build_visuals(summary_df: pd.DataFrame, layer_df: pd.DataFrame, overlap_df: pd.DataFrame, figures_dir: Path) -> None:
    summary_df = summary_df.copy()
    summary_df["metric_condition"] = summary_df["metric"] + " | " + summary_df["condition"]
    piv = summary_df.pivot(index="metric_condition", columns="run", values="delta_vs_reference").fillna(0)
    heat = px.imshow(
        piv,
        aspect="auto",
        color_continuous_scale="RdBu",
        origin="lower",
        title="Delta vs Reference by Run (Metric | Condition)",
    )
    heat.write_html(str(figures_dir / "language_summary_heatmap.html"), include_plotlyjs="cdn")

    line = px.line(
        layer_df,
        x="layer",
        y="delta_vs_reference",
        color="run",
        facet_row="metric",
        facet_col="condition",
        title="Layer-wise Delta vs Reference Across Runs",
        height=2600,
    )
    line.write_html(str(figures_dir / "language_layer_comparison.html"), include_plotlyjs="cdn")

    if not overlap_df.empty:
        ov = overlap_df.copy()
        ov["run_pair"] = ov["run_a"] + " vs " + ov["run_b"]
        bar = px.bar(
            ov,
            x="condition",
            y="jaccard",
            color="run_pair",
            barmode="group",
            title="Top-K Neuron Overlap (Jaccard) Across Runs",
        )
        bar.write_html(str(figures_dir / "language_neuron_overlap.html"), include_plotlyjs="cdn")


def write_summary_markdown(
    out_path: Path,
    runs: dict[str, Path],
    summary_diff_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
) -> None:
    lines = ["# Language Run Comparison", "", "## Runs"]
    for label, p in runs.items():
        lines.append(f"- {label}: {p}")

    lines.append("")
    lines.append("## Largest Metric Differences")
    if summary_diff_df.empty:
        lines.append("- No differences computed")
    else:
        top = summary_diff_df.reindex(summary_diff_df["difference_a_minus_b"].abs().sort_values(ascending=False).index).head(20)
        for _, r in top.iterrows():
            lines.append(
                f"- {r['metric']} | {r['condition']} | {r['run_a']} - {r['run_b']} = {r['difference_a_minus_b']:.6f}"
            )

    lines.append("")
    lines.append("## Neuron Overlap")
    if overlap_df.empty:
        lines.append("- No overlap rows computed")
    else:
        for _, r in overlap_df.sort_values("jaccard", ascending=False).iterrows():
            lines.append(
                f"- {r['condition']} | {r['run_a']} vs {r['run_b']} | jaccard={r['jaccard']:.4f} ({int(r['intersection'])}/{int(r['union'])})"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = _parse_runs(args)

    out_root = Path(args.output_dir)
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df, layer_df, neuron_df = load_tables(runs)
    summary_diff_df = compute_summary_differences(summary_df)
    overlap_df = compute_neuron_overlap(neuron_df, topk=args.topk_neurons)

    summary_df.to_csv(tables_dir / "summary_merged.csv", index=False)
    layer_df.to_csv(tables_dir / "layer_metrics_diff_merged.csv", index=False)
    neuron_df.to_csv(tables_dir / "neuron_tendency_merged.csv", index=False)
    summary_diff_df.to_csv(tables_dir / "summary_diff_between_runs.csv", index=False)
    overlap_df.to_csv(tables_dir / "neuron_overlap_topk.csv", index=False)

    build_visuals(summary_df, layer_df, overlap_df, figures_dir)
    write_summary_markdown(out_root / "LANGUAGE_COMPARISON_SUMMARY.md", runs, summary_diff_df, overlap_df)

    meta = {
        "runs": {k: str(v) for k, v in runs.items()},
        "topk_neurons": args.topk_neurons,
        "rows": {
            "summary": len(summary_df),
            "layer": len(layer_df),
            "neuron": len(neuron_df),
            "summary_diffs": len(summary_diff_df),
            "overlap": len(overlap_df),
        },
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Comparison written to: {out_root}")


if __name__ == "__main__":
    main()
