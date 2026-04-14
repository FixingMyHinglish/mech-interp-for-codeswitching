#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze Exp4 outputs and produce a report of neurons that are "
            "consistently more active in the code-switched condition than in both "
            "monolingual baselines, across samples."
        )
    )
    p.add_argument(
        "--results_dir",
        required=True,
        help="Exp4 results directory containing summary.json and tables/.",
    )
    p.add_argument(
        "--focus_offset",
        type=int,
        default=0,
        help="Relative token offset to treat as the primary switch point (default: 0).",
    )
    p.add_argument(
        "--min_consistency_fraction",
        type=float,
        default=0.5,
        help="Consistency threshold used when run.py was executed (default: 0.5).",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for analysis artifacts (default: <results_dir>/analysis).",
    )
    return p.parse_args()


def _format_offset(relative_offset: int) -> str:
    return f"{relative_offset:+d}".replace("+", "plus_").replace("-", "minus_")


def _read_run_summary(results_dir: Path) -> dict:
    path = results_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _render_markdown_table(
    df: pd.DataFrame,
    *,
    columns: list[str],
    max_rows: int = 20,
    float_decimals: int = 4,
) -> str:
    if df.empty:
        return "_No rows._"

    use = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()

    def _fmt(v: object) -> str:
        if isinstance(v, (np.floating, float)):
            if np.isnan(float(v)):
                return "nan"
            return f"{float(v):.{float_decimals}f}"
        return str(v)

    headers = list(use.columns)
    rows = [[_fmt(v) for v in row] for row in use.itertuples(index=False, name=None)]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(out)


def _layer_breakdown(
    consistent_df: pd.DataFrame,
    all_neurons_df: pd.DataFrame,
    *,
    focus_offset: int,
) -> pd.DataFrame:
    """Per-layer counts of consistent neurons at the focus offset."""
    if consistent_df.empty:
        return pd.DataFrame()

    sub_c = consistent_df[consistent_df["relative_offset"] == focus_offset].copy()
    sub_a = all_neurons_df[all_neurons_df["relative_offset"] == focus_offset].copy()
    if sub_c.empty or sub_a.empty:
        return pd.DataFrame()

    total_per_layer = (
        sub_a.groupby("layer")["neuron"].count().rename("n_neurons_total")
    )
    consistent_per_layer = (
        sub_c.groupby("layer")
        .agg(
            n_consistent=("neuron", "count"),
            max_consistency=("consistency_fraction", "max"),
            mean_consistency=("consistency_fraction", "mean"),
            max_n_samples=("n_samples_switch_specific", "max"),
        )
    )
    out = consistent_per_layer.join(total_per_layer, how="left").reset_index()
    out["pct_consistent"] = 100.0 * out["n_consistent"] / out["n_neurons_total"].clip(lower=1)
    return out.sort_values(["n_consistent", "max_consistency"], ascending=[False, False])


def _consistency_distribution(
    all_neurons_df: pd.DataFrame,
    *,
    focus_offset: int,
    bins: list[float] | None = None,
) -> pd.DataFrame:
    """Histogram of consistency_fraction values across all neurons at the focus offset."""
    sub = all_neurons_df[all_neurons_df["relative_offset"] == focus_offset]
    if sub.empty:
        return pd.DataFrame()

    if bins is None:
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    counts, edges = np.histogram(sub["consistency_fraction"].to_numpy(), bins=bins)
    rows = []
    for i, cnt in enumerate(counts):
        rows.append(
            {
                "bin_lo": round(edges[i], 2),
                "bin_hi": round(edges[i + 1], 2),
                "n_neurons": int(cnt),
                "pct": round(100.0 * cnt / max(len(sub), 1), 2),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    *,
    run_summary: dict,
    offset_summary_df: pd.DataFrame,
    consistent_df: pd.DataFrame,
    all_neurons_df: pd.DataFrame,
    focus_offset: int,
    min_consistency_fraction: float,
) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append("# Exp4 Analysis Report: Consistently Switch-Specific Neurons")
    lines.append("")
    lines.append(f"- Generated: {now_utc}")
    lines.append(f"- Focus offset: {focus_offset:+d}")
    lines.append(f"- Consistency threshold: >= {min_consistency_fraction}")
    if run_summary:
        lines.append(f"- Dataset: `{run_summary.get('dataset_csv', '')}`")
        lines.append(f"- Model: `{run_summary.get('model_name', '')}`")
        lines.append(f"- Focus condition: `{run_summary.get('focus_condition', '')}`")
        lines.append(f"- Baseline conditions: `{run_summary.get('baseline_conditions', '')}`")
        lines.append(f"- Samples scanned: {run_summary.get('n_source_groups_scanned', '')}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 1: Summary across all offsets
    # ------------------------------------------------------------------
    lines.append("## Section 1: Per-Offset Summary")
    lines.append("")
    lines.append(
        "For each token offset, how many (layer, neuron) pairs are consistently "
        "switch-specific (i.e. more active in the code-switched version than in every "
        "monolingual baseline, in at least the required fraction of samples)."
    )
    lines.append("")
    lines.append(
        _render_markdown_table(
            offset_summary_df,
            columns=[
                "relative_offset",
                "n_samples_total",
                "n_neuron_layer_pairs_total",
                "n_consistent_neuron_layer_pairs",
                "pct_consistent",
                "max_consistency_fraction",
                "mean_consistency_fraction",
            ],
            max_rows=20,
        )
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Section 2: Distribution of consistency fractions at focus offset
    # ------------------------------------------------------------------
    lines.append(f"## Section 2: Consistency Distribution at Offset {focus_offset:+d}")
    lines.append("")
    lines.append(
        "Distribution of consistency fractions across all (layer, neuron) pairs "
        f"at offset {focus_offset:+d}, showing how many neurons achieve each level "
        "of consistency."
    )
    lines.append("")
    dist_df = _consistency_distribution(all_neurons_df, focus_offset=focus_offset)
    lines.append(
        _render_markdown_table(
            dist_df,
            columns=["bin_lo", "bin_hi", "n_neurons", "pct"],
            max_rows=15,
        )
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Section 3: Layer breakdown of consistent neurons at focus offset
    # ------------------------------------------------------------------
    lines.append(f"## Section 3: Layer Breakdown at Offset {focus_offset:+d}")
    lines.append("")
    lines.append(
        f"Per-layer counts of consistently switch-specific neurons at offset {focus_offset:+d}, "
        f"sorted by number of consistent neurons descending."
    )
    lines.append("")
    layer_df = _layer_breakdown(
        consistent_df, all_neurons_df, focus_offset=focus_offset
    )
    lines.append(
        _render_markdown_table(
            layer_df,
            columns=[
                "layer",
                "n_consistent",
                "n_neurons_total",
                "pct_consistent",
                "max_consistency",
                "mean_consistency",
                "max_n_samples",
            ],
            max_rows=30,
        )
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Section 4: All consistent neurons at focus offset (no top-k cap)
    # ------------------------------------------------------------------
    lines.append(f"## Section 4: All Consistent Neurons at Offset {focus_offset:+d}")
    lines.append("")
    lines.append(
        f"Every (layer, neuron) pair with consistency_fraction >= {min_consistency_fraction} "
        f"at offset {focus_offset:+d}, sorted by consistency descending. "
        "No top-k filtering is applied."
    )
    lines.append("")

    focus_consistent = pd.DataFrame()
    if not consistent_df.empty:
        focus_consistent = (
            consistent_df[consistent_df["relative_offset"] == focus_offset]
            .sort_values(
                ["consistency_fraction", "n_samples_switch_specific"],
                ascending=[False, False],
            )
            .copy()
        )
        focus_consistent.insert(
            0, "rank", np.arange(1, len(focus_consistent) + 1, dtype=int)
        )

    lines.append(
        _render_markdown_table(
            focus_consistent,
            columns=[
                "rank",
                "layer",
                "neuron",
                "n_samples_switch_specific",
                "n_samples_total",
                "consistency_fraction",
                # Unconditional means (all samples)
                "mean_cs_activation",
                "mean_eng_activation",
                "mean_tgt_activation",
                # Conditional means when the neuron WAS switch-specific
                # (benchmark for CS-specific cluster characterisation)
                "mean_cs_act_switch_on",
                "mean_eng_act_switch_on",
                "mean_tgt_act_switch_on",
                # Conditional means when the neuron was NOT switch-specific
                # (baseline / control)
                "mean_cs_act_switch_off",
                "mean_eng_act_switch_off",
                "mean_tgt_act_switch_off",
            ],
            max_rows=200,
        )
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load run outputs.
    counts_path = tables_dir / "switch_neuron_counts.csv.gz"
    if not counts_path.exists():
        raise FileNotFoundError(
            f"Missing required file: {counts_path}\n"
            "Run run.py first to generate experiment outputs."
        )

    run_summary = _read_run_summary(results_dir)
    all_neurons_df = pd.read_csv(counts_path)
    consistent_df = pd.DataFrame()
    consistent_path = tables_dir / "consistent_switch_neurons.csv"
    if consistent_path.exists():
        consistent_df = pd.read_csv(consistent_path)

    offset_summary_path = tables_dir / "switch_neuron_summary.csv"
    offset_summary_df = pd.DataFrame()
    if offset_summary_path.exists():
        offset_summary_df = pd.read_csv(offset_summary_path)

    # Derived tables.
    focus_offset = int(args.focus_offset)
    min_cf = float(args.min_consistency_fraction)

    layer_df = _layer_breakdown(consistent_df, all_neurons_df, focus_offset=focus_offset)
    dist_df = _consistency_distribution(all_neurons_df, focus_offset=focus_offset)

    # Save derived tables.
    offset_tag = _format_offset(focus_offset)
    layer_df.to_csv(out_dir / f"layer_breakdown_offset_{offset_tag}.csv", index=False)
    dist_df.to_csv(out_dir / f"consistency_distribution_offset_{offset_tag}.csv", index=False)

    if not consistent_df.empty:
        focus_consistent = (
            consistent_df[consistent_df["relative_offset"] == focus_offset]
            .sort_values(
                ["consistency_fraction", "n_samples_switch_specific"],
                ascending=[False, False],
            )
        )
        focus_consistent.to_csv(
            out_dir / f"all_consistent_neurons_offset_{offset_tag}.csv", index=False
        )

    # Generate and save the report.
    report = build_report(
        run_summary=run_summary,
        offset_summary_df=offset_summary_df,
        consistent_df=consistent_df,
        all_neurons_df=all_neurons_df,
        focus_offset=focus_offset,
        min_consistency_fraction=min_cf,
    )
    report_path = out_dir / f"exp4_analysis_report_offset_{offset_tag}.md"
    report_path.write_text(report, encoding="utf-8")

    analysis_summary = {
        "results_dir": str(results_dir),
        "analysis_dir": str(out_dir),
        "focus_offset": focus_offset,
        "min_consistency_fraction": min_cf,
        "n_rows_all_neurons": int(len(all_neurons_df)),
        "n_rows_consistent_neurons": int(len(consistent_df)),
        "n_consistent_at_focus_offset": int(
            len(consistent_df[consistent_df["relative_offset"] == focus_offset])
            if not consistent_df.empty
            else 0
        ),
        "report_path": str(report_path),
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis_summary, indent=2))


if __name__ == "__main__":
    main()
