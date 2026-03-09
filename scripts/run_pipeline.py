#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running as `python scripts/run_pipeline.py` without setting PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mechinterp_pipeline.analysis import (
    aggregate_attention_metrics,
    aggregate_layer_metrics,
    aggregate_neuron_metrics,
    aggregate_neuron_tendency,
    build_neuron_event_table,
    compare_conditions,
)
from mechinterp_pipeline.io_utils import ensure_dirs, load_config, load_dataset
from mechinterp_pipeline.modeling import analyze_text, prepare_model_and_lens
from mechinterp_pipeline.visualize import (
    plot_domain_metric_heatmap,
    plot_domain_metric_heatmap_absolute,
    plot_attention_heatmap,
    plot_attention_heatmap_absolute,
    plot_layer_metrics,
    plot_layer_metrics_absolute,
    plot_neuron_deltas,
    plot_neuron_absolute,
    plot_neuron_layer_3d_absolute,
    plot_neuron_layer_3d_delta,
    plot_neuron_layer_heatmap_absolute,
    plot_neuron_layer_heatmap_delta,
    render_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare LLM internals for code-switched vs language-confused text using Tuned Lens"
    )
    p.add_argument("--config", required=True, help="Path to YAML config")
    return p.parse_args()


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _write_important_numbers_text(
    out_path: Path,
    summary_df,
    pairwise_df,
    neuron_tendency_df,
    reference_condition: str,
) -> None:
    lines = []
    lines.append("IMPORTANT NUMBERS")
    lines.append(f"Reference condition: {reference_condition}")
    lines.append("")
    lines.append("Global metric deltas vs reference:")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['metric']}: {row['condition']} - {reference_condition} = {row['delta_vs_reference']:.6f}"
        )
    lines.append("")
    lines.append("Top pairwise metric deltas (absolute):")
    pair = pairwise_df.copy()
    pair["abs_delta"] = pair["delta_b_minus_a"].abs()
    top_pair = pair.sort_values("abs_delta", ascending=False).head(30)
    for _, row in top_pair.iterrows():
        lines.append(
            f"- {row['metric']}: {row['condition_b']} - {row['condition_a']} = {row['delta_b_minus_a']:.6f}"
        )
    lines.append("")
    lines.append("Top neuron tendencies by event_count (first 100 rows):")
    top_neurons = neuron_tendency_df.sort_values(
        ["event_count", "activation_mean"], ascending=[False, False]
    ).head(100)
    for _, row in top_neurons.iterrows():
        lines.append(
            f"- cond={row['condition']} domain={row['domain']} layer={int(row['layer'])} "
            f"neuron={int(row['neuron'])} events={int(row['event_count'])} "
            f"act_mean={row['activation_mean']:.6f} avg_rank={row['avg_rank_in_sample_layer']:.3f}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _compress_full_neuron_layers(
    layers: dict[int, list[float]] | dict[str, list[float]],
    round_decimals: int,
    keep_layers: list[int] | None,
    min_layer_exclusive: int | None,
    topk_per_layer: int,
) -> dict[str, object]:
    keep_set = set(keep_layers) if keep_layers is not None else None
    result: dict[str, object] = {}

    for layer_key, vec in layers.items():
        layer = int(layer_key)
        if keep_set is not None and layer not in keep_set:
            continue
        if min_layer_exclusive is not None and layer <= min_layer_exclusive:
            continue

        if topk_per_layer > 0 and topk_per_layer < len(vec):
            idx_vals = sorted(
                enumerate(vec),
                key=lambda x: abs(float(x[1])),
                reverse=True,
            )[:topk_per_layer]
            idxs = [int(i) for i, _ in idx_vals]
            vals = [round(float(v), round_decimals) for _, v in idx_vals]
            result[str(layer)] = {"indices": idxs, "values": vals}
        else:
            result[str(layer)] = [round(float(v), round_decimals) for v in vec]

    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _setup_logging(cfg.log_level)
    logger = logging.getLogger("run_pipeline")

    logger.info("Pipeline start")
    logger.info("Config loaded: model=%s input=%s output=%s", cfg.model_name, cfg.input_path, cfg.output_dir)

    out_root = cfg.output_dir
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    text_dir = out_root / "text_exports"
    ensure_dirs([out_root, tables_dir, figures_dir, text_dir])
    logger.info("Ensured output directories")

    df = load_dataset(cfg.input_path)
    logger.info(
        "Dataset loaded: rows=%d conditions=%s domains=%s",
        len(df),
        sorted(df["condition"].unique().tolist()),
        sorted(df["domain"].unique().tolist()),
    )

    logger.info("Preparing model and tuned lens")
    prepared = prepare_model_and_lens(
        model_name=cfg.model_name,
        tuned_lens_resource_id=cfg.tuned_lens_resource_id,
        device=cfg.device,
    )
    logger.info("Model ready; lens_used=%s", prepared.lens is not None)

    per_sample = []
    skipped_samples: list[dict[str, str]] = []
    full_activation_path = (
        text_dir / "full_neuron_activations.jsonl.gz"
        if cfg.full_neuron_export_gzip
        else text_dir / "full_neuron_activations.jsonl"
    )
    full_activation_fh = None
    if cfg.save_full_neuron_activations:
        if cfg.full_neuron_export_gzip:
            full_activation_fh = gzip.open(full_activation_path, "wt", encoding="utf-8")
        else:
            full_activation_fh = full_activation_path.open("w", encoding="utf-8")
        logger.info(
            "Full neuron activation export enabled: %s (reduce_mode=%s gzip=%s round_decimals=%d topk_per_layer=%d stride=%d)",
            full_activation_path,
            cfg.full_neuron_reduce_mode,
            cfg.full_neuron_export_gzip,
            cfg.full_neuron_round_decimals,
            cfg.full_neuron_topk_per_layer,
            cfg.full_neuron_sample_stride,
        )
    logger.info("Starting per-sample analysis")
    try:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Analyzing samples"), start=1):
            if i == 1 or i % max(1, cfg.log_every_n_samples) == 0 or i == len(df):
                logger.info(
                    "Sample %d/%d id=%s condition=%s domain=%s",
                    i,
                    len(df),
                    row["id"],
                    row["condition"],
                    row["domain"],
                )
            try:
                rec = analyze_text(
                    prepared=prepared,
                    sample_id=str(row["id"]),
                    text=str(row["text"]),
                    condition=str(row["condition"]),
                    domain=str(row["domain"]),
                    max_length=cfg.max_length,
                    topk_neurons=cfg.topk_neurons,
                    collect_full_neuron_activations=cfg.save_full_neuron_activations,
                    full_neuron_reduce_mode=cfg.full_neuron_reduce_mode,
                )
                if (
                    full_activation_fh is not None
                    and "full_neuron_activations" in rec
                    and i % cfg.full_neuron_sample_stride == 0
                ):
                    full_payload = {
                        "id": rec["id"],
                        "condition": rec["condition"],
                        "domain": rec["domain"],
                        "n_tokens": rec["n_tokens"],
                        "reduce_mode": rec.get("full_neuron_reduce_mode", cfg.full_neuron_reduce_mode),
                        "layers": _compress_full_neuron_layers(
                            rec["full_neuron_activations"],
                            round_decimals=cfg.full_neuron_round_decimals,
                            keep_layers=cfg.full_neuron_layers,
                            min_layer_exclusive=cfg.full_neuron_min_layer_exclusive,
                            topk_per_layer=cfg.full_neuron_topk_per_layer,
                        ),
                    }
                    full_activation_fh.write(json.dumps(full_payload) + "\n")
                rec.pop("full_neuron_activations", None)
                rec.pop("full_neuron_reduce_mode", None)
                per_sample.append(rec)
            except ValueError as e:
                # Keep long jobs running when a sample is invalid for trajectory stats.
                if "too short after tokenization" in str(e):
                    skipped_samples.append(
                        {
                            "id": str(row["id"]),
                            "condition": str(row["condition"]),
                            "domain": str(row["domain"]),
                            "reason": str(e),
                        }
                    )
                    logger.warning("Skipping sample id=%s: %s", row["id"], e)
                    continue
                raise
    finally:
        if full_activation_fh is not None:
            full_activation_fh.close()
            logger.info("Wrote full neuron activations: %s", full_activation_path)
    logger.info("Per-sample analysis complete")
    logger.info("Samples kept=%d skipped=%d", len(per_sample), len(skipped_samples))

    if not per_sample:
        raise RuntimeError("No valid samples remained after filtering/skips")

    logger.info("Aggregating metrics")
    layer_df = aggregate_layer_metrics(per_sample)
    attn_df = aggregate_attention_metrics(per_sample)
    neuron_df = aggregate_neuron_metrics(per_sample)
    neuron_events_df = build_neuron_event_table(per_sample)
    neuron_tendency_df = aggregate_neuron_tendency(neuron_events_df)
    logger.info(
        "Aggregate sizes: layer=%d attention=%d neuron=%d",
        len(layer_df),
        len(attn_df),
        len(neuron_df),
    )
    logger.info(
        "Neuron event exports: events=%d tendency_rows=%d",
        len(neuron_events_df),
        len(neuron_tendency_df),
    )

    logger.info("Computing condition comparisons")
    comparisons = compare_conditions(
        layer_df,
        attn_df,
        neuron_df,
        reference_condition=cfg.reference_condition,
    )
    ref = comparisons.reference_condition
    non_ref_conditions = [c for c in comparisons.conditions if c != ref]
    logger.info("Comparison done: reference=%s others=%s", ref, non_ref_conditions)

    logger.info("Writing tabular outputs")
    layer_df.to_csv(tables_dir / "layer_metrics_raw.csv", index=False)
    attn_df.to_csv(tables_dir / "attention_entropy_raw.csv", index=False)
    neuron_df.to_csv(tables_dir / "neuron_proxy_raw.csv", index=False)

    comparisons.layer_metrics.to_csv(tables_dir / "layer_metrics_diff.csv", index=False)
    comparisons.attention_diff.to_csv(tables_dir / "attention_diff.csv", index=False)
    comparisons.neuron_diff.to_csv(tables_dir / "neuron_diff.csv", index=False)
    comparisons.summary.to_csv(tables_dir / "summary.csv", index=False)
    comparisons.pairwise_summary.to_csv(tables_dir / "pairwise_summary.csv", index=False)
    neuron_events_df.to_csv(tables_dir / "neuron_events.csv", index=False)
    neuron_tendency_df.to_csv(tables_dir / "neuron_tendency.csv", index=False)

    neuron_events_df.to_json(text_dir / "neuron_events.jsonl", orient="records", lines=True)
    neuron_tendency_df.to_json(text_dir / "neuron_tendency.jsonl", orient="records", lines=True)

    (out_root / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": cfg.model_name,
                "conditions": comparisons.conditions,
                "reference_condition": ref,
                "n_samples": len(df),
                "n_samples_analyzed": len(per_sample),
                "n_samples_skipped": len(skipped_samples),
                "lens_used": prepared.lens is not None,
                "full_neuron_activations_saved": bool(cfg.save_full_neuron_activations),
                "full_neuron_activation_reduce_mode": cfg.full_neuron_reduce_mode,
                "full_neuron_export_gzip": bool(cfg.full_neuron_export_gzip),
                "full_neuron_round_decimals": int(cfg.full_neuron_round_decimals),
                "full_neuron_layers": cfg.full_neuron_layers,
                "full_neuron_min_layer_exclusive": cfg.full_neuron_min_layer_exclusive,
                "full_neuron_topk_per_layer": int(cfg.full_neuron_topk_per_layer),
                "full_neuron_sample_stride": int(cfg.full_neuron_sample_stride),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote tables and metadata")

    if skipped_samples:
        import csv

        skipped_path = text_dir / "skipped_samples.csv"
        with skipped_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "condition", "domain", "reason"])
            writer.writeheader()
            writer.writerows(skipped_samples)
        logger.warning("Wrote skipped sample report: %s", skipped_path)

    _write_important_numbers_text(
        text_dir / "IMPORTANT_NUMBERS.txt",
        comparisons.summary,
        comparisons.pairwise_summary,
        neuron_tendency_df,
        ref,
    )
    logger.info("Wrote text exports")

    logger.info("Rendering figures")
    plot_layer_metrics(comparisons.layer_metrics, figures_dir / "layer_metric_deltas.html", ref)

    for condition in comparisons.conditions:
        logger.info("Rendering absolute figures for condition=%s", condition)
        plot_layer_metrics_absolute(
            layer_df,
            figures_dir / f"layer_metric_absolute_{condition}.html",
            condition,
        )
        plot_neuron_absolute(
            neuron_df,
            figures_dir / f"neuron_top100_absolute_{condition}.html",
            condition,
        )
        plot_neuron_layer_heatmap_absolute(
            neuron_df,
            figures_dir / f"neuron_layer_heatmap_absolute_{condition}.html",
            condition,
        )
        plot_neuron_layer_3d_absolute(
            neuron_df,
            figures_dir / f"neuron_layer_3d_absolute_{condition}.html",
            condition,
        )
        plot_domain_metric_heatmap_absolute(
            layer_df,
            figures_dir / f"domain_metric_absolute_{condition}.html",
            condition,
        )
        if not attn_df.empty:
            plot_attention_heatmap_absolute(
                attn_df,
                figures_dir / f"attention_entropy_absolute_{condition}.html",
                condition,
            )
        else:
            logger.warning("Skipping absolute attention figure for %s (no attention data)", condition)

    for condition in non_ref_conditions:
        logger.info("Rendering delta figures for condition=%s vs ref=%s", condition, ref)
        if not comparisons.attention_diff.empty:
            plot_attention_heatmap(
                comparisons.attention_diff,
                figures_dir / f"attention_entropy_heatmap_{condition}_vs_{ref}.html",
                ref,
                condition,
            )
        else:
            logger.warning("Skipping delta attention figure for %s (no attention diff data)", condition)
        plot_neuron_deltas(
            comparisons.neuron_diff,
            figures_dir / f"neuron_shift_top100_{condition}_vs_{ref}.html",
            ref,
            condition,
        )
        plot_neuron_layer_heatmap_delta(
            comparisons.neuron_diff,
            figures_dir / f"neuron_layer_heatmap_{condition}_vs_{ref}.html",
            ref,
            condition,
        )
        plot_neuron_layer_3d_delta(
            comparisons.neuron_diff,
            figures_dir / f"neuron_layer_3d_{condition}_vs_{ref}.html",
            ref,
            condition,
        )
        plot_domain_metric_heatmap(
            layer_df,
            figures_dir / f"domain_metric_heatmap_{condition}_vs_{ref}.html",
            ref,
            condition,
        )

    render_summary_markdown(comparisons.summary, comparisons.pairwise_summary, out_root / "SUMMARY.md", ref)
    logger.info("Summary written")
    logger.info("Pipeline finished successfully")

    print(f"Done. Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
