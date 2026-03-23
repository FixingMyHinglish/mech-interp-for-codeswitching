from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from .activations import (
    ModelBundle,
    SequenceSummary,
    prepare_model,
    summarize_text_positions,
    tokenize_text,
)
from .config import RunConfig
from .data import build_group_index, collect_required_conditions, load_dataset, matched_group_counts
from .plots import layer_summary_heatmap, neuron_heatmap
from .stats import ComparisonMeta, ConditionMeanAccumulator, PairedDeltaAccumulator


@dataclass(frozen=True)
class EventAlignment:
    source_id: str
    comparison: str
    focus_condition: str
    baseline_condition: str
    prefix_len: int
    focus_event_token_index: int
    baseline_event_token_index: int
    focus_anchor_position: int
    baseline_anchor_position: int
    focus_event_token_text: str
    baseline_event_token_text: str


def run_pipeline(config: RunConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = config.output_dir / "tables"
    figures_dir = config.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(config.input_path)
    group_index = build_group_index(df)
    required_conditions = collect_required_conditions(config.comparisons)

    matched_counts = matched_group_counts(group_index, config.comparisons)
    pd.DataFrame([row.__dict__ for row in matched_counts]).to_csv(
        tables_dir / "matched_group_counts.csv",
        index=False,
    )

    model_bundle = prepare_model(config.model_name, config.device, config.chunk_size)
    condition_acc = ConditionMeanAccumulator()
    comparison_accs = {
        comp.name: PairedDeltaAccumulator(
            ComparisonMeta(
                name=comp.name,
                focus_condition=comp.focus_condition,
                baseline_condition=comp.baseline_condition,
            )
        )
        for comp in config.comparisons
    }

    processed_conditions = 0
    skipped_samples = []
    event_rows = []
    position_rows = []
    source_items = list(group_index.items())
    if config.max_groups is not None:
        source_items = source_items[: config.max_groups]

    for idx, (source_id, cond_map) in enumerate(tqdm(source_items, desc="Analyzing matched groups"), start=1):
        available_required = sorted(required_conditions.intersection(cond_map))
        if not available_required:
            continue

        token_cache = {
            condition: tokenize_text(model_bundle, cond_map[condition]["text"], config.max_length)
            for condition in available_required
        }

        alignments: dict[str, EventAlignment] = {}
        positions_needed: dict[str, set[int]] = {condition: set() for condition in available_required}
        for comp in config.comparisons:
            if comp.focus_condition not in token_cache or comp.baseline_condition not in token_cache:
                continue
            alignment = build_event_alignment(
                source_id=source_id,
                comparison=comp.name,
                focus_condition=comp.focus_condition,
                baseline_condition=comp.baseline_condition,
                focus_tokens=token_cache[comp.focus_condition],
                baseline_tokens=token_cache[comp.baseline_condition],
            )
            alignments[comp.name] = alignment
            event_rows.append(alignment.__dict__)

            for relative_offset in config.token_offsets:
                focus_position = alignment.focus_anchor_position + relative_offset
                baseline_position = alignment.baseline_anchor_position + relative_offset
                if 0 <= focus_position < len(token_cache[comp.focus_condition].input_ids) - 1 and 0 <= baseline_position < len(
                    token_cache[comp.baseline_condition].input_ids
                ) - 1:
                    positions_needed[comp.focus_condition].add(focus_position)
                    positions_needed[comp.baseline_condition].add(baseline_position)

        sequence_cache: dict[str, SequenceSummary] = {}
        for condition, positions in positions_needed.items():
            if not positions:
                continue
            record = cond_map[condition]
            try:
                summary = summarize_text_positions(
                    model_bundle,
                    record["text"],
                    config.max_length,
                    positions=sorted(positions),
                    target_mode=config.target_mode,
                )
            except Exception as exc:  # pragma: no cover - runtime/data dependent
                skipped_samples.append(
                    {
                        "source_id": source_id,
                        "condition": condition,
                        "error": str(exc),
                    }
                )
                continue
            sequence_cache[condition] = summary
            processed_conditions += 1

        for comp in config.comparisons:
            if comp.name not in alignments:
                continue
            if comp.focus_condition not in sequence_cache or comp.baseline_condition not in sequence_cache:
                continue

            alignment = alignments[comp.name]
            focus_vectors, focus_position_rows = collect_relative_vectors(
                source_id=source_id,
                comparison=comp.name,
                condition=comp.focus_condition,
                anchor_position=alignment.focus_anchor_position,
                sequence_summary=sequence_cache[comp.focus_condition],
                token_offsets=config.token_offsets,
            )
            baseline_vectors, baseline_position_rows = collect_relative_vectors(
                source_id=source_id,
                comparison=comp.name,
                condition=comp.baseline_condition,
                anchor_position=alignment.baseline_anchor_position,
                sequence_summary=sequence_cache[comp.baseline_condition],
                token_offsets=config.token_offsets,
            )
            if not focus_vectors or not baseline_vectors:
                continue

            comparison_accs[comp.name].update(focus_vectors, baseline_vectors)
            condition_acc.update(comp.name, comp.focus_condition, focus_vectors)
            condition_acc.update(comp.name, comp.baseline_condition, baseline_vectors)
            position_rows.extend(focus_position_rows)
            position_rows.extend(baseline_position_rows)

        if idx % config.log_every == 0:
            pass

    condition_df = condition_acc.to_frame()
    if not condition_df.empty:
        condition_df = condition_df.sort_values(["comparison", "condition", "relative_offset", "layer", "neuron"])
    condition_df.to_csv(
        tables_dir / "condition_means.csv.gz",
        index=False,
        compression="gzip",
    )

    if event_rows:
        pd.DataFrame(event_rows).to_csv(tables_dir / "event_alignments.csv", index=False)
    if position_rows:
        pd.DataFrame(position_rows).to_csv(tables_dir / "position_targets.csv", index=False)

    comparison_frames = []
    for comp in config.comparisons:
        frame = comparison_accs[comp.name].to_frame(alpha=config.alpha)
        comparison_frames.append(frame)
        for relative_offset in sorted(frame["relative_offset"].unique()) if not frame.empty else []:
            subset = frame[frame["relative_offset"] == relative_offset].copy()
            neuron_heatmap(
                subset,
                value_col="mean_delta",
                title=f"{comp.name} offset {format_offset(relative_offset)} logit-effect heatmap",
                out_path=figures_dir / f"{comp.name}_offset_{format_offset(relative_offset)}_effect_heatmap.html",
            )

    comparison_df = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(["comparison", "relative_offset", "layer", "neuron"])
    comparison_df.to_csv(
        tables_dir / "comparison_stats.csv.gz",
        index=False,
        compression="gzip",
    )

    phenomenon_df = build_phenomenon_consensus(comparison_df, config)
    phenomenon_df.to_csv(
        tables_dir / "phenomenon_consensus.csv.gz",
        index=False,
        compression="gzip",
    )

    for phenomenon in config.phenomena:
        subset = phenomenon_df[phenomenon_df["phenomenon"] == phenomenon.name].copy()
        top = subset.sort_values(
            ["passes", "rank_score", "consensus_effect"],
            ascending=[False, False, False],
        ).head(phenomenon.top_k)
        top.to_csv(tables_dir / f"top_neurons_{phenomenon.name}.csv", index=False)
        for relative_offset in sorted(subset["relative_offset"].unique()) if not subset.empty else []:
            offset_subset = subset[subset["relative_offset"] == relative_offset].copy()
            neuron_heatmap(
                offset_subset,
                value_col="consensus_effect",
                title=f"{phenomenon.name} offset {format_offset(relative_offset)} consensus heatmap",
                out_path=figures_dir / f"{phenomenon.name}_offset_{format_offset(relative_offset)}_consensus_heatmap.html",
            )

    layer_summary_df = build_layer_summary(comparison_df)
    layer_summary_df.to_csv(tables_dir / "layer_summary.csv", index=False)
    layer_summary_heatmap(layer_summary_df, figures_dir / "layer_summary_heatmap.html")

    summary = {
        "model_name": config.model_name,
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "processed_conditions": processed_conditions,
        "skipped_condition_count": len(skipped_samples),
        "chunk_size": config.chunk_size,
        "max_groups": config.max_groups,
        "token_offsets": list(config.token_offsets),
        "target_mode": config.target_mode,
        "score_definition": "Predicted or observed next-token logit delta caused by adding one MLP neuron contribution at token positions aligned to the first matched divergence point.",
        "comparison_pair_counts": {
            comp.name: {str(offset): int(comparison_accs[comp.name].counts.get(offset, 0)) for offset in config.token_offsets}
            for comp in config.comparisons
        },
    }
    with open(tables_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if skipped_samples:
        pd.DataFrame(skipped_samples).to_csv(tables_dir / "skipped_conditions.csv", index=False)


def build_event_alignment(
    source_id: str,
    comparison: str,
    focus_condition: str,
    baseline_condition: str,
    focus_tokens,
    baseline_tokens,
) -> EventAlignment:
    prefix_len = longest_common_prefix(focus_tokens.input_ids, baseline_tokens.input_ids)
    focus_event_token_index = min(prefix_len, len(focus_tokens.input_ids) - 1)
    baseline_event_token_index = min(prefix_len, len(baseline_tokens.input_ids) - 1)
    focus_anchor_position = clamp_prediction_position(len(focus_tokens.input_ids), focus_event_token_index)
    baseline_anchor_position = clamp_prediction_position(len(baseline_tokens.input_ids), baseline_event_token_index)
    return EventAlignment(
        source_id=source_id,
        comparison=comparison,
        focus_condition=focus_condition,
        baseline_condition=baseline_condition,
        prefix_len=int(prefix_len),
        focus_event_token_index=int(focus_event_token_index),
        baseline_event_token_index=int(baseline_event_token_index),
        focus_anchor_position=int(focus_anchor_position),
        baseline_anchor_position=int(baseline_anchor_position),
        focus_event_token_text=focus_tokens.token_texts[focus_event_token_index],
        baseline_event_token_text=baseline_tokens.token_texts[baseline_event_token_index],
    )


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    common = min(len(a), len(b))
    idx = 0
    while idx < common and a[idx] == b[idx]:
        idx += 1
    return idx


def clamp_prediction_position(n_tokens: int, event_token_index: int) -> int:
    if n_tokens < 2:
        return 0
    return min(max(event_token_index - 1, 0), n_tokens - 2)


def collect_relative_vectors(
    source_id: str,
    comparison: str,
    condition: str,
    anchor_position: int,
    sequence_summary: SequenceSummary,
    token_offsets: tuple[int, ...],
) -> tuple[dict[int, dict[int, pd.Series]], list[dict[str, object]]]:
    relative_vectors = {}
    rows = []
    for relative_offset in token_offsets:
        position = anchor_position + relative_offset
        if position not in sequence_summary.positions:
            continue
        summary = sequence_summary.positions[position]
        relative_vectors[relative_offset] = summary.layer_vectors
        rows.append(
            {
                "source_id": source_id,
                "comparison": comparison,
                "condition": condition,
                "relative_offset": int(relative_offset),
                "anchor_position": int(anchor_position),
                "position": int(position),
                "current_token_id": int(summary.current_token_id),
                "current_token_text": summary.current_token_text,
                "target_token_id": int(summary.target_token_id),
                "target_token_text": summary.target_token_text,
            }
        )
    return relative_vectors, rows


def build_phenomenon_consensus(comparison_df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    rows = []
    if comparison_df.empty:
        return pd.DataFrame(
            columns=[
                "phenomenon",
                "focus_condition",
                "relative_offset",
                "layer",
                "neuron",
                "consensus_effect",
                "min_effect_size_dz",
                "max_q_value",
                "min_pairs",
                "passes",
                "rank_score",
                "required_baselines",
            ]
        )

    for phenomenon in config.phenomena:
        subset = comparison_df[
            (comparison_df["focus_condition"] == phenomenon.focus_condition)
            & (comparison_df["baseline_condition"].isin(phenomenon.required_baselines))
        ].copy()
        if subset.empty:
            continue

        required = list(phenomenon.required_baselines)
        mean_pivot = subset.pivot_table(
            index=["relative_offset", "layer", "neuron"],
            columns="baseline_condition",
            values="mean_delta",
            aggfunc="first",
        )
        dz_pivot = subset.pivot_table(
            index=["relative_offset", "layer", "neuron"],
            columns="baseline_condition",
            values="effect_size_dz",
            aggfunc="first",
        )
        q_pivot = subset.pivot_table(
            index=["relative_offset", "layer", "neuron"],
            columns="baseline_condition",
            values="q_value",
            aggfunc="first",
        )
        n_pivot = subset.pivot_table(
            index=["relative_offset", "layer", "neuron"],
            columns="baseline_condition",
            values="n_pairs",
            aggfunc="first",
        )

        if not all(baseline in mean_pivot.columns for baseline in required):
            continue

        joined = (
            mean_pivot.add_prefix("mean__")
            .join(dz_pivot.add_prefix("dz__"), how="inner")
            .join(q_pivot.add_prefix("q__"), how="inner")
            .join(n_pivot.add_prefix("n__"), how="inner")
            .reset_index()
        )
        joined = joined.dropna(subset=[f"mean__{baseline}" for baseline in required])
        if joined.empty:
            continue

        joined["consensus_effect"] = joined[[f"mean__{baseline}" for baseline in required]].min(axis=1)
        joined["min_effect_size_dz"] = joined[[f"dz__{baseline}" for baseline in required]].min(axis=1)
        joined["max_q_value"] = joined[[f"q__{baseline}" for baseline in required]].max(axis=1)
        joined["min_pairs"] = joined[[f"n__{baseline}" for baseline in required]].min(axis=1).astype(int)
        joined["passes"] = (
            (joined["min_pairs"] >= config.min_pairs)
            & (joined[[f"mean__{baseline}" for baseline in required]] > 0.0).all(axis=1)
            & (joined[[f"q__{baseline}" for baseline in required]] <= config.alpha).all(axis=1)
        )
        joined["rank_score"] = joined["consensus_effect"] * joined["min_effect_size_dz"].clip(lower=0.0)
        joined["phenomenon"] = phenomenon.name
        joined["focus_condition"] = phenomenon.focus_condition
        joined["required_baselines"] = ",".join(required)

        rows.extend(
            joined[
                [
                    "phenomenon",
                    "focus_condition",
                    "relative_offset",
                    "layer",
                    "neuron",
                    "consensus_effect",
                    "min_effect_size_dz",
                    "max_q_value",
                    "min_pairs",
                    "passes",
                    "rank_score",
                    "required_baselines",
                ]
            ].to_dict(orient="records")
        )

    return pd.DataFrame(rows)


def build_layer_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame(
            columns=["comparison", "relative_offset", "comparison_label", "layer", "mean_abs_delta", "positive_significant_count"]
        )

    layer_summary = (
        comparison_df.assign(
            abs_delta=comparison_df["mean_delta"].abs(),
            positive_significant=(
                comparison_df["significant"].astype(bool) & (comparison_df["mean_delta"] > 0.0)
            ),
            comparison_label=comparison_df.apply(
                lambda row: f"{row['comparison']}@{format_offset(int(row['relative_offset']))}",
                axis=1,
            ),
        )
        .groupby(["comparison", "relative_offset", "comparison_label", "layer"], as_index=False)
        .agg(
            mean_abs_delta=("abs_delta", "mean"),
            positive_significant_count=("positive_significant", lambda s: int(s.sum())),
        )
        .sort_values(["comparison", "relative_offset", "layer"])
    )
    return layer_summary


def format_offset(relative_offset: int) -> str:
    return f"{relative_offset:+d}".replace("+", "plus_").replace("-", "minus_")
