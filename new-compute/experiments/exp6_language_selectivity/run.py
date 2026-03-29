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
    infer_target_script,
    label_token_language,
    load_dataset,
    load_tl_model,
    resolve_device,
    write_neuron_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 6: language-selective neurons in code-switched text "
            "using switch-window token activations."
        )
    )
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--out_dir", default="new-compute/experiments/exp6_language_selectivity/results")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--focus_condition", default="code_switched")
    p.add_argument("--target_label", default="target_language")
    p.add_argument("--token_offsets", nargs="+", type=int, default=[-1, 0, 1])
    p.add_argument("--selectivity_threshold", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=50)
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


def _find_switch_points(labels: list[str]) -> list[int]:
    points = []
    for idx in range(1, len(labels)):
        prev_label = labels[idx - 1]
        cur_label = labels[idx]
        if prev_label in {"english", "target"} and cur_label in {"english", "target"} and prev_label != cur_label:
            points.append(idx)
    return points


def _ensure_slot(slot_map: dict, key: tuple[int, int], vector: np.ndarray) -> dict:
    if key not in slot_map:
        zeros = np.zeros_like(vector, dtype=np.float64)
        slot_map[key] = {
            "english_sum": zeros.copy(),
            "target_sum": zeros.copy(),
            "english_count": 0,
            "target_count": 0,
        }
    return slot_map[key]


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

    target_script = infer_target_script(df, target_label=args.target_label)
    focus_df = df[df["condition"] == args.focus_condition].copy()
    if args.max_groups is not None:
        focus_df = focus_df.head(max(1, int(args.max_groups)))

    stats_map: dict[tuple[int, int], dict[str, np.ndarray | int]] = {}
    sample_rows: list[dict[str, object]] = []
    n_switch_events = 0

    for idx, row in enumerate(
        tqdm(focus_df.itertuples(index=False), total=len(focus_df), desc="Exp6 focus rows"),
        start=1,
    ):
        input_ids = encode_text(
            tokenizer=tokenizer,
            text=row.text,
            max_length=int(args.max_length),
            device=device,
        )
        ids = input_ids[0].detach().cpu().tolist()
        if len(ids) < 2:
            continue

        token_texts = tokenizer.convert_ids_to_tokens(ids)
        token_labels = [label_token_language(tok, target_script=target_script) for tok in token_texts]
        switch_points = _find_switch_points(token_labels)
        if not switch_points:
            continue

        activations = extract_post_activations(model, input_ids)
        seq_len = len(ids)

        for switch_idx in switch_points:
            n_switch_events += 1
            for rel_off in args.token_offsets:
                pos = int(switch_idx + rel_off)
                if pos < 0 or pos >= seq_len:
                    continue
                lang_label = token_labels[pos]
                if lang_label not in {"english", "target"}:
                    continue

                sample_rows.append(
                    {
                        "source_id": row.source_id,
                        "id": row.id,
                        "domain": row.domain,
                        "switch_index": int(switch_idx),
                        "relative_offset": int(rel_off),
                        "position": int(pos),
                        "token_text": str(token_texts[pos]),
                        "token_language_label": lang_label,
                    }
                )

                for layer, layer_mat in activations.items():
                    vec = layer_mat[pos].astype(np.float64, copy=False)
                    slot = _ensure_slot(stats_map, (int(rel_off), int(layer)), vec)
                    if lang_label == "english":
                        slot["english_sum"] += vec
                        slot["english_count"] = int(slot["english_count"]) + 1
                    elif lang_label == "target":
                        slot["target_sum"] += vec
                        slot["target_count"] = int(slot["target_count"]) + 1
        if bool(args.gpu_friendly) and device.type == "cuda" and idx % 64 == 0:
            torch.cuda.empty_cache()

    if sample_rows:
        pd.DataFrame(sample_rows).to_csv(tables_dir / "switch_window_tokens.csv", index=False)

    eps = 1e-9
    frames: list[pd.DataFrame] = []
    for (rel_off, layer), slot in stats_map.items():
        n_eng = int(slot["english_count"])
        n_tgt = int(slot["target_count"])
        if n_eng <= 0 or n_tgt <= 0:
            continue
        mean_eng = slot["english_sum"] / n_eng
        mean_tgt = slot["target_sum"] / n_tgt
        selectivity = (mean_tgt - mean_eng) / (mean_tgt + mean_eng + eps)
        neurons = np.arange(len(selectivity), dtype=int)
        frames.append(
            pd.DataFrame(
                {
                    "relative_offset": int(rel_off),
                    "layer": int(layer),
                    "neuron": neurons,
                    "n_english_tokens": n_eng,
                    "n_target_tokens": n_tgt,
                    "mean_english_activation": mean_eng,
                    "mean_target_activation": mean_tgt,
                    "selectivity_index": selectivity,
                    "abs_selectivity_index": np.abs(selectivity),
                }
            )
        )

    selectivity_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not selectivity_df.empty:
        selectivity_df = selectivity_df.sort_values(["relative_offset", "layer", "neuron"])
    selectivity_df.to_csv(
        tables_dir / "language_selectivity.csv.gz",
        index=False,
        compression="gzip",
    )

    layer_summary = pd.DataFrame()
    if not selectivity_df.empty:
        thr = float(args.selectivity_threshold)
        layer_summary = (
            selectivity_df.assign(
                is_target_selective=selectivity_df["selectivity_index"] >= thr,
                is_english_selective=selectivity_df["selectivity_index"] <= -thr,
                is_abs_selective=selectivity_df["abs_selectivity_index"] >= thr,
            )
            .groupby(["relative_offset", "layer"], as_index=False)
            .agg(
                n_neurons=("neuron", "size"),
                target_selective_count=("is_target_selective", "sum"),
                english_selective_count=("is_english_selective", "sum"),
                abs_selective_count=("is_abs_selective", "sum"),
                mean_abs_selectivity=("abs_selectivity_index", "mean"),
            )
        )
        layer_summary["pct_target_selective"] = (
            100.0 * layer_summary["target_selective_count"] / layer_summary["n_neurons"].clip(lower=1)
        )
        layer_summary["pct_english_selective"] = (
            100.0 * layer_summary["english_selective_count"] / layer_summary["n_neurons"].clip(lower=1)
        )
        layer_summary["pct_abs_selective"] = (
            100.0 * layer_summary["abs_selective_count"] / layer_summary["n_neurons"].clip(lower=1)
        )
    layer_summary.to_csv(tables_dir / "layer_selectivity_summary.csv", index=False)

    if not selectivity_df.empty:
        for rel_off, sub in selectivity_df.groupby("relative_offset"):
            top_target = sub.sort_values("selectivity_index", ascending=False).head(int(args.top_k))
            top_english = sub.sort_values("selectivity_index", ascending=True).head(int(args.top_k))
            top_abs = sub.sort_values("abs_selectivity_index", ascending=False).head(int(args.top_k))

            top_target.to_csv(
                tables_dir / f"top_target_selective_offset_{format_offset(int(rel_off))}.csv",
                index=False,
            )
            top_english.to_csv(
                tables_dir / f"top_english_selective_offset_{format_offset(int(rel_off))}.csv",
                index=False,
            )
            top_abs.to_csv(
                tables_dir / f"top_abs_selective_offset_{format_offset(int(rel_off))}.csv",
                index=False,
            )

            write_neuron_heatmap(
                sub,
                value_col="selectivity_index",
                title=f"language selectivity offset {int(rel_off):+d}",
                out_path=figures_dir / f"language_selectivity_offset_{format_offset(int(rel_off))}_heatmap.html",
            )

    summary = {
        "experiment": "exp6_language_selectivity",
        "dataset_csv": str(args.dataset_csv),
        "model_name": str(args.model_name),
        "focus_condition": str(args.focus_condition),
        "target_label": str(args.target_label),
        "target_script_inferred": target_script,
        "token_offsets": [int(x) for x in args.token_offsets],
        "selectivity_threshold": float(args.selectivity_threshold),
        "top_k": int(args.top_k),
        "gpu_friendly": bool(args.gpu_friendly),
        "n_focus_rows_scanned": int(len(focus_df)),
        "n_switch_events": int(n_switch_events),
        "n_switch_window_rows": int(len(sample_rows)),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
