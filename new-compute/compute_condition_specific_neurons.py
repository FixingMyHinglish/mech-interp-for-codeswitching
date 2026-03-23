#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from tqdm import tqdm

try:
    from transformer_lens import HookedTransformer
    from transformer_lens import utils as tl_utils
except Exception:  # pragma: no cover - optional dependency
    HookedTransformer = None
    tl_utils = None

from neuron_selection import NeuronKey, jaccard, run_selection, to_keys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute condition-specific neuron sets using either pipeline proxy tables "
            "or TransformerLens MLP hook activations."
        )
    )
    p.add_argument(
        "--backend",
        default="transformer_lens",
        choices=["transformer_lens", "pipeline_proxy"],
        help="Selection backend. transformer_lens uses standard TL hook activations.",
    )
    p.add_argument(
        "--run_dir",
        default=None,
        help="Pipeline output directory (used for pipeline_proxy input and/or model_name metadata fallback).",
    )
    p.add_argument(
        "--dataset_csv",
        default=None,
        help="Required for transformer_lens backend; must include id/text/condition/domain.",
    )
    p.add_argument(
        "--model_name",
        default=None,
        help="Model name for transformer_lens backend; defaults to run_dir/metadata.json model_name.",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for TransformerLens inference.",
    )
    p.add_argument(
        "--reduce_mode",
        default="mean_abs",
        choices=["mean_abs", "mean", "max_abs"],
        help=(
            "How to collapse the token-position dimension. Pad tokens are always "
            "excluded from the reduction. mean_abs averages |activation| over "
            "valid (non-pad) positions."
        ),
    )
    p.add_argument(
        "--out_dir",
        default="new-compute/results",
        help="Directory for new-compute outputs.",
    )
    p.add_argument(
        "--activation_cutoff",
        type=float,
        default=0.0,
        help="Minimum activation to count as 'firing'.",
    )
    p.add_argument(
        "--importance_quantile",
        type=float,
        default=0.90,
        help=(
            "Per condition-domain quantile for importance cutoff when --importance_min is not set. "
            "Example: 0.90 keeps top 10%% most important neurons."
        ),
    )
    p.add_argument(
        "--importance_min",
        type=float,
        default=None,
        help="Absolute importance cutoff. If set, overrides quantile cutoff.",
    )
    p.add_argument(
        "--min_domain_consistency",
        type=float,
        default=0.50,
        help="Minimum fraction of domains in which a neuron must appear to be marked consistent.",
    )
    p.add_argument("--cs_label", default="code_switched")
    p.add_argument("--confused_label", default="confused")
    p.add_argument("--english_label", default="english")
    p.add_argument("--target_label", default="target_language")
    args = p.parse_args()
    if args.backend == "pipeline_proxy" and not args.run_dir:
        p.error("--run_dir is required when --backend pipeline_proxy")
    if args.backend == "transformer_lens" and not args.dataset_csv:
        p.error("--dataset_csv is required when --backend transformer_lens")
    return args


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _derive_source_id(row_id: str, condition: str) -> str:
    suffix = f"_{condition}"
    if row_id.endswith(suffix):
        return row_id[: -len(suffix)]
    return row_id


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("dataset_csv must be .csv, .jsonl, or .json")

    required = {"id", "text", "condition", "domain"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["text"] = out["text"].astype(str)
    out["condition"] = out["condition"].astype(str)
    out["domain"] = out["domain"].astype(str)
    if "source_id" not in out.columns:
        out["source_id"] = [
            _derive_source_id(rid, cond)
            for rid, cond in zip(out["id"], out["condition"])
        ]
    else:
        out["source_id"] = out["source_id"].astype(str)
    return out


def load_model_name(run_dir: str | None, override: str | None) -> str:
    if override:
        return str(override)
    if not run_dir:
        raise ValueError(
            "model_name is required for transformer_lens backend when run_dir is not set"
        )
    meta_path = Path(run_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; pass --model_name explicitly")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError(f"{meta_path} does not contain model_name")
    return str(model_name)


def _reduce_tensor(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce over the token-position dimension (dim 0) for a single-example slice."""
    if mode == "mean_abs":
        return x.abs().mean(dim=0)
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "max_abs":
        return x.abs().max(dim=0).values
    raise ValueError(f"Unsupported reduce_mode: {mode}")


_PROXY_COLS = ["id", "source_id", "domain", "condition", "layer", "neuron", "activation"]
# Flush the in-memory row buffer to disk every this many batches.
# Keeps peak RAM ~ batch_size * FLUSH_EVERY * n_layers * d_mlp dicts
# instead of the full dataset, avoiding OOM on large corpora.
_FLUSH_EVERY = 10


def build_proxy_from_transformer_lens(
    dataset_df: pd.DataFrame,
    model_name: str,
    device: torch.device,
    max_length: int,
    reduce_mode: str,
    stream_path: Path,
    activation_cutoff: float = 0.0,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Extract MLP post-activation neuron scores using TransformerLens hooks.

    Hooks on ``blocks.{layer}.mlp.hook_post`` capture activations after the
    nonlinearity, which is the canonical TL location for neuron analysis.
    All neurons that exceed ``activation_cutoff`` are stored; the cutoff is
    applied as a tensor mask at extraction time so inactive neurons never
    reach the proxy file. The importance quantile in ``run_selection`` is the
    sole gate on which active neurons are ultimately kept, so no signal is
    discarded before cross-example aggregation.
    Examples are processed in batches; pad tokens are masked out before the
    position-level reduction so they don't affect per-example scores.

    Rows are streamed to ``stream_path`` (gzipped CSV) every ``_FLUSH_EVERY``
    batches and the in-memory buffer is cleared, so peak RAM is proportional
    to the buffer size rather than the full dataset. The final DataFrame is
    read back from disk, which is ~10x more memory-efficient than a list of
    Python dicts.
    """
    if HookedTransformer is None:
        raise ImportError(
            "transformer_lens is not installed. Install it or use --backend pipeline_proxy."
        )

    model = HookedTransformer.from_pretrained(model_name, device=str(device))
    model.eval()

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = int(model.cfg.n_layers)
    # get_act_name ensures hook names stay correct across TL versions.
    hook_names = {tl_utils.get_act_name("post", layer) for layer in range(n_layers)}

    n_batches = (len(dataset_df) + batch_size - 1) // batch_size
    stream_path.parent.mkdir(parents=True, exist_ok=True)

    row_buffer: list[dict[str, object]] = []

    def _flush(writer: csv.DictWriter) -> None:
        if row_buffer:
            writer.writerows(row_buffer)
            row_buffer.clear()

    with gzip.open(stream_path, "wt", newline="", encoding="utf-8") as gz_f:
        writer = csv.DictWriter(gz_f, fieldnames=_PROXY_COLS)
        writer.writeheader()

        for batch_idx, batch_start in enumerate(
            tqdm(range(0, len(dataset_df), batch_size), total=n_batches,
                 desc="TL neuron extraction")
        ):
            batch_df = dataset_df.iloc[batch_start : batch_start + batch_size]
            texts = batch_df["text"].tolist()

            encoded = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                padding_side="right",
            )
            input_ids = encoded["input_ids"].to(device)           # [B, T]
            attention_mask = encoded["attention_mask"].to(device)  # [B, T]

            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    input_ids,
                    return_type="logits",
                    names_filter=lambda n: n in hook_names,
                )

            for i, row in enumerate(batch_df.itertuples(index=False)):
                valid_mask = attention_mask[i].bool()  # [T]
                valid_len = int(valid_mask.sum().item())
                if valid_len < 2:
                    continue

                for layer in range(n_layers):
                    key = tl_utils.get_act_name("post", layer)
                    if key not in cache:
                        continue
                    # Mask out pad positions before reducing over the sequence.
                    acts = cache[key][i][valid_mask].detach().float()  # [valid_T, d_mlp]
                    reduced = _reduce_tensor(acts, reduce_mode)         # [d_mlp]
                    if reduced.numel() == 0:
                        continue
                    # Apply activation cutoff as a tensor mask — neurons at or
                    # below the cutoff are dropped here so they never reach disk.
                    # The importance quantile in run_selection() handles the rest.
                    active = reduced > activation_cutoff
                    if not active.any():
                        continue
                    active_idxs = active.nonzero(as_tuple=True)[0]
                    for neuron_idx, score in zip(active_idxs.tolist(), reduced[active].tolist()):
                        row_buffer.append(
                            {
                                "id": str(row.id),
                                "source_id": str(row.source_id),
                                "domain": str(row.domain),
                                "condition": str(row.condition),
                                "layer": int(layer),
                                "neuron": int(neuron_idx),
                                "activation": float(score),
                            }
                        )

            # Periodically flush to disk to avoid accumulating a huge list.
            if (batch_idx + 1) % _FLUSH_EVERY == 0:
                _flush(writer)

        _flush(writer)  # write any remaining rows

    print(f"       Streaming proxy written to {stream_path}")
    return pd.read_csv(stream_path, compression="gzip")


def load_proxy_from_pipeline(run_dir: str) -> pd.DataFrame:
    in_path = Path(run_dir) / "tables" / "neuron_proxy_raw.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input table: {in_path}")
    return pd.read_csv(in_path)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/5] Backend: {args.backend}")
    if args.backend == "pipeline_proxy":
        df = load_proxy_from_pipeline(args.run_dir)
        source_note = f"run_dir={args.run_dir}"
    else:
        device = resolve_device(args.device)
        model_name = load_model_name(args.run_dir, args.model_name)
        dataset_df = load_dataset(args.dataset_csv)
        tl_raw_path = Path(args.out_dir) / "neuron_proxy_transformer_lens.csv.gz"
        df = build_proxy_from_transformer_lens(
            dataset_df=dataset_df,
            model_name=model_name,
            device=device,
            max_length=int(args.max_length),
            reduce_mode=args.reduce_mode,
            stream_path=tl_raw_path,
            activation_cutoff=float(args.activation_cutoff),
            batch_size=int(args.batch_size),
        )
        source_note = (
            f"dataset={args.dataset_csv} model={model_name} device={device} "
            f"max_length={args.max_length} reduce_mode={args.reduce_mode} "
            f"activation_cutoff={args.activation_cutoff} batch_size={args.batch_size}"
        )

    req = {"source_id", "domain", "condition", "layer", "neuron", "activation"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    print(f"[2/5] Rows loaded: {len(df):,}  ({source_note})")

    print("[3/5] Running neuron selection (firing → importance → filtering → consistency)")
    out = run_selection(
        proxy_df=df,
        activation_cutoff=args.activation_cutoff,
        importance_quantile=args.importance_quantile,
        importance_min=args.importance_min,
        min_domain_consistency=args.min_domain_consistency,
        cs_label=args.cs_label,
        confused_label=args.confused_label,
        english_label=args.english_label,
        target_label=args.target_label,
    )

    fired = out["fired"]
    consistent = out["consistent"]
    print(
        f"       Fired rows (activation > {args.activation_cutoff}): {len(fired):,} "
        f"({(len(fired) / max(len(df), 1)) * 100:.1f}%)"
    )

    print("[4/5] Writing outputs")
    fired.to_csv(
        os.path.join(args.out_dir, "all_fired_rows.csv.gz"), index=False, compression="gzip"
    )
    out["importance"].to_csv(
        os.path.join(args.out_dir, "neuron_importance_condition_domain.csv.gz"),
        index=False, compression="gzip",
    )
    out["important"].to_csv(
        os.path.join(args.out_dir, "important_neurons_condition_domain.csv.gz"),
        index=False, compression="gzip",
    )
    out["filtered"].to_csv(
        os.path.join(args.out_dir, "cs_confused_after_filtering_base.csv.gz"),
        index=False, compression="gzip",
    )
    consistent.to_csv(
        os.path.join(args.out_dir, "consistent_neurons.csv.gz"),
        index=False, compression="gzip",
    )

    print("[5/5] Comparing CS vs confused neuron sets")
    base_set = to_keys(
        out["important"][
            out["important"]["condition"].isin([args.english_label, args.target_label])
        ][["layer", "neuron"]].drop_duplicates()
    )

    def _build_cond_set(cond: str, consistent_only: bool) -> set[NeuronKey]:
        sub = consistent[consistent["condition"] == cond]
        if consistent_only:
            sub = sub[sub["passes_consistency"]]
        return to_keys(sub[["layer", "neuron"]].drop_duplicates())

    cs_set = _build_cond_set(args.cs_label, consistent_only=False)
    conf_set = _build_cond_set(args.confused_label, consistent_only=False)
    cs_cons = _build_cond_set(args.cs_label, consistent_only=True)
    conf_cons = _build_cond_set(args.confused_label, consistent_only=True)

    cs_filtered_keys = to_keys(
        out["filtered"][out["filtered"]["condition"] == args.cs_label][
            ["layer", "neuron"]
        ].drop_duplicates()
    )
    conf_filtered_keys = to_keys(
        out["filtered"][out["filtered"]["condition"] == args.confused_label][
            ["layer", "neuron"]
        ].drop_duplicates()
    )

    summary: Dict[str, object] = {
        "backend": args.backend,
        "run_dir": args.run_dir,
        "dataset_csv": args.dataset_csv,
        "model_name": args.model_name,
        "input_rows": int(len(df)),
        "fired_rows": int(len(fired)),
        "activation_cutoff": float(args.activation_cutoff),
        "importance_quantile": (
            None if args.importance_min is not None else float(args.importance_quantile)
        ),
        "importance_min": (
            None if args.importance_min is None else float(args.importance_min)
        ),
        "min_domain_consistency": float(args.min_domain_consistency),
        "base_neuron_count_english_or_target": int(len(base_set)),
        "cs_filtered_neuron_count": int(len(cs_filtered_keys)),
        "confused_filtered_neuron_count": int(len(conf_filtered_keys)),
        "cs_conf_overlap_count_all": int(len(cs_set & conf_set)),
        "cs_conf_jaccard_all": float(jaccard(cs_set, conf_set)),
        "cs_conf_overlap_count_consistent": int(len(cs_cons & conf_cons)),
        "cs_conf_jaccard_consistent": float(jaccard(cs_cons, conf_cons)),
        "cs_only_consistent_count": int(len(cs_cons - conf_cons)),
        "confused_only_consistent_count": int(len(conf_cons - cs_cons)),
        "source_note": source_note,
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Outputs:\n"
            "- all_fired_rows.csv.gz\n"
            "- neuron_importance_condition_domain.csv.gz\n"
            "- important_neurons_condition_domain.csv.gz\n"
            "- cs_confused_after_filtering_base.csv.gz\n"
            "- consistent_neurons.csv.gz\n"
            "- summary.json\n"
            "- neuron_proxy_transformer_lens.csv.gz  (transformer_lens backend only)\n"
        )

    print("Done. Outputs written to:", args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
