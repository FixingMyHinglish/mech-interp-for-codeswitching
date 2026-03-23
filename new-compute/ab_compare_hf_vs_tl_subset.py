#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from tqdm import tqdm

from neuron_selection import NeuronKey, jaccard, run_selection, to_keys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "A/B compare current new-compute selection vs TransformerLens-based selection "
            "on a subset of data."
        )
    )
    p.add_argument("--run_dir", required=True, help="Pipeline output dir with tables/neuron_proxy_raw.csv")
    p.add_argument("--dataset_csv", required=True, help="Dataset CSV containing id/text/condition/domain")
    p.add_argument("--out_dir", default="new-compute/ab_subset")
    p.add_argument("--model_name", default=None, help="Override model name (default: run_dir/metadata.json)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for TransformerLens inference.",
    )
    p.add_argument("--reduce_mode", default="mean_abs", choices=["mean_abs", "mean", "max_abs"])
    p.add_argument("--n_source_ids", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--activation_cutoff", type=float, default=0.0)
    p.add_argument("--importance_quantile", type=float, default=0.90)
    p.add_argument("--importance_min", type=float, default=None)
    p.add_argument("--min_domain_consistency", type=float, default=0.50)
    p.add_argument("--cs_label", default="code_switched")
    p.add_argument("--confused_label", default="confused")
    p.add_argument("--english_label", default="english")
    p.add_argument("--target_label", default="target_language")
    return p.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def derive_source_id(row_id: str, condition: str) -> str:
    suffix = f"_{condition}"
    if row_id.endswith(suffix):
        return row_id[: -len(suffix)]
    return row_id


def load_model_name(run_dir: Path, override: str | None) -> str:
    if override:
        return override
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; pass --model_name")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError(f"{meta_path} missing model_name")
    return str(model_name)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "text", "condition", "domain"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["condition"] = out["condition"].astype(str)
    out["domain"] = out["domain"].astype(str)
    out["text"] = out["text"].astype(str)
    if "source_id" not in out.columns:
        out["source_id"] = [
            derive_source_id(rid, cond)
            for rid, cond in zip(out["id"], out["condition"])
        ]
    else:
        out["source_id"] = out["source_id"].astype(str)
    return out


def sample_matched_subset(
    df: pd.DataFrame,
    n_source_ids: int,
    seed: int,
    required_conditions: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    cond_sets = (
        df.groupby("source_id")["condition"]
        .apply(lambda s: set(s.astype(str).tolist()))
        .to_dict()
    )
    candidates = [
        sid for sid, conds in cond_sets.items()
        if all(c in conds for c in required_conditions)
    ]
    if not candidates:
        raise ValueError("No source_ids contain all required conditions for A/B subset")
    rng = random.Random(seed)
    selected = rng.sample(candidates, k=min(n_source_ids, len(candidates)))
    return df[df["source_id"].isin(selected)].copy(), selected


def _reduce_tensor(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean_abs":
        return x.abs().mean(dim=0)
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "max_abs":
        return x.abs().max(dim=0).values
    raise ValueError(f"Unsupported reduce_mode: {mode}")


def build_tl_proxy_rows(
    subset_df: pd.DataFrame,
    model_name: str,
    device: torch.device,
    max_length: int,
    reduce_mode: str,
    activation_cutoff: float = 0.0,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Extract MLP post-activation neuron scores via TL hooks, batched.

    Mirrors the extraction logic in ``compute_condition_specific_neurons.py``
    exactly: activation cutoff applied as a tensor mask at extraction time,
    no top-k pre-filter, pad tokens masked before position-level reduction.
    """
    model = HookedTransformer.from_pretrained(model_name, device=str(device))
    model.eval()
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = int(model.cfg.n_layers)
    hook_names = {tl_utils.get_act_name("post", layer) for layer in range(n_layers)}

    rows: list[dict[str, object]] = []
    n_batches = (len(subset_df) + batch_size - 1) // batch_size

    for batch_start in tqdm(range(0, len(subset_df), batch_size), total=n_batches,
                            desc="TL proxy extraction (A/B)"):
        batch_df = subset_df.iloc[batch_start : batch_start + batch_size]
        texts = batch_df["text"].tolist()

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
            padding_side="right",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                return_type="logits",
                names_filter=lambda n: n in hook_names,
            )

        for i, row in enumerate(batch_df.itertuples(index=False)):
            valid_mask = attention_mask[i].bool()
            if int(valid_mask.sum().item()) < 2:
                continue

            for layer in range(n_layers):
                key = tl_utils.get_act_name("post", layer)
                if key not in cache:
                    continue
                acts = cache[key][i][valid_mask].detach().float()  # [valid_T, d_mlp]
                reduced = _reduce_tensor(acts, reduce_mode)
                if reduced.numel() == 0:
                    continue
                active = reduced > activation_cutoff
                if not active.any():
                    continue
                active_idxs = active.nonzero(as_tuple=True)[0]
                for neuron_idx, score in zip(active_idxs.tolist(), reduced[active].tolist()):
                    rows.append(
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

    return pd.DataFrame(rows)


def compare_methods(
    a_consistent: pd.DataFrame,
    b_consistent: pd.DataFrame,
    cs_label: str,
    confused_label: str,
) -> pd.DataFrame:
    rows = []
    for condition in [cs_label, confused_label]:
        a_set = to_keys(
            a_consistent[
                (a_consistent["condition"] == condition)
                & (a_consistent["passes_consistency"].astype(bool))
            ][["layer", "neuron"]].drop_duplicates()
        )
        b_set = to_keys(
            b_consistent[
                (b_consistent["condition"] == condition)
                & (b_consistent["passes_consistency"].astype(bool))
            ][["layer", "neuron"]].drop_duplicates()
        )
        rows.append(
            {
                "condition": condition,
                "a_count": int(len(a_set)),
                "b_count": int(len(b_set)),
                "overlap_count": int(len(a_set & b_set)),
                "a_only_count": int(len(a_set - b_set)),
                "b_only_count": int(len(b_set - a_set)),
                "jaccard": float(jaccard(a_set, b_set)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model_name = load_model_name(run_dir, args.model_name)
    dataset_df = load_dataset(Path(args.dataset_csv))

    required_conditions = (
        args.english_label, args.target_label, args.cs_label, args.confused_label
    )
    subset_df, selected_source_ids = sample_matched_subset(
        dataset_df,
        n_source_ids=args.n_source_ids,
        seed=args.seed,
        required_conditions=required_conditions,
    )
    subset_df.to_csv(out_dir / "subset_rows.csv", index=False)
    (out_dir / "subset_source_ids.txt").write_text(
        "\n".join(selected_source_ids), encoding="utf-8"
    )

    baseline_proxy_path = run_dir / "tables" / "neuron_proxy_raw.csv"
    if not baseline_proxy_path.exists():
        raise FileNotFoundError(f"Missing baseline proxy table: {baseline_proxy_path}")
    baseline_proxy = pd.read_csv(baseline_proxy_path)
    baseline_proxy_sub = baseline_proxy[
        baseline_proxy["source_id"].astype(str).isin(selected_source_ids)
    ].copy()
    baseline_proxy_sub.to_csv(
        out_dir / "baseline_proxy_subset.csv.gz", index=False, compression="gzip"
    )

    tl_proxy = build_tl_proxy_rows(
        subset_df=subset_df,
        model_name=model_name,
        device=device,
        max_length=args.max_length,
        reduce_mode=args.reduce_mode,
        activation_cutoff=args.activation_cutoff,
        batch_size=args.batch_size,
    )
    tl_proxy.to_csv(out_dir / "tl_proxy_subset.csv.gz", index=False, compression="gzip")

    selection_kwargs = dict(
        activation_cutoff=args.activation_cutoff,
        importance_quantile=args.importance_quantile,
        importance_min=args.importance_min,
        min_domain_consistency=args.min_domain_consistency,
        cs_label=args.cs_label,
        confused_label=args.confused_label,
        english_label=args.english_label,
        target_label=args.target_label,
    )
    a_out = run_selection(proxy_df=baseline_proxy_sub, **selection_kwargs)
    b_out = run_selection(proxy_df=tl_proxy, **selection_kwargs)

    a_out["consistent"].to_csv(
        out_dir / "a_consistent_baseline.csv.gz", index=False, compression="gzip"
    )
    b_out["consistent"].to_csv(
        out_dir / "b_consistent_tl.csv.gz", index=False, compression="gzip"
    )

    ab = compare_methods(
        a_consistent=a_out["consistent"],
        b_consistent=b_out["consistent"],
        cs_label=args.cs_label,
        confused_label=args.confused_label,
    )
    ab.to_csv(out_dir / "ab_condition_overlap.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "dataset_csv": str(args.dataset_csv),
        "model_name": model_name,
        "device": str(device),
        "seed": int(args.seed),
        "subset_n_source_ids_requested": int(args.n_source_ids),
        "subset_n_source_ids_used": int(len(selected_source_ids)),
        "subset_rows": int(len(subset_df)),
        "baseline_proxy_subset_rows": int(len(baseline_proxy_sub)),
        "tl_proxy_subset_rows": int(len(tl_proxy)),
        "reduce_mode": args.reduce_mode,
        "activation_cutoff": float(args.activation_cutoff),
        "importance_quantile": (
            None if args.importance_min is not None else float(args.importance_quantile)
        ),
        "importance_min": (
            None if args.importance_min is None else float(args.importance_min)
        ),
        "min_domain_consistency": float(args.min_domain_consistency),
        "ab_condition_overlap": ab.to_dict(orient="records"),
    }
    (out_dir / "ab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("A/B complete.")
    print(f"Subset source_ids: {len(selected_source_ids)}")
    print("Condition overlap:")
    print(ab.to_string(index=False))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
