#!/usr/bin/env python3
"""
extract_activations.py ? Run inference and save MLP + residual activations.

Saves per sentence+condition (one .pt file each):
  - mlp/{id}_{condition}.pt       : MLP post-activations  [n_layers, n_tokens, d_mlp]
  - residual/{id}_{condition}.pt  : Residual stream        [n_layers, n_tokens, d_model]

Attention, logits, and sentence_emb are intentionally excluded to save disk space.
These two outputs cover all requirements for Exp 4, 6, 7, and 21.

All tensors saved in fp16.

Usage:
  python extract_activations.py \\
    --input combined_dataset_preprocessed_qwen.csv \\
    --model_name Qwen/Qwen2-7B-Instruct \\
    --out_dir /scratch0/jabraham/qwen_activations \\
    --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract MLP + residual activations.")
    p.add_argument("--input",      required=True, help="Path to preprocessed CSV")
    p.add_argument("--model_name", default="Qwen/Qwen2-7B-Instruct")
    p.add_argument("--out_dir",    default="outputs/activations")
    p.add_argument("--device",     default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["cs_fr", "cs_hi", "english", "french", "hindi"],
    )
    return p.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def make_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "mlp":     out_dir / "mlp",
        "residual": out_dir / "residual",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_id(row_id: str, condition: str) -> str:
    clean = row_id.replace(":", "_").replace("/", "_")
    return f"{clean}_{condition}"


def extract_and_save(
    row: pd.Series,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dirs: dict[str, Path],
    device: torch.device,
    max_length: int,
) -> None:
    text      = str(row["text"])
    row_id    = str(row["id"])
    condition = str(row["condition"])
    fname     = safe_id(row_id, condition)

    # Skip if already extracted (resume support)
    if (dirs["mlp"] / f"{fname}.pt").exists() and \
       (dirs["residual"] / f"{fname}.pt").exists():
        return

    # Tokenize
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=False,
    )
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    valid_len      = int(attention_mask.sum().item())

    if valid_len < 2:
        return

    mlp_acts   = {}
    resid_acts = {}
    hooks      = []

    for layer_idx, layer in enumerate(model.model.layers):

        def make_mlp_hook(li):
            def hook(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                mlp_acts[li] = act[0, :valid_len].detach().cpu().to(torch.float16)
            return hook

        def make_resid_hook(li):
            def hook(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                resid_acts[li] = act[0, :valid_len].detach().cpu().to(torch.float16)
            return hook

        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(layer_idx)))
        hooks.append(layer.register_forward_hook(make_resid_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )
    finally:
        for h in hooks:
            h.remove()

    n_layers = len(model.model.layers)

    # Stack MLP: [n_layers, valid_T, d_mlp]
    mlp_tensor = torch.stack([
        mlp_acts[l] for l in range(n_layers) if l in mlp_acts
    ])

    # Stack residual: [n_layers, valid_T, d_model]
    resid_tensor = torch.stack([
        resid_acts[l] for l in range(n_layers) if l in resid_acts
    ])

    torch.save(mlp_tensor,   dirs["mlp"]      / f"{fname}.pt")
    torch.save(resid_tensor, dirs["residual"] / f"{fname}.pt")


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Loading dataset from {input_path} ...")
    df = pd.read_csv(input_path)
    df = df[df["condition"].isin(args.conditions)].reset_index(drop=True)
    print(f"  {len(df):,} rows | conditions: {args.conditions}")

    dirs = make_dirs(out_dir)

    print(f"\nLoading model: {args.model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        device_map="auto",
    )
    model.eval()
    print(f"  Layers : {len(model.model.layers)}")
    print(f"  d_model: {model.config.hidden_size}")

    print(f"\nExtracting activations ? {out_dir}")
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        fname = safe_id(str(row["id"]), str(row["condition"]))
        if (dirs["mlp"] / f"{fname}.pt").exists() and \
           (dirs["residual"] / f"{fname}.pt").exists():
            skipped += 1
            continue
        extract_and_save(row, model, tokenizer, dirs, device, args.max_length)

    print(f"\nDone.")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  Outputs written to: {out_dir}")
    print(f"\n  Structure:")
    for name, d in dirs.items():
        n_files = len(list(d.glob("*.pt")))
        print(f"    {name:<15}: {n_files:>6} files")


if __name__ == "__main__":
    main()
