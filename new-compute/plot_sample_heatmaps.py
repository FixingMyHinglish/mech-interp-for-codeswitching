#!/usr/bin/env python3
"""
plot_sample_heatmaps.py
-----------------------
Generate residual attribution heatmaps for the first N samples of each condition.

Layout:
  Y-axis  = layer (0..n_layers-1)
  X-axis  = token positions (actual token text)
  Colour  = normalised gradient × activation (RdBu, centred at 0)
            blue = pushing toward predicted token, red = suppressing it

Usage:
  python new-compute/plot_sample_heatmaps.py \
    --dataset_csv  data/french.csv \
    --out_dir      new-compute/sample_heatmaps_french \
    --n_samples    3 \
    --model_name   gpt2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

try:
    from transformer_lens import HookedTransformer
except ImportError:
    raise SystemExit("transformer_lens is required. Install it in your venv.")


# ── residual attribution ──────────────────────────────────────────────────────

def get_residual_attribution(
    model: HookedTransformer,
    text: str,
    max_length: int,
    device: torch.device,
) -> tuple[np.ndarray, str, int, list[str]]:
    """Return (attr [n_layers, seq], target_token_text, target_position, token_labels)."""
    tokens = model.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"].to(device)

    seq_len = int(tokens.shape[1])
    if seq_len < 2:
        raise ValueError("Sequence too short")

    n_layers = int(model.cfg.n_layers)
    resid_by_layer: dict[int, torch.Tensor] = {}

    def make_hook(l: int):
        def _hook(resid: torch.Tensor, hook):
            resid.retain_grad()
            resid_by_layer[l] = resid
            return resid
        return _hook

    fwd_hooks = [
        (f"blocks.{l}.hook_resid_post", make_hook(l))
        for l in range(n_layers)
    ]

    model.zero_grad(set_to_none=True)
    logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=fwd_hooks)

    pred_pos        = seq_len - 2
    target_token_id = int(torch.argmax(logits[0, pred_pos]).item())
    logits[0, pred_pos, target_token_id].backward()

    attr = np.zeros((n_layers, seq_len), dtype=np.float32)
    for l in range(n_layers):
        resid = resid_by_layer.get(l)
        if resid is not None and resid.grad is not None:
            gxa = (resid.grad[0] * resid[0]).sum(dim=-1).detach().float().cpu().numpy()
            attr[l] = gxa.astype(np.float32, copy=False)

    denom = float(np.max(np.abs(attr)))
    if denom > 0:
        attr = attr / denom

    # Token labels
    token_ids   = tokens[0].tolist()
    token_labels = [
        model.tokenizer.decode([tid]).replace(" ", "·") or f"[{tid}]"
        for tid in token_ids
    ]
    target_text = model.tokenizer.decode([target_token_id]).strip() or str(target_token_id)

    return attr, target_text, pred_pos, token_labels


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_residual_attribution(
    attr: np.ndarray,
    token_labels: list[str],
    target_token: str,
    target_position: int,
    title: str,
    out_path: Path,
) -> None:
    n_layers, n_pos = attr.shape
    fig = go.Figure(go.Heatmap(
        z=attr,
        x=token_labels[:n_pos],
        y=[str(l) for l in range(n_layers)],
        colorscale="RdBu",
        zmid=0.0,
        colorbar=dict(title="Normalised\nattribution"),
    ))
    fig.update_layout(
        title=(
            f"Residual Attribution  (target='{target_token}', pred_pos={target_position})"
            f"<br>{title}"
        ),
        xaxis_title="Token",
        yaxis_title="Layer",
        width=max(950, 30 * n_pos),
        height=max(520, 30 * n_layers),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=70, r=40, t=85, b=70),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"    saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--out_dir",     default="new-compute/sample_heatmaps")
    p.add_argument("--n_samples",   type=int, default=3)
    p.add_argument("--model_name",  default="gpt2")
    p.add_argument("--max_length",  type=int, default=96)
    p.add_argument("--device",      default="auto")
    return p.parse_args()


def resolve_device(d: str) -> torch.device:
    if d == "auto":
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset_csv)
    print(f"Dataset: {len(df)} rows  |  conditions: {sorted(df.condition.unique())}")

    print(f"\nLoading '{args.model_name}' on {device}...")
    model = HookedTransformer.from_pretrained(args.model_name)
    model.eval()
    model.to(device)
    print(f"  {model.cfg.n_layers} layers\n")

    for cond in sorted(df.condition.unique()):
        cond_df  = df[df.condition == cond].reset_index(drop=True)
        n_take   = min(args.n_samples, len(cond_df))
        cond_dir = out / cond
        cond_dir.mkdir(exist_ok=True)
        print(f"[{cond}]  {len(cond_df)} rows — plotting first {n_take}")

        for i in range(n_take):
            row  = cond_df.iloc[i]
            sid  = str(row["id"])
            text = str(row["text"])
            print(f"  sample {i+1}/{n_take}  id={sid}")
            try:
                attr, tgt_text, tgt_pos, tok_labels = get_residual_attribution(
                    model, text, args.max_length, device,
                )
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)[:60]
                plot_residual_attribution(
                    attr, tok_labels, tgt_text, tgt_pos,
                    title=f"{cond} | sample {i+1} | id={sid}",
                    out_path=cond_dir / f"sample_{i+1:02d}_{safe}.html",
                )
            except Exception as exc:
                print(f"    ERROR: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
