#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformer_lens import HookedTransformer
    from transformer_lens import utils as tl_utils
except Exception:  # pragma: no cover - optional dependency
    HookedTransformer = None
    tl_utils = None


@dataclass
class BackendBundle:
    backend: str
    model: Any
    tokenizer: AutoTokenizer
    device: torch.device
    n_layers: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create attention + residual-stream attribution visuals for random samples, "
            "plus overall aggregate visuals."
        )
    )
    p.add_argument("--run_dir", required=True, help="Pipeline output dir (used for metadata.json).")
    p.add_argument("--dataset_csv", required=True, help="Dataset CSV/JSON/JSONL containing id/text/condition.")
    p.add_argument(
        "--results_dir",
        default=None,
        help="Optional new-compute results dir (for overall importance heatmap).",
    )
    p.add_argument("--out_dir", default="new-compute/visuals", help="Output directory for HTML figures.")
    p.add_argument("--model_name", default=None, help="Override model name (default: read from run_dir/metadata.json).")
    p.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "transformer_lens"],
        help="Model backend for attention/attribution extraction.",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=96, help="Tokenizer max length for plots.")
    p.add_argument("--n_random_samples", type=int, default=5, help="Number of random rows to visualize.")
    p.add_argument(
        "--overall_sample_cap",
        type=int,
        default=0,
        help="Max rows per condition used to compute overall aggregate visuals (0 means all rows in each condition).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--conditions",
        default=None,
        nargs="+",
        help="Conditions to visualise. Defaults to all conditions found in the dataset.",
    )
    p.add_argument("--attention_layer", type=int, default=0, help="Layer index for attention visual panels.")
    p.add_argument(
        "--all_attention_layers",
        action="store_true",
        help="If set, produce attention visuals for every layer instead of just --attention_layer.",
    )
    p.add_argument(
        "--focus_head",
        type=int,
        default=-1,
        help="Head index for single-head plot. Use -1 to auto-pick by strongest off-diagonal mass.",
    )
    p.add_argument(
        "--target_mode",
        default="predicted_next_token",
        choices=["predicted_next_token", "observed_next_token"],
        help="Token choice for residual-stream attribution objective.",
    )
    p.add_argument(
        "--overall_max_positions",
        type=int,
        default=40,
        help="Maximum token positions included in overall aggregate plots.",
    )
    p.add_argument(
        "--with_logit_lens",
        action="store_true",
        help="(TL backend only) Add logit-lens heatmap: top predicted token at every layer/position.",
    )
    p.add_argument(
        "--with_dla",
        action="store_true",
        help="(TL backend only) Add direct logit attribution: per-layer MLP vs attention contribution.",
    )
    p.add_argument(
        "--with_activation_heatmap",
        action="store_true",
        help="(TL backend only) Add neuron activation heatmap for condition-specific consistent neurons. "
             "Requires --results_dir.",
    )
    p.add_argument(
        "--with_scalar_activation",
        action="store_true",
        help="(TL backend only) Add scalar MLP activation heatmap: rows=layers, cols=token positions, "
             "colour=mean absolute MLP post-activation at that layer/position. No gradient needed.",
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


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("dataset_csv must be .csv, .jsonl, or .json")

    required = {"id", "text", "condition"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df.copy()


def resolve_model_name(run_dir: Path, override: str | None) -> str:
    if override:
        return override
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Pass --model_name explicitly if metadata.json is unavailable."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError(f"{meta_path} does not contain model_name")
    return str(model_name)


def normalize_token(token: str) -> str:
    if token == "":
        return "<empty>"
    clean = token.replace("\n", "\\n").replace("\t", "\\t")
    clean = clean.replace("Ġ", " ").replace("▁", " ")
    return clean


def load_backend(model_name: str, backend: str, device: torch.device) -> BackendBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if backend == "transformer_lens":
        if HookedTransformer is None:
            raise ImportError(
                "transformer_lens is not installed. Install it to use --backend transformer_lens."
            )
        tl_model = HookedTransformer.from_pretrained(
            model_name,
            device=str(device),
        )
        tl_model.eval()
        return BackendBundle(
            backend="transformer_lens",
            model=tl_model,
            tokenizer=tokenizer,
            device=device,
            n_layers=int(tl_model.cfg.n_layers),
        )

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    except TypeError:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_model.to(device)
    hf_model.eval()
    n_layers = int(getattr(hf_model.config, "num_hidden_layers", 0))
    return BackendBundle(
        backend="hf",
        model=hf_model,
        tokenizer=tokenizer,
        device=device,
        n_layers=n_layers,
    )


def tokenize_inputs(tokenizer: AutoTokenizer, text: str, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def _get_attention_maps_hf(
    model: AutoModelForCausalLM,
    encoded: dict[str, torch.Tensor],
    layer_idx: int,
) -> np.ndarray:
    with torch.no_grad():
        out = model(**encoded, output_attentions=True, use_cache=False)
    attentions = out.attentions
    if attentions is None or len(attentions) == 0:
        raise ValueError("Model did not return attentions")
    if layer_idx < 0 or layer_idx >= len(attentions):
        raise ValueError(f"attention_layer={layer_idx} out of range [0, {len(attentions)-1}]")
    return attentions[layer_idx][0].detach().float().cpu().numpy()


def _get_attention_maps_tl(
    model: Any,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> np.ndarray:
    if layer_idx < 0 or layer_idx >= int(model.cfg.n_layers):
        raise ValueError(f"attention_layer={layer_idx} out of range [0, {int(model.cfg.n_layers)-1}]")
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, return_type="logits")
    key = f"blocks.{layer_idx}.attn.hook_pattern"
    try:
        pattern = cache[key]
    except KeyError:
        # Compatibility fallback for older cache key access patterns.
        pattern = cache["pattern", layer_idx]
    return pattern[0].detach().float().cpu().numpy()


def get_attention_maps(
    bundle: BackendBundle,
    encoded: dict[str, torch.Tensor],
    layer_idx: int,
) -> np.ndarray:
    if bundle.backend == "transformer_lens":
        return _get_attention_maps_tl(bundle.model, encoded["input_ids"], layer_idx)
    return _get_attention_maps_hf(bundle.model, encoded, layer_idx)


def get_all_attention_maps(
    bundle: BackendBundle,
    encoded: dict[str, torch.Tensor],
) -> dict[int, np.ndarray]:
    """Run the model ONCE and return attention maps for every layer.

    Using this instead of calling get_attention_maps() per layer avoids
    re-running the full forward pass N times (once per layer).
    Returns {layer_idx: np.ndarray[n_heads, seq, seq]}.
    """
    if bundle.backend == "transformer_lens":
        model = bundle.model
        n_layers = int(model.cfg.n_layers)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                encoded["input_ids"], return_type="logits"
            )
        result: dict[int, np.ndarray] = {}
        for l in range(n_layers):
            key = f"blocks.{l}.attn.hook_pattern"
            try:
                pattern = cache[key]
            except KeyError:
                pattern = cache["pattern", l]
            result[l] = pattern[0].detach().float().cpu().numpy()
        return result
    else:
        with torch.no_grad():
            out = bundle.model(**encoded, output_attentions=True, use_cache=False)
        attentions = out.attentions
        if attentions is None:
            raise ValueError("Model did not return attentions")
        return {
            l: attentions[l][0].detach().float().cpu().numpy()
            for l in range(len(attentions))
        }


def _get_residual_attribution_hf(
    model: AutoModelForCausalLM,
    encoded: dict[str, torch.Tensor],
    target_mode: str,
) -> tuple[np.ndarray, int, int]:
    model.zero_grad(set_to_none=True)
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states
    if hidden_states is None or len(hidden_states) < 2:
        raise ValueError("Model did not return hidden states")

    logits = out.logits[0]
    seq_len = int(logits.shape[0])
    if seq_len < 2:
        raise ValueError("Text too short after tokenization for attribution plot")

    pred_position = seq_len - 2
    if target_mode == "observed_next_token":
        target_token_id = int(encoded["input_ids"][0, pred_position + 1].item())
    else:
        target_token_id = int(torch.argmax(logits[pred_position]).item())

    objective = logits[pred_position, target_token_id]
    layer_states = list(hidden_states[1:])
    grads = torch.autograd.grad(
        outputs=objective,
        inputs=layer_states,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    attr = np.zeros((len(layer_states), seq_len), dtype=np.float32)
    for layer_idx, (state, grad) in enumerate(zip(layer_states, grads)):
        if grad is None:
            continue
        gxa = (state[0] * grad[0]).sum(dim=-1).detach().float().cpu().numpy()
        attr[layer_idx, :] = gxa.astype(np.float32, copy=False)

    denom = float(np.max(np.abs(attr)))
    if denom > 0:
        attr = attr / denom
    return attr, target_token_id, pred_position


def _get_residual_attribution_tl(
    model: Any,
    input_ids: torch.Tensor,
    target_mode: str,
) -> tuple[np.ndarray, int, int]:
    seq_len = int(input_ids.shape[1])
    if seq_len < 2:
        raise ValueError("Text too short after tokenization for attribution plot")

    resid_by_layer: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def _hook(resid: torch.Tensor, hook):
            resid.retain_grad()
            resid_by_layer[layer_idx] = resid
            return resid

        return _hook

    fwd_hooks = [(f"blocks.{layer_idx}.hook_resid_post", make_hook(layer_idx)) for layer_idx in range(int(model.cfg.n_layers))]

    model.zero_grad(set_to_none=True)
    logits = model.run_with_hooks(input_ids, return_type="logits", fwd_hooks=fwd_hooks)

    pred_position = seq_len - 2
    if target_mode == "observed_next_token":
        target_token_id = int(input_ids[0, pred_position + 1].item())
    else:
        target_token_id = int(torch.argmax(logits[0, pred_position]).item())

    objective = logits[0, pred_position, target_token_id]
    objective.backward()

    attr = np.zeros((int(model.cfg.n_layers), seq_len), dtype=np.float32)
    for layer_idx in range(int(model.cfg.n_layers)):
        resid = resid_by_layer.get(layer_idx)
        if resid is None or resid.grad is None:
            continue
        gxa = (resid.grad[0] * resid[0]).sum(dim=-1).detach().float().cpu().numpy()
        attr[layer_idx, :] = gxa.astype(np.float32, copy=False)

    denom = float(np.max(np.abs(attr)))
    if denom > 0:
        attr = attr / denom
    return attr, target_token_id, pred_position


def get_residual_attribution(
    bundle: BackendBundle,
    encoded: dict[str, torch.Tensor],
    target_mode: str,
) -> tuple[np.ndarray, int, int]:
    if bundle.backend == "transformer_lens":
        return _get_residual_attribution_tl(bundle.model, encoded["input_ids"], target_mode)
    return _get_residual_attribution_hf(bundle.model, encoded, target_mode)


# ── Logit lens ───────────────────────────────────────────────────────────────

def compute_tl_extras(
    model: Any,
    input_ids: torch.Tensor,
    condition_neurons: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Single no-grad forward pass collecting logit-lens, DLA, and optional
    neuron activations for a specific set of (layer, neuron) pairs.

    Returns a dict with keys:
      ll_tokens   [n_layers, seq]  int32  — top-1 token id at each (layer, pos)
      ll_probs    [n_layers, seq]  float32 — top-1 probability
      mlp_contribs  [n_layers]    float32 — MLP logit contribution at pred pos
      attn_contribs [n_layers]    float32 — Attn logit contribution at pred pos
      target_token_id  int
      pred_pos         int
      neuron_acts  dict{(layer,neuron): [seq] float32}
    """
    n_layers = int(model.cfg.n_layers)
    seq_len  = int(input_ids.shape[1])

    needed: set[str] = set()
    needed.update(f"blocks.{l}.hook_resid_post" for l in range(n_layers))
    needed.update(tl_utils.get_act_name("mlp_out", l)  for l in range(n_layers))
    needed.update(tl_utils.get_act_name("attn_out", l) for l in range(n_layers))
    if condition_neurons:
        layers_needed = {l for l, _ in condition_neurons}
        needed.update(tl_utils.get_act_name("post", l) for l in layers_needed)

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            input_ids, return_type="logits",
            names_filter=lambda n: n in needed,
        )

    # ── logit lens ──
    ll_tokens = np.zeros((n_layers, seq_len), dtype=np.int32)
    ll_probs  = np.zeros((n_layers, seq_len), dtype=np.float32)
    for layer in range(n_layers):
        resid  = cache[f"blocks.{layer}.hook_resid_post"]   # [1, seq, d_model]
        normed = model.ln_final(resid)
        lgt    = model.unembed(normed)                       # [1, seq, vocab]
        probs  = torch.softmax(lgt[0], dim=-1)
        top_p, top_id = probs.max(dim=-1)
        ll_tokens[layer] = top_id.cpu().numpy()
        ll_probs[layer]  = top_p.detach().cpu().numpy()

    # ── direct logit attribution ──
    pred_pos        = max(0, seq_len - 2)
    target_token_id = int(logits[0, pred_pos, :].argmax().item())
    W_U             = model.unembed.W_U                      # [d_model, vocab]
    target_dir      = W_U[:, target_token_id]                # [d_model]

    mlp_contribs  = np.zeros(n_layers, dtype=np.float32)
    attn_contribs = np.zeros(n_layers, dtype=np.float32)
    for layer in range(n_layers):
        mk = tl_utils.get_act_name("mlp_out", layer)
        ak = tl_utils.get_act_name("attn_out", layer)
        if mk in cache:
            mlp_contribs[layer]  = float((cache[mk][0, pred_pos]  @ target_dir).item())
        if ak in cache:
            attn_contribs[layer] = float((cache[ak][0, pred_pos]  @ target_dir).item())

    # ── condition-specific neuron activations ──
    neuron_acts: dict[tuple[int, int], np.ndarray] = {}
    if condition_neurons:
        for layer, neuron_id in condition_neurons:
            key = tl_utils.get_act_name("post", layer)
            if key in cache:
                acts = cache[key][0, :, neuron_id].detach().float().cpu().numpy()
                neuron_acts[(layer, neuron_id)] = acts

    return {
        "ll_tokens":       ll_tokens,
        "ll_probs":        ll_probs,
        "mlp_contribs":    mlp_contribs,
        "attn_contribs":   attn_contribs,
        "target_token_id": target_token_id,
        "pred_pos":        pred_pos,
        "neuron_acts":     neuron_acts,
    }


def compute_scalar_activations(model: Any, input_ids: torch.Tensor) -> np.ndarray:
    """Return a [n_layers, seq_len] float32 array of mean-absolute MLP post-activations.

    For each (layer, position) cell we take the MLP post-nonlinearity vector
    (shape [d_mlp]) and reduce it to a single scalar via mean(|activations|).
    This gives a raw magnitude picture of where the MLP is most active, with
    no gradient or attribution computation required.
    """
    n_layers = int(model.cfg.n_layers)
    seq_len  = int(input_ids.shape[1])
    needed   = {tl_utils.get_act_name("post", l) for l in range(n_layers)}

    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, return_type=None,
            names_filter=lambda n: n in needed,
        )

    out = np.zeros((n_layers, seq_len), dtype=np.float32)
    for layer in range(n_layers):
        key = tl_utils.get_act_name("post", layer)
        if key in cache:
            # cache[key]: [1, seq, d_mlp]
            out[layer] = cache[key][0].abs().mean(dim=-1).cpu().numpy()
    return out


def load_condition_neurons(results_dir: Path, condition: str) -> list[tuple[int, int]]:
    """Load consistent neurons for a condition from consistent_neurons.csv.gz."""
    path = results_dir / "consistent_neurons.csv.gz"
    if not path.exists():
        return []
    df = pd.read_csv(path, compression="gzip")
    sub = df[(df["condition"] == condition) & df["passes_consistency"]]
    return [(int(r["layer"]), int(r["neuron"])) for _, r in sub.iterrows()]


# ── plot: logit lens ──────────────────────────────────────────────────────────

def plot_logit_lens(
    ll_tokens: np.ndarray,
    ll_probs:  np.ndarray,
    tokenizer: Any,
    title: str,
    out_path: Path,
) -> None:
    """Heatmap: rows=layer, cols=position, cell text=predicted token, colour=probability.

    Shows what the model is predicting at each layer before the final output —
    a standard mechanistic-interpretability diagnostic for tracing how
    predictions build up across layers.
    """
    n_layers, seq_len = ll_tokens.shape
    text_grid = [
        [normalize_token(tokenizer.decode([int(ll_tokens[l, p])]))[:8]
         for p in range(seq_len)]
        for l in range(n_layers)
    ]
    fig = go.Figure(go.Heatmap(
        z=ll_probs,
        text=text_grid,
        texttemplate="%{text}",
        textfont=dict(size=7),
        x=[str(p) for p in range(seq_len)],
        y=[str(l) for l in range(n_layers)],
        colorscale="Blues",
        zmin=0.0, zmax=1.0,
        colorbar=dict(title="Top-1<br>prob"),
    ))
    fig.update_layout(
        title=f"Logit Lens — {title}",
        xaxis_title="Token position",
        yaxis_title="Layer",
        width=max(900, seq_len * 22),
        height=max(400, n_layers * 38),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=40, t=70, b=50),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


# ── plot: direct logit attribution ───────────────────────────────────────────

def plot_dla(
    mlp_contribs:  np.ndarray,
    attn_contribs: np.ndarray,
    target_token:  str,
    title: str,
    out_path: Path,
) -> None:
    """Grouped bar chart: per-layer MLP vs attention contribution to the
    predicted token's logit.  Positive = boosting the prediction,
    negative = suppressing it.
    """
    n_layers = len(mlp_contribs)
    layers   = list(range(n_layers))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MLP",
        x=layers,
        y=mlp_contribs.tolist(),
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        name="Attention",
        x=layers,
        y=attn_contribs.tolist(),
        marker_color="salmon",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    fig.update_layout(
        barmode="group",
        title=f"Direct Logit Attribution → '{target_token}' | {title}",
        xaxis_title="Layer",
        yaxis_title="Logit contribution",
        xaxis=dict(tickmode="linear", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=max(800, n_layers * 60),
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=40, t=80, b=50),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


# ── plot: neuron activation heatmap ──────────────────────────────────────────

def plot_neuron_activation_heatmap(
    neuron_acts:       dict[tuple[int, int], np.ndarray],
    condition_neurons: list[tuple[int, int]],
    token_labels:      list[str],
    title: str,
    out_path: Path,
    n_layers: int | None = None,
) -> None:
    """Heatmap of condition-specific neuron activations.

    Y-axis  = all layers 0..n_layers-1 (rows with no condition-specific neuron are blank)
    X-axis  = neuron ID
    Colour  = mean activation of that neuron across all token positions (RdBu, centred at 0)

    Each cell represents one (layer, neuron) pair. Cells with no condition-specific
    neuron for that combination are left as NaN and rendered white.
    """
    plotable = [(l, n) for l, n in sorted(condition_neurons) if (l, n) in neuron_acts]
    if not plotable:
        return

    # All layers 0..n_layers-1; fall back to max observed layer if n_layers not given
    max_layer = max(l for l, _ in plotable)
    layers  = list(range(n_layers if n_layers is not None else max_layer + 1))
    neurons = sorted({n for _, n in plotable})
    layer_idx  = {l: i for i, l in enumerate(layers)}
    neuron_idx = {n: i for i, n in enumerate(neurons)}

    # Build grid: rows=layers, cols=neurons, fill NaN by default
    z = np.full((len(layers), len(neurons)), np.nan, dtype=np.float32)
    for layer, nid in plotable:
        acts = neuron_acts[(layer, nid)]
        # Scalar: mean activation across all token positions
        z[layer_idx[layer], neuron_idx[nid]] = float(np.mean(acts))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(n) for n in neurons],
        y=[f"Layer {l}" for l in layers],
        colorscale="RdBu",
        zmid=0.0,
        colorbar=dict(title="Mean activation"),
    ))
    fig.update_layout(
        title=f"Condition-Specific Neuron Activations | {title}",
        xaxis_title="Neuron ID",
        yaxis_title="Layer",
        yaxis=dict(autorange="reversed"),
        width=max(900, len(neurons) * 30 + 120),
        height=max(300, len(layers) * 50 + 150),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=80),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_scalar_activation_heatmap(
    scalar_acts: np.ndarray,
    token_labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Heatmap of raw MLP activation magnitudes.

    Rows    = transformer layers (0 … n_layers-1)
    Columns = token positions
    Colour  = mean |MLP post-activation| at that (layer, position)
              — always ≥ 0, so colourscale is sequential (Viridis).

    Unlike residual attribution there is no red/blue: this simply shows
    *where* in the network neurons are firing strongly, with no causal
    direction information.
    """
    n_layers, n_pos = scalar_acts.shape
    use = min(n_pos, len(token_labels))
    fig = go.Figure(go.Heatmap(
        z=scalar_acts[:, :use],
        x=token_labels[:use],
        y=[f"Layer {l}" for l in range(n_layers)],
        colorscale="Viridis",
        colorbar=dict(title="Mean |activation|"),
    ))
    fig.update_layout(
        title=f"Scalar MLP Activation Magnitude | {title}",
        xaxis_title="Token position",
        yaxis_title="Layer",
        yaxis=dict(autorange="reversed"),
        width=max(900, use * 22),
        height=max(300, n_layers * 40 + 120),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=80, r=40, t=70, b=80),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def auto_focus_head(attn: np.ndarray) -> int:
    # Pick the head with strongest non-diagonal mass to surface non-trivial routing.
    off_diag = np.tril(np.ones(attn.shape[1:], dtype=np.float32), k=-1)
    scores = (attn * off_diag[None, :, :]).sum(axis=(1, 2))
    return int(np.argmax(scores))


def build_token_labels(tokenizer: AutoTokenizer, input_ids: list[int], with_index: bool = False) -> list[str]:
    labels = []
    for idx, token_id in enumerate(input_ids):
        token = normalize_token(tokenizer.decode([int(token_id)]))
        if with_index:
            labels.append(f"{token}_{idx}")
        else:
            labels.append(token)
    return labels


def plot_attention_head_grid(attn: np.ndarray, layer_idx: int, title: str, out_path: Path) -> None:
    n_heads = int(attn.shape[0])
    cells = n_heads + 1
    cols = min(6, cells)
    rows = int(math.ceil(cells / cols))
    subplot_titles = ["Mean"] + [f"Head {h}" for h in range(n_heads)]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, horizontal_spacing=0.02, vertical_spacing=0.07)

    panel_data = [attn.mean(axis=0)] + [attn[h] for h in range(n_heads)]
    for i, panel in enumerate(panel_data, start=1):
        r = (i - 1) // cols + 1
        c = (i - 1) % cols + 1
        fig.add_trace(
            go.Heatmap(
                z=panel,
                colorscale="YlGnBu",
                zmin=0.0,
                zmax=1.0,
                showscale=False,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, row=r, col=c)

    fig.update_layout(
        title=f"Layer {layer_idx} Attention Patterns: {title}",
        height=max(240, rows * 220),
        width=320 * cols,
        paper_bgcolor="#2f2f2f",
        plot_bgcolor="#2f2f2f",
        font=dict(color="white"),
        margin=dict(l=30, r=30, t=60, b=20),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_single_head(attn_head: np.ndarray, token_labels: list[str], layer_idx: int, head_idx: int, title: str, out_path: Path) -> None:
    n = len(token_labels)
    annotate = n <= 22
    text_vals = np.round(attn_head, 2).astype(str) if annotate else None
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=attn_head,
                x=token_labels,
                y=token_labels,
                colorscale="Blues",
                zmin=0.0,
                zmax=1.0,
                text=text_vals,
                texttemplate="%{text}" if annotate else None,
                textfont=dict(size=10),
                colorbar=dict(title="Attention"),
            )
        ]
    )
    fig.update_layout(
        title=f"Layer {layer_idx} Head {head_idx} Attention Pattern: {title}",
        xaxis_title="Attention to tokens",
        yaxis_title="Tokens",
        width=max(850, 34 * n),
        height=max(700, 30 * n),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=40, t=70, b=50),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_residual_attribution(
    attr: np.ndarray,
    token_labels: list[str],
    target_token: str,
    target_position: int,
    title: str,
    out_path: Path,
) -> None:
    n_layers, n_pos = attr.shape
    x_labels = token_labels[:n_pos]
    y_labels = [str(i) for i in range(n_layers)]
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=attr,
                x=x_labels,
                y=y_labels,
                colorscale="RdBu",
                zmid=0.0,
                colorbar=dict(title="Normalized\nattribution"),
            )
        ]
    )
    fig.update_layout(
        title=(
            "Normalized Residual-Stream Logit Attribution "
            f"(target='{target_token}', pred_position={target_position})<br>{title}"
        ),
        xaxis_title="Position",
        yaxis_title="Layer",
        width=max(950, 30 * n_pos),
        height=max(520, 30 * n_layers),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=70, r=40, t=85, b=70),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_overall_importance_heatmap(results_dir: Path, out_path: Path) -> None:
    imp_path = results_dir / "neuron_importance_condition_domain.csv.gz"
    if not imp_path.exists():
        return
    df = pd.read_csv(imp_path)
    if df.empty:
        return
    agg = (
        df.groupby(["condition", "layer"], as_index=False)["importance_score"]
        .mean()
        .sort_values(["condition", "layer"])
    )
    pivot = agg.pivot(index="layer", columns="condition", values="importance_score").fillna(0.0)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=pivot.to_numpy(),
                x=[str(c) for c in pivot.columns],
                y=[int(v) for v in pivot.index],
                colorscale="YlOrRd",
                colorbar=dict(title="Mean importance"),
            )
        ]
    )
    fig.update_layout(
        title="Overall New-Compute Importance by Layer and Condition",
        xaxis_title="Condition",
        yaxis_title="Layer",
        width=max(800, 120 * len(pivot.columns)),
        height=max(500, 30 * len(pivot.index)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=70, r=40, t=70, b=60),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def aggregate_attention(mats: list[np.ndarray], max_positions: int) -> np.ndarray:
    if not mats:
        raise ValueError("No attention matrices to aggregate")
    n_heads = mats[0].shape[0]
    max_pos = min(max_positions, max(mat.shape[1] for mat in mats))
    sum_mat = np.zeros((n_heads, max_pos, max_pos), dtype=np.float64)
    cnt = np.zeros((max_pos, max_pos), dtype=np.float64)
    for mat in mats:
        n = min(max_pos, mat.shape[1])
        sum_mat[:, :n, :n] += mat[:, :n, :n]
        cnt[:n, :n] += 1.0
    mean = np.divide(sum_mat, cnt[None, :, :], out=np.zeros_like(sum_mat), where=cnt[None, :, :] > 0)
    return mean.astype(np.float32, copy=False)


def aggregate_attr(mats: list[np.ndarray], max_positions: int) -> np.ndarray:
    if not mats:
        raise ValueError("No attribution matrices to aggregate")
    n_layers = mats[0].shape[0]
    max_pos = min(max_positions, max(mat.shape[1] for mat in mats))
    sum_mat = np.zeros((n_layers, max_pos), dtype=np.float64)
    cnt = np.zeros((max_pos,), dtype=np.float64)
    for mat in mats:
        n = min(max_pos, mat.shape[1])
        sum_mat[:, :n] += mat[:, :n]
        cnt[:n] += 1.0
    mean = np.divide(sum_mat, cnt[None, :], out=np.zeros_like(sum_mat), where=cnt[None, :] > 0)
    denom = float(np.max(np.abs(mean)))
    if denom > 0:
        mean = mean / denom
    return mean.astype(np.float32, copy=False)


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _run_sample(
    bundle: BackendBundle,
    row: Any,
    sample_num: int,
    cond_dir: Path,
    layers_to_plot: list[int],
    focus_head_arg: int,
    target_mode: str,
    max_length: int,
    with_logit_lens: bool = False,
    with_dla: bool = False,
    with_scalar_activation: bool = False,
    condition_neurons: list[tuple[int, int]] | None = None,
) -> dict[str, object] | None:
    """Compute and write plots for a single sample row.

    Runs the model once to collect attention maps for all requested layers,
    then writes one attention-grid + one focused-head plot per layer, plus a
    single residual-attribution plot.  Returns a manifest entry dict, or None
    if the row is too short to plot.
    """
    sample_id = str(row["id"])
    text = str(row["text"])
    condition = str(row["condition"])

    encoded = tokenize_inputs(bundle.tokenizer, text, max_length, bundle.device)
    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    if len(input_ids) < 2:
        return None

    token_labels_plain = build_token_labels(bundle.tokenizer, input_ids, with_index=False)
    token_labels_indexed = build_token_labels(bundle.tokenizer, input_ids, with_index=True)

    # Single forward pass for all layers.
    all_attn = get_all_attention_maps(bundle, encoded)

    attr, target_token_id, target_position = get_residual_attribution(bundle, encoded, target_mode)
    target_token_text = normalize_token(bundle.tokenizer.decode([target_token_id]))

    safe_id = _safe_name(sample_id)[:60]
    prefix = f"{sample_num:02d}_{safe_id}"
    title = f"id={sample_id} | condition={condition}"
    multi = len(layers_to_plot) > 1

    # Residual attribution — one per sample regardless of layer count.
    attr_path = cond_dir / f"{prefix}_residual_attribution.html"
    plot_residual_attribution(attr, token_labels_indexed, target_token_text, target_position, title, attr_path)

    # ── TL-only extras: logit lens, DLA, activation heatmap ──────────────────
    want_extras = bundle.backend == "transformer_lens" and (
        with_logit_lens or with_dla or bool(condition_neurons)
    )
    if want_extras:
        extras = compute_tl_extras(bundle.model, encoded["input_ids"], condition_neurons)
        tgt_text = normalize_token(bundle.tokenizer.decode([extras["target_token_id"]]))
        if with_logit_lens:
            plot_logit_lens(
                extras["ll_tokens"], extras["ll_probs"],
                bundle.tokenizer,
                title,
                cond_dir / f"{prefix}_logit_lens.html",
            )
        if with_dla:
            plot_dla(
                extras["mlp_contribs"], extras["attn_contribs"],
                tgt_text, title,
                cond_dir / f"{prefix}_dla.html",
            )
        if condition_neurons and extras["neuron_acts"]:
            plot_neuron_activation_heatmap(
                extras["neuron_acts"], condition_neurons,
                token_labels_plain, title,
                cond_dir / f"{prefix}_neuron_heatmap.html",
                n_layers=bundle.n_layers,
            )

    # ── Scalar MLP activation magnitude heatmap (no gradients needed) ─────────
    if bundle.backend == "transformer_lens" and with_scalar_activation:
        scalar_acts = compute_scalar_activations(bundle.model, encoded["input_ids"])
        plot_scalar_activation_heatmap(
            scalar_acts, token_labels_plain, title,
            cond_dir / f"{prefix}_scalar_activation.html",
        )

    layer_files: dict[int, dict[str, str]] = {}
    for l in layers_to_plot:
        attn = all_attn.get(l)
        if attn is None:
            continue
        focus_head = focus_head_arg if focus_head_arg >= 0 else auto_focus_head(attn)
        l_suffix = f"_layer{l}" if multi else ""
        grid_path = cond_dir / f"{prefix}_attention_grid{l_suffix}.html"
        head_path = cond_dir / f"{prefix}_attention_head{l_suffix}_{focus_head}.html"
        plot_attention_head_grid(attn, l, title, grid_path)
        plot_single_head(attn[focus_head], token_labels_plain, l, focus_head, title, head_path)
        layer_files[l] = {
            "attention_grid": str(grid_path),
            "single_head": str(head_path),
            "focus_head": int(focus_head),
        }

    # Flat backward-compat keys pointing at the first layer's files.
    first = next(iter(layer_files.values())) if layer_files else {}
    return {
        "sample_num": sample_num,
        "id": sample_id,
        "condition": condition,
        "text_preview": text[:180],
        "n_tokens": len(input_ids),
        "residual_attribution": str(attr_path),
        "layers": layer_files,
        "attention_grid": first.get("attention_grid", ""),
        "single_head": first.get("single_head", ""),
        "focus_head": first.get("focus_head", 0),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = load_dataset(Path(args.dataset_csv))
    if df.empty:
        raise ValueError("Dataset is empty")

    all_conditions = sorted(df["condition"].astype(str).unique().tolist())
    target_conditions = args.conditions if args.conditions else all_conditions
    # Validate any user-specified conditions against what's in the data.
    unknown = set(target_conditions) - set(all_conditions)
    if unknown:
        raise ValueError(f"Conditions not found in dataset: {sorted(unknown)}")

    model_name = resolve_model_name(run_dir, args.model_name)
    device = resolve_device(args.device)
    bundle = load_backend(model_name=model_name, backend=args.backend, device=device)

    layers_to_plot: list[int] = (
        list(range(bundle.n_layers)) if args.all_attention_layers else [args.attention_layer]
    )
    print(f"Attention layers to visualise: {layers_to_plot}")

    by_condition: dict[str, dict[str, object]] = {}

    for cond in target_conditions:
        safe_cond = _safe_name(cond)
        cond_dir = out_dir / safe_cond
        samples_dir = cond_dir / "samples"
        aggregate_dir = cond_dir / "aggregate"
        samples_dir.mkdir(parents=True, exist_ok=True)
        aggregate_dir.mkdir(parents=True, exist_ok=True)

        cond_df = df[df["condition"].astype(str) == cond].copy().reset_index(drop=True)
        print(f"\n[{cond}]  {len(cond_df)} rows total")

        # Load condition-specific consistent neurons (for activation heatmap).
        cond_neurons: list[tuple[int, int]] | None = None
        if args.with_activation_heatmap and args.results_dir:
            cond_neurons = load_condition_neurons(Path(args.results_dir), cond)
            print(f"  condition-specific neurons loaded: {len(cond_neurons)}")

        # ── Per-sample visuals ────────────────────────────────────────────────
        n_take = min(args.n_random_samples, len(cond_df))
        sample_indices = rng.sample(range(len(cond_df)), k=n_take)
        sample_entries: list[dict[str, object]] = []

        for sample_num, idx in enumerate(sample_indices, start=1):
            row = cond_df.iloc[idx]
            try:
                entry = _run_sample(
                    bundle=bundle,
                    row=row,
                    sample_num=sample_num,
                    cond_dir=samples_dir,
                    layers_to_plot=layers_to_plot,
                    focus_head_arg=args.focus_head,
                    target_mode=args.target_mode,
                    max_length=args.max_length,
                    with_logit_lens=args.with_logit_lens,
                    with_dla=args.with_dla,
                    with_scalar_activation=args.with_scalar_activation,
                    condition_neurons=cond_neurons,
                )
                if entry:
                    sample_entries.append(entry)
                    print(f"  sample {sample_num}/{n_take}  id={row['id']}  tokens={entry['n_tokens']}")
            except Exception as exc:
                print(f"  sample {sample_num}/{n_take}  SKIPPED  ({exc})")

        # ── Aggregate visuals ─────────────────────────────────────────────────
        # Cap the number of rows used for aggregation if requested.
        if int(args.overall_sample_cap) > 0 and len(cond_df) > int(args.overall_sample_cap):
            agg_indices = rng.sample(range(len(cond_df)), k=int(args.overall_sample_cap))
            agg_df = cond_df.iloc[agg_indices].reset_index(drop=True)
        else:
            agg_df = cond_df

        # Accumulate per-layer attention mats and attribution mats separately.
        agg_attn_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in layers_to_plot}
        agg_attr_mats: list[np.ndarray] = []

        for _, row in agg_df.iterrows():
            try:
                encoded = tokenize_inputs(bundle.tokenizer, str(row["text"]), args.max_length, bundle.device)
                all_attn = get_all_attention_maps(bundle, encoded)
                for l in layers_to_plot:
                    if l in all_attn:
                        agg_attn_by_layer[l].append(all_attn[l])
                attr, _, _ = get_residual_attribution(bundle, encoded, args.target_mode)
                agg_attr_mats.append(attr)
            except Exception:
                continue

        n_agg_used = max((len(v) for v in agg_attn_by_layer.values()), default=0)
        print(f"  aggregate: {n_agg_used}/{len(agg_df)} rows used")

        multi = len(layers_to_plot) > 1
        for l in layers_to_plot:
            mats = agg_attn_by_layer[l]
            if not mats:
                continue
            mean_attn = aggregate_attention(mats, args.overall_max_positions)
            agg_focus_head = args.focus_head if args.focus_head >= 0 else auto_focus_head(mean_attn)
            pos_labels = [f"pos_{i}" for i in range(mean_attn.shape[-1])]
            agg_title = f"condition={cond} | layer {l} | mean over {len(mats)} samples"
            l_suffix = f"_layer{l}" if multi else ""

            plot_attention_head_grid(
                mean_attn, l, agg_title,
                aggregate_dir / f"attention_grid{l_suffix}.html",
            )
            plot_single_head(
                mean_attn[agg_focus_head], pos_labels,
                l, agg_focus_head, agg_title,
                aggregate_dir / f"attention_head{l_suffix}_{agg_focus_head}.html",
            )

        if agg_attr_mats:
            mean_attr = aggregate_attr(agg_attr_mats, args.overall_max_positions)
            pos_labels = [f"pos_{i}" for i in range(mean_attr.shape[1])]
            plot_residual_attribution(
                mean_attr, pos_labels,
                target_token="aggregate", target_position=-1,
                title=f"condition={cond} | mean over {len(agg_attr_mats)} samples",
                out_path=aggregate_dir / "residual_attribution.html",
            )

        # ── TL-only aggregate extras ──────────────────────────────────────────
        if bundle.backend == "transformer_lens":
            agg_ll_probs_list:    list[np.ndarray] = []
            agg_mlp_list:         list[np.ndarray] = []
            agg_attn_list:        list[np.ndarray] = []
            agg_neuron_acts_list: list[dict]       = []
            agg_scalar_acts_list: list[np.ndarray] = []

            want_extras_agg = (
                args.with_logit_lens or args.with_dla
                or bool(cond_neurons) or args.with_scalar_activation
            )
            if want_extras_agg:
                n_extras_ok = 0
                for _, row in agg_df.iterrows():
                    try:
                        enc = tokenize_inputs(bundle.tokenizer, str(row["text"]),
                                              args.max_length, bundle.device)
                        ex  = compute_tl_extras(bundle.model, enc["input_ids"], cond_neurons)
                        agg_ll_probs_list.append(ex["ll_probs"])
                        agg_mlp_list.append(ex["mlp_contribs"])
                        agg_attn_list.append(ex["attn_contribs"])
                        if ex["neuron_acts"]:
                            agg_neuron_acts_list.append(ex["neuron_acts"])
                        if args.with_scalar_activation:
                            sa = compute_scalar_activations(bundle.model, enc["input_ids"])
                            agg_scalar_acts_list.append(sa)
                        n_extras_ok += 1
                    except Exception:
                        continue
                print(f"  aggregate extras: {n_extras_ok}/{len(agg_df)} rows used")

                agg_title = f"condition={cond} | mean over {n_extras_ok} samples"
                pos_labels_agg = [f"pos_{i}" for i in range(args.overall_max_positions)]

                if args.with_logit_lens and agg_ll_probs_list:
                    # Truncate/pad all to overall_max_positions then average
                    max_p = args.overall_max_positions
                    padded = []
                    for m in agg_ll_probs_list:
                        p = m[:, :max_p]
                        if p.shape[1] < max_p:
                            pad = np.zeros((p.shape[0], max_p - p.shape[1]), dtype=np.float32)
                            p = np.concatenate([p, pad], axis=1)
                        padded.append(p)
                    mean_ll_probs  = np.mean(padded, axis=0)
                    # Use zeros for token text in aggregate (no single token to show)
                    dummy_tokens   = np.zeros_like(mean_ll_probs, dtype=np.int32)
                    plot_logit_lens(
                        dummy_tokens, mean_ll_probs,
                        bundle.tokenizer, agg_title,
                        aggregate_dir / "logit_lens.html",
                    )

                if args.with_dla and agg_mlp_list:
                    mean_mlp  = np.mean(agg_mlp_list,  axis=0)
                    mean_attn = np.mean(agg_attn_list, axis=0)
                    plot_dla(mean_mlp, mean_attn, "aggregate", agg_title,
                             aggregate_dir / "dla.html")

                if cond_neurons and agg_neuron_acts_list:
                    # Average per-neuron activations across samples.
                    # Samples have different sequence lengths, so
                    # truncate/pad every vector to overall_max_positions first.
                    max_p = args.overall_max_positions
                    merged: dict[tuple[int, int], list[np.ndarray]] = {}
                    for na in agg_neuron_acts_list:
                        for key, acts in na.items():
                            a = acts[:max_p]
                            if len(a) < max_p:
                                a = np.pad(a, (0, max_p - len(a)))
                            merged.setdefault(key, []).append(a)
                    mean_acts = {k: np.mean(np.stack(vs), axis=0)
                                 for k, vs in merged.items()}
                    plot_neuron_activation_heatmap(
                        mean_acts, cond_neurons,
                        pos_labels_agg, agg_title,
                        aggregate_dir / "neuron_heatmap.html",
                        n_layers=bundle.n_layers,
                    )

                if args.with_scalar_activation and agg_scalar_acts_list:
                    # Truncate/pad each [n_layers, seq] mat to overall_max_positions then mean.
                    max_p = args.overall_max_positions
                    padded_sa = []
                    for sa in agg_scalar_acts_list:
                        p = sa[:, :max_p]
                        if p.shape[1] < max_p:
                            p = np.pad(p, ((0, 0), (0, max_p - p.shape[1])))
                        padded_sa.append(p)
                    mean_scalar = np.mean(padded_sa, axis=0)
                    plot_scalar_activation_heatmap(
                        mean_scalar, pos_labels_agg, agg_title,
                        aggregate_dir / "scalar_activation.html",
                    )

        by_condition[cond] = {
            "condition": cond,
            "total_rows": int(len(cond_df)),
            "samples_requested": int(n_take),
            "samples_plotted": int(len(sample_entries)),
            "aggregate_rows_used": int(n_agg_used),
            "aggregate_focus_head": None if agg_focus_head is None else int(agg_focus_head),
            "samples": sample_entries,
            "output_dir": str(cond_dir),
        }

    # Optional cross-condition importance heatmap.
    if args.results_dir:
        plot_overall_importance_heatmap(
            Path(args.results_dir),
            out_dir / "importance_by_layer_condition.html",
        )

    manifest = {
        "run_dir": str(run_dir),
        "dataset_csv": str(args.dataset_csv),
        "results_dir": str(args.results_dir) if args.results_dir else None,
        "model_name": model_name,
        "backend": args.backend,
        "device": str(device),
        "seed": int(args.seed),
        "attention_layers": layers_to_plot,
        "n_random_samples_per_condition": int(args.n_random_samples),
        "overall_sample_cap_per_condition": int(args.overall_sample_cap),
        "conditions": by_condition,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nOutputs written to: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
