#!/usr/bin/env python3
"""
causal_analysis.py
==================
Activation patching experiments to establish *causal* evidence for which neurons
drive code-switching vs language-confusion behaviour in the LLM.

The script builds on candidate neurons already identified by the main mechinterp
pipeline (stored in neuron_tendency.csv from a completed run), then runs two
activation-patching experiments:

  Experiment A  (CS → confused)
    Take samples labelled `confused`, patch the pre-down_proj MLP activations
    at the code-switching-specific neurons with mean values collected from
    `code_switched` samples.  If those neurons are causally responsible for
    CS behaviour, the model should process confused text *more like* code-switched
    text (lower NLL, shifted output distribution).

  Experiment B  (confused → CS)
    Reverse direction: patch code-switched samples with confusion-neuron values.
    If confusion neurons drive the confused-processing pathway, this should
    disrupt the model's handling of CS text (higher NLL, more entropy).

Usage
-----
  # From the project root:
  PYTHONPATH=src python scripts/causal_analysis.py \\
      --config config.yaml \\
      --source_run outputs/serious_run_french_002 \\
      --output_dir causal_results \\
      --top_k 20 \\
      --max_samples 50

Outputs (all in --output_dir)
------------------------------
  CAUSAL_SUMMARY.txt          Human-readable findings summary
  tables/
    cs_candidate_neurons.csv  Neurons specific to code_switched
    conf_candidate_neurons.csv Neurons specific to confused
    patching_effects.csv      Per-neuron delta metrics for both experiments
    circuit_effects.csv       Effect of patching ALL top-k neurons together
  figures/
    patching_effects.html     Interactive bar chart of per-neuron ΔNLL
  metadata.json               Run parameters and circuit-level numbers
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import plotly.express as px
from tqdm import tqdm

# Allow running as `python scripts/causal_analysis.py` without PYTHONPATH set.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.io_utils import load_config, load_dataset
from pipeline.modeling import prepare_model_and_lens

logger = logging.getLogger("causal_analysis")


@dataclass
class CircuitStats:
    metric: str
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float


# ---------------------------------------------------------------------------
# Step 1 – Identify candidate neurons from a completed pipeline run
# ---------------------------------------------------------------------------

def load_candidate_neurons(
    source_run: Path,
    top_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read neuron_tendency.csv from a completed pipeline run and find neurons
    that are uniquely active for *one* condition while silent in all others.

    Specificity score = activation_mean(target_condition)
                        - max(activation_mean across all other conditions)

    A high positive score means the neuron fires for that condition and barely
    fires (or is completely silent) everywhere else — exactly the candidate we
    want for a causal patching experiment.

    Returns
    -------
    cs_candidates   – top_k neurons most specific to code_switched
    conf_candidates – top_k neurons most specific to confused
    """
    tendency_path = source_run / "tables" / "neuron_tendency.csv"
    if not tendency_path.exists():
        raise FileNotFoundError(
            f"neuron_tendency.csv not found at {tendency_path}.\n"
            "Make sure --source_run points to a completed pipeline output directory."
        )

    nt = pd.read_csv(tendency_path)

    # Build a (layer, neuron) × condition activation table
    pivot = (
        nt.groupby(["layer", "neuron", "condition"])["activation_mean"]
        .mean()
        .unstack("condition")
        .fillna(0.0)
    )

    for cond in ["code_switched", "confused"]:
        if cond not in pivot.columns:
            raise ValueError(
                f"Condition '{cond}' missing from neuron_tendency.csv. "
                f"Found: {list(pivot.columns)}"
            )

    # Specificity = how much higher a neuron fires for one condition vs all others
    other_than_cs   = [c for c in pivot.columns if c != "code_switched"]
    other_than_conf = [c for c in pivot.columns if c != "confused"]

    pivot["cs_specificity"]   = pivot["code_switched"] - pivot[other_than_cs].max(axis=1)
    pivot["conf_specificity"] = pivot["confused"]       - pivot[other_than_conf].max(axis=1)

    cs_candidates   = pivot.nlargest(top_k, "cs_specificity").reset_index()
    conf_candidates = pivot.nlargest(top_k, "conf_specificity").reset_index()

    return cs_candidates, conf_candidates


# ---------------------------------------------------------------------------
# Step 2 – PyTorch hooks for capturing and patching MLP internals
# ---------------------------------------------------------------------------

def _find_down_proj(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module | None:
    """
    Return the down_proj linear layer for transformer block `layer_idx`.
    Works for LLaMA (model.layers), GPT-2 (transformer.h), and GPT-NeoX.

    The pre-down_proj activation — the tensor passed as input to this module —
    is `silu(gate_proj(x)) * up_proj(x)` in SwiGLU architectures.  Patching
    it is the cleanest place to intervene: it directly controls what flows
    through down_proj into the residual stream.
    """
    named = dict(model.named_modules())
    for prefix in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        key = f"{prefix}.{layer_idx}.mlp.down_proj"
        if key in named:
            return named[key]
    return None


@contextmanager
def capture_pre_down_proj(
    model: torch.nn.Module,
    layer_neuron_pairs: list[tuple[int, int]],
    storage: dict,
):
    """
    Context manager: registers forward pre-hooks on each relevant down_proj
    to record the mean absolute activation value at each requested neuron index.

    After the forward pass, storage[(layer, neuron)] contains a list of scalar
    values (one per sample processed inside this context).
    """
    handles = []
    by_layer: dict[int, list[int]] = {}
    for layer, neuron in layer_neuron_pairs:
        by_layer.setdefault(layer, []).append(neuron)

    for layer_idx, neuron_indices in by_layer.items():
        down_proj = _find_down_proj(model, layer_idx)
        if down_proj is None:
            logger.warning("Could not locate down_proj for layer %d — skipping", layer_idx)
            continue

        def make_hook(l_idx: int, n_indices: list[int]):
            def hook(module, args):
                # args[0]: shape [batch=1, seq_len, intermediate_size]
                act = args[0]
                for n_idx in n_indices:
                    val = act[0, :, n_idx].float().abs().mean().item()
                    storage.setdefault((l_idx, n_idx), []).append(val)
            return hook

        handle = down_proj.register_forward_pre_hook(make_hook(layer_idx, neuron_indices))
        handles.append(handle)

    try:
        yield storage
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_pre_down_proj(
    model: torch.nn.Module,
    patch_values: dict[tuple[int, int], float],
):
    """
    Context manager: registers forward pre-hooks that replace specific neuron
    activations in the pre-down_proj tensor with the given constant values.

    patch_values: {(layer_idx, neuron_idx): scalar_value}
    """
    handles = []
    by_layer: dict[int, dict[int, float]] = {}
    for (layer, neuron), val in patch_values.items():
        by_layer.setdefault(layer, {})[neuron] = val

    for layer_idx, neuron_vals in by_layer.items():
        down_proj = _find_down_proj(model, layer_idx)
        if down_proj is None:
            continue

        def make_patch_hook(n_vals: dict[int, float]):
            def hook(module, args):
                inp = list(args)
                inp[0] = inp[0].clone()
                for n_idx, val in n_vals.items():
                    # Broadcast the scalar across all token positions
                    inp[0][:, :, n_idx] = val
                return tuple(inp)
            return hook

        handle = down_proj.register_forward_pre_hook(make_patch_hook(neuron_vals))
        handles.append(handle)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Step 3 – Collect mean source activations from one condition
# ---------------------------------------------------------------------------

def collect_source_activations(
    model: torch.nn.Module,
    tokenizer,
    samples: pd.DataFrame,
    layer_neuron_pairs: list[tuple[int, int]],
    max_length: int,
    device: torch.device,
) -> dict[tuple[int, int], float]:
    """
    Run all samples from the source condition through the model and compute
    the mean absolute pre-down_proj activation at each (layer, neuron) pair.

    These mean values become the patch values injected into target samples.
    Using means (rather than per-sample values) mirrors the standard
    'mean activation patching' approach in the mech interp literature.

    Returns: {(layer, neuron): mean_activation_value}
    """
    raw: dict[tuple[int, int], list[float]] = {}

    for _, row in tqdm(
        samples.iterrows(), total=len(samples), desc="  Collecting source activations"
    ):
        encoded = tokenizer(
            str(row["text"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        if encoded["input_ids"].shape[1] < 3:
            continue

        with capture_pre_down_proj(model, layer_neuron_pairs, raw):
            with torch.no_grad():
                model(**encoded, use_cache=False)

    return {k: float(np.mean(v)) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Step 4 – Output metrics
# ---------------------------------------------------------------------------

def compute_output_metrics(
    model: torch.nn.Module,
    encoded: dict,
    device: torch.device,
) -> dict[str, float]:
    """
    Run a forward pass and return three scalars summarising the model's output:

    nll       – mean next-token negative log-likelihood.
                Lower = model is more confident in predicting the next token.
    entropy   – mean output entropy across token positions.
                Higher = model is less certain about what comes next.
    top1_prob – mean probability assigned to the top-1 predicted token.
                Higher = model is more confident.
    """
    with torch.no_grad():
        outputs = model(**encoded, use_cache=False)
        logits = outputs.logits.squeeze(0)          # [seq, vocab]

    log_probs   = F.log_softmax(logits, dim=-1)
    labels      = encoded["input_ids"].squeeze(0)
    next_labels = labels[1:]
    n           = min(log_probs.shape[0] - 1, next_labels.shape[0])

    nll      = float((-log_probs[:n].gather(1, next_labels[:n].unsqueeze(-1)).squeeze(-1)).mean())
    entropy  = float((-(log_probs[:n].exp() * log_probs[:n]).sum(dim=-1)).mean())
    top1     = float(log_probs[:n].exp().max(dim=-1).values.mean())

    return {"nll": nll, "entropy": entropy, "top1_prob": top1}


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) == 1 or n_bootstrap <= 0:
        return (mean, mean, mean)
    idx = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    boots = arr[idx].mean(axis=1)
    low = float(np.quantile(boots, alpha / 2))
    high = float(np.quantile(boots, 1 - alpha / 2))
    return (mean, low, high)


def sign_flip_permutation_pvalue(
    values: list[float],
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    observed = abs(float(arr.mean()))
    if len(arr) == 1 or n_permutations <= 0:
        return float(1.0 if observed == 0.0 else 0.0)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(arr)))
    perm_means = np.abs((arr * signs).mean(axis=1))
    p = (np.sum(perm_means >= observed) + 1) / (n_permutations + 1)
    return float(p)


def split_samples_train_eval(
    samples: pd.DataFrame,
    train_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(samples) <= 1:
        return samples.reset_index(drop=True), samples.reset_index(drop=True)
    if train_frac <= 0 or train_frac >= 1:
        raise ValueError(f"--train_frac must be in (0,1), got {train_frac}")
    train_n = max(1, min(len(samples) - 1, int(round(len(samples) * train_frac))))
    train = samples.sample(n=train_n, random_state=seed)
    eval_df = samples.drop(train.index)
    return train.reset_index(drop=True), eval_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5 – Patching experiments
# ---------------------------------------------------------------------------

def run_patching_experiment(
    model: torch.nn.Module,
    tokenizer,
    target_samples: pd.DataFrame,
    patch_values: dict[tuple[int, int], float],
    neuron_list: list[tuple[int, int]],
    max_length: int,
    device: torch.device,
    desc: str = "Patching",
    compute_individual: bool = True,
) -> dict[str, Any]:
    """
    For each target sample, measure three things:
      1. Baseline metrics (no patching)
      2. Circuit effect: patch ALL neurons in neuron_list simultaneously
      3. Individual effect: patch ONE neuron at a time → per-neuron causal score

    Returns
    -------
    {
      "circuit":    {"nll": Δ, "entropy": Δ, "top1_prob": Δ},   # mean across samples
      "individual": {(layer, neuron): {"nll": Δ, ...}, ...}      # mean across samples
    }
    """
    circuit_deltas: dict[str, list[float]]                             = {m: [] for m in ["nll", "entropy", "top1_prob"]}
    indiv_deltas: dict[tuple[int, int], dict[str, list[float]]] = {}
    if compute_individual:
        indiv_deltas = {
            (l, n): {m: [] for m in ["nll", "entropy", "top1_prob"]}
            for l, n in neuron_list
        }
    per_sample_circuit: list[dict[str, float]] = []

    for _, row in tqdm(target_samples.iterrows(), total=len(target_samples), desc=f"  {desc}"):
        encoded = tokenizer(
            str(row["text"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(device)

        if encoded["input_ids"].shape[1] < 3:
            continue

        # --- Baseline ---
        baseline = compute_output_metrics(model, encoded, device)

        # --- Circuit effect (patch everything at once) ---
        with patch_pre_down_proj(model, patch_values):
            circuit = compute_output_metrics(model, encoded, device)
        for m in ["nll", "entropy", "top1_prob"]:
            circuit_deltas[m].append(circuit[m] - baseline[m])
        per_sample_circuit.append(
            {
                "delta_nll": circuit["nll"] - baseline["nll"],
                "delta_entropy": circuit["entropy"] - baseline["entropy"],
                "delta_top1_prob": circuit["top1_prob"] - baseline["top1_prob"],
            }
        )

        # --- Individual neuron effects ---
        if compute_individual:
            for layer, neuron in neuron_list:
                if (layer, neuron) not in patch_values:
                    continue
                single = {(layer, neuron): patch_values[(layer, neuron)]}
                with patch_pre_down_proj(model, single):
                    patched = compute_output_metrics(model, encoded, device)
                for m in ["nll", "entropy", "top1_prob"]:
                    indiv_deltas[(layer, neuron)][m].append(patched[m] - baseline[m])

    circuit_summary = {
        m: float(np.mean(v)) for m, v in circuit_deltas.items() if v
    }
    individual_summary = {}
    if compute_individual:
        individual_summary = {
            (l, n): {m: float(np.mean(v)) for m, v in deltas.items() if v}
            for (l, n), deltas in indiv_deltas.items()
        }

    return {
        "circuit": circuit_summary,
        "individual": individual_summary,
        "circuit_per_sample": per_sample_circuit,
    }


# ---------------------------------------------------------------------------
# Step 6 – Format and save all results
# ---------------------------------------------------------------------------

def build_effects_dataframe(
    individual_results: dict[tuple[int, int], dict[str, float]],
    experiment_label: str,
) -> pd.DataFrame:
    rows = []
    for (layer, neuron), m in individual_results.items():
        rows.append({
            "experiment":    experiment_label,
            "layer":         layer,
            "neuron":        neuron,
            "layer_neuron":  f"L{layer}:N{neuron}",
            "delta_nll":     m.get("nll",      float("nan")),
            "delta_entropy": m.get("entropy",  float("nan")),
            "delta_top1":    m.get("top1_prob", float("nan")),
            "abs_delta_nll": abs(m.get("nll", 0.0)),
        })
    return pd.DataFrame(rows).sort_values("abs_delta_nll", ascending=False)


def plot_patching_effects(effects_df: pd.DataFrame, out_path: Path) -> None:
    if effects_df.empty:
        return
    fig = px.bar(
        effects_df,
        x="layer_neuron",
        y="delta_nll",
        color="experiment",
        barmode="group",
        title="Activation Patching Effect on Next-Token NLL (per neuron)<br>"
              "<sub>Negative ΔNLL = model shifted towards source condition (causal signal)</sub>",
        labels={
            "delta_nll":    "ΔNLL vs baseline",
            "layer_neuron": "Layer : Neuron",
            "experiment":   "Experiment",
        },
        color_discrete_map={
            "CS → confused":  "#2196F3",
            "confused → CS":  "#F44336",
        },
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=60, height=620)
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info("Saved patching effects figure → %s", out_path)


def write_summary(
    cs_candidates:     pd.DataFrame,
    conf_candidates:   pd.DataFrame,
    cs_to_conf:        dict,
    conf_to_cs:        dict,
    model_name:        str,
    source_run:        str,
    n_cs_samples:      int,
    n_conf_samples:    int,
    out_path:          Path,
) -> None:
    """Write a plain-text human-readable summary of every finding."""
    SEP  = "=" * 72
    THIN = "-" * 72

    def fmt_circuit(results: dict) -> list[str]:
        c = results["circuit"]
        lines = [
            f"  ΔNLL       = {c.get('nll',      0):+.4f}",
            f"  ΔEntropy   = {c.get('entropy',  0):+.4f}",
            f"  ΔTop1-prob = {c.get('top1_prob',0):+.4f}",
        ]
        return lines

    def fmt_individual(results: dict, n: int = 15) -> list[str]:
        ind = results["individual"]
        ranked = sorted(ind.items(), key=lambda x: abs(x[1].get("nll", 0)), reverse=True)
        lines = []
        for (layer, neuron), m in ranked[:n]:
            lines.append(
                f"  L{layer}:N{neuron:<5}  "
                f"ΔNLL={m.get('nll',0):+.5f}  "
                f"ΔEntropy={m.get('entropy',0):+.5f}  "
                f"ΔTop1={m.get('top1_prob',0):+.5f}"
            )
        return lines

    lines = [
        SEP,
        "  CAUSAL ANALYSIS — Code-Switching vs Language Confusion",
        SEP,
        f"  Model      : {model_name}",
        f"  Source run : {source_run}",
        f"  CS samples : {n_cs_samples}    Confused samples : {n_conf_samples}",
        "",
        "  HOW TO READ THIS FILE",
        "  ─────────────────────",
        "  Activation patching replaces specific MLP neuron values in one",
        "  condition's samples with the mean values recorded for another.",
        "",
        "  ΔNLL < 0  →  model became MORE confident after patching.",
        "              This is the causal signal: patching in code-switching",
        "              activations shifts the model's processing towards CS.",
        "",
        "  ΔNLL > 0  →  model became LESS confident (more confused) after",
        "              patching — injecting confusion activations disrupts CS.",
        "",
        "  The 'circuit effect' patches all top-k neurons simultaneously.",
        "  Individual effects show which single neurons contribute most.",
        "",
        THIN,
        "  CANDIDATE NEURONS",
        THIN,
    ]

    lines.append("")
    lines.append("  Code-switching-specific neurons")
    lines.append("  (fire for code_switched, silent in all other conditions):")
    for _, row in cs_candidates.head(15).iterrows():
        lines.append(
            f"    L{int(row['layer'])}:N{int(row['neuron']):<5}  "
            f"cs_mean={row.get('code_switched', 0):.3f}  "
            f"confused_mean={row.get('confused', 0):.3f}  "
            f"english_mean={row.get('english', 0):.3f}  "
            f"specificity={row.get('cs_specificity', 0):.3f}"
        )

    lines.append("")
    lines.append("  Confusion-specific neurons")
    lines.append("  (fire for confused, silent in all other conditions):")
    for _, row in conf_candidates.head(15).iterrows():
        lines.append(
            f"    L{int(row['layer'])}:N{int(row['neuron']):<5}  "
            f"confused_mean={row.get('confused', 0):.3f}  "
            f"cs_mean={row.get('code_switched', 0):.3f}  "
            f"english_mean={row.get('english', 0):.3f}  "
            f"specificity={row.get('conf_specificity', 0):.3f}"
        )

    lines += [
        "",
        THIN,
        "  EXPERIMENT A  —  CS → confused",
        "  (Confused samples patched with code-switching neuron activations)",
        THIN,
        "",
        "  Circuit effect (all CS candidate neurons patched together):",
    ]
    lines += fmt_circuit(cs_to_conf)
    lines += [
        "",
        "  If ΔNLL is negative here, patching CS activations into confused text",
        "  makes the model process it more like code-switched text.",
        "  This is causal evidence those neurons drive CS behaviour.",
        "",
        "  Individual neuron effects (ranked by |ΔNLL|):",
    ]
    lines += fmt_individual(cs_to_conf)

    lines += [
        "",
        THIN,
        "  EXPERIMENT B  —  confused → CS",
        "  (Code-switched samples patched with confusion neuron activations)",
        THIN,
        "",
        "  Circuit effect (all confusion candidate neurons patched together):",
    ]
    lines += fmt_circuit(conf_to_cs)
    lines += [
        "",
        "  If ΔNLL is positive here, injecting confusion activations disrupts",
        "  the model's handling of CS text — causal evidence those neurons are",
        "  involved in language-confusion processing.",
        "",
        "  Individual neuron effects (ranked by |ΔNLL|):",
    ]
    lines += fmt_individual(conf_to_cs)

    lines += [
        "",
        THIN,
        "  WHAT TO DO NEXT WITH THESE RESULTS",
        THIN,
        "",
        "  1. The neurons with the largest individual |ΔNLL| are your primary",
        "     causal candidates — report them in your paper.",
        "",
        "  2. The circuit effect tells you how much of the total behavioural",
        "     difference is explained by the top-k neurons collectively.",
        "",
        "  3. For further validation: try patching only the top-3 or top-5",
        "     neurons (reduce --top_k) and check if the circuit effect holds.",
        "     A small number of neurons explaining most of the effect is the",
        "     cleanest finding.",
        "",
        "  4. Cross-reference with the layer_metrics_diff.csv from your source",
        "     run: the lens_to_final_kl divergence anomaly in confused text",
        "     (negative vs English) likely traces back to these late-layer neurons.",
        SEP,
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved summary → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal activation patching: code-switching vs language confusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     required=True,  help="Path to pipeline config YAML")
    p.add_argument("--source_run", required=True,  help="Completed pipeline output directory")
    p.add_argument("--output_dir", default="causal_results", help="Where to write outputs")
    p.add_argument("--top_k",      type=int, default=20, help="Candidate neurons per condition")
    p.add_argument(
        "--topk_sweep",
        default="",
        help="Comma-separated top-k sweep values (e.g. 1,3,5,10,20). Uses circuit-only eval.",
    )
    p.add_argument(
        "--max_samples", type=int, default=50,
        help="Max samples per condition to use (-1 = all)",
    )
    p.add_argument("--train_frac", type=float, default=0.5, help="Train split fraction for source activations")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling/splits")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CI")
    p.add_argument("--permute", type=int, default=1000, help="Sign-flip permutations for p-values")
    p.add_argument("--alpha", type=float, default=0.05, help="CI alpha")
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg        = load_config(args.config)
    source_run = Path(args.source_run)
    out_root   = Path(args.output_dir)
    tables_dir = out_root / "tables"
    figs_dir   = out_root / "figures"
    for d in [out_root, tables_dir, figs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Candidate neurons ────────────────────────────────────────────────
    logger.info("Loading candidate neurons from %s", source_run)
    cs_cands, conf_cands = load_candidate_neurons(source_run, top_k=args.top_k)
    logger.info(
        "  %d CS-specific  |  %d confusion-specific  candidate neurons",
        len(cs_cands), len(conf_cands),
    )
    cs_cands.to_csv(tables_dir   / "cs_candidate_neurons.csv",   index=False)
    conf_cands.to_csv(tables_dir / "conf_candidate_neurons.csv", index=False)

    # ── 2. Dataset ──────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", cfg.input_path)
    df = load_dataset(cfg.input_path)

    def get_samples(cond: str) -> pd.DataFrame:
        sub = df[df["condition"] == cond]
        if args.max_samples > 0 and len(sub) > args.max_samples:
            sub = sub.sample(n=args.max_samples, random_state=args.seed)
        return sub.reset_index(drop=True)

    cs_samples = get_samples("code_switched")
    conf_samples = get_samples("confused")
    cs_train, cs_eval = split_samples_train_eval(cs_samples, train_frac=args.train_frac, seed=args.seed)
    conf_train, conf_eval = split_samples_train_eval(conf_samples, train_frac=args.train_frac, seed=args.seed + 1)
    logger.info(
        "  Using %d CS (%d train / %d eval)  |  %d confused (%d train / %d eval)",
        len(cs_samples), len(cs_train), len(cs_eval),
        len(conf_samples), len(conf_train), len(conf_eval),
    )

    # ── 3. Model ────────────────────────────────────────────────────────────
    logger.info("Loading model: %s  (device: %s)", cfg.model_name, cfg.device)
    prepared  = prepare_model_and_lens(cfg.model_name, cfg.tuned_lens_resource_id, cfg.device)
    model     = prepared.model
    tokenizer = prepared.tokenizer
    device    = prepared.device
    logger.info("Model ready")

    cs_layer_neurons   = [(int(r["layer"]), int(r["neuron"])) for _, r in cs_cands.iterrows()]
    conf_layer_neurons = [(int(r["layer"]), int(r["neuron"])) for _, r in conf_cands.iterrows()]

    # ── 4. Collect source activations ───────────────────────────────────────
    logger.info("Collecting mean activations — code_switched condition")
    cs_src_acts = collect_source_activations(
        model, tokenizer, cs_train, cs_layer_neurons, cfg.max_length, device
    )

    logger.info("Collecting mean activations — confused condition")
    conf_src_acts = collect_source_activations(
        model, tokenizer, conf_train, conf_layer_neurons, cfg.max_length, device
    )

    # ── 5. Patching experiments ─────────────────────────────────────────────
    logger.info("Experiment A: CS → confused  (patch confused with CS values)")
    cs_to_conf = run_patching_experiment(
        model, tokenizer, conf_eval,
        patch_values=cs_src_acts,
        neuron_list=cs_layer_neurons,
        max_length=cfg.max_length,
        device=device,
        desc="Exp A  CS → confused",
    )

    logger.info("Experiment B: confused → CS  (patch CS with confusion values)")
    conf_to_cs = run_patching_experiment(
        model, tokenizer, cs_eval,
        patch_values=conf_src_acts,
        neuron_list=conf_layer_neurons,
        max_length=cfg.max_length,
        device=device,
        desc="Exp B  confused → CS",
    )

    # ── 5b. Statistical evaluation on held-out per-sample deltas ──────────
    rng = np.random.default_rng(args.seed)
    stats_rows = []
    for label, res in [("CS → confused", cs_to_conf), ("confused → CS", conf_to_cs)]:
        per = pd.DataFrame(res.get("circuit_per_sample", []))
        if per.empty:
            continue
        metric_map = {
            "delta_nll": "nll",
            "delta_entropy": "entropy",
            "delta_top1_prob": "top1_prob",
        }
        for col, metric_name in metric_map.items():
            values = per[col].tolist()
            mean_delta, ci_low, ci_high = bootstrap_ci(
                values=values,
                n_bootstrap=args.bootstrap,
                alpha=args.alpha,
                rng=rng,
            )
            p_val = sign_flip_permutation_pvalue(
                values=values,
                n_permutations=args.permute,
                rng=rng,
            )
            stats_rows.append(
                {
                    "experiment": label,
                    "metric": metric_name,
                    "n_samples": len(values),
                    "mean_delta": mean_delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value_sign_flip": p_val,
                }
            )

    # ── 5c. Top-k sweep on held-out set (circuit-only) ────────────────────
    sweep_rows = []
    sweep_values: list[int] = []
    if str(args.topk_sweep).strip():
        sweep_values = sorted({int(x.strip()) for x in args.topk_sweep.split(",") if x.strip()})
        if any(k <= 0 for k in sweep_values):
            raise ValueError("--topk_sweep must contain positive integers")

    if sweep_values:
        logger.info("Running top-k sweep: %s", sweep_values)
        if max(sweep_values) > args.top_k:
            raise ValueError(
                f"--topk_sweep max ({max(sweep_values)}) cannot exceed --top_k ({args.top_k}) "
                "because candidates are selected up to top_k."
            )
        for k in sweep_values:
            cs_neurons_k = cs_layer_neurons[:k]
            conf_neurons_k = conf_layer_neurons[:k]
            cs_patch_k = {ln: cs_src_acts[ln] for ln in cs_neurons_k if ln in cs_src_acts}
            conf_patch_k = {ln: conf_src_acts[ln] for ln in conf_neurons_k if ln in conf_src_acts}

            a = run_patching_experiment(
                model, tokenizer, conf_eval,
                patch_values=cs_patch_k,
                neuron_list=cs_neurons_k,
                max_length=cfg.max_length,
                device=device,
                desc=f"Sweep k={k} A",
                compute_individual=False,
            )
            b = run_patching_experiment(
                model, tokenizer, cs_eval,
                patch_values=conf_patch_k,
                neuron_list=conf_neurons_k,
                max_length=cfg.max_length,
                device=device,
                desc=f"Sweep k={k} B",
                compute_individual=False,
            )
            sweep_rows.append(
                {
                    "top_k": k,
                    "experiment": "CS → confused",
                    "circuit_delta_nll": a["circuit"].get("nll", float("nan")),
                    "circuit_delta_entropy": a["circuit"].get("entropy", float("nan")),
                    "circuit_delta_top1_prob": a["circuit"].get("top1_prob", float("nan")),
                }
            )
            sweep_rows.append(
                {
                    "top_k": k,
                    "experiment": "confused → CS",
                    "circuit_delta_nll": b["circuit"].get("nll", float("nan")),
                    "circuit_delta_entropy": b["circuit"].get("entropy", float("nan")),
                    "circuit_delta_top1_prob": b["circuit"].get("top1_prob", float("nan")),
                }
            )

    # ── 6. Save outputs ─────────────────────────────────────────────────────
    logger.info("Saving results to %s", out_root)

    effects_a = build_effects_dataframe(cs_to_conf["individual"],  "CS → confused")
    effects_b = build_effects_dataframe(conf_to_cs["individual"],  "confused → CS")
    effects_all = pd.concat([effects_a, effects_b], ignore_index=True)
    effects_all.to_csv(tables_dir / "patching_effects.csv", index=False)

    circuit_rows = []
    for label, res in [("CS → confused", cs_to_conf), ("confused → CS", conf_to_cs)]:
        row = {"experiment": label}
        row.update({f"circuit_delta_{k}": v for k, v in res["circuit"].items()})
        circuit_rows.append(row)
    pd.DataFrame(circuit_rows).to_csv(tables_dir / "circuit_effects.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(tables_dir / "circuit_stats.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(tables_dir / "topk_sweep.csv", index=False)

    plot_patching_effects(effects_all, figs_dir / "patching_effects.html")

    write_summary(
        cs_candidates=cs_cands,
        conf_candidates=conf_cands,
        cs_to_conf=cs_to_conf,
        conf_to_cs=conf_to_cs,
        model_name=cfg.model_name,
        source_run=str(source_run),
        n_cs_samples=len(cs_eval),
        n_conf_samples=len(conf_eval),
        out_path=out_root / "CAUSAL_SUMMARY.txt",
    )

    metadata = {
        "model_name":       cfg.model_name,
        "source_run":       str(source_run),
        "top_k":            args.top_k,
        "topk_sweep":       sweep_values,
        "max_samples":      args.max_samples,
        "train_frac":       args.train_frac,
        "seed":             args.seed,
        "bootstrap":        args.bootstrap,
        "permute":          args.permute,
        "alpha":            args.alpha,
        "cs_samples_total": len(cs_samples),
        "conf_samples_total": len(conf_samples),
        "cs_samples_train": len(cs_train),
        "cs_samples_eval": len(cs_eval),
        "conf_samples_train": len(conf_train),
        "conf_samples_eval": len(conf_eval),
        "circuit_cs_to_conf":  cs_to_conf["circuit"],
        "circuit_conf_to_cs":  conf_to_cs["circuit"],
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ── 7. Print quick summary to terminal ──────────────────────────────────
    c_a = cs_to_conf["circuit"]
    c_b = conf_to_cs["circuit"]
    print("\n" + "=" * 60)
    print("  CAUSAL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Output directory : {out_root.resolve()}")
    print()
    print("  Circuit-level results:")
    print(f"    Exp A  CS → confused  :  ΔNLL = {c_a.get('nll',0):+.4f}  ΔEntropy = {c_a.get('entropy',0):+.4f}")
    print(f"    Exp B  confused → CS  :  ΔNLL = {c_b.get('nll',0):+.4f}  ΔEntropy = {c_b.get('entropy',0):+.4f}")
    print()
    print("  Key outputs:")
    print(f"    {out_root / 'CAUSAL_SUMMARY.txt'}")
    print(f"    {tables_dir / 'patching_effects.csv'}")
    print(f"    {tables_dir / 'circuit_stats.csv'}")
    print(f"    {tables_dir / 'topk_sweep.csv'}")
    print(f"    {figs_dir   / 'patching_effects.html'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
