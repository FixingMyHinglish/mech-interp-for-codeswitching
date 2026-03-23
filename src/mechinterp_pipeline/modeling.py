from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    from tuned_lens.nn.lenses import TunedLens
    from tuned_lens.plotting.prediction_trajectory import PredictionTrajectory
except Exception:  # pragma: no cover - fallback when package shape changes
    TunedLens = None
    PredictionTrajectory = None

logger = logging.getLogger(__name__)


@dataclass
class PreparedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    lens: Any
    device: torch.device



def prepare_model_and_lens(model_name: str, tuned_lens_resource_id: str | None, device: str) -> PreparedModel:
    if device == "auto":
        if torch.cuda.is_available():
            device_t = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_t = torch.device("mps")
        else:
            device_t = torch.device("cpu")
    else:
        device_t = torch.device(device)
    logger.info("Using device: %s", device_t)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
        logger.info("Loaded model with eager attention: %s", model_name)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("Loaded model without eager-attention override: %s", model_name)
    model.eval()
    model.to(device_t)

    lens = None
    if TunedLens is not None:
        try:
            if tuned_lens_resource_id:
                lens = TunedLens.from_model_and_pretrained(model, resource_id=tuned_lens_resource_id)
            else:
                lens = TunedLens.from_model_and_pretrained(model)
            lens.to(device_t)
            lens.eval()
            logger.info("Loaded Tuned Lens successfully")
        except Exception:
            lens = None
            logger.warning("Tuned Lens unavailable for this model/config; using fallback probing")
    else:
        logger.warning("tuned-lens package import unavailable; using fallback probing")

    return PreparedModel(model=model, tokenizer=tokenizer, lens=lens, device=device_t)



def _mean_entropy_from_attentions(attn: torch.Tensor) -> np.ndarray:
    probs = attn.to(torch.float32).clamp_min(1e-9)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.mean(dim=-1).squeeze(0).detach().cpu().numpy()



def _reduce_proxy(proxy: torch.Tensor, reduce_mode: str) -> torch.Tensor:
    if reduce_mode == "mean_abs":
        return proxy.abs().mean(dim=0)
    if reduce_mode == "mean":
        return proxy.mean(dim=0)
    if reduce_mode == "max_abs":
        return proxy.abs().max(dim=0).values
    raise ValueError(f"Unsupported neuron activation reduce mode: {reduce_mode}")


def _collect_mlp_neuron_proxy(
    model: torch.nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    topk: int,
    collect_full: bool = False,
    full_reduce_mode: str = "mean_abs",
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[float]] | None]:
    top_by_layer: dict[int, list[tuple[int, float]]] = {}
    full_by_layer: dict[int, list[float]] | None = {} if collect_full else None

    named_modules = dict(model.named_modules())
    layer_prefixes = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]

    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx].squeeze(0)
        layer_mod = None
        for prefix in layer_prefixes:
            key = f"{prefix}.{layer_idx - 1}.mlp"
            if key in named_modules:
                layer_mod = named_modules[key]
                break
        if layer_mod is None:
            acts = h.abs().mean(dim=0).to(torch.float32)
            vals, idxs = torch.topk(acts, k=min(topk, acts.shape[0]))
            top_by_layer[layer_idx - 1] = [
                (int(i.item()), float(v.item())) for i, v in zip(idxs, vals)
            ]
            if full_by_layer is not None:
                full_by_layer[layer_idx - 1] = acts.detach().cpu().tolist()
            continue

        projection = None
        for cand in ["up_proj", "c_fc", "fc_in", "gate_proj"]:
            if hasattr(layer_mod, cand):
                projection = getattr(layer_mod, cand)
                break

        if projection is None:
            acts = h.abs().mean(dim=0).to(torch.float32)
            vals, idxs = torch.topk(acts, k=min(topk, acts.shape[0]))
            top_by_layer[layer_idx - 1] = [
                (int(i.item()), float(v.item())) for i, v in zip(idxs, vals)
            ]
            if full_by_layer is not None:
                full_by_layer[layer_idx - 1] = acts.detach().cpu().tolist()
            continue

        with torch.no_grad():
            proxy = projection(h)
            proxy = F.silu(proxy) if proxy.dtype.is_floating_point else proxy
            acts = _reduce_proxy(proxy.to(torch.float32), reduce_mode=full_reduce_mode)
            vals, idxs = torch.topk(acts, k=min(topk, acts.shape[0]))
            top_by_layer[layer_idx - 1] = [
                (int(i.item()), float(v.item())) for i, v in zip(idxs, vals)
            ]
            if full_by_layer is not None:
                full_by_layer[layer_idx - 1] = acts.detach().cpu().tolist()

    return top_by_layer, full_by_layer



def _manual_lens_logits(
    hidden_state: torch.Tensor,
    lens_layer_idx: int | None,
    model: torch.nn.Module,
    lens: Any,
) -> torch.Tensor:
    if lens is not None and lens_layer_idx is not None:
        try:
            transformed = lens.transform_hidden(hidden_state, lens_layer_idx)
        except (IndexError, KeyError):
            # Some model/lens pairs expose fewer translators than hidden-state entries
            # (e.g. final residual state). Fall back to identity for unmatched layers.
            transformed = hidden_state
    else:
        transformed = hidden_state

    lm_head = model.get_output_embeddings()
    return lm_head(transformed)



def analyze_text(
    prepared: PreparedModel,
    sample_id: str,
    text: str,
    condition: str,
    domain: str,
    max_length: int,
    topk_neurons: int,
    collect_full_neuron_activations: bool = False,
    full_neuron_reduce_mode: str = "mean_abs",
) -> dict[str, Any]:
    tokenizer = prepared.tokenizer
    model = prepared.model
    lens = prepared.lens
    device = prepared.device

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)
    logger.debug("Sample %s tokenized to %d tokens", sample_id, int(encoded["input_ids"].shape[1]))

    if encoded["input_ids"].shape[1] < 3:
        raise ValueError(f"Sample {sample_id} is too short after tokenization for trajectory stats")

    with torch.no_grad():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    final_logits = outputs.logits.squeeze(0)

    final_log_probs = F.log_softmax(final_logits, dim=-1)

    labels = encoded["input_ids"].squeeze(0)
    next_labels = labels[1:]

    layer_metrics: dict[int, dict[str, float]] = {}
    final_h = hidden_states[-1].squeeze(0)
    lens_n_layers = len(lens.layer_translators) if lens is not None else 0

    for layer_idx in range(len(hidden_states)):
        h = hidden_states[layer_idx].squeeze(0)
        lens_layer_idx: int | None
        if lens is None:
            lens_layer_idx = None
        elif len(hidden_states) == lens_n_layers + 1:
            # Common case (e.g. GPT-2): hidden_states includes embeddings at index 0.
            lens_layer_idx = layer_idx - 1 if layer_idx > 0 else None
        elif layer_idx < lens_n_layers:
            lens_layer_idx = layer_idx
        else:
            lens_layer_idx = None

        lens_logits = _manual_lens_logits(h, lens_layer_idx, model, lens)
        lens_log_probs = F.log_softmax(lens_logits, dim=-1)

        n_positions = min(lens_log_probs.shape[0] - 1, next_labels.shape[0])
        step_log_probs = lens_log_probs[:n_positions]
        step_final_log_probs = final_log_probs[:n_positions]
        step_targets = next_labels[:n_positions]

        entropy = float((-(step_log_probs.exp() * step_log_probs).sum(dim=-1)).mean().item())
        top1_prob = float(step_log_probs.exp().max(dim=-1).values.mean().item())

        kl = float(
            F.kl_div(
                step_log_probs,
                step_final_log_probs.exp(),
                reduction="batchmean",
                log_target=False,
            ).item()
        )

        nll = float((-step_log_probs.gather(1, step_targets.unsqueeze(-1)).squeeze(-1)).mean().item())

        hidden_norm = float(h.norm(dim=-1).mean().item())
        cos_to_final = float(
            F.cosine_similarity(h, final_h, dim=-1).mean().item()
        )
        if layer_idx == 0:
            delta_norm = 0.0
        else:
            prev = hidden_states[layer_idx - 1].squeeze(0)
            delta_norm = float((h - prev).norm(dim=-1).mean().item())

        layer_metrics[layer_idx] = {
            "hidden_norm": hidden_norm,
            "delta_norm": delta_norm,
            "cosine_to_final": cos_to_final,
            "lens_entropy": entropy,
            "lens_top1_prob": top1_prob,
            "lens_to_final_kl": kl,
            "next_token_nll": nll,
        }

    attention_entropy: dict[int, np.ndarray] = {}
    if attentions is not None:
        for layer_idx, attn in enumerate(attentions):
            if attn is None:
                continue
            attention_entropy[layer_idx] = _mean_entropy_from_attentions(attn)
    else:
        logger.warning("Sample %s returned no attentions", sample_id)

    top_neurons, full_neurons = _collect_mlp_neuron_proxy(
        model,
        hidden_states,
        topk=topk_neurons,
        collect_full=collect_full_neuron_activations,
        full_reduce_mode=full_neuron_reduce_mode,
    )

    trajectory_summary: dict[str, float] = {}
    if lens is not None and PredictionTrajectory is not None:
        try:
            with torch.no_grad():
                trajectory = PredictionTrajectory.from_lens_and_model(
                    lens,
                    model,
                    input_ids=encoded["input_ids"],
                    tokenizer=tokenizer,
                )
            # These are robust scalar exports from the trajectory object.
            trajectory_summary = {
                "trajectory_layers": float(len(trajectory.log_probs)),
                "trajectory_positions": float(trajectory.log_probs.shape[-2]),
                "trajectory_vocab": float(trajectory.log_probs.shape[-1]),
            }
        except Exception:
            trajectory_summary = {}

    result = {
        "id": sample_id,
        "condition": condition,
        "domain": domain,
        "n_tokens": int(encoded["input_ids"].shape[1]),
        "layer_metrics": layer_metrics,
        "attention_entropy": attention_entropy,
        "top_neurons": top_neurons,
        "trajectory": trajectory_summary,
    }
    if full_neurons is not None:
        result["full_neuron_activations"] = full_neurons
        result["full_neuron_reduce_mode"] = full_neuron_reduce_mode
    return result
