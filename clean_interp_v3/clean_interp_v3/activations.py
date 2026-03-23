from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D


@dataclass(frozen=True)
class LayerInfo:
    layer_idx: int
    block: torch.nn.Module
    projection: torch.nn.Module
    output_matrix: torch.Tensor


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    layers: list[LayerInfo]
    final_norm: torch.nn.Module | None
    lm_head: torch.nn.Module
    chunk_size: int


@dataclass(frozen=True)
class TokenizedText:
    input_ids: list[int]
    token_texts: list[str]


@dataclass(frozen=True)
class PositionSummary:
    position: int
    current_token_id: int
    current_token_text: str
    target_token_id: int
    target_token_text: str
    layer_vectors: dict[int, np.ndarray]


@dataclass(frozen=True)
class SequenceSummary:
    n_tokens: int
    input_ids: list[int]
    token_texts: list[str]
    positions: dict[int, PositionSummary]


@dataclass
class LayerCapture:
    neuron_activations: torch.Tensor | None = None
    mlp_output: torch.Tensor | None = None
    block_output: torch.Tensor | None = None


def prepare_model(model_name: str, device: str, chunk_size: int) -> ModelBundle:
    device_t = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()
    model.to(device_t)

    layers = _find_layers(model, device_t)
    if not layers:
        raise ValueError("Could not find supported MLP output projection modules in this model")

    final_norm = _find_final_norm(model)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("Could not find lm_head on model")

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device_t,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
        chunk_size=max(16, int(chunk_size)),
    )


def tokenize_text(bundle: ModelBundle, text: str, max_length: int) -> TokenizedText:
    encoded = bundle.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = encoded["input_ids"][0].tolist()
    token_texts = [bundle.tokenizer.decode([token_id]) for token_id in input_ids]
    return TokenizedText(input_ids=input_ids, token_texts=token_texts)


def summarize_text_positions(
    bundle: ModelBundle,
    text: str,
    max_length: int,
    positions: list[int],
    target_mode: str,
) -> SequenceSummary:
    tokenized = tokenize_text(bundle, text, max_length)
    n_tokens = len(tokenized.input_ids)
    if n_tokens < 2:
        raise ValueError("Text is too short after tokenization")

    valid_positions = sorted({int(pos) for pos in positions if 0 <= int(pos) < n_tokens - 1})
    if not valid_positions:
        raise ValueError("No valid positions to analyze")

    encoded = bundle.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(bundle.device)

    with capture_sequence_state(bundle) as storage:
        with torch.inference_mode():
            outputs = bundle.model(**encoded, use_cache=False)

    logits = outputs.logits[0].detach().float()
    position_map: dict[int, PositionSummary] = {}
    for position in valid_positions:
        if target_mode == "observed_next_token":
            target_token_id = int(tokenized.input_ids[position + 1])
        else:
            target_token_id = int(torch.argmax(logits[position]).item())

        current_token_id = int(tokenized.input_ids[position])
        current_token_text = tokenized.token_texts[position]
        target_token_text = bundle.tokenizer.decode([target_token_id])

        layer_vectors: dict[int, np.ndarray] = {}
        for layer in bundle.layers:
            capture = storage[layer.layer_idx]
            if capture.neuron_activations is None or capture.mlp_output is None or capture.block_output is None:
                raise ValueError(f"Missing captured tensors for layer {layer.layer_idx}")

            residual_before_mlp = capture.block_output[position] - capture.mlp_output[position]
            neuron_activations = capture.neuron_activations[position]
            scores = _score_neurons_for_target_token(
                bundle=bundle,
                layer=layer,
                residual_before_mlp=residual_before_mlp,
                neuron_activations=neuron_activations,
                target_token_id=target_token_id,
            )
            layer_vectors[layer.layer_idx] = scores

        position_map[position] = PositionSummary(
            position=position,
            current_token_id=current_token_id,
            current_token_text=current_token_text,
            target_token_id=target_token_id,
            target_token_text=target_token_text,
            layer_vectors=layer_vectors,
        )

    return SequenceSummary(
        n_tokens=n_tokens,
        input_ids=tokenized.input_ids,
        token_texts=tokenized.token_texts,
        positions=position_map,
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _find_transformer_layers(model: torch.nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    return None


def _find_layers(model: torch.nn.Module, device: torch.device) -> list[LayerInfo]:
    blocks = _find_transformer_layers(model)
    if blocks is None:
        return []

    result: list[LayerInfo] = []
    for layer_idx, block in enumerate(blocks):
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            continue
        projection = None
        for name in ["down_proj", "c_proj", "dense_4h_to_h"]:
            if hasattr(mlp, name):
                projection = getattr(mlp, name)
                break
        if projection is None:
            continue
        output_matrix = _projection_output_matrix(projection).detach().to(device=device, dtype=torch.float32)
        result.append(
            LayerInfo(
                layer_idx=layer_idx,
                block=block,
                projection=projection,
                output_matrix=output_matrix,
            )
        )
    return result


def _projection_output_matrix(module: torch.nn.Module) -> torch.Tensor:
    if isinstance(module, torch.nn.Linear):
        return module.weight.detach().float()
    if isinstance(module, Conv1D):
        return module.weight.detach().float().T
    raise TypeError(f"Unsupported projection module type: {type(module)!r}")


def _find_final_norm(model: torch.nn.Module) -> torch.nn.Module | None:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    return None


@contextmanager
def capture_sequence_state(bundle: ModelBundle) -> Iterator[dict[int, LayerCapture]]:
    storage = {layer.layer_idx: LayerCapture() for layer in bundle.layers}
    handles = []

    for layer in bundle.layers:
        layer_capture = storage[layer.layer_idx]

        def make_pre_hook(capture: LayerCapture, current_layer: int):
            def hook(_module, args):
                tensor = args[0]
                if not torch.is_tensor(tensor):
                    raise TypeError(f"Expected tensor input for layer {current_layer}")
                capture.neuron_activations = tensor[0].detach().float()
            return hook

        def make_proj_hook(capture: LayerCapture):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                capture.mlp_output = tensor[0].detach().float()
            return hook

        def make_block_hook(capture: LayerCapture):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                capture.block_output = tensor[0].detach().float()
            return hook

        handles.append(layer.projection.register_forward_pre_hook(make_pre_hook(layer_capture, layer.layer_idx)))
        handles.append(layer.projection.register_forward_hook(make_proj_hook(layer_capture)))
        handles.append(layer.block.register_forward_hook(make_block_hook(layer_capture)))

    try:
        yield storage
    finally:
        for handle in handles:
            handle.remove()


def _score_neurons_for_target_token(
    bundle: ModelBundle,
    layer: LayerInfo,
    residual_before_mlp: torch.Tensor,
    neuron_activations: torch.Tensor,
    target_token_id: int,
) -> np.ndarray:
    residual_before_mlp = residual_before_mlp.to(bundle.device, dtype=torch.float32)
    neuron_activations = neuron_activations.to(bundle.device, dtype=torch.float32)
    contribution_matrix = layer.output_matrix * neuron_activations.unsqueeze(0)
    baseline_logit = _target_logit(bundle, residual_before_mlp.unsqueeze(0), target_token_id)[0]

    scores = []
    n_neurons = int(neuron_activations.shape[0])
    for start in range(0, n_neurons, bundle.chunk_size):
        stop = min(start + bundle.chunk_size, n_neurons)
        contribution_chunk = contribution_matrix[:, start:stop].T.contiguous()
        candidate_states = residual_before_mlp.unsqueeze(0) + contribution_chunk
        chunk_logits = _target_logit(bundle, candidate_states, target_token_id)
        scores.append((chunk_logits - baseline_logit).detach().cpu())

    return torch.cat(scores, dim=0).numpy()


def _target_logit(bundle: ModelBundle, hidden_states: torch.Tensor, target_token_id: int) -> torch.Tensor:
    hidden_states = hidden_states.to(bundle.device, dtype=torch.float32)
    if bundle.final_norm is not None:
        normed = bundle.final_norm(hidden_states)
    else:
        normed = hidden_states

    weight = bundle.lm_head.weight[target_token_id].detach().to(bundle.device, dtype=torch.float32)
    logits = normed @ weight
    bias = getattr(bundle.lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[target_token_id].detach().to(bundle.device, dtype=torch.float32)
    return logits.reshape(-1)
