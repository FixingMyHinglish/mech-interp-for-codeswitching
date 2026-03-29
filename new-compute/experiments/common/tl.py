from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from transformer_lens import HookedTransformer
    from transformer_lens import utils as tl_utils
except Exception:  # pragma: no cover - optional dependency
    HookedTransformer = None
    tl_utils = None


def configure_gpu_runtime(device: torch.device, gpu_friendly: bool) -> None:
    if not gpu_friendly:
        return
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_tl_model(model_name: str, device: torch.device, gpu_friendly: bool = False) -> Any:
    if HookedTransformer is None:
        raise ImportError(
            "transformer_lens is not installed. Install it to run these experiments."
        )
    model = None
    if gpu_friendly and device.type == "cuda":
        for dtype in [torch.bfloat16, torch.float16]:
            try:
                model = HookedTransformer.from_pretrained(
                    model_name,
                    device=str(device),
                    dtype=dtype,
                )
                break
            except Exception:
                model = None
    if model is None:
        model = HookedTransformer.from_pretrained(model_name, device=str(device))
    model.eval()
    return model


def encode_text(tokenizer: Any, text: str, max_length: int, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return encoded["input_ids"].to(device)


def extract_post_activations(model: Any, input_ids: torch.Tensor) -> dict[int, np.ndarray]:
    n_layers = int(model.cfg.n_layers)
    hook_names = {tl_utils.get_act_name("post", layer) for layer in range(n_layers)}
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids,
            return_type="logits",
            names_filter=lambda name: name in hook_names,
        )

    out: dict[int, np.ndarray] = {}
    for layer in range(n_layers):
        key = tl_utils.get_act_name("post", layer)
        if key not in cache:
            continue
        out[layer] = cache[key][0].detach().float().cpu().numpy()
    return out
