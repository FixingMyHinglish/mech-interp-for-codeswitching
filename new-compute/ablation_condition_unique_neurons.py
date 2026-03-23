#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils


logger = logging.getLogger("ablation_unique")

Neuron = Tuple[int, int]


# ---------------------------------------------------------------------------
# Script-range helpers
# ---------------------------------------------------------------------------

SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "latin": [(0x0041, 0x007A), (0x00C0, 0x024F)],
    "devanagari": [(0x0900, 0x097F)],
    "cyrillic": [(0x0400, 0x04FF)],
    "arabic": [(0x0600, 0x06FF)],
    "cjk": [(0x4E00, 0x9FFF)],
    "hiragana": [(0x3040, 0x309F)],
    "katakana": [(0x30A0, 0x30FF)],
    "hangul": [(0xAC00, 0xD7AF)],
}


def _contains_script_char(ch: str, ranges: Sequence[tuple[int, int]]) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def _char_counts(text: str, target_script: str) -> tuple[int, int, int]:
    latin_ranges = SCRIPT_RANGES["latin"]
    target_ranges = SCRIPT_RANGES[target_script]
    latin = target = other = 0
    for ch in text:
        if ch.isspace():
            continue
        if _contains_script_char(ch, latin_ranges):
            latin += 1
        elif _contains_script_char(ch, target_ranges):
            target += 1
        else:
            other += 1
    return latin, target, other


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _token_script(tok: str, target_script: str) -> str:
    latin_ranges = SCRIPT_RANGES["latin"]
    target_ranges = SCRIPT_RANGES[target_script]
    has_latin = any(_contains_script_char(ch, latin_ranges) for ch in tok)
    has_target = any(_contains_script_char(ch, target_ranges) for ch in tok)
    if has_latin and has_target:
        return "mixed"
    if has_latin:
        return "latin"
    if has_target:
        return "target"
    return "other"


def compute_text_features(text: str, target_script: str) -> np.ndarray:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return np.zeros(8, dtype=np.float32)

    scripts = [_token_script(t, target_script) for t in tokens]
    n = len(tokens)
    latin_tok = sum(s == "latin" for s in scripts)
    target_tok = sum(s == "target" for s in scripts)
    mixed_tok = sum(s == "mixed" for s in scripts)
    other_tok = sum(s == "other" for s in scripts)

    switches = sum(scripts[i] != scripts[i - 1] for i in range(1, len(scripts)))
    switch_rate = switches / max(1, n - 1)

    latin_ch, target_ch, other_ch = _char_counts(text, target_script)
    total_ch = max(1, latin_ch + target_ch + other_ch)

    return np.array(
        [
            latin_tok / n,
            target_tok / n,
            mixed_tok / n,
            other_tok / n,
            switch_rate,
            latin_ch / total_ch,
            target_ch / total_ch,
            math.log1p(n),
        ],
        dtype=np.float32,
    )


def compute_monolinguality_metrics(text: str, target_script: str) -> dict[str, float]:
    feats = compute_text_features(text, target_script)
    latin_tok, target_tok, mixed_tok, _, switch_rate, latin_char, target_char, _ = feats.tolist()

    script_char_total = max(1e-9, latin_char + target_char)
    dominant_script_ratio = max(latin_char, target_char) / script_char_total
    mixedness = float(np.clip((mixed_tok + switch_rate) / 2.0, 0.0, 1.0))
    monolinguality_score = float(np.clip(dominant_script_ratio * (1.0 - mixedness), 0.0, 1.0))
    target_language_monolinguality_score = float(
        np.clip(target_char * (1.0 - mixedness), 0.0, 1.0)
    )

    return {
        "latin_token_ratio": float(latin_tok),
        "target_token_ratio": float(target_tok),
        "mixed_token_ratio": float(mixed_tok),
        "switch_rate": float(switch_rate),
        "latin_char_ratio": float(latin_char),
        "target_char_ratio": float(target_char),
        "dominant_script_ratio": float(dominant_script_ratio),
        "mixedness_score": float(mixedness),
        "monolinguality_score": monolinguality_score,
        "target_language_monolinguality_score": target_language_monolinguality_score,
    }


def infer_target_script(df: pd.DataFrame, target_label: str) -> str:
    subset = df[df["condition"] == target_label]
    if subset.empty:
        return "devanagari"
    counts: dict[str, int] = defaultdict(int)
    for text in subset["text"].astype(str).head(300):
        for script, ranges in SCRIPT_RANGES.items():
            if script == "latin":
                continue
            counts[script] += sum(
                1 for ch in text if _contains_script_char(ch, ranges)
            )
    if not counts:
        return "devanagari"
    return max(counts.items(), key=lambda x: x[1])[0]


def load_unique_neurons(
    consistent_csv: Path, cs_label: str, confused_label: str
) -> tuple[list[Neuron], list[Neuron]]:
    df = pd.read_csv(consistent_csv)
    if "passes_consistency" in df.columns:
        df = df[df["passes_consistency"].astype(bool)]
    cs = set(
        zip(
            df[df["condition"] == cs_label]["layer"].astype(int),
            df[df["condition"] == cs_label]["neuron"].astype(int),
        )
    )
    confused = set(
        zip(
            df[df["condition"] == confused_label]["layer"].astype(int),
            df[df["condition"] == confused_label]["neuron"].astype(int),
        )
    )
    return sorted(cs - confused), sorted(confused - cs)


# ---------------------------------------------------------------------------
# TL-native ablation via activation hooks
# ---------------------------------------------------------------------------

def _zero_ablate_hook(
    value: torch.Tensor, hook: Any, neuron_ids: list[int]
) -> torch.Tensor:
    """Zero out specific neurons in an MLP post-activation tensor.

    ``value`` has shape ``[batch, pos, d_mlp]``.  We zero the selected neuron
    indices along the last dimension, silencing their contribution to the
    residual stream for every token position in the sequence.

    This is the canonical TransformerLens approach: no weight surgery is
    needed, the hook is fully reversible, and it works identically across
    GPT-2, LLaMA, GPT-NeoX, and any other architecture TL supports.
    """
    value[:, :, neuron_ids] = 0.0
    return value


def build_ablation_hooks(
    neurons: list[Neuron],
) -> list[tuple[str, Callable]]:
    """Return ``(hook_name, hook_fn)`` pairs for zeroing the given neurons.

    Uses ``transformer_lens.utils.get_act_name`` so hook names stay correct
    across TL versions and model architectures.  Neurons are grouped by layer
    so only one hook is registered per layer.
    """
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, neuron in neurons:
        by_layer[int(layer)].append(int(neuron))

    return [
        (
            tl_utils.get_act_name("post", layer_idx),
            partial(_zero_ablate_hook, neuron_ids=sorted(set(ids))),
        )
        for layer_idx, ids in by_layer.items()
    ]


def generate_continuation(
    model: HookedTransformer,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    ablation_hooks: list[tuple[str, Callable]] | None = None,
) -> str:
    """Generate a continuation, optionally with neuron ablation hooks active.

    ``model.hooks()`` registers the hooks for all forward passes within the
    context, including each auto-regressive step inside ``model.generate()``,
    without touching the underlying weight matrices.
    """
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding=False,
    ).to(device)
    input_ids = encoded["input_ids"]
    input_len = int(input_ids.shape[1])

    hooks = ablation_hooks or []

    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        # generate() returns the full sequence: input + generated tokens.
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_at_eos=True,
            verbose=False,
        )

    new_ids = out[0, input_len:]
    return tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)


def sample_eval_rows(
    df: pd.DataFrame,
    cs_label: str,
    confused_label: str,
    max_per_condition: int,
    seed: int,
) -> pd.DataFrame:
    parts = []
    rng = random.Random(seed)
    for cond in [cs_label, confused_label]:
        sub = df[df["condition"] == cond].copy()
        idx = list(sub.index)
        rng.shuffle(idx)
        take = idx[: min(max_per_condition, len(idx))]
        parts.append(sub.loc[take])
    return pd.concat(parts, ignore_index=True)


def run_setting(
    setting_name: str,
    model: HookedTransformer,
    tokenizer: Any,
    device: torch.device,
    eval_rows: pd.DataFrame,
    target_script: str,
    neurons_to_ablate: list[Neuron],
    max_input_length: int,
    max_new_tokens: int,
) -> pd.DataFrame:
    ablation_hooks = build_ablation_hooks(neurons_to_ablate) if neurons_to_ablate else []
    out_rows = []
    for i, (_, row) in enumerate(eval_rows.iterrows()):
        gen = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=str(row["text"]),
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            ablation_hooks=ablation_hooks,
        )
        m = compute_monolinguality_metrics(gen, target_script)
        out_rows.append(
            {
                "setting": setting_name,
                "source_id": row.get("source_id", row["id"]),
                "prompt_id": row["id"],
                "source_condition": row["condition"],
                "domain": row["domain"],
                "generated_text": gen,
                **m,
            }
        )
        if (i + 1) % 25 == 0:
            logger.info("[%s] processed %d/%d prompts", setting_name, i + 1, len(eval_rows))
    return pd.DataFrame(out_rows)


def summarize_predictions(df: pd.DataFrame) -> dict:
    summary = {}
    for setting, sub in df.groupby("setting"):
        metric_means = {
            "mean_monolinguality_score": float(sub["monolinguality_score"].mean()),
            "mean_target_language_monolinguality_score": float(
                sub["target_language_monolinguality_score"].mean()
            ),
            "mean_mixedness_score": float(sub["mixedness_score"].mean()),
            "mean_switch_rate": float(sub["switch_rate"].mean()),
            "mean_dominant_script_ratio": float(sub["dominant_script_ratio"].mean()),
            "mean_latin_char_ratio": float(sub["latin_char_ratio"].mean()),
            "mean_target_char_ratio": float(sub["target_char_ratio"].mean()),
        }
        by_source = (
            sub.groupby("source_condition")[
                ["monolinguality_score", "mixedness_score", "switch_rate", "dominant_script_ratio"]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )
        summary[setting] = {
            "n_samples": int(len(sub)),
            **metric_means,
            "by_source_condition": by_source,
        }

    if "baseline" in summary:
        base = summary["baseline"]
        for k in list(summary.keys()):
            if k == "baseline":
                continue
            cur = summary[k]
            summary[k]["delta_monolinguality_vs_baseline"] = float(
                cur["mean_monolinguality_score"] - base["mean_monolinguality_score"]
            )
            summary[k]["delta_target_language_monolinguality_vs_baseline"] = float(
                cur["mean_target_language_monolinguality_score"]
                - base["mean_target_language_monolinguality_score"]
            )
            summary[k]["delta_mixedness_vs_baseline"] = float(
                cur["mean_mixedness_score"] - base["mean_mixedness_score"]
            )
            summary[k]["delta_switch_rate_vs_baseline"] = float(
                cur["mean_switch_rate"] - base["mean_switch_rate"]
            )

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablate CS-/confused-unique neurons and measure change in output monolinguality."
    )
    p.add_argument("--run_dir", required=True, help="Pipeline run dir (used for model metadata).")
    p.add_argument("--dataset_csv", required=True, help="Dataset CSV with id,text,condition,domain.")
    p.add_argument(
        "--consistent_neurons_csv",
        required=True,
        help="Path to consistent_neurons.csv(.gz) from new-compute results.",
    )
    p.add_argument("--out_dir", default="new-compute/ablation_results")
    p.add_argument("--model_name", default=None, help="Override model name (else from run metadata).")
    p.add_argument("--device", default="auto", help="Device (auto/cpu/mps/cuda).")
    p.add_argument("--cs_label", default="code_switched")
    p.add_argument("--confused_label", default="confused")
    p.add_argument("--target_label", default="target_language")
    p.add_argument(
        "--target_script",
        default="auto",
        choices=["auto", "devanagari", "cyrillic", "arabic", "cjk", "hiragana", "katakana", "hangul"],
    )
    p.add_argument("--max_eval_per_condition", type=int, default=150)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--max_input_length", type=int, default=192)
    p.add_argument("--max_neurons_per_set", type=int, default=0, help="0 = use all unique neurons.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model name
    model_name = args.model_name
    if model_name is None:
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing {meta_path}. Pass --model_name explicitly."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = str(meta["model_name"])

    # Resolve device
    device_str = args.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_t = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_t = torch.device("mps")
        else:
            device_t = torch.device("cpu")
    else:
        device_t = torch.device(device_str)

    df = pd.read_csv(args.dataset_csv)
    for col in ["id", "text", "condition", "domain"]:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    target_script = args.target_script
    if target_script == "auto":
        target_script = infer_target_script(df, args.target_label)
    logger.info("Target script for monolinguality metrics: %s", target_script)

    cs_unique, confused_unique = load_unique_neurons(
        Path(args.consistent_neurons_csv),
        cs_label=args.cs_label,
        confused_label=args.confused_label,
    )
    if args.max_neurons_per_set > 0:
        cs_unique = cs_unique[: args.max_neurons_per_set]
        confused_unique = confused_unique[: args.max_neurons_per_set]

    logger.info(
        "Unique neurons loaded: cs_unique=%d confused_unique=%d",
        len(cs_unique), len(confused_unique),
    )

    eval_rows = sample_eval_rows(
        df=df,
        cs_label=args.cs_label,
        confused_label=args.confused_label,
        max_per_condition=args.max_eval_per_condition,
        seed=args.seed,
    )
    logger.info("Eval prompts selected: %d", len(eval_rows))

    # Load a single HookedTransformer — the same instance is used for both
    # activation extraction (to identify neurons) and ablation (to silence them).
    logger.info("Loading HookedTransformer: %s on %s", model_name, device_t)
    model = HookedTransformer.from_pretrained(model_name, device=str(device_t))
    model.eval()
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    results.append(
        run_setting(
            setting_name="baseline",
            model=model,
            tokenizer=tokenizer,
            device=device_t,
            eval_rows=eval_rows,
            target_script=target_script,
            neurons_to_ablate=[],
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )
    )
    if cs_unique:
        results.append(
            run_setting(
                setting_name="ablate_cs_unique",
                model=model,
                tokenizer=tokenizer,
                device=device_t,
                eval_rows=eval_rows,
                target_script=target_script,
                neurons_to_ablate=cs_unique,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )
        )
    if confused_unique:
        results.append(
            run_setting(
                setting_name="ablate_confused_unique",
                model=model,
                tokenizer=tokenizer,
                device=device_t,
                eval_rows=eval_rows,
                target_script=target_script,
                neurons_to_ablate=confused_unique,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )
        )

    all_preds = pd.concat(results, ignore_index=True)
    summary = summarize_predictions(all_preds)

    all_preds.to_csv(out_dir / "ablation_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(cs_unique, columns=["layer", "neuron"]).to_csv(
        out_dir / "cs_unique_neurons.csv", index=False
    )
    pd.DataFrame(confused_unique, columns=["layer", "neuron"]).to_csv(
        out_dir / "confused_unique_neurons.csv", index=False
    )

    meta_out = {
        "run_dir": str(run_dir),
        "dataset_csv": str(Path(args.dataset_csv)),
        "consistent_neurons_csv": str(Path(args.consistent_neurons_csv)),
        "model_name": model_name,
        "device": str(device_t),
        "target_script": target_script,
        "max_eval_per_condition": int(args.max_eval_per_condition),
        "max_new_tokens": int(args.max_new_tokens),
        "max_input_length": int(args.max_input_length),
        "cs_unique_count_used": int(len(cs_unique)),
        "confused_unique_count_used": int(len(confused_unique)),
        "settings_summary": summary,
    }
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )

    logger.info("Saved: %s", out_dir / "ablation_predictions.csv.gz")
    logger.info("Saved: %s", out_dir / "ablation_summary.json")
    print(json.dumps(meta_out, indent=2))


if __name__ == "__main__":
    main()
