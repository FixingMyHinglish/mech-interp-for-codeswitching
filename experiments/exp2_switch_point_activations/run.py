#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common import (  # noqa: E402
    configure_gpu_runtime,
    encode_text,
    extract_post_activations,
    format_offset,
    infer_target_language_code_fasttext,
    infer_target_script,
    label_token_language,
    load_dataset,
    load_fasttext_model,
    load_tl_model,
    longest_common_prefix,
    resolve_device,
    write_neuron_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment 2: switch-point neuron activation patterns. "
            "For each sample, identifies neurons that are more active in the "
            "code-switched version than in EVERY monolingual baseline at the switch "
            "point. Reports neurons that show this pattern consistently across samples."
        )
    )
    p.add_argument("--dataset_csv", required=True, help="Dataset CSV with id/text/condition/domain/source_id.")
    p.add_argument("--model_name", required=True, help="Hugging Face model name for TransformerLens.")
    p.add_argument("--out_dir", default="experiments/exp2_switch_point_activations/results")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--focus_condition", default="code_switched")
    p.add_argument("--baseline_conditions", nargs="+", default=["english", "target_language"])
    p.add_argument("--token_offsets", nargs="+", type=int, default=[-1, 0, 1])
    p.add_argument(
        "--switch_detection_level",
        choices=["token", "word"],
        default="word",
        help=(
            "Granularity used to detect switch boundaries before mapping them back "
            "to token positions for activation lookup. 'word' is more robust for "
            "same-script pairs such as French-English; 'token' reproduces the old behaviour."
        ),
    )
    p.add_argument(
        "--min_consistency_fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of samples in which a neuron must show switch-specific "
            "activation to be included in the consistent output (default: 0.5)."
        ),
    )
    p.add_argument("--max_groups", type=int, default=None)
    p.add_argument(
        "--language_id_method",
        choices=["prefix", "script", "sliding_window", "vocab"],
        default="vocab",
        help=(
            "Method used to find switch points. "
            "'vocab' (default) labels every CS token by checking whether it appears "
            "in the English or target-language baseline vocabulary for that sample, "
            "then finds every transition — no external libraries required and works "
            "for both same-script (French-English) and different-script (Hindi-English) pairs. "
            "'prefix' uses longest-common-prefix alignment and finds only the first switch "
            "point per sample. "
            "'script' labels every CS token by Unicode script — works well for "
            "script-separated pairs (e.g. Hindi-English). "
            "'sliding_window' uses FastText on a decoded context window — works for "
            "same-script pairs but requires --fasttext_model_path."
        ),
    )
    p.add_argument(
        "--fasttext_model_path",
        default=None,
        help="Path to a FastText language-ID model (e.g. lid.176.bin). Required for --language_id_method sliding_window.",
    )
    p.add_argument(
        "--fasttext_min_prob",
        type=float,
        default=0.5,
        help="Minimum FastText confidence to assign a language label (default: 0.5). Tokens below this threshold are labeled 'other'.",
    )
    p.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Number of tokens in the sliding context window used for FastText classification (default: 5, must be odd or even — center token is at position i).",
    )
    p.add_argument(
        "--gpu_friendly",
        action="store_true",
        help=(
            "Enable GPU-friendly settings: TF32, cuDNN benchmark, and mixed-precision "
            "model loading (bfloat16/float16 fallback) when running on CUDA."
        ),
    )
    return p.parse_args()


WORD_RE = re.compile(r"\w+(?:['’\-]\w+)*", flags=re.UNICODE)


def encode_text_with_offsets(tokenizer, text: str, max_length: int, device: torch.device) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise ValueError(
            "Tokenizer did not return offset mappings. "
            "Use --switch_detection_level token to reproduce the old token-level detector."
        )
    input_ids = encoded["input_ids"].to(device)
    raw_offsets = encoded["offset_mapping"]
    if isinstance(raw_offsets, torch.Tensor):
        raw_offsets = raw_offsets[0].detach().cpu().tolist()
    else:
        raw_offsets = raw_offsets[0]
    offsets = [(int(start), int(end)) for start, end in raw_offsets]
    return input_ids, offsets


def _normalize_word(word: str) -> str:
    norm = unicodedata.normalize("NFKC", word).casefold()
    return norm.strip("_'’`-")


def extract_words_with_token_spans(
    text: str,
    token_offsets: list[tuple[int, int]],
) -> list[dict[str, int | str]]:
    """
    Build word spans on the original text, then map each word back to the token
    indices whose offset ranges overlap that word.
    """
    max_char = max((end for start, end in token_offsets if end > start), default=0)
    if max_char <= 0:
        return []

    words: list[dict[str, int | str]] = []
    token_cursor = 0
    visible_text = text[:max_char]

    for match in WORD_RE.finditer(visible_text):
        char_start, char_end = match.span()
        token_indices: list[int] = []

        while token_cursor < len(token_offsets) and token_offsets[token_cursor][1] <= char_start:
            token_cursor += 1

        probe = token_cursor
        while probe < len(token_offsets):
            tok_start, tok_end = token_offsets[probe]
            if tok_end <= tok_start:
                probe += 1
                continue
            if tok_start >= char_end:
                break
            if tok_end > char_start and tok_start < char_end:
                token_indices.append(probe)
            probe += 1

        if not token_indices:
            continue

        words.append(
            {
                "text": match.group(0),
                "norm": _normalize_word(match.group(0)),
                "char_start": int(char_start),
                "char_end": int(char_end),
                "token_start": int(token_indices[0]),
                "token_end": int(token_indices[-1] + 1),
            }
        )

    return words


def _forward_fill_labels(raw_labels: list[str]) -> list[str]:
    labels: list[str] = []
    last_known = "other"
    for label in raw_labels:
        clean = label if label in {"english", "target"} else "other"
        if clean != "other":
            last_known = clean
            labels.append(clean)
        else:
            labels.append(last_known if last_known != "other" else "other")
    return labels


def label_words_vocab(
    cs_words: list[dict[str, int | str]],
    en_words: list[dict[str, int | str]],
    tl_words: list[dict[str, int | str]],
) -> list[str]:
    en_vocab = {str(word["norm"]) for word in en_words if str(word["norm"])}
    tl_vocab = {str(word["norm"]) for word in tl_words if str(word["norm"])}

    raw: list[str] = []
    for word in cs_words:
        norm = str(word["norm"])
        in_en = norm in en_vocab
        in_tl = norm in tl_vocab
        if in_en and not in_tl:
            raw.append("english")
        elif in_tl and not in_en:
            raw.append("target")
        else:
            raw.append("other")
    return _forward_fill_labels(raw)


def label_words_script(
    words: list[dict[str, int | str]],
    target_script: str,
) -> list[str]:
    raw = [label_token_language(str(word["text"]), target_script=target_script) for word in words]
    return _forward_fill_labels(raw)


def label_words_sliding_window(
    words: list[dict[str, int | str]],
    *,
    text: str,
    fasttext_model,
    target_lang_code: str,
    window_size: int = 5,
    min_prob: float = 0.5,
) -> list[str]:
    half = window_size // 2
    raw: list[str] = []
    for idx in range(len(words)):
        start_word = max(0, idx - half)
        end_word = min(len(words), idx + half + 1)
        char_start = int(words[start_word]["char_start"])
        char_end = int(words[end_word - 1]["char_end"])
        window_text = text[char_start:char_end].strip().replace("\n", " ")
        if not window_text:
            raw.append("other")
            continue
        preds, probs = fasttext_model.predict(window_text, k=1)
        if not preds:
            raw.append("other")
            continue
        lang = preds[0].replace("__label__", "")
        prob = float(probs[0])
        if prob < min_prob:
            raw.append("other")
        elif lang == "en":
            raw.append("english")
        elif lang == target_lang_code:
            raw.append("target")
        else:
            raw.append("other")
    return _forward_fill_labels(raw)


def label_tokens_sliding_window(
    token_ids: list[int],
    tokenizer,
    fasttext_model,
    target_lang_code: str,
    window_size: int = 5,
    min_prob: float = 0.5,
) -> list[str]:
    """
    Assign a language label to every token position by classifying a sliding
    context window with FastText.

    For position i the window spans [i - half, i + half + 1] (clamped to the
    sequence boundaries), where half = window_size // 2.  The window tokens are
    decoded back to a text string and passed to FastText.  Using a window rather
    than a single token gives FastText enough context to reliably distinguish
    same-script language pairs such as French and English.

    Labels returned per position: 'english', 'target', or 'other'.
    Positions whose window classification falls below min_prob are labeled 'other'.
    """
    token_texts = tokenizer.convert_ids_to_tokens(token_ids)
    half = window_size // 2
    labels: list[str] = []
    for i in range(len(token_texts)):
        start = max(0, i - half)
        end = min(len(token_texts), i + half + 1)
        window_text = tokenizer.convert_tokens_to_string(
            token_texts[start:end]
        ).strip().replace("\n", " ")
        if not window_text:
            labels.append("other")
            continue
        preds, probs = fasttext_model.predict(window_text, k=1)
        if not preds:
            labels.append("other")
            continue
        lang = preds[0].replace("__label__", "")
        prob = float(probs[0])
        if prob < min_prob:
            labels.append("other")
        elif lang == "en":
            labels.append("english")
        elif lang == target_lang_code:
            labels.append("target")
        else:
            labels.append("other")
    return labels


def label_tokens_vocab(
    cs_token_ids: list[int],
    en_token_ids: list[int],
    tl_token_ids: list[int],
) -> list[str]:
    """
    Assign a language label to every CS token position using the monolingual
    baseline token vocabularies for that sample.

    Logic per token:
      - appears in en_vocab only  → 'english'
      - appears in tl_vocab only  → 'target'
      - appears in both or neither → 'other'

    Ambiguous ('other') tokens carry forward the most recent non-other label
    so that switch transitions through shared/unknown tokens are still detected.

    No external libraries required. Works for both same-script pairs
    (French-English) and different-script pairs (Hindi-English).
    """
    en_vocab = set(en_token_ids)
    tl_vocab = set(tl_token_ids)

    raw: list[str] = []
    for tid in cs_token_ids:
        in_en = tid in en_vocab
        in_tl = tid in tl_vocab
        if in_en and not in_tl:
            raw.append("english")
        elif in_tl and not in_en:
            raw.append("target")
        else:
            raw.append("other")

    # Forward-fill: ambiguous tokens inherit the last known language label.
    labels: list[str] = []
    last_known = "other"
    for lab in raw:
        if lab != "other":
            last_known = lab
        labels.append(last_known if lab == "other" else lab)

    return labels


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset_csv)
    device = resolve_device(args.device)
    configure_gpu_runtime(device=device, gpu_friendly=bool(args.gpu_friendly))
    model = load_tl_model(
        args.model_name,
        device=device,
        gpu_friendly=bool(args.gpu_friendly),
    )
    tokenizer = model.tokenizer

    # Pre-compute language-ID resources once, before the main loop.
    target_script: str | None = None
    fasttext_model = None
    target_lang_code: str | None = None

    if args.language_id_method == "script":
        target_script = infer_target_script(df, target_label="target_language")
    elif args.language_id_method == "sliding_window":
        if not args.fasttext_model_path:
            raise ValueError(
                "--fasttext_model_path is required when --language_id_method sliding_window. "
                "Example: --fasttext_model_path /path/to/lid.176.bin"
            )
        fasttext_model = load_fasttext_model(args.fasttext_model_path)
        target_lang_code = infer_target_language_code_fasttext(
            df,
            target_label="target_language",
            model=fasttext_model,
        )

    # Group samples by source_id so we can match code-switched with its monolinguals.
    by_source: dict[str, dict[str, dict[str, str]]] = {}
    for row in df.itertuples(index=False):
        by_source.setdefault(row.source_id, {})[row.condition] = {
            "id": row.id,
            "text": row.text,
            "domain": row.domain,
            "condition": row.condition,
        }

    source_items = list(by_source.items())
    if args.max_groups is not None:
        source_items = source_items[: max(1, int(args.max_groups))]

    # Per-(offset, layer) accumulation.
    # count_map[(rel_off, layer)] is a 1-D int array of length n_neurons where
    # count_map[key][n] = number of samples where neuron n was switch-specific.
    # total_map[rel_off] = number of samples that contributed to that offset.
    count_map: dict[tuple[int, int], np.ndarray] = {}
    total_map: dict[int, int] = {}

    # Activation accumulation split by whether each neuron was switch-specific.
    # act_spec_sum[(rel_off, layer, condition)][n]    = sum of activations for neuron n
    #                                                   across samples where n WAS switch-specific.
    # act_nonspec_sum[(rel_off, layer, condition)][n] = same, for samples where n was NOT.
    # Dividing by count_map[key][n] and (n_total - count_map[key][n]) respectively
    # gives the per-neuron conditional means needed to benchmark CS-specific clusters.
    act_spec_sum: dict[tuple[int, int, str], np.ndarray] = {}
    act_nonspec_sum: dict[tuple[int, int, str], np.ndarray] = {}

    event_rows: list[dict] = []
    skipped: list[dict] = []

    for idx, (source_id, cond_map) in enumerate(
        tqdm(source_items, desc="Exp4 matched groups"), start=1
    ):
        if args.focus_condition not in cond_map:
            continue

        # Require ALL baseline conditions to be present so the intersection
        # "active in cs but not in ANY monolingual" is meaningful.
        valid_baselines = [b for b in args.baseline_conditions if b in cond_map]
        if not valid_baselines:
            continue

        # ------------------------------------------------------------------
        # Tokenise all needed conditions and run the model.
        # ------------------------------------------------------------------
        needed_conditions = [args.focus_condition] + valid_baselines
        token_id_cache: dict[str, list[int]] = {}
        act_cache: dict[str, dict[int, np.ndarray]] = {}
        seq_len_cache: dict[str, int] = {}
        token_offset_cache: dict[str, list[tuple[int, int]]] = {}
        word_cache: dict[str, list[dict[str, int | str]]] = {}

        skip_sample = False
        for cond in needed_conditions:
            if args.switch_detection_level == "word":
                input_ids, token_offsets = encode_text_with_offsets(
                    tokenizer=tokenizer,
                    text=cond_map[cond]["text"],
                    max_length=int(args.max_length),
                    device=device,
                )
                token_offset_cache[cond] = token_offsets
                word_cache[cond] = extract_words_with_token_spans(
                    cond_map[cond]["text"],
                    token_offsets,
                )
            else:
                input_ids = encode_text(
                    tokenizer=tokenizer,
                    text=cond_map[cond]["text"],
                    max_length=int(args.max_length),
                    device=device,
                )
            ids = input_ids[0].detach().cpu().tolist()
            if len(ids) < 2:
                skip_sample = True
                break
            token_id_cache[cond] = ids
            seq_len_cache[cond] = len(ids)
            act_cache[cond] = extract_post_activations(model, input_ids)

        if skip_sample or any(c not in token_id_cache for c in needed_conditions):
            skipped.append({"source_id": source_id, "reason": "short_sequence"})
            continue

        # ------------------------------------------------------------------
        # Find switch point(s) in the CS sequence.
        #
        # 'prefix' mode (original behaviour):
        #   Compare CS token-by-token against each baseline to find the longest
        #   common prefix.  The baseline with the longest shared prefix is the
        #   target-language-like one; the first divergence is the switch point.
        #   Only ONE switch point is returned.
        #
        # 'script' mode (multi-switch):
        #   Label every CS token by script (latin = English, non-latin = target).
        #   Every position where the label changes is a switch point, so sentences
        #   with more than one code-switch (e.g. "mi número es four five six volver")
        #   contribute multiple events.  All comparisons use position-aligned
        #   anchors against the monolingual baselines.
        # ------------------------------------------------------------------
        cs_ids = token_id_cache[args.focus_condition]

        if args.language_id_method in {"script", "sliding_window", "vocab"}:
            # ------------------------------------------------------------------
            # Label every CS token by language, then find all positions where
            # the label transitions between 'english' and 'target'.
            #
            # 'script'         — Unicode script ranges; reliable for script-
            #                    separated pairs (Hindi-English).
            # 'sliding_window' — FastText on a decoded context window; works for
            #                    same-script pairs (French-English) where single-
            #                    token classification is noisy.
            # ------------------------------------------------------------------
            switch_positions: list[int] = []
            switch_word_indices: list[int | None] = []
            if args.switch_detection_level == "word":
                cs_words = word_cache.get(args.focus_condition, [])
                if not cs_words:
                    skipped.append({"source_id": source_id, "reason": "no_word_spans"})
                    continue

                if args.language_id_method == "vocab":
                    word_labels = label_words_vocab(
                        cs_words=cs_words,
                        en_words=word_cache.get("english", []),
                        tl_words=word_cache.get("target_language", []),
                    )
                elif args.language_id_method == "script":
                    assert target_script is not None
                    word_labels = label_words_script(cs_words, target_script=target_script)
                else:
                    assert fasttext_model is not None
                    assert target_lang_code is not None
                    word_labels = label_words_sliding_window(
                        cs_words,
                        text=cond_map[args.focus_condition]["text"],
                        fasttext_model=fasttext_model,
                        target_lang_code=target_lang_code,
                        window_size=int(args.window_size),
                        min_prob=float(args.fasttext_min_prob),
                    )

                for word_idx in range(1, len(word_labels)):
                    prev, cur = word_labels[word_idx - 1], word_labels[word_idx]
                    if prev in {"english", "target"} and cur in {"english", "target"} and prev != cur:
                        switch_positions.append(int(cs_words[word_idx]["token_start"]))
                        switch_word_indices.append(int(word_idx))
            else:
                if args.language_id_method == "vocab":
                    en_ids = token_id_cache.get("english", [])
                    tl_ids = token_id_cache.get("target_language", [])
                    token_labels = label_tokens_vocab(
                        cs_token_ids=cs_ids,
                        en_token_ids=en_ids,
                        tl_token_ids=tl_ids,
                    )
                elif args.language_id_method == "script":
                    assert target_script is not None
                    token_texts = tokenizer.convert_ids_to_tokens(cs_ids)
                    token_labels = [
                        label_token_language(tok, target_script=target_script)
                        for tok in token_texts
                    ]
                else:
                    assert fasttext_model is not None
                    assert target_lang_code is not None
                    token_labels = label_tokens_sliding_window(
                        token_ids=cs_ids,
                        tokenizer=tokenizer,
                        fasttext_model=fasttext_model,
                        target_lang_code=target_lang_code,
                        window_size=int(args.window_size),
                        min_prob=float(args.fasttext_min_prob),
                    )
                for i in range(1, len(token_labels)):
                    prev, cur = token_labels[i - 1], token_labels[i]
                    if prev in {"english", "target"} and cur in {"english", "target"} and prev != cur:
                        switch_positions.append(i)
                        switch_word_indices.append(None)

            if not switch_positions:
                skipped.append({"source_id": source_id, "reason": "no_switch_point"})
                continue
            # Position-aligned anchors: compare CS[p] vs baseline[p] for every baseline.
            tl_baseline = next(
                (b for b in valid_baselines if b != "english"), valid_baselines[0]
            )
            all_valid = True
            anchor_maps: list[dict[str, dict[int, tuple[int, int]]]] = []
            for switch_pos, switch_word_idx in zip(switch_positions, switch_word_indices):
                anchor_map: dict[str, dict[int, tuple[int, int]]] = {}
                for baseline in valid_baselines:
                    base_anchor = int(switch_pos)
                    if args.switch_detection_level == "word" and switch_word_idx is not None:
                        baseline_words = word_cache.get(baseline, [])
                        if switch_word_idx < len(baseline_words):
                            base_anchor = int(baseline_words[switch_word_idx]["token_start"])
                    offsets_for_pair: dict[int, tuple[int, int]] = {}
                    for rel_off in args.token_offsets:
                        fp = switch_pos + rel_off
                        bp = base_anchor + rel_off
                        if (
                            0 <= fp < seq_len_cache[args.focus_condition]
                            and 0 <= bp < seq_len_cache[baseline]
                        ):
                            offsets_for_pair[rel_off] = (fp, bp)
                    anchor_map[baseline] = offsets_for_pair
                    event_rows.append(
                        {
                            "source_id": source_id,
                            "focus_condition": args.focus_condition,
                            "baseline_condition": baseline,
                            "focus_event_token_index": switch_pos,
                            "baseline_event_token_index": base_anchor,
                            "switch_token_id": int(cs_ids[switch_pos]),
                            "focus_event_word_index": switch_word_idx,
                            "tl_baseline": tl_baseline,
                        }
                    )
                anchor_maps.append(anchor_map)
        else:
            # ------------------------------------------------------------------
            # Original prefix mode: one switch point per sample.
            # ------------------------------------------------------------------
            prefix_lengths: dict[str, int] = {}
            for baseline in valid_baselines:
                base_ids = token_id_cache[baseline]
                pos = 0
                while pos < len(cs_ids) and pos < len(base_ids) and cs_ids[pos] == base_ids[pos]:
                    pos += 1
                prefix_lengths[baseline] = pos

            tl_baseline = max(prefix_lengths, key=lambda b: prefix_lengths[b])
            switch_pos = prefix_lengths[tl_baseline]

            if switch_pos >= min(len(cs_ids), len(token_id_cache[tl_baseline])):
                skipped.append({"source_id": source_id, "reason": "no_switch_point"})
                continue

            switch_token_id = cs_ids[switch_pos]

            all_valid = True
            anchor_map = {}
            for baseline in valid_baselines:
                base_ids = token_id_cache[baseline]
                focus_anchor = int(switch_pos)

                if baseline == tl_baseline:
                    base_anchor = int(switch_pos)
                else:
                    try:
                        base_anchor = base_ids.index(switch_token_id)
                    except ValueError:
                        skipped.append(
                            {"source_id": source_id, "reason": f"switch_token_not_in_{baseline}"}
                        )
                        all_valid = False
                        break

                event_rows.append(
                    {
                        "source_id": source_id,
                        "focus_condition": args.focus_condition,
                        "baseline_condition": baseline,
                        "focus_event_token_index": focus_anchor,
                        "baseline_event_token_index": base_anchor,
                        "switch_token_id": switch_token_id,
                        "tl_baseline": tl_baseline,
                    }
                )

                offsets_for_pair = {}
                for rel_off in args.token_offsets:
                    fp = focus_anchor + rel_off
                    bp = base_anchor + rel_off
                    if (
                        0 <= fp < seq_len_cache[args.focus_condition]
                        and 0 <= bp < seq_len_cache[baseline]
                    ):
                        offsets_for_pair[rel_off] = (fp, bp)

                anchor_map[baseline] = offsets_for_pair

            if not all_valid:
                continue
            anchor_maps = [anchor_map]

        # ------------------------------------------------------------------
        # For each switch point and each offset, build the switch-specific
        # boolean mask, then accumulate counts.
        #
        # A neuron is "switch-specific" for a given event if its activation in
        # the code-switched text is strictly greater than its activation in
        # EVERY monolingual baseline at the corresponding switch position.
        # In 'script' mode there may be multiple switch points per sample,
        # each treated as an independent event.
        # ------------------------------------------------------------------
        focus_acts = act_cache[args.focus_condition]
        common_layers = set(focus_acts.keys())
        for baseline in valid_baselines:
            common_layers &= set(act_cache[baseline].keys())

        any_event_contributed = False
        for anchor_map in anchor_maps:
            # An offset is usable only if every baseline has a valid position for it.
            usable_offsets = [
                rel_off
                for rel_off in args.token_offsets
                if all(rel_off in anchor_map[b] for b in valid_baselines)
            ]

            if not usable_offsets:
                continue

            for rel_off in usable_offsets:
                offset_contributed = False

                for layer in sorted(common_layers):
                    # n_neurons determined from the first position of the focus activations.
                    n_neurons = focus_acts[layer][0].shape[0]
                    # Start with every neuron as a candidate.
                    switch_mask = np.ones(n_neurons, dtype=bool)

                    # Collect per-baseline activations to build switch mask and accumulate means.
                    baseline_vecs: dict[str, np.ndarray] = {}
                    focus_vecs_for_layer: list[np.ndarray] = []

                    for baseline in valid_baselines:
                        focus_pos, base_pos = anchor_map[baseline][rel_off]
                        focus_vec = focus_acts[layer][focus_pos].astype(np.float64)
                        base_vec = act_cache[baseline][layer][base_pos].astype(np.float64)
                        # Neuron must be strictly more active in cs than this baseline.
                        switch_mask &= focus_vec > base_vec
                        baseline_vecs[baseline] = base_vec
                        focus_vecs_for_layer.append(focus_vec)

                    key = (int(rel_off), int(layer))
                    if key not in count_map:
                        count_map[key] = np.zeros(n_neurons, dtype=np.int64)
                    count_map[key] += switch_mask.astype(np.int64)
                    offset_contributed = True

                    # Accumulate split activation sums.
                    mean_focus_vec = np.mean(focus_vecs_for_layer, axis=0)
                    spec_mask_f = switch_mask.astype(np.float64)
                    nonspec_mask_f = (~switch_mask).astype(np.float64)
                    for cond, vec in [(args.focus_condition, mean_focus_vec)] + list(baseline_vecs.items()):
                        act_key = (int(rel_off), int(layer), str(cond))
                        if act_key not in act_spec_sum:
                            act_spec_sum[act_key] = np.zeros(n_neurons, dtype=np.float64)
                            act_nonspec_sum[act_key] = np.zeros(n_neurons, dtype=np.float64)
                        act_spec_sum[act_key] += vec * spec_mask_f
                        act_nonspec_sum[act_key] += vec * nonspec_mask_f

                # Count this event once per offset once at least one layer was processed.
                if offset_contributed:
                    total_map[rel_off] = total_map.get(rel_off, 0) + 1
                    any_event_contributed = True

        if not any_event_contributed:
            skipped.append({"source_id": source_id, "reason": "no_usable_offsets"})

        if bool(args.gpu_friendly) and device.type == "cuda" and idx % 64 == 0:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save event alignments and skipped log.
    # ------------------------------------------------------------------
    if event_rows:
        pd.DataFrame(event_rows).to_csv(tables_dir / "event_alignments.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(tables_dir / "skipped_samples.csv", index=False)

    # ------------------------------------------------------------------
    # Build the per-neuron count table and the consistency-filtered table.
    # ------------------------------------------------------------------
    all_rows: list[pd.DataFrame] = []
    consistent_rows: list[pd.DataFrame] = []

    # Condition name → short label used in output column names.
    _cond_label = {
        args.focus_condition: "cs",
        "english": "eng",
        "target_language": "tgt",
    }

    for (rel_off, layer), counts in sorted(count_map.items()):
        n_total = total_map.get(rel_off, 0)
        if n_total == 0:
            continue

        n_neurons = len(counts)
        neurons = np.arange(n_neurons, dtype=int)
        consistency = counts / float(n_total)
        n_specific = counts.astype(np.float64)          # per-neuron count of switch-specific samples
        n_nonspecific = float(n_total) - n_specific     # per-neuron count of non-specific samples

        frame = pd.DataFrame(
            {
                "relative_offset": int(rel_off),
                "layer": int(layer),
                "neuron": neurons,
                "n_samples_switch_specific": counts,
                "n_samples_total": n_total,
                "consistency_fraction": consistency,
            }
        )

        # Add per-condition activation columns, split by switch-specificity.
        # For each condition three columns are emitted:
        #   mean_<cond>_activation         — unconditional mean across all samples
        #   mean_<cond>_act_switch_on      — mean only for samples where neuron WAS switch-specific
        #   mean_<cond>_act_switch_off     — mean only for samples where neuron was NOT switch-specific
        for cond, label in _cond_label.items():
            act_key = (int(rel_off), int(layer), cond)
            if act_key not in act_spec_sum:
                for col in [f"mean_{label}_activation", f"mean_{label}_act_switch_on", f"mean_{label}_act_switch_off"]:
                    frame[col] = np.nan
                continue

            spec_s = act_spec_sum[act_key]
            nonspec_s = act_nonspec_sum[act_key]

            # Unconditional mean (sum of both splits / n_total).
            frame[f"mean_{label}_activation"] = (spec_s + nonspec_s) / float(n_total)

            # Conditional mean when switch-specific (nan where neuron was never specific).
            with np.errstate(invalid="ignore"):
                frame[f"mean_{label}_act_switch_on"] = np.where(
                    n_specific > 0, spec_s / np.maximum(n_specific, 1), np.nan
                )
                frame[f"mean_{label}_act_switch_off"] = np.where(
                    n_nonspecific > 0, nonspec_s / np.maximum(n_nonspecific, 1), np.nan
                )

        all_rows.append(frame)

        mask = consistency >= float(args.min_consistency_fraction)
        if mask.any():
            consistent_rows.append(frame[mask].copy())

    # Full counts table (every neuron, every offset/layer).
    _activation_cols = [
        "mean_cs_activation", "mean_cs_act_switch_on", "mean_cs_act_switch_off",
        "mean_eng_activation", "mean_eng_act_switch_on", "mean_eng_act_switch_off",
        "mean_tgt_activation", "mean_tgt_act_switch_on", "mean_tgt_act_switch_off",
    ]
    all_neurons_df = (
        pd.concat(all_rows, ignore_index=True)
        if all_rows
        else pd.DataFrame(
            columns=[
                "relative_offset", "layer", "neuron",
                "n_samples_switch_specific", "n_samples_total", "consistency_fraction",
            ] + _activation_cols
        )
    )
    if not all_neurons_df.empty:
        all_neurons_df = all_neurons_df.sort_values(
            ["relative_offset", "layer", "neuron"]
        ).reset_index(drop=True)

    all_neurons_df.to_csv(
        tables_dir / "switch_neuron_counts.csv.gz",
        index=False,
        compression="gzip",
    )

    # Consistent-neurons table (filtered, sorted by consistency desc).
    consistent_df = (
        pd.concat(consistent_rows, ignore_index=True)
        if consistent_rows
        else pd.DataFrame(
            columns=[
                "relative_offset", "layer", "neuron",
                "n_samples_switch_specific", "n_samples_total", "consistency_fraction",
            ] + _activation_cols
        )
    )
    if not consistent_df.empty:
        consistent_df = consistent_df.sort_values(
            ["relative_offset", "consistency_fraction", "n_samples_switch_specific"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    consistent_df.to_csv(tables_dir / "consistent_switch_neurons.csv", index=False)

    # ------------------------------------------------------------------
    # Per-offset summary table.
    # ------------------------------------------------------------------
    summary_rows: list[dict] = []
    for rel_off in sorted(total_map.keys()):
        n_total = total_map[rel_off]
        sub = all_neurons_df[all_neurons_df["relative_offset"] == rel_off]
        if sub.empty:
            continue
        n_consistent = int(
            (sub["consistency_fraction"] >= float(args.min_consistency_fraction)).sum()
        )
        summary_rows.append(
            {
                "relative_offset": int(rel_off),
                "n_samples_total": n_total,
                "n_neuron_layer_pairs_total": int(len(sub)),
                "n_consistent_neuron_layer_pairs": n_consistent,
                "pct_consistent": round(100.0 * n_consistent / max(len(sub), 1), 4),
                "max_consistency_fraction": float(sub["consistency_fraction"].max()),
                "mean_consistency_fraction": float(sub["consistency_fraction"].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(tables_dir / "switch_neuron_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Heatmaps of consistency fraction and per-offset consistent-neuron tables.
    # ------------------------------------------------------------------
    if not all_neurons_df.empty:
        for rel_off, sub in all_neurons_df.groupby("relative_offset"):
            write_neuron_heatmap(
                sub,
                value_col="consistency_fraction",
                title=f"Switch-specific consistency offset {int(rel_off):+d}",
                out_path=figures_dir
                / f"switch_consistency_offset_{format_offset(int(rel_off))}_heatmap.html",
            )

    if not consistent_df.empty:
        for rel_off, sub in consistent_df.groupby("relative_offset"):
            sub.sort_values(
                ["consistency_fraction", "n_samples_switch_specific"],
                ascending=[False, False],
            ).to_csv(
                tables_dir
                / f"consistent_neurons_offset_{format_offset(int(rel_off))}.csv",
                index=False,
            )

    # ------------------------------------------------------------------
    # Run summary JSON.
    # ------------------------------------------------------------------
    run_summary = {
        "experiment": "exp2_switch_point_activations",
        "dataset_csv": str(args.dataset_csv),
        "model_name": str(args.model_name),
        "focus_condition": str(args.focus_condition),
        "baseline_conditions": list(args.baseline_conditions),
        "token_offsets": [int(x) for x in args.token_offsets],
        "switch_detection_level": str(args.switch_detection_level),
        "language_id_method": str(args.language_id_method),
        "fasttext_model_path": str(args.fasttext_model_path) if args.fasttext_model_path else None,
        "fasttext_min_prob": float(args.fasttext_min_prob),
        "window_size": int(args.window_size),
        "target_lang_code": str(target_lang_code) if target_lang_code else None,
        "min_consistency_fraction": float(args.min_consistency_fraction),
        "gpu_friendly": bool(args.gpu_friendly),
        "n_source_groups_scanned": int(len(source_items)),
        "n_event_pairs": int(len(event_rows)),
        "n_skipped": int(len(skipped)),
        "n_consistent_neuron_layer_pairs": int(len(consistent_df)),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
