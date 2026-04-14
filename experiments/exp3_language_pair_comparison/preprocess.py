#!/usr/bin/env python3
"""
preprocess.py  Tokenize dataset and label tokens with language using Lingua.

Reads:
  - combined_dataset.csv (id, text, condition, domain)

For each sentence:
  - Tokenizes using the model tokenizer
  - Labels each word with Lingua, then maps labels back to tokens via char offsets
  - Identifies switch positions (token indices where language changes)

Outputs:
  - combined_dataset_preprocessed.csv
    (id, text, condition, domain, token_ids, token_strings, token_lang_labels, switch_positions)
    where list columns are stored as JSON strings

Usage:
  python preprocess.py --input combined_dataset.csv --model_name modelname --out combined_dataset_preprocessed_modelname.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from lingua import Language, LanguageDetectorBuilder


#  Language config 

CONDITION_UNIFORM_LABEL = {
    "english": "EN",
    "french":  "FR",
    "hindi":   "HI",
}

SWITCHED_CONDITIONS = {"cs_fr", "cs_hi"}

CONDITION_LANGUAGES = {
    "cs_fr": [Language.ENGLISH, Language.FRENCH],
    "cs_hi": [Language.ENGLISH, Language.HINDI],
}

LINGUA_TO_LABEL = {
    Language.ENGLISH: "EN",
    Language.FRENCH:  "FR",
    Language.HINDI:   "HI",
}


#  Word-level labelling 

def label_words(text: str, detector) -> list[tuple[int, int, str]]:
    """Label each word in text with a language label.

    Returns list of (start_char, end_char, label) for each word.
    Uses Lingua on the full word string  works for both Latin and Devanagari.
    """
    # Split on whitespace, keeping track of character positions
    word_spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', text)]

    labelled = []
    prev_label = "EN"
    for start, end, word in word_spans:
        detected = detector.detect_language_of(word)
        label = LINGUA_TO_LABEL.get(detected, prev_label)
        prev_label = label
        labelled.append((start, end, label))

    return labelled


def map_tokens_to_labels(
    offsets: list[tuple[int, int]],
    word_labels: list[tuple[int, int, str]],
    default_label: str = "EN",
) -> list[str]:
    """Map token char offsets to word-level language labels.

    For each token, find which word it overlaps with and assign that word's label.
    Special tokens have offset (0, 0)  they get the default label.
    """
    token_labels = []

    for tok_start, tok_end in offsets:
        # Special token (e.g. <s>, </s>)
        if tok_start == 0 and tok_end == 0:
            token_labels.append(default_label)
            continue

        # Find the word this token overlaps with
        matched_label = default_label
        for w_start, w_end, label in word_labels:
            # Token overlaps with word if ranges intersect
            if tok_start < w_end and tok_end > w_start:
                matched_label = label
                break

        token_labels.append(matched_label)

    return token_labels


def get_switch_positions(labels: list[str]) -> list[int]:
    """Return token indices where the language label changes."""
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


#  Main 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenize and label dataset tokens with language.")
    p.add_argument("--input",      required=True,  help="Path to combined_dataset.csv")
    p.add_argument("--model_name", default="Qwen/Qwen2-7B-Instruct", help="HuggingFace model name")
    p.add_argument("--out",        default="combined_dataset_preprocessed.csv")
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading dataset from {input_path} ...")
    df = pd.read_csv(input_path)

    # Remove known bad row (too short, no switching)
    df = df[df["id"] != "fr_eng_s_e_s_n:835"].reset_index(drop=True)

    print(f"  {len(df):,} rows  |  conditions: {sorted(df['condition'].unique())}")

    print(f"\nLoading tokenizer: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building Lingua detectors ...")
    detectors = {
        cond: LanguageDetectorBuilder.from_languages(*langs).build()
        for cond, langs in CONDITION_LANGUAGES.items()
    }
    print(f"  Built detectors for: {list(detectors.keys())}")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        text      = str(row["text"])
        condition = str(row["condition"])

        # Tokenize with char offset mapping
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
        )
        token_ids     = encoded["input_ids"]
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)
        offsets       = encoded["offset_mapping"]  # [(start_char, end_char), ...]

        # Label tokens
        if condition in CONDITION_UNIFORM_LABEL:
            uniform           = CONDITION_UNIFORM_LABEL[condition]
            token_lang_labels = [uniform] * len(token_strings)
            switch_positions  = []

        elif condition in SWITCHED_CONDITIONS:
            detector   = detectors[condition]
            word_labels = label_words(text, detector)
            token_lang_labels = map_tokens_to_labels(offsets, word_labels)
            switch_positions  = get_switch_positions(token_lang_labels)

        else:
            token_lang_labels = ["EN"] * len(token_strings)
            switch_positions  = []

        results.append({
            "id":                row["id"],
            "text":              text,
            "condition":         condition,
            "domain":            row["domain"],
            "token_ids":         json.dumps(token_ids),
            "token_strings":     json.dumps(token_strings),
            "token_lang_labels": json.dumps(token_lang_labels),
            "switch_positions":  json.dumps(switch_positions),
        })

    out_df   = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    # Summary
    cs_rows      = out_df[out_df["condition"].isin(SWITCHED_CONDITIONS)]
    switch_counts = cs_rows["switch_positions"].apply(lambda x: len(json.loads(x)))

    print(f"\nDone. {len(out_df):,} rows written to {out_path}")
    print(f"\nSwitch point stats (cs_fr + cs_hi rows only):")
    print(f"  Mean switches per sentence : {switch_counts.mean():.2f}")
    print(f"  Max switches per sentence  : {switch_counts.max()}")
    print(f"  Min switches per sentence  : {switch_counts.min()}")
    print(f"  Sentences with 0 switches  : {(switch_counts == 0).sum()}")
    print(f"\nSwitches by condition:")
    for cond in SWITCHED_CONDITIONS:
        cond_counts = out_df[out_df["condition"] == cond]["switch_positions"].apply(
            lambda x: len(json.loads(x))
        )
        print(f"  {cond:<8}: mean={cond_counts.mean():.2f}  min={cond_counts.min()}  max={cond_counts.max()}  zeros={( cond_counts==0).sum()}")


if __name__ == "__main__":
    main()

    
