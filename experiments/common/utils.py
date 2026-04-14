from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


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


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    idx = 0
    while idx < n and a[idx] == b[idx]:
        idx += 1
    return idx


def _contains_script_char(ch: str, ranges: Sequence[tuple[int, int]]) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


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


def _clean_token(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").strip()


def label_token_language(token: str, target_script: str) -> str:
    token = _clean_token(token)
    if not token:
        return "other"
    latin_ranges = SCRIPT_RANGES["latin"]
    target_ranges = SCRIPT_RANGES.get(target_script, [])
    has_latin = any(_contains_script_char(ch, latin_ranges) for ch in token)
    has_target = any(_contains_script_char(ch, target_ranges) for ch in token)
    if has_latin and not has_target:
        return "english"
    if has_target and not has_latin:
        return "target"
    if has_latin and has_target:
        return "mixed"
    return "other"


def load_fasttext_model(model_path: str | Path) -> Any:
    try:
        import fasttext
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fasttext is required for --language_id_method fasttext. "
            "Install with: python3 -m pip install fasttext"
        ) from exc
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"FastText model file not found: {path}")
    return fasttext.load_model(str(path))


def _fasttext_label_to_lang(label: str) -> str:
    prefix = "__label__"
    if label.startswith(prefix):
        return label[len(prefix) :]
    return label


def infer_target_language_code_fasttext(
    df: pd.DataFrame,
    *,
    target_label: str,
    model: Any,
) -> str:
    subset = df[df["condition"] == target_label]
    if subset.empty:
        return "unknown"

    counts: dict[str, int] = defaultdict(int)
    for text in subset["text"].astype(str).head(500):
        labels, _probs = model.predict(text.replace("\n", " "), k=1)
        if not labels:
            continue
        lang = _fasttext_label_to_lang(str(labels[0]))
        counts[lang] += 1

    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda item: item[1])[0]


def label_token_language_fasttext(
    token: str,
    *,
    model: Any,
    target_lang_code: str,
    english_lang_code: str = "en",
    min_prob: float = 0.0,
) -> str:
    clean = _clean_token(token)
    if not clean:
        return "other"
    labels, probs = model.predict(clean, k=1)
    if not labels:
        return "other"
    lang = _fasttext_label_to_lang(str(labels[0]))
    prob = float(probs[0]) if probs else 0.0
    if prob < float(min_prob):
        return "other"
    if lang == english_lang_code:
        return "english"
    if target_lang_code != "unknown" and lang == target_lang_code:
        return "target"
    return "other"
