#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


KEY_ALIASES = {
    "english": ["eng", "english"],
    "target_language": ["fr", "hindi", "hi", "target_language"],
    "code_switched": ["codeswitching", "code_switched", "hinglish"],
    "confused": ["language_confusion", "confused"],
    "domain": ["domain", "topic", "source"],
    "strategy": ["confusion_strategy", "type_of_confusion"],
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        for candidate in ["records", "data", "items", "examples"]:
            if candidate in obj and isinstance(obj[candidate], list):
                records = obj[candidate]
                break
        else:
            raise ValueError(f"{path}: JSON dict did not contain a list under records/data/items/examples")
    else:
        raise ValueError(f"{path}: Input JSON must be a list or dict containing a list")

    if not records:
        raise ValueError(f"{path}: Input JSON contains no records")
    if not isinstance(records[0], dict):
        raise ValueError(f"{path}: Input list elements must be objects")
    return records


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_key(rec: dict[str, Any], preferred: str | None, aliases: list[str]) -> str | None:
    if preferred and preferred in rec:
        return preferred
    for k in aliases:
        if k in rec:
            return k
    return None


def _iter_input_files(in_path: Path) -> list[Path]:
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
        if not files:
            raise ValueError(f"No .json files found in directory: {in_path}")
        return files
    raise ValueError(f"Input path does not exist: {in_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert JSON file/folder into pipeline format: id,text,condition,domain"
    )
    p.add_argument("--input", required=True, help="Path to source JSON file OR directory of JSON files")
    p.add_argument("--output", required=True, help="Path to output CSV file")

    p.add_argument("--english-key", default=None, help="Override source key for English text")
    p.add_argument("--target-lang-key", default=None, help="Override source key for target-language text")
    p.add_argument("--code-switched-key", default=None, help="Override source key for code-switched text")
    p.add_argument("--confused-key", default=None, help="Override source key for confused text")
    p.add_argument("--domain-key", default=None, help="Override source key for domain")
    p.add_argument("--strategy-key", default=None, help="Override source key for confusion strategy")

    p.add_argument(
        "--target-lang-label",
        default="target_language",
        help="Condition label to use for target-language rows",
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow records missing one or more condition texts (not recommended for balanced comparisons)",
    )
    p.add_argument(
        "--domain-from-filename",
        action="store_true",
        help="Use JSON filename stem as domain if domain key is missing/empty",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    files = _iter_input_files(in_path)

    out_rows: list[dict[str, Any]] = []
    dropped_incomplete_records = 0
    skipped_condition_rows = 0
    total_source_records = 0

    for file_path in files:
        records = _load_records(file_path)
        total_source_records += len(records)

        first = records[0]
        english_key = _resolve_key(first, args.english_key, KEY_ALIASES["english"])
        target_lang_key = _resolve_key(first, args.target_lang_key, KEY_ALIASES["target_language"])
        cs_key = _resolve_key(first, args.code_switched_key, KEY_ALIASES["code_switched"])
        conf_key = _resolve_key(first, args.confused_key, KEY_ALIASES["confused"])
        domain_key = _resolve_key(first, args.domain_key, KEY_ALIASES["domain"])
        strategy_key = _resolve_key(first, args.strategy_key, KEY_ALIASES["strategy"])

        missing_required = []
        for label, key in [
            ("english", english_key),
            ("target_language", target_lang_key),
            ("code_switched", cs_key),
            ("confused", conf_key),
        ]:
            if key is None:
                missing_required.append(label)
        if missing_required:
            raise ValueError(
                f"{file_path}: missing keys for {missing_required}. "
                f"Use explicit --*-key args to map your schema."
            )

        for idx, rec in enumerate(records, start=1):
            source_id = rec.get("id", idx)
            strategy = _clean_text(rec.get(strategy_key, "")) if strategy_key else ""

            domain = ""
            if domain_key:
                domain = _clean_text(rec.get(domain_key, ""))
            if (not domain) and args.domain_from_filename:
                domain = file_path.stem
            if not domain:
                domain = "unknown"

            condition_texts: dict[str, str] = {
                "english": _clean_text(rec.get(english_key, "")),
                args.target_lang_label: _clean_text(rec.get(target_lang_key, "")),
                "code_switched": _clean_text(rec.get(cs_key, "")),
                "confused": _clean_text(rec.get(conf_key, "")),
            }

            missing_conditions = [c for c, text in condition_texts.items() if not text]
            if missing_conditions and not args.allow_incomplete:
                dropped_incomplete_records += 1
                continue

            for condition, text in condition_texts.items():
                if not text:
                    skipped_condition_rows += 1
                    continue

                out_rows.append(
                    {
                        "id": f"{file_path.stem}:{source_id}_{condition}",
                        "text": text,
                        "condition": condition,
                        "domain": domain,
                        "source_id": source_id,
                        "source_file": file_path.name,
                        "confusion_strategy": strategy,
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "text", "condition", "domain", "source_id", "source_file", "confusion_strategy"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    condition_counts = Counter(r["condition"] for r in out_rows)
    domain_counts = Counter(r["domain"] for r in out_rows)

    print(f"Input files processed: {len(files)}")
    print(f"Wrote {len(out_rows)} rows to {out_path}")
    print(f"Source records total: {total_source_records}")
    print(f"Dropped incomplete source records: {dropped_incomplete_records}")
    print(f"Skipped condition rows (only when --allow-incomplete): {skipped_condition_rows}")
    print("Condition counts:")
    for c, n in sorted(condition_counts.items()):
        print(f"  - {c}: {n}")
    print(f"Unique domains: {len(domain_counts)}")


if __name__ == "__main__":
    main()
