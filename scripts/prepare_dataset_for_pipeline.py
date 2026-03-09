#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MAPPING = {
    "english": "eng",
    "target_language": "fr",
    "code_switched": "codeswitching",
    "confused": "language_confusion",
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
            raise ValueError("JSON dict did not contain a list under records/data/items/examples")
    else:
        raise ValueError("Input JSON must be a list or dict containing a list")

    if not records:
        raise ValueError("Input JSON contains no records")
    if not isinstance(records[0], dict):
        raise ValueError("Input list elements must be objects")
    return records


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert dataset JSON into pipeline format: id,text,condition,domain"
    )
    p.add_argument("--input", required=True, help="Path to source JSON file")
    p.add_argument("--output", required=True, help="Path to output CSV file")
    p.add_argument(
        "--target-lang-key",
        default="fr",
        help="Source key for target-language text (default: fr)",
    )
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    records = _load_records(in_path)

    mapping = {
        "english": "eng",
        args.target_lang_label: args.target_lang_key,
        "code_switched": "codeswitching",
        "confused": "language_confusion",
    }

    required_source_keys = {"eng", args.target_lang_key, "codeswitching", "language_confusion", "domain"}
    missing_source = sorted(k for k in required_source_keys if k not in records[0])
    if missing_source:
        raise ValueError(f"Missing required source keys in first record: {missing_source}")

    out_rows: list[dict[str, Any]] = []
    dropped_incomplete_records = 0
    skipped_condition_rows = 0

    for idx, rec in enumerate(records, start=1):
        source_id = rec.get("id", idx)
        domain = _clean_text(rec.get("domain", "unknown")) or "unknown"
        strategy = _clean_text(rec.get("confusion_strategy", ""))

        condition_texts: dict[str, str] = {}
        for condition, source_key in mapping.items():
            condition_texts[condition] = _clean_text(rec.get(source_key))

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
                    "id": f"{source_id}_{condition}",
                    "text": text,
                    "condition": condition,
                    "domain": domain,
                    "source_id": source_id,
                    "confusion_strategy": strategy,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "text", "condition", "domain", "source_id", "confusion_strategy"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    condition_counts = Counter(r["condition"] for r in out_rows)
    domain_counts = Counter(r["domain"] for r in out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")
    print(f"Source records in JSON: {len(records)}")
    print(f"Dropped incomplete source records: {dropped_incomplete_records}")
    print(f"Skipped condition rows (only when --allow-incomplete): {skipped_condition_rows}")
    print("Condition counts:")
    for c, n in sorted(condition_counts.items()):
        print(f"  - {c}: {n}")
    print(f"Unique domains: {len(domain_counts)}")


if __name__ == "__main__":
    main()
