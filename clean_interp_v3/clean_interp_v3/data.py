from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"id", "text", "condition", "domain"}


@dataclass(frozen=True)
class MatchedGroupInfo:
    comparison_name: str
    focus_condition: str
    baseline_condition: str
    n_pairs: int


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("Input must be .csv, .jsonl, or .json")

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {sorted(missing)}")

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].astype(str)
    df["condition"] = df["condition"].astype(str)
    df["domain"] = df["domain"].astype(str)

    if "source_id" not in df.columns:
        df["source_id"] = [
            _derive_source_id(row_id, condition)
            for row_id, condition in zip(df["id"], df["condition"])
        ]
    else:
        df["source_id"] = df["source_id"].astype(str)

    return df


def _derive_source_id(row_id: str, condition: str) -> str:
    suffix = f"_{condition}"
    if row_id.endswith(suffix):
        return row_id[: -len(suffix)]
    return row_id


def collect_required_conditions(comparisons) -> set[str]:
    required: set[str] = set()
    for comp in comparisons:
        required.add(comp.focus_condition)
        required.add(comp.baseline_condition)
    return required


def build_group_index(df: pd.DataFrame) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in df.itertuples(index=False):
        grouped.setdefault(row.source_id, {})[row.condition] = {
            "id": row.id,
            "text": row.text,
            "domain": row.domain,
            "condition": row.condition,
            "source_id": row.source_id,
        }
    return grouped


def matched_group_counts(group_index: dict[str, dict[str, dict[str, str]]], comparisons) -> list[MatchedGroupInfo]:
    rows: list[MatchedGroupInfo] = []
    for comp in comparisons:
        n_pairs = sum(
            1
            for _, cond_map in group_index.items()
            if comp.focus_condition in cond_map and comp.baseline_condition in cond_map
        )
        rows.append(
            MatchedGroupInfo(
                comparison_name=comp.name,
                focus_condition=comp.focus_condition,
                baseline_condition=comp.baseline_condition,
                n_pairs=n_pairs,
            )
        )
    return rows
